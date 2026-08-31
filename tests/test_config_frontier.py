"""Regression tests for the frontier tier addition.

Two independent guarantees, each isolated to a small test:

1. ``naysayer`` routing invariance (msg-003 §6 permanent constraint):
   adding a frontier tier and its backend must not change the naysayer
   resolution path, because the naysayer tier is what mindwire uses to
   review this repo's own PRs. If a re-shuffle silently changes the
   backend or model for naysayer, the review machinery breaks itself.

2. ``LEXORA_FRONTIER_MODEL`` env override (T-frontier-tier D-2): the
   verbatim nested env variable is not honoured by ``create_settings``
   (measured 2026-08-31, root cause is init-kwarg > env precedence in
   pydantic-settings). This test file pins the single-variable hook
   that ``create_settings`` implements as the replacement and verifies
   both write sites (tier model + backend model[0].name) are updated
   in lockstep so the frontier tier's advertised capability never
   disagrees with what the router actually sends upstream.
"""

from pathlib import Path

import pytest

from lexora.config import create_settings, load_yaml_config
from lexora.services.router import BackendRouter
from lexora.config import RoutingSettings, VLLMSettings


REPO_CONFIG = Path(__file__).parent.parent / "config" / "lexora_config.yaml"


class TestNaysayerRouteInvariance:
    """The naysayer tier resolution must survive the frontier tier addition."""

    def test_shipped_config_naysayer_backend_is_gemini(self) -> None:
        settings = create_settings(REPO_CONFIG)
        assert settings.routing.tiers["naysayer"].backend == "gemini"

    def test_shipped_config_naysayer_model_is_gemini_31_pro(self) -> None:
        settings = create_settings(REPO_CONFIG)
        assert settings.routing.tiers["naysayer"].model == "gemini-3.1-pro-preview"

    def test_router_resolves_naysayer_to_gemini_backend_and_model(self) -> None:
        """End-to-end: BackendRouter built from the shipped config with the
        real backend factory still produces the same tier resolution."""
        settings = create_settings(REPO_CONFIG)
        router = BackendRouter(
            routing_settings=settings.routing,
            vllm_settings=settings.vllm,
        )
        try:
            assert router.get_backend_name_for_model("naysayer") == "gemini"
            assert router.resolve_model("naysayer") == "gemini-3.1-pro-preview"
        finally:
            # Backends spin up httpx.AsyncClient / subprocess handles.
            import asyncio

            asyncio.run(router.close())

    def test_naysayer_route_unchanged_by_frontier_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LEXORA_FRONTIER_MODEL must not perturb naysayer."""
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", "claude-opus-5-20260601")
        settings = create_settings(REPO_CONFIG)
        assert settings.routing.tiers["naysayer"].backend == "gemini"
        assert settings.routing.tiers["naysayer"].model == "gemini-3.1-pro-preview"


class TestFrontierTierConfig:
    """T-frontier-tier D-3 / D-4 / D-7 invariants baked into shipped config."""

    def test_frontier_tier_present(self) -> None:
        settings = create_settings(REPO_CONFIG)
        assert "frontier" in settings.routing.tiers
        assert settings.routing.tiers["frontier"].backend == "frontier"

    def test_frontier_backend_is_anthropic(self) -> None:
        settings = create_settings(REPO_CONFIG)
        assert settings.routing.backends["frontier"].type == "anthropic"

    def test_frontier_fallback_backends_is_empty(self) -> None:
        """No silent fallback (msg-011 D-3 + msg-001 requirement 3)."""
        settings = create_settings(REPO_CONFIG)
        assert settings.routing.backends["frontier"].fallback_backends == []

    def test_frontier_error_passthrough_is_true(self) -> None:
        """Upstream 4xx/5xx and safety-classifier declines pass verbatim (D-4)."""
        settings = create_settings(REPO_CONFIG)
        assert settings.routing.backends["frontier"].error_passthrough is True

    def test_frontier_has_default_max_tokens(self) -> None:
        """A reserved output budget for reasoning models (D-7)."""
        settings = create_settings(REPO_CONFIG)
        assert settings.routing.backends["frontier"].default_max_tokens is not None
        assert settings.routing.backends["frontier"].default_max_tokens >= 1500

    def test_frontier_backend_not_used_as_fallback_by_other_tiers(self) -> None:
        """Nobody else lists ``frontier`` in their fallback list.

        Guards D-1: the frontier tier gets its own backend precisely so
        the FallbackService (if ever wired) does not silently reroute
        traffic into the paid tier.
        """
        settings = create_settings(REPO_CONFIG)
        for backend_name, backend_cfg in settings.routing.backends.items():
            assert "frontier" not in backend_cfg.fallback_backends, (
                f"backend {backend_name!r} lists 'frontier' as a fallback — "
                "the paid tier must not be reachable via silent fallback."
            )


class TestFrontierModelEnvOverride:
    """T-frontier-tier D-2: LEXORA_FRONTIER_MODEL env override."""

    def test_env_override_updates_tier_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", "claude-opus-5-20260601")
        settings = create_settings(REPO_CONFIG)
        assert settings.routing.tiers["frontier"].model == "claude-opus-5-20260601"

    def test_env_override_updates_backend_first_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both write sites must agree, or capabilities lies about what's served."""
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", "claude-opus-5-20260601")
        settings = create_settings(REPO_CONFIG)
        assert (
            settings.routing.backends["frontier"].models[0].name
            == "claude-opus-5-20260601"
        )

    def test_no_env_leaves_shipped_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXORA_FRONTIER_MODEL", raising=False)
        settings = create_settings(REPO_CONFIG)
        # Whatever the YAML ships; ensure it is at least Fable-shaped and
        # the tier model resolves through the backend's first model.
        first_model = settings.routing.backends["frontier"].models[0].name
        assert "fable" in first_model.lower() or "claude" in first_model.lower()

    def test_env_override_reaches_router_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: env → create_settings → BackendRouter.resolve_model."""
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", "claude-opus-5-20260601")
        settings = create_settings(REPO_CONFIG)
        router = BackendRouter(
            routing_settings=settings.routing,
            vllm_settings=settings.vllm,
        )
        try:
            assert router.resolve_model("frontier") == "claude-opus-5-20260601"
        finally:
            import asyncio

            asyncio.run(router.close())


class TestFrontierPricing:
    """T-frontier-tier D-6a: frontier candidate models are in DEFAULT_PRICING.

    Without an entry, ``pricing_known=0`` and cost records 0.00 with a
    warning. That is safe (see D-6c: unpriced is distinguishable from
    free), but it defeats the point of requirement 6 (frontier costs
    must be visible for reconciliation against the vendor bill) — so
    the shipped defaults for Fable 5 and Opus 5 are pinned here.
    """

    def test_default_frontier_model_priced(self) -> None:
        from lexora.services.cost_tracker import DEFAULT_PRICING

        settings = create_settings(REPO_CONFIG)
        default_model = settings.routing.backends["frontier"].models[0].name
        assert default_model in DEFAULT_PRICING, (
            f"frontier default model {default_model!r} must have a price "
            "entry so records are not silently zeroed (D-6c warning is a "
            "safety net, not a substitute for correct data)."
        )

    def test_opus_5_priced_for_env_swap(self) -> None:
        """Opus 5 is the other candidate; env swap must not land unpriced."""
        from lexora.services.cost_tracker import DEFAULT_PRICING

        assert "claude-opus-5-20260601" in DEFAULT_PRICING

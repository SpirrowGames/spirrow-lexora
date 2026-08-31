"""Regression tests for the frontier tier addition.

Three independent guarantees, each isolated to a small test (the third was
added 2026-08-31 when the fallback machinery was removed — see
``TestFrontierCannotBeSilentlyDowngraded``):

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

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

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

    def test_frontier_error_passthrough_is_true(self) -> None:
        """Upstream 4xx/5xx and safety-classifier declines pass verbatim (D-4)."""
        settings = create_settings(REPO_CONFIG)
        assert settings.routing.backends["frontier"].error_passthrough is True

    def test_frontier_has_default_max_tokens(self) -> None:
        """A reserved output budget for reasoning models (D-7)."""
        settings = create_settings(REPO_CONFIG)
        assert settings.routing.backends["frontier"].default_max_tokens is not None
        assert settings.routing.backends["frontier"].default_max_tokens >= 1500


class TestFrontierCannotBeSilentlyDowngraded:
    """要件 3 の構造テスト — 「宣言」ではなく「機構が無い」ことで保証する。

    以前ここには 2 本のテストがあった: `frontier` の `fallback_backends` が
    空リストであること、および他のどの backend も `frontier` をフォールバック
    先に挙げていないこと。どちらも「設定がそうなっている」の確認であり、設定を
    書き換えれば破れた。

    2026-08-31 にフォールバック機構そのものを撤去した (T-frontier-tier msg-025
    R-1。PR #9 の独立 gate が「未配線の機構の config を出荷するのは運用者への
    偽の約束だ」として止めた結果)。∴ 保証は設定値からスキーマの性質へ移った —
    フォールバック先を書く場所が存在しない。空リストの宣言より強い。

    共通の不変条件テストは `tests/test_config_no_fallback.py`。ここは frontier
    の視点から「格下げされうる経路が無い」ことだけを固定する。
    """

    def test_fallback_service_module_does_not_exist(self) -> None:
        """`FallbackService` は撤去済み ∴ import 自体が失敗すること。"""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("lexora.services.fallback")

    def test_router_exposes_no_fallback_surface(self) -> None:
        """router にフォールバックを引く API が無いこと。

        あれば「誰かが将来呼ぶ」経路が残る = frontier が黙って格下げされうる。
        """
        assert not hasattr(BackendRouter, "get_fallback_backends")
        assert not hasattr(BackendRouter, "fallback_map")

    def test_frontier_config_with_fallback_backends_fails_to_load(
        self, tmp_path: Path
    ) -> None:
        """frontier にフォールバックを書き戻した config は起動できないこと。

        `BackendSettings` は extra="forbid" ∴ 削除済みキーは黙って無視される
        のではなく ValidationError になる。壊れ方が fail-loud であることが
        「どんな config でも格下げできない」の実体。
        """
        config_file = tmp_path / "lexora_config.yaml"
        config_file.write_text(
            """
routing:
  enabled: true
  default_backend: "frontier"
  backends:
    frontier:
      type: "anthropic"
      url: "https://api.anthropic.com"
      models:
        - name: "some-frontier-model"
      fallback_backends:
        - "heavy"
  tiers:
    frontier:
      backend: "frontier"
""",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            create_settings(config_file)


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


class TestFrontierPricingHonestUnpriced:
    """T-frontier-tier F-1 (msg-016): placeholder IDs must land unpriced.

    Original design tried to seed both Fable 5 and Opus 5 in
    DEFAULT_PRICING so requirement 6 (reconcilable frontier costs)
    would work at the shipped defaults. Bohr's spec review
    (msg-016 F-1) — endorsed by Einstein (msg-018) — observed that
    the price *and* the ID are both placeholders, so any entry
    keyed to them writes a confident-wrong cost to the ledger,
    which is the exact failure class D-6c was built to prevent
    re-entering through the constant instead of the lookup.

    The correct honest state until Anthropic's published IDs and
    per-MTok figures are known: no Fable 5 / Opus 5 entries in
    DEFAULT_PRICING; frontier requests degrade safely to
    ``pricing_known=0`` and log ``cost_pricing_unknown``. These
    tests pin the honest path so a future edit re-adding an
    invented price is caught immediately.
    """

    def test_default_frontier_model_is_unpriced_until_verified(self) -> None:
        """Placeholder IDs stay OUT of DEFAULT_PRICING until real numbers land."""
        from lexora.services.cost_tracker import DEFAULT_PRICING

        settings = create_settings(REPO_CONFIG)
        default_model = settings.routing.backends["frontier"].models[0].name
        assert default_model not in DEFAULT_PRICING, (
            f"frontier default model {default_model!r} appears in "
            "DEFAULT_PRICING; if you have added it, the price must be "
            "from Anthropic's published pricing page with a citation "
            "date in the constant's comment (msg-016 F-1). If the ID "
            "is still a placeholder, remove the pricing entry — a "
            "confident-wrong ledger row is worse than an unpriced one."
        )

    def test_opus_5_placeholder_is_unpriced(self) -> None:
        """The Opus 5 placeholder ID must not carry an invented price either."""
        from lexora.services.cost_tracker import DEFAULT_PRICING

        assert "claude-opus-5-20260601" not in DEFAULT_PRICING, (
            "Placeholder ID with placeholder price violates F-1 — "
            "when the real Opus 5 ID and price are known, add both "
            "together (never one without the other)."
        )

    def test_unpriced_frontier_records_pricing_known_zero(
        self, tmp_path
    ) -> None:
        """End-to-end honest-path check: recording a frontier request against
        the shipped default model yields ``pricing_known=0`` and the model
        surfaces in ``unpriced_models``. If someone re-adds a fabricated
        price entry this test flips (which is exactly what F-1 forbids)."""
        from lexora.services.cost_tracker import CostTracker

        settings = create_settings(REPO_CONFIG)
        default_model = settings.routing.backends["frontier"].models[0].name

        tracker = CostTracker(db_path=tmp_path / "costs.db")
        tracker.record(
            model=default_model,
            endpoint="/v1/chat/completions",
            tokens_input=100,
            tokens_output=50,
            tier="frontier",
        )
        rows = tracker.get_recent(limit=1)
        assert rows[0]["pricing_known"] == 0
        assert rows[0]["cost_usd"] == 0.0

        report = tracker.get_costs(period="all")
        assert default_model in report["unpriced_models"]
        assert report["summary"]["unpriced_requests"] == 1


class TestFrontierTierBackendModelAgreement:
    """T-frontier-tier D-2 residual hazard (msg-016 (c) 2nd para): the
    router derives ``_tier_to_model["frontier"]`` from
    ``backends.frontier.models[0].name`` when ``tiers.frontier.model`` is
    unset. The env hook writes both, so env-driven swaps cannot diverge.
    A future YAML edit that sets ``tiers.frontier.model`` without
    updating the backend entry (or vice versa) *would* reintroduce
    the D-2 observability lie. This one-line invariant catches it.
    """

    def test_resolved_frontier_matches_backend_first_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LEXORA_FRONTIER_MODEL", raising=False)
        settings = create_settings(REPO_CONFIG)
        backend_model = settings.routing.backends["frontier"].models[0].name
        router = BackendRouter(
            routing_settings=settings.routing,
            vllm_settings=settings.vllm,
        )
        try:
            assert router.resolve_model("frontier") == backend_model, (
                "tiers.frontier.model and backends.frontier.models[0].name "
                "have diverged. `/v1/models` advertises the backend's "
                "first model, but the router will send the tier's model "
                "upstream — the same observability lie D-2 was written "
                "to prevent."
            )
        finally:
            import asyncio

            asyncio.run(router.close())

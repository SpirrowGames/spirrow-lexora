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
   both write sites -- the tier's model and `models[0].name` on the
   backend `tiers.frontier.backend` names -- are updated in lockstep,
   so the frontier tier's advertised capability never disagrees with
   what the router actually sends upstream. The second site is reached
   through the tier, not by the literal backend name `frontier`; a
   config where it cannot be reached refuses to start rather than
   applying half the override (B-19, added 2026-09-01 after PR #11's
   gate measured the half-applied case).
"""

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from lexora.config import Settings, create_settings, load_yaml_config
from lexora.services.model_registry import ModelRegistry
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


class TestFrontierPricingRealAndReconcilable:
    """T-frontier-tier F-1 (msg-016), after the real numbers landed.

    The original rule was: a placeholder ID must never carry a price,
    because an entry keyed to an invented ID writes a confident-wrong
    cost to the ledger — the exact failure D-6c exists to prevent,
    re-entering through the constant instead of the lookup. The rule
    was never "frontier stays unpriced forever"; it was **add the ID
    and the price together, or neither**, with a citation date.

    2026-09-01: both landed. `claude-fable-5` is the real Anthropic ID
    (the old `claude-fable-5-20260101` carried a date suffix that no
    Anthropic Claude 5 model ID has) and $10 / $50 per MTok are the
    published base rates. So the invariant flips from "must be absent"
    to "must be present, and must still refuse date-suffixed
    placeholders" — same rule, other side of the transition.

    The unpriced degrade path is NOT dropped: it still has coverage
    below against a genuinely unknown model, because that path is what
    protects the *next* model added before its price is known.
    """

    def test_shipped_frontier_model_is_priced(self) -> None:
        """ID and price must travel together — here, both present."""
        from lexora.services.cost_tracker import DEFAULT_PRICING

        settings = create_settings(REPO_CONFIG)
        default_model = settings.routing.backends["frontier"].models[0].name
        assert default_model in DEFAULT_PRICING, (
            f"frontier default model {default_model!r} is missing from "
            "DEFAULT_PRICING. Shipping a frontier tier whose model has no "
            "price puts every billed frontier request in the "
            "pricing_known=0 bucket, so requirement 6 (reconcilable "
            "frontier costs) cannot be met. Add the price from Anthropic's "
            "published pricing page with a citation date — or, if the ID is "
            "not verified yet, revert the ID too. Never one without the "
            "other (msg-016 F-1)."
        )

    def test_placeholder_ids_are_never_priced(self) -> None:
        """Date-suffixed placeholder IDs must not carry prices."""
        from lexora.services.cost_tracker import DEFAULT_PRICING

        for placeholder in (
            "claude-fable-5-20260101",
            "claude-opus-5-20260601",
        ):
            assert placeholder not in DEFAULT_PRICING, (
                f"Placeholder ID {placeholder!r} carries a price. Anthropic's "
                "Claude 5 model IDs have no date suffix, so this ID cannot be "
                "billed — a price keyed to it can only ever be "
                "confident-wrong (msg-016 F-1)."
            )

    def test_priced_frontier_records_pricing_known_one(self, tmp_path) -> None:
        """End-to-end: the shipped default now yields a reconcilable row."""
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
        assert rows[0]["pricing_known"] == 1
        # 100 in @ $10/MTok + 50 out @ $50/MTok
        assert rows[0]["cost_usd"] == pytest.approx(0.0035)

        report = tracker.get_costs(period="all")
        assert default_model not in report["unpriced_models"]
        assert report["summary"]["unpriced_requests"] == 0

    def test_unknown_model_still_degrades_to_unpriced(self, tmp_path) -> None:
        """The D-6c honest-degrade path stays covered for the next new model."""
        from lexora.services.cost_tracker import CostTracker

        tracker = CostTracker(db_path=tmp_path / "costs.db")
        tracker.record(
            model="claude-not-yet-priced-9",
            endpoint="/v1/chat/completions",
            tokens_input=100,
            tokens_output=50,
            tier="frontier",
        )
        rows = tracker.get_recent(limit=1)
        assert rows[0]["pricing_known"] == 0
        assert rows[0]["cost_usd"] == 0.0

        report = tracker.get_costs(period="all")
        assert "claude-not-yet-priced-9" in report["unpriced_models"]
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


# --- B-19: the frontier surface invariant, exercised across config shapes ---
#
# PR #11's gate found that the LEXORA_FRONTIER_MODEL hook reached its second
# write site by the literal backend name "frontier" instead of through
# tiers["frontier"].backend, so a YAML naming that backend anything else got
# the tier updated and the advertised model left behind. The tests below are
# written as one invariant applied to several config shapes rather than as a
# list of cases, so the next config topology that breaks the invariant is
# caught without anyone having thought to enumerate it.

_HEAVY_BACKEND = """\
    heavy:
      type: "vllm"
      url: "http://localhost:8000"
      models:
        - name: "Qwen3.8-27B"
"""

_PAID_BACKEND = """\
    anthropic_paid:
      type: "anthropic"
      url: "https://api.anthropic.com"
      error_passthrough: true
      models:
        - name: "claude-fable-5-20260101"
          capabilities: ["code", "reasoning", "frontier"]
          description: "Anthropic Claude Fable 5"
"""

_DECOY_BACKEND = """\
    frontier:
      type: "anthropic"
      url: "https://api.anthropic.com"
      error_passthrough: true
      models:
        - name: "claude-fable-5-20260101"
          capabilities: ["code", "reasoning", "frontier"]
          description: "unused - no tier points here"
"""

_FRONTIER_TIER_ON_PAID = """\
    frontier:
      backend: "anthropic_paid"
"""


def _routing_config(backends: str, tiers: str) -> str:
    return (
        "routing:\n"
        "  enabled: true\n"
        '  default_backend: "heavy"\n'
        "  backends:\n"
        f"{backends}"
        "  tiers:\n"
        f"{tiers}"
    )


def _write_config(tmp_path: Path, body: str) -> Path:
    config_file = tmp_path / "lexora_config.yaml"
    config_file.write_text(body, encoding="utf-8")
    return config_file


def assert_frontier_surface_agrees(settings: Settings) -> None:
    """The frontier invariant, in one place.

    What the router sends upstream for the ``frontier`` tier must be exactly
    what ``/v1/models`` and ``/v1/models/capabilities`` advertise for it. The
    advertised ID is ``models[0].name`` on the backend the tier points at --
    found through ``tiers["frontier"].backend``, the same way the router finds
    it, and deliberately not by the literal name ``frontier``: reading a
    literal there is the defect this asserts against.
    """
    tier = settings.routing.tiers["frontier"]
    advertised = settings.routing.backends[tier.backend].models[0].name
    router = BackendRouter(
        routing_settings=settings.routing,
        vllm_settings=settings.vllm,
    )
    try:
        assert router.get_backend_name_for_model("frontier") == tier.backend
        assert router.resolve_model("frontier") == advertised, (
            f"the frontier tier resolves to "
            f"{router.resolve_model('frontier')!r} but backend "
            f"{tier.backend!r} advertises {advertised!r}. The router would "
            f"bill one model while /v1/models/capabilities named another -- "
            f"the D-2 observability lie."
        )
    finally:
        import asyncio

        asyncio.run(router.close())


class TestFrontierOverrideFollowsTheTierNotTheBackendName:
    """``LEXORA_FRONTIER_MODEL`` must reach whichever backend the tier names.

    ``TierSettings.backend`` is a free-form string the YAML chooses; only the
    *tier* name ``frontier`` is public API (callers send ``model:
    "frontier"``). The shipped config happens to name the backend ``frontier``
    as well, which is exactly why the shipped-config tests above cannot see
    this class of break.
    """

    OVERRIDE = "claude-opus-5-20260601"

    def test_shipped_config_surface_agrees(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", self.OVERRIDE)
        settings = create_settings(REPO_CONFIG)
        assert_frontier_surface_agrees(settings)
        assert settings.routing.tiers["frontier"].model == self.OVERRIDE

    def test_renamed_backend_surface_agrees(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The tier points at a backend that is NOT named ``frontier``.

        Measured at e0fe315 (pre-fix): the tier resolved to Opus 5 while the
        backend kept advertising Fable 5, and the override ID appeared on no
        backend at all.
        """
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", self.OVERRIDE)
        config_file = _write_config(
            tmp_path,
            _routing_config(
                _HEAVY_BACKEND + _PAID_BACKEND, _FRONTIER_TIER_ON_PAID
            ),
        )
        settings = create_settings(config_file)
        assert_frontier_surface_agrees(settings)
        assert (
            settings.routing.backends["anthropic_paid"].models[0].name
            == self.OVERRIDE
        )

    def test_renamed_backend_keeps_model_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Only the ID is swapped; capabilities / description survive.

        The name assert is what makes this a detector rather than a
        tautology: at e0fe315 the metadata survived only because nothing was
        written to this backend at all.
        """
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", self.OVERRIDE)
        config_file = _write_config(
            tmp_path,
            _routing_config(
                _HEAVY_BACKEND + _PAID_BACKEND, _FRONTIER_TIER_ON_PAID
            ),
        )
        settings = create_settings(config_file)
        model = settings.routing.backends["anthropic_paid"].models[0]
        assert model.name == self.OVERRIDE
        assert model.capabilities == ["code", "reasoning", "frontier"]
        assert model.description == "Anthropic Claude Fable 5"

    def test_decoy_backend_named_frontier_is_not_rewritten(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A backend named ``frontier`` that no tier points at must be untouched.

        Measured at e0fe315 (pre-fix) this was worse than a skipped update:
        the override landed on the unused backend, so
        ``/v1/models/capabilities`` advertised Opus 5 on a backend no request
        could reach while the tier's real backend still advertised Fable 5.
        """
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", self.OVERRIDE)
        config_file = _write_config(
            tmp_path,
            _routing_config(
                _HEAVY_BACKEND + _PAID_BACKEND + _DECOY_BACKEND,
                _FRONTIER_TIER_ON_PAID,
            ),
        )
        settings = create_settings(config_file)
        assert_frontier_surface_agrees(settings)
        assert (
            settings.routing.backends["frontier"].models[0].name
            == "claude-fable-5-20260101"
        )
        registry = ModelRegistry(routing_settings=settings.routing)
        unreachable = [
            entry
            for entry in registry.get_all_models()
            if entry.id == self.OVERRIDE and entry.backend != "anthropic_paid"
        ]
        assert unreachable == [], (
            "the override was advertised on a backend no tier resolves to: "
            f"{unreachable}"
        )


class TestFrontierOverrideRefusesRatherThanApplyingHalf:
    """B-19: "both or refuse", not "both or neither".

    Setting ``LEXORA_FRONTIER_MODEL`` is an operator saying "bill me for this
    model". Applying none of it silently bills them for the YAML default under
    the name of the model they asked for, so a config the override cannot be
    applied to in full does not start -- the same judgement as
    ``BackendSettings._reject_unimplemented_error_passthrough`` ("a config that
    lies is worse than a config that will not load").

    ``match=`` is not decoration in these tests: pydantic's ``ValidationError``
    subclasses ``ValueError``, so a bare ``pytest.raises(ValueError)`` would
    pass just as happily if the config had failed to load for some entirely
    unrelated reason.
    """

    OVERRIDE = "claude-opus-5-20260601"

    UNRESOLVABLE_BACKEND = _routing_config(
        _HEAVY_BACKEND,
        '    frontier:\n      backend: "no_such_backend"\n',
    )
    EMPTY_MODELS = _routing_config(
        _HEAVY_BACKEND
        + "    anthropic_paid:\n"
        '      type: "anthropic"\n'
        '      url: "https://api.anthropic.com"\n'
        "      models: []\n",
        _FRONTIER_TIER_ON_PAID,
    )
    NO_FRONTIER_TIER = _routing_config(
        _HEAVY_BACKEND + _DECOY_BACKEND,
        '    heavy:\n      backend: "heavy"\n',
    )

    def test_refuses_when_tier_backend_does_not_exist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", self.OVERRIDE)
        config_file = _write_config(tmp_path, self.UNRESOLVABLE_BACKEND)
        with pytest.raises(
            ValueError,
            match=(
                r"names 'no_such_backend', which is not defined in "
                r"routing\.backends"
            ),
        ):
            create_settings(config_file)

    def test_refuses_when_tier_backend_has_no_models(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", self.OVERRIDE)
        config_file = _write_config(tmp_path, self.EMPTY_MODELS)
        with pytest.raises(ValueError, match=r"has an empty 'models' list"):
            create_settings(config_file)

    def test_refuses_when_there_is_no_frontier_tier(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", self.OVERRIDE)
        config_file = _write_config(tmp_path, self.NO_FRONTIER_TIER)
        with pytest.raises(ValueError, match=r"has no 'frontier' tier"):
            create_settings(config_file)

    def test_refusal_message_names_the_variable_and_the_fix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The operator reads this in a crash loop, not in a code review."""
        monkeypatch.setenv("LEXORA_FRONTIER_MODEL", self.OVERRIDE)
        config_file = _write_config(tmp_path, self.UNRESOLVABLE_BACKEND)
        with pytest.raises(ValueError) as excinfo:
            create_settings(config_file)
        message = str(excinfo.value)
        assert "LEXORA_FRONTIER_MODEL" in message
        assert self.OVERRIDE in message
        assert "routing.tiers.frontier.backend" in message

    @pytest.mark.parametrize(
        "body",
        [UNRESOLVABLE_BACKEND, EMPTY_MODELS, NO_FRONTIER_TIER],
        ids=["unresolvable_backend", "empty_models", "no_frontier_tier"],
    )
    def test_without_the_env_var_none_of_these_configs_are_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
    ) -> None:
        """The refusal is scoped to the env-override path and nothing else.

        Without this, the next reader of the check is invited to promote it
        into a whole-config validator and start refusing tier topologies that
        have nothing to do with ``LEXORA_FRONTIER_MODEL``.
        """
        monkeypatch.delenv("LEXORA_FRONTIER_MODEL", raising=False)
        config_file = _write_config(tmp_path, body)
        create_settings(config_file)

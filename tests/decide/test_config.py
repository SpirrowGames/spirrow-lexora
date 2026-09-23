"""Tests for ``lexora.decide.config`` — DecisionSettings + env check.

The env check is a startup guarantee (msg-240 §1); the tests here
exercise it in isolation from FastAPI so a failure in main.py's wiring
cannot mask a regression in the check itself. The paired tests in
:mod:`tests.decide.test_startup_env_check` cover the wiring.
"""

from __future__ import annotations

import pytest

from pydantic import ValidationError

from lexora.config import create_settings
from lexora.decide.config import (
    TYPESAFE_API_KEY_ENV,
    DecisionSettings,
    check_typesafe_api_key,
    references_jev,
)


class TestDecisionSettingsDefaults:
    def test_default_is_null_off(self) -> None:
        """Fresh settings default to the safest configuration.

        ``primary=null`` + ``mode=off`` means an operator who ships
        this branch without touching ``[decision]`` never talks to an
        external service.
        """
        settings = DecisionSettings()
        assert settings.primary == "null"
        assert settings.mode == "off"
        assert not hasattr(settings, "fallback")
        assert settings.timeout_ms == 2000

    def test_default_log_path_is_on_disk(self) -> None:
        """The decision log defaults to a durable path.

        msg-251 blocking objection: ``:memory:`` would make shadow-mode
        data collection (msg-237's calibration curves and mindwire's
        116 decision-point replay) forget everything at every process
        restart. The default must be on disk. The specific path
        ``data/decisions.db`` matches the sibling ``data/costs.db``
        convention in ``services/cost_tracker.py`` so an operator does
        not have to learn two directory layouts.
        """
        settings = DecisionSettings()
        assert settings.log_path == "data/decisions.db"

    def test_log_path_accepts_memory_for_tests(self) -> None:
        """Tests may pass ``:memory:`` explicitly to keep the run hermetic."""
        settings = DecisionSettings(log_path=":memory:")
        assert settings.log_path == ":memory:"

    def test_timeout_ms_must_be_positive(self) -> None:
        """0 / negative timeouts are refused (ge=1 in the field)."""
        with pytest.raises(Exception):
            DecisionSettings(timeout_ms=0)


class TestReferencesJev:
    def test_neither_slot_is_jev(self) -> None:
        assert references_jev(DecisionSettings(primary="null")) is False

    def test_primary_is_jev(self) -> None:
        assert references_jev(DecisionSettings(primary="jev")) is True

    def test_mode_off_still_references(self) -> None:
        """``mode=off`` does not soften the reference detection.

        Rationale: the fail-closed rule is about *config intent*, not
        current runtime behaviour. An operator flipping ``mode`` from
        ``off`` to ``active`` at runtime is a normal deployment step;
        it should not depend on happening to remember the env variable
        at that moment.
        """
        settings = DecisionSettings(primary="jev", mode="off")
        assert references_jev(settings) is True


class TestCheckTypesafeApiKey:
    def test_no_jev_reference_passes_without_env(self) -> None:
        """A config that never mentions Jev never needs the env variable."""
        check_typesafe_api_key(
            DecisionSettings(primary="null"),
            environ={},
        )

    def test_primary_jev_missing_env_raises(self) -> None:
        """primary=jev + missing env → startup fails."""
        with pytest.raises(RuntimeError) as excinfo:
            check_typesafe_api_key(
                DecisionSettings(primary="jev", mode="active"),
                environ={},
            )
        # Fixed message shape (msg-240 §1); do not check the value at
        # all because the check must never receive one.
        assert TYPESAFE_API_KEY_ENV in str(excinfo.value)
        assert "required" in str(excinfo.value).lower()

    def test_mode_off_with_jev_still_raises(self) -> None:
        """Even ``mode=off`` fails when a config names Jev without the env.

        Endorsed with rationale in msg-243 / msg-245: a config that
        names ``jev`` is intent, not behaviour. ``off`` is the
        deployment safety net, not a bypass of the env check.
        """
        with pytest.raises(RuntimeError):
            check_typesafe_api_key(
                DecisionSettings(primary="jev", mode="off"),
                environ={},
            )

    def test_env_present_passes(self) -> None:
        """A non-empty env value satisfies the check.

        The check does NOT inspect the value's shape (length / prefix
        / character class); a syntactically valid key that TypeSafe
        would reject at runtime is not the startup check's job to
        catch.
        """
        check_typesafe_api_key(
            DecisionSettings(primary="jev", mode="active"),
            environ={TYPESAFE_API_KEY_ENV: "sk-anything-nonempty"},
        )

    def test_empty_env_treated_as_missing(self) -> None:
        """An empty string is treated the same as unset.

        An operator who set the variable but left the value empty had
        the same effect on the process as not setting it, so the check
        should behave the same way. Silently accepting the empty
        string would defer the failure to the first Jev call.
        """
        with pytest.raises(RuntimeError):
            check_typesafe_api_key(
                DecisionSettings(primary="jev", mode="active"),
                environ={TYPESAFE_API_KEY_ENV: ""},
            )

    def test_error_message_does_not_leak_value(self) -> None:
        """The fixed message never carries the caller's value.

        A leaked value in a log capture is the concrete failure this
        rule exists to prevent (msg-240 §1). This test pins two
        properties:

        1. The caller-supplied value never appears verbatim in the
           message (checked by passing a distinctive sentinel and
           looking for it).
        2. No length / prefix / suffix / char-count adjective slips
           in — those are the derivative facts msg-240 §1 also bars.
        """
        settings = DecisionSettings(primary="jev", mode="active")
        # Sentinel a real leak would show; the message must not
        # contain it because the check refuses BEFORE reading the
        # value, and the fixed text was crafted without it.
        sentinel_value = "sk-do-not-leak-this-XZQ42"
        try:
            check_typesafe_api_key(settings, environ={"OTHER": sentinel_value})
            assert False, "should have raised"
        except RuntimeError as e:
            msg = str(e)
        assert sentinel_value not in msg
        lowered = msg.lower()
        # None of these derivative facts belong in a message that a
        # log-capture pipeline may forward downstream.
        for banned in ("length", "prefix", "suffix", "char", "byte"):
            assert banned not in lowered


class TestSchemaMatchesImplementation:
    """Values the code does not implement fail validation (Bohr msg-387 v3).

    ``shadow``, ``llm`` and a ``fallback`` selector used to be accepted and
    then silently did nothing (msg-383 / msg-384 / msg-386). The PR that
    implements one of them puts it back into the schema and rewrites the
    matching test here on purpose.
    """

    def test_mode_shadow_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            DecisionSettings(mode="shadow")  # type: ignore[arg-type]
        assert "mode" in str(excinfo.value)

    @pytest.mark.parametrize("primary", ["null", "jev"])
    def test_mode_shadow_rejected_regardless_of_primary(self, primary: str) -> None:
        with pytest.raises(ValidationError):
            DecisionSettings(primary=primary, mode="shadow")  # type: ignore[arg-type]

    def test_primary_llm_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            DecisionSettings(primary="llm")  # type: ignore[arg-type]
        assert "primary" in str(excinfo.value)

    @pytest.mark.parametrize("value", ["null", "llm", "jev"])
    def test_fallback_key_is_rejected(self, value: str) -> None:
        """A leftover ``fallback`` key stops load instead of being ignored.

        ``DecisionSettings`` is a ``BaseModel`` (msg-390 v4), whose
        default would be ``extra="ignore"``; ``forbid`` is set explicitly
        and pinned here so dropping it shows up.
        """
        assert DecisionSettings.model_config.get("extra") == "forbid"
        with pytest.raises(ValidationError) as excinfo:
            DecisionSettings(fallback=value)  # type: ignore[call-arg]
        assert "fallback" in str(excinfo.value)

    @pytest.mark.parametrize(
        "body",
        ['  mode: "shadow"', '  primary: "llm"', '  fallback: "null"'],
    )
    def test_yaml_config_is_rejected_at_load(self, tmp_path, body: str) -> None:  # type: ignore[no-untyped-def]
        """The operator surface: ``[decision]`` in the YAML config.

        ``create_settings`` is what ``lexora.main`` calls, and it builds
        ``DecisionSettings(**yaml["decision"])``, so this is the path a
        real deployment takes.
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"decision:\n{body}\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            create_settings(config_file)

    @pytest.mark.parametrize("mode", ["off", "active"])
    def test_implemented_modes_pass(self, mode: str) -> None:
        assert DecisionSettings(mode=mode).mode == mode  # type: ignore[arg-type]


class TestYamlOnlySource:
    """``[decision]`` comes from YAML only (Bohr msg-390 v4).

    On develop b1ef20d (``DecisionSettings(BaseSettings)`` without
    ``env_prefix``) the unprefixed variables below were read into the
    decision config by ``create_settings`` — this test failed there.
    """

    def test_unprefixed_env_is_ignored(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        for name, value in {
            "LOG_PATH": "/elsewhere/other.db",
            "PRIMARY": "jev",
            "MODE": "active",
            "TIMEOUT_MS": "1234",
            "JEV_MODEL": "hijack",
        }.items():
            monkeypatch.setenv(name, value)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("decision:\n  timeout_ms: 3000\n", encoding="utf-8")
        decision = create_settings(config_file).decision
        assert decision.primary == "null"
        assert decision.mode == "off"
        assert decision.timeout_ms == 3000
        assert decision.jev_model == "jev-latest"
        assert decision.log_path == "data/decisions.db"

    def test_prefixed_env_is_ignored(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("LEXORA_DECISION__PRIMARY", "jev")
        monkeypatch.setenv("LEXORA_DECISION__MODE", "active")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("decision:\n  timeout_ms: 3000\n", encoding="utf-8")
        decision = create_settings(config_file).decision
        assert (decision.primary, decision.mode) == ("null", "off")

"""Tests for ``lexora.decide.config`` — DecisionSettings + env check.

The env check is a startup guarantee (msg-240 §1); the tests here
exercise it in isolation from FastAPI so a failure in main.py's wiring
cannot mask a regression in the check itself. The paired tests in
:mod:`tests.decide.test_startup_env_check` cover the wiring.
"""

from __future__ import annotations

import pytest

from lexora.decide.config import (
    TYPESAFE_API_KEY_ENV,
    DecisionSettings,
    check_typesafe_api_key,
    references_jev,
)


class TestDecisionSettingsDefaults:
    def test_default_is_null_off_llm(self) -> None:
        """Fresh settings default to the safest configuration.

        ``primary=null`` + ``mode=off`` means an operator who ships
        this branch without touching ``[decision]`` never talks to an
        external service.
        """
        settings = DecisionSettings()
        assert settings.primary == "null"
        assert settings.mode == "off"
        assert settings.fallback == "llm"
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
        assert references_jev(DecisionSettings(primary="null", fallback="llm")) is False

    def test_primary_is_jev(self) -> None:
        assert references_jev(DecisionSettings(primary="jev", fallback="llm")) is True

    def test_fallback_is_jev(self) -> None:
        """The fallback slot triggers the reference just like primary.

        Endorsed by Einstein msg-243: catching a fallback-only Jev
        reference is not overzealous — a config that names ``jev`` only
        for fallback but omits the key is a time bomb that goes off the
        moment the primary starts failing.
        """
        assert references_jev(DecisionSettings(primary="llm", fallback="jev")) is True

    def test_mode_off_still_references(self) -> None:
        """``mode=off`` does not soften the reference detection.

        Rationale: the fail-closed rule is about *config intent*, not
        current runtime behaviour. An operator flipping ``mode`` from
        ``off`` to ``active`` at runtime is a normal deployment step;
        it should not depend on happening to remember the env variable
        at that moment.
        """
        settings = DecisionSettings(primary="jev", fallback="llm", mode="off")
        assert references_jev(settings) is True


class TestCheckTypesafeApiKey:
    def test_no_jev_reference_passes_without_env(self) -> None:
        """A config that never mentions Jev never needs the env variable."""
        check_typesafe_api_key(
            DecisionSettings(primary="null", fallback="llm"),
            environ={},
        )

    def test_primary_jev_missing_env_raises(self) -> None:
        """primary=jev + missing env → startup fails."""
        with pytest.raises(RuntimeError) as excinfo:
            check_typesafe_api_key(
                DecisionSettings(primary="jev", fallback="llm", mode="active"),
                environ={},
            )
        # Fixed message shape (msg-240 §1); do not check the value at
        # all because the check must never receive one.
        assert TYPESAFE_API_KEY_ENV in str(excinfo.value)
        assert "required" in str(excinfo.value).lower()

    def test_fallback_jev_missing_env_raises(self) -> None:
        """fallback=jev alone (primary=llm) also fails at startup."""
        with pytest.raises(RuntimeError):
            check_typesafe_api_key(
                DecisionSettings(primary="llm", fallback="jev", mode="active"),
                environ={},
            )

    def test_mode_off_with_jev_still_raises(self) -> None:
        """Even ``mode=off`` fails when a config names Jev without the env.

        Endorsed with rationale in msg-243 / msg-245: a config that
        names ``jev`` is intent, not behaviour. ``off`` is the
        deployment safety net, not a bypass of the env check.
        """
        with pytest.raises(RuntimeError):
            check_typesafe_api_key(
                DecisionSettings(primary="jev", fallback="llm", mode="off"),
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
            DecisionSettings(primary="jev", fallback="llm", mode="active"),
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
                DecisionSettings(primary="jev", fallback="llm", mode="active"),
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
        settings = DecisionSettings(primary="jev", fallback="llm", mode="active")
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

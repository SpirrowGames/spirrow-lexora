"""End-to-end tests for the ``/v1/decide`` route.

Uses FastAPI's TestClient against the real application factory
(``create_app``) so the wiring in ``main.py`` — settings loading, env
check, provider registry, decision-log storage — is exercised together.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lexora.config import Settings
from lexora.decide.config import DecisionSettings
from lexora.main import create_app


def _app_with_null_provider() -> TestClient:
    """Build the app with a safe (NullProvider-only) decision config."""
    settings = Settings(decision=DecisionSettings(primary="null", mode="off"))
    return TestClient(create_app(settings=settings))


class TestDecideRoute:
    def test_returns_null_answer_off_mode(self) -> None:
        client = _app_with_null_provider()
        resp = client.post(
            "/v1/decide",
            json={
                "state": "customer said the product was great",
                "questions": {
                    "positive": {
                        "type": "noul",
                        "instructions": "Is the customer positive?",
                    }
                },
                "policy": "test.caller",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["provider"] == "null"
        assert body["answers"] == {"positive": {"noul": 0.5}}
        assert body["decision_id"]
        assert body["latency_ms"] >= 0

    def test_questions_version_optional(self) -> None:
        """A request without ``questions_version`` still succeeds."""
        client = _app_with_null_provider()
        resp = client.post(
            "/v1/decide",
            json={
                "state": "s",
                "questions": {
                    "q": {"type": "noul", "instructions": "i"},
                },
                "policy": "p",
            },
        )
        assert resp.status_code == 200

    def test_questions_version_recorded_in_log(self) -> None:
        """A request WITH ``questions_version`` lands on the log row."""
        settings = Settings(decision=DecisionSettings(primary="null", mode="off"))
        app = create_app(settings=settings)
        client = TestClient(app)
        resp = client.post(
            "/v1/decide",
            json={
                "state": "s",
                "questions": {
                    "q": {"type": "noul", "instructions": "i"},
                },
                "policy": "p",
                "questions_version": "vx",
            },
        )
        assert resp.status_code == 200
        rows = app.state.decision_log.fetch_all()
        assert len(rows) == 1
        assert rows[0]["questions_version"] == "vx"
        assert rows[0]["policy"] == "p"
        assert rows[0]["provider"] == "null"

    def test_multiple_questions_in_one_request(self) -> None:
        """msg-237: multiple questions may be mixed in one call."""
        client = _app_with_null_provider()
        resp = client.post(
            "/v1/decide",
            json={
                "state": "s",
                "questions": {
                    "a": {"type": "noul", "instructions": "A"},
                    "b": {
                        "type": "choice",
                        "instructions": "B",
                        "criteria": ["x", "y"],
                    },
                    "c": {
                        "type": "score",
                        "instructions": "C",
                        "criteria": ["low", "high"],
                    },
                },
                "policy": "p",
            },
        )
        assert resp.status_code == 200
        answers = resp.json()["answers"]
        assert set(answers) == {"a", "b", "c"}
        assert answers["a"] == {"noul": 0.5}
        assert answers["b"]["confidence"] == 0.0
        assert answers["c"] == {
            "score": 0.0,
            "legend": ["low", "high"],
            "confidence": 0.0,
        }

    def test_missing_required_field_rejected(self) -> None:
        """policy is required — a missing value is a 422, not silent."""
        client = _app_with_null_provider()
        resp = client.post(
            "/v1/decide",
            json={
                "state": "s",
                "questions": {"q": {"type": "noul", "instructions": "i"}},
            },
        )
        assert resp.status_code == 422


class TestDecideStartupEnvCheck:
    def test_create_app_refuses_when_jev_config_and_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The env check runs inside ``create_app`` (fail-closed).

        Rationale from msg-240 §1: the check runs after config load
        and before any provider is instantiated. Building the app is
        the "provider instantiation" moment, so the exception should
        surface from ``create_app`` — not from the first request.
        """
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        settings = Settings(
            decision=DecisionSettings(primary="jev", fallback="llm", mode="active")
        )
        with pytest.raises(RuntimeError) as excinfo:
            create_app(settings=settings)
        assert "TYPESAFE_API_KEY" in str(excinfo.value)

    def test_create_app_refuses_even_when_mode_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mode=off does not soften the refusal — see test_config."""
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        settings = Settings(
            decision=DecisionSettings(primary="jev", fallback="llm", mode="off")
        )
        with pytest.raises(RuntimeError):
            create_app(settings=settings)

    def test_create_app_passes_with_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TYPESAFE_API_KEY", "sk-not-a-real-key")
        settings = Settings(
            decision=DecisionSettings(primary="jev", fallback="llm", mode="active")
        )
        # No exception; a happy startup is the whole assertion.
        app = create_app(settings=settings)
        # And the app carries the config it was built with.
        assert app.state.decision_settings.primary == "jev"

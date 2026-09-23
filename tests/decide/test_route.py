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
    """Build the app with a safe (NullProvider-only) decision config.

    ``log_path=":memory:"`` is passed explicitly so this test does not
    write a ``data/decisions.db`` file into the working tree on every
    run. Production defaults to on-disk (see :class:`DecisionSettings`);
    tests that need to introspect the log use ``app.state.decision_log.
    fetch_all()`` on the in-memory instance.
    """
    settings = Settings(
        decision=DecisionSettings(primary="null", mode="off", log_path=":memory:")
    )
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
        settings = Settings(
            decision=DecisionSettings(
                primary="null", mode="off", log_path=":memory:"
            )
        )
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
            decision=DecisionSettings(primary="jev", mode="active")
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
            decision=DecisionSettings(primary="jev", mode="off")
        )
        with pytest.raises(RuntimeError):
            create_app(settings=settings)

    def test_create_app_passes_with_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TYPESAFE_API_KEY", "sk-not-a-real-key")
        settings = Settings(
            decision=DecisionSettings(
                primary="jev", mode="active", log_path=":memory:"
            )
        )
        # No exception; a happy startup is the whole assertion.
        app = create_app(settings=settings)
        # And the app carries the config it was built with.
        assert app.state.decision_settings.primary == "jev"


class TestDecisionLogPathHonoured:
    """``create_app`` places the decision log at ``settings.decision.log_path``.

    Rationale (msg-251): before this fix the wiring in ``main.py``
    hard-coded ``DecisionLog(path=":memory:")``, which made the
    shadow-mode data-collection promise in msg-237 unachievable — a
    process restart forgot every logged row. The test below pins the
    write against the configured path so a regression that reintroduces
    the literal is caught here rather than in production traffic.
    """

    def test_writes_land_on_configured_path(self, tmp_path) -> None:
        """A write through the endpoint appears in the configured SQLite file."""
        db_path = tmp_path / "sub" / "decisions.db"
        settings = Settings(
            decision=DecisionSettings(
                primary="null", mode="off", log_path=str(db_path)
            )
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        resp = client.post(
            "/v1/decide",
            json={
                "state": "s",
                "questions": {"q": {"type": "noul", "instructions": "i"}},
                "policy": "p",
            },
        )
        assert resp.status_code == 200, resp.text
        assert db_path.exists(), "log file was not created at the configured path"

        # Cross-check the row landed via a fresh connection so this
        # test does not lean on the same in-memory handle it wrote
        # through — the whole point of on-disk persistence is that a
        # second reader can see the row.
        import sqlite3

        with sqlite3.connect(str(db_path)) as ro:
            (count,) = ro.execute("SELECT COUNT(*) FROM decisions").fetchone()
        assert count == 1


class _FakeJev:
    """Stands in for JevProvider in the registry (route-level tests)."""

    name = "jev"

    def __init__(
        self, *, error: Exception | None = None, upstream: object = None
    ) -> None:
        self._error = error
        self._upstream = upstream
        self.calls = 0

    async def evaluate(self, *, state, questions):  # type: ignore[no-untyped-def]
        from lexora.decide.providers import ProviderResult

        self.calls += 1
        if self._error is not None:
            raise self._error
        return ProviderResult(
            answers={name: {"noul": 0.91} for name in questions},
            upstream=self._upstream,  # type: ignore[arg-type]
        )


def _perr(code: str, **kw):  # type: ignore[no-untyped-def]
    from lexora.decide.providers import ProviderError, _Failure

    return ProviderError(_Failure(code, **kw))  # type: ignore[arg-type]


_JEV_KEY = "sk-route-SECRET-0123456789"
_STATE_SENTINEL = "STATE-SENTINEL-do-not-log-7f3a"


def _jev_app(
    monkeypatch: pytest.MonkeyPatch, fake, log_path: str = ":memory:"
):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TYPESAFE_API_KEY", _JEV_KEY)
    settings = Settings(
        decision=DecisionSettings(
            primary="jev", mode="active", log_path=log_path
        )
    )
    app = create_app(settings=settings)
    app.state.decision_providers["jev"] = fake
    return app


def _real_jev(handler):  # type: ignore[no-untyped-def]
    import httpx

    from lexora.decide.providers import JevProvider

    return JevProvider(
        _JEV_KEY,
        timeout_ms=2000,
        base_url="https://jev.test",
        transport=httpx.MockTransport(handler),
    )


_BODY = {
    "state": "s",
    "questions": {"esc": {"type": "noul", "instructions": "Escalate?"}},
    "policy": "mindwire.tier_c",
}

_UPSTREAM_COLS = ("provider_model", "provider_input_tokens", "provider_output_tokens")


class TestJevRouting:
    def test_create_app_registers_real_jev_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lexora.decide.providers import JevProvider

        monkeypatch.setenv("TYPESAFE_API_KEY", _JEV_KEY)
        settings = Settings(
            decision=DecisionSettings(
                primary="jev", mode="active",
                log_path=":memory:",
                jev_model="jev-1.13.0",
            )
        )
        app = create_app(settings=settings)
        provider = app.state.decision_providers["jev"]
        assert isinstance(provider, JevProvider)
        assert provider._model == "jev-1.13.0"  # noqa: SLF001
        # The key is not parked on app.state / settings.
        assert _JEV_KEY not in repr(vars(app.state))

    def test_default_config_does_not_register_jev(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TYPESAFE_API_KEY", _JEV_KEY)
        app = create_app(
            settings=Settings(decision=DecisionSettings(log_path=":memory:"))
        )
        assert set(app.state.decision_providers) == {"null"}

    def test_active_jev_success_logs_upstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lexora.decide.providers import UpstreamMeta

        fake = _FakeJev(upstream=UpstreamMeta("jev-1.13.0", 120, 9))
        app = _jev_app(monkeypatch, fake)
        resp = TestClient(app).post("/v1/decide", json=_BODY)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["provider"] == "jev"
        assert body["answers"] == {"esc": {"noul": 0.91}}
        # DecideResponse contract unchanged: upstream meta is log-only.
        assert set(body) == {"answers", "provider", "decision_id", "latency_ms"}
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider"] == "jev"
        assert row["provider_error"] is None
        assert row["provider_model"] == "jev-1.13.0"
        assert row["provider_input_tokens"] == 120
        assert row["provider_output_tokens"] == 9

    @pytest.mark.parametrize(
        ("code", "level"),
        [
            ("timeout", "warning"),
            ("network", "warning"),
            ("rate_limited", "warning"),
            ("overloaded", "warning"),
            ("http_status", "warning"),
            ("auth", "error"),
            ("invalid_request", "error"),
            ("invalid_response", "error"),
            ("internal_error", "error"),
        ],
    )
    def test_active_jev_failure_falls_back_to_null(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        code: str,
        level: str,
    ) -> None:
        fake = _FakeJev(error=_perr(code))
        app = _jev_app(monkeypatch, fake)
        capsys.readouterr()
        resp = TestClient(app).post("/v1/decide", json=_BODY)
        captured = capsys.readouterr()
        logs = captured.out + captured.err
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["provider"] == "null"
        assert body["answers"] == {"esc": {"noul": 0.5}}
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider"] == "null"
        assert row["provider_error"] == f"jev:{code}"
        for col in _UPSTREAM_COLS:
            assert row[col] is None, col
        # Structlog renders to stderr; the renderer (console / json) is
        # config-dependent, so assert on content, not layout.
        fallback_lines = [ln for ln in logs.splitlines() if "decide_provider_fallback" in ln]
        assert len(fallback_lines) == 1
        assert "jev" in fallback_lines[0]
        assert code in fallback_lines[0]
        assert level in fallback_lines[0].lower()
        assert _JEV_KEY not in logs
        assert _JEV_KEY not in resp.text
        assert _JEV_KEY not in repr(row)

    def test_invalid_response_fallback_keeps_upstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bohr msg-342 #2: ``provider="null"`` next to a billed Jev call."""
        from lexora.decide.providers import UpstreamMeta

        err = _perr("invalid_response", upstream=UpstreamMeta("jev-1.13.0", 50, 3))
        app = _jev_app(monkeypatch, _FakeJev(error=err))
        assert TestClient(app).post("/v1/decide", json=_BODY).status_code == 200
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider"] == "null"
        assert row["provider_error"] == "jev:invalid_response"
        assert row["provider_model"] == "jev-1.13.0"
        assert (row["provider_input_tokens"], row["provider_output_tokens"]) == (50, 3)

    def test_fallback_row_lands_on_disk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        import sqlite3

        db_path = tmp_path / "decisions.db"
        app = _jev_app(monkeypatch, _FakeJev(error=_perr("network")), str(db_path))
        assert TestClient(app).post("/v1/decide", json=_BODY).status_code == 200
        with sqlite3.connect(str(db_path)) as ro:
            assert ro.execute(
                "SELECT provider, provider_error, provider_model FROM decisions"
            ).fetchall() == [("null", "jev:network", None)]

    def test_mode_off_never_calls_jev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TYPESAFE_API_KEY", _JEV_KEY)
        app = create_app(
            settings=Settings(
                decision=DecisionSettings(
                    primary="jev", mode="off", log_path=":memory:"
                )
            )
        )
        fake = _FakeJev()
        app.state.decision_providers["jev"] = fake
        resp = TestClient(app).post("/v1/decide", json=_BODY)
        assert resp.json()["provider"] == "null"
        assert fake.calls == 0


_MALFORMED_2XX = [
    pytest.param({"model": "jev-1.13.0", "answers": [], "usage": {}}, id="answers-list"),
    pytest.param(
        {"model": "jev-1.13.0", "answers": {"esc": {"type": "noul", "noul": "high"}}},
        id="noul-string",
    ),
    pytest.param("<html>not json", id="not-json"),
]


class TestRealJevProviderThroughRoute:
    """Bohr msg-342 #1: a real JevProvider over MockTransport; every upstream
    fault ends in a 200 with a NullProvider answer, never a 500."""

    @pytest.mark.parametrize("payload", _MALFORMED_2XX)
    def test_malformed_2xx_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        payload: object,
    ) -> None:
        import httpx

        async def handler(request: httpx.Request) -> httpx.Response:
            if isinstance(payload, str):
                return httpx.Response(200, text=payload)
            return httpx.Response(200, json=payload)

        app = _jev_app(monkeypatch, _real_jev(handler))
        capsys.readouterr()
        body = dict(_BODY, state=_STATE_SENTINEL)
        resp = TestClient(app).post("/v1/decide", json=body)
        captured = capsys.readouterr()
        assert resp.status_code == 200, resp.text
        assert resp.json()["provider"] == "null"
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider"] == "null"
        assert row["provider_error"] == "jev:invalid_response"
        assert _STATE_SENTINEL not in captured.out + captured.err
        assert _JEV_KEY not in captured.out + captured.err

    def test_malformed_score_legend_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "jev-1.13.0",
                    "answers": {
                        "sev": {
                            "type": "score",
                            "score": 1.0,
                            "legend": None,
                            "confidence": 0.5,
                        }
                    },
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            )

        app = _jev_app(monkeypatch, _real_jev(handler))
        body = {
            "state": "s",
            "questions": {"sev": {"type": "score", "instructions": "i"}},
            "policy": "p",
        }
        resp = TestClient(app).post("/v1/decide", json=body)
        assert resp.status_code == 200, resp.text
        assert resp.json()["provider"] == "null"
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider_error"] == "jev:invalid_response"
        # Billed but discarded: the SQL-visible trace (msg-342 #2).
        assert row["provider_model"] == "jev-1.13.0"
        assert (row["provider_input_tokens"], row["provider_output_tokens"]) == (11, 1)

    def test_422_loc_logged_without_input(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import httpx

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                422,
                json={
                    "detail": [
                        {
                            "loc": ["body", "questions", "esc", "criteria"],
                            "msg": f"bad value {_STATE_SENTINEL}",
                            "input": _STATE_SENTINEL,
                        }
                    ]
                },
            )

        app = _jev_app(monkeypatch, _real_jev(handler))
        capsys.readouterr()
        resp = TestClient(app).post("/v1/decide", json=dict(_BODY, state=_STATE_SENTINEL))
        captured = capsys.readouterr()
        logs = captured.out + captured.err
        assert resp.status_code == 200, resp.text
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider_error"] == "jev:invalid_request"
        (line,) = [ln for ln in logs.splitlines() if "decide_provider_fallback" in ln]
        assert "criteria" in line
        assert "error" in line.lower()
        assert _STATE_SENTINEL not in logs
        assert _JEV_KEY not in logs

    def test_400_is_error_level_invalid_request_without_body(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """msg-368 v9: 400 (measured in msg-367) → invalid_request at error
        level, loc None, and nothing from the 400 body reaches the log."""
        import httpx

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "detail": {
                        "error_type": "api_usage_error",
                        "message": f"Invalid request. {_STATE_SENTINEL}",
                    }
                },
            )

        app = _jev_app(monkeypatch, _real_jev(handler))
        capsys.readouterr()
        resp = TestClient(app).post("/v1/decide", json=dict(_BODY, state=_STATE_SENTINEL))
        captured = capsys.readouterr()
        logs = captured.out + captured.err
        assert resp.status_code == 200, resp.text
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider"] == "null"
        assert row["provider_error"] == "jev:invalid_request"
        (line,) = [ln for ln in logs.splitlines() if "decide_provider_fallback" in ln]
        assert "error" in line.lower()
        assert "api_usage_error" not in logs
        assert _STATE_SENTINEL not in logs
        assert _JEV_KEY not in logs

    def test_success_through_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "jev-1.13.0",
                    "answers": {"esc": {"type": "noul", "noul": 0.95}},
                    "usage": {"input_tokens": 30, "output_tokens": 2},
                },
            )

        app = _jev_app(monkeypatch, _real_jev(handler))
        resp = TestClient(app).post("/v1/decide", json=_BODY)
        assert resp.status_code == 200, resp.text
        assert resp.json()["provider"] == "jev"
        assert resp.json()["answers"] == {"esc": {"type": "noul", "noul": 0.95}}
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider"] == "jev"
        assert row["provider_model"] == "jev-1.13.0"
        assert (row["provider_input_tokens"], row["provider_output_tokens"]) == (30, 2)


class TestSelectProvider:
    """Two-way routing, no silent Null substitution (Bohr msg-387 v3 #3)."""

    def test_off_answers_from_null(self) -> None:
        from lexora.decide.providers import NullProvider
        from lexora.decide.routes import _select_provider

        null = NullProvider()
        settings = DecisionSettings(primary="jev", mode="off")
        assert _select_provider(settings, {"null": null}) is null

    def test_active_unregistered_primary_is_loud(self) -> None:
        """An active primary missing from the registry is a Lexora bug.

        The startup key check guarantees ``jev`` is registered whenever
        ``primary="jev"``; if that ever breaks, the request must fail
        rather than quietly answer from NullProvider.
        """
        from lexora.decide.providers import NullProvider
        from lexora.decide.routes import _select_provider

        settings = DecisionSettings(primary="jev", mode="active")
        with pytest.raises(KeyError):
            _select_provider(settings, {"null": NullProvider()})

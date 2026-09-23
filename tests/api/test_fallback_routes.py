"""PR-2b through the real routes: the ledger row says who answered (B-2),
and the status endpoint unwraps the fallback backend (msg-441 #1, B-6).

The handlers are the unmodified ones; the router is a stub that hands out a
real ``FallbackBackend`` over the fake codex CLI and a fake Gemini, exactly
as ``test_streaming_ledger_row`` does for its backends.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexora.api.routes import (
    get_backend,
    get_backend_router,
    get_cost_tracker,
    get_metrics_collector,
    get_rate_limiter,
    get_retry_handler,
    get_stats_collector,
    is_rate_limit_enabled,
    router,
)
from lexora.backends.fallback import FallbackBackend
from lexora.services.cost_tracker import CostTracker
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector
from tests.backends.test_codex import make_backend, record_pass
from tests.backends.test_fallback import GEMINI_MODEL, FakeGemini

WRAPPER = "naysayer-fb"


def _wrapper(tmp_path: Path, *, verified: bool, mode: str = "fallback") -> FallbackBackend:
    codex = make_backend(tmp_path, "ok")
    if verified:
        record_pass(codex)
    w = FallbackBackend(
        name=WRAPPER, primary=codex, fallback=FakeGemini(), fallback_name="gemini", mode=mode, webhook_url=None
    )
    w.tier_label = "naysayer"
    return w


def _client(backend: Any, ledger: CostTracker) -> TestClient:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=backend)
    backend_router.resolve_model = MagicMock(return_value=GEMINI_MODEL)
    backend_router.get_backend_name_for_model = MagicMock(return_value=WRAPPER)
    backend_router.get_backend_by_name = MagicMock(side_effect=lambda n: backend if n == WRAPPER else None)
    backend_router.is_tier = MagicMock(side_effect=lambda t: t == "naysayer")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: backend
    app.dependency_overrides[get_backend_router] = lambda: backend_router
    app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
    app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(max_retries=1, base_delay=0.01, max_delay=0.1, jitter=False)
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(default_rate=1000.0, default_burst=1000)
    app.dependency_overrides[is_rate_limit_enabled] = lambda: False
    app.dependency_overrides[get_metrics_collector] = lambda: None
    app.dependency_overrides[get_cost_tracker] = lambda: ledger
    return TestClient(app)


ROUTES = [
    pytest.param("/v1/chat/completions", {"model": "naysayer", "messages": [{"role": "user", "content": "review"}]}, id="chat"),
    pytest.param(
        "/v1/messages",
        {"model": "naysayer", "max_tokens": 16, "messages": [{"role": "user", "content": "review"}]},
        id="messages",
    ),
]


def _rows(db: Path) -> list[tuple[Any, ...]]:
    with sqlite3.connect(db) as conn:
        return conn.execute(
            "SELECT backend, answered_by, model, cost_usd, pricing_known, tier FROM request_costs ORDER BY id"
        ).fetchall()


class TestLedgerThroughTheRoutes:
    @pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
    @pytest.mark.parametrize(("path", "body"), ROUTES)
    def test_codex_answer(self, tmp_path: Path, path: str, body: dict[str, Any], stream: bool) -> None:
        ledger = CostTracker(tmp_path / "costs.db")
        response = _client(_wrapper(tmp_path, verified=True), ledger).post(path, json={**body, "stream": stream})
        assert response.status_code == 200, response.text
        assert _rows(tmp_path / "costs.db") == [("codex", "codex", "gpt-5-codex", 0.0, 1, "naysayer")]

    @pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
    @pytest.mark.parametrize(("path", "body"), ROUTES)
    def test_gemini_fallback_answer(self, tmp_path: Path, path: str, body: dict[str, Any], stream: bool) -> None:
        ledger = CostTracker(tmp_path / "costs.db")
        response = _client(_wrapper(tmp_path, verified=False), ledger).post(path, json={**body, "stream": stream})
        assert response.status_code == 200, response.text
        [(backend, answered_by, model, cost, known, tier)] = _rows(tmp_path / "costs.db")
        assert (backend, answered_by, model, known, tier) == ("gemini", "gemini-fallback", GEMINI_MODEL, 1, "naysayer")
        assert cost > 0

    def test_shadow_gemini_answer_is_not_a_fallback(self, tmp_path: Path) -> None:
        ledger = CostTracker(tmp_path / "costs.db")
        w = _wrapper(tmp_path, verified=False, mode="shadow")  # closed codex: no shadow run
        response = _client(w, ledger).post("/v1/chat/completions", json={"model": "naysayer", "messages": [{"role": "user", "content": "x"}]})
        assert response.status_code == 200
        assert [r[:2] for r in _rows(tmp_path / "costs.db")] == [("gemini", None)]


class TestStatusUnwrapsTheWrapper:
    def test_codex_block_describes_the_primary(self, tmp_path: Path) -> None:
        w = _wrapper(tmp_path, verified=True)
        body = _client(w, CostTracker(tmp_path / "costs.db")).get("/v1/naysayer/status").json()
        assert body["backend"] == WRAPPER and body["primary"] == "codex"
        assert body["codex"] == {"codex_disabled_reason": None, "quota_hold_until": None, "inflight_runs": 0}
        assert body["mode"] == "codex" and body["fallback_since"] is None and body["shadow_skipped"] == 0

    def test_closed_codex_reads_fallback_with_counts(self, tmp_path: Path) -> None:
        ledger = CostTracker(tmp_path / "costs.db")
        w = _wrapper(tmp_path, verified=False)
        w.attach_ledger(ledger)
        client = _client(w, ledger)
        before = client.get("/v1/naysayer/status").json()
        assert before["mode"] == "fallback" and before["fallback_calls"] is None  # nothing fell back yet
        client.post("/v1/chat/completions", json={"model": "naysayer", "messages": [{"role": "user", "content": "x"}]})
        after = client.get("/v1/naysayer/status").json()
        assert after["codex"]["codex_disabled_reason"] == "verification_missing"
        assert after["fallback_since"] is not None
        assert after["fallback_calls"] == 1 and after["fallback_cost_usd"] > 0

    def test_shadow_primary_is_gemini(self, tmp_path: Path) -> None:
        w = _wrapper(tmp_path, verified=True, mode="shadow")
        w.primary._quota_hold_until = datetime.now(timezone.utc) + timedelta(hours=1)
        body = _client(w, CostTracker(tmp_path / "costs.db")).get("/v1/naysayer/status").json()
        assert (body["primary"], body["mode"]) == ("gemini", "shadow")
        assert body["codex"]["codex_disabled_reason"] == "quota_hold"

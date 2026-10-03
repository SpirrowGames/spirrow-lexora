"""``X-Mindwire-Trace`` reaches the cost row at every record site, and
``/stats/costs/recent?trace_id=`` filters on it (T-cost-row-trace-id).

The ten ``record`` sites (msg-627 §1): six non-streaming handlers, three
streaming ``stream_generator`` closures, and the shadow run in
``backends/fallback.py``. Each is driven through the real route and read back
out of a real SQLite ledger, so "the value arrived in the row" is what is
measured -- not that a keyword was passed.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexora.api import routes
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
from lexora.services import trace
from lexora.services.cost_tracker import CostTracker
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector
from lexora.services.trace import TRACE_HEADER
from tests.backends.test_codex import make_backend, record_pass
from tests.backends.test_fallback import GEMINI_MODEL, FakeGemini

TRACE = "01J9Z3K4M5N6P7Q8R9S0T1V2W3"
OTHER = "01J9Z3K4M5N6P7Q8R9S0T1V2W4"

REQUESTED = "heavy"
RESOLVED = "claude-fable-5"
USAGE = {"prompt_tokens": 11, "completion_tokens": 5}

CHAT_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
    ],
    "usage": USAGE,
}
COMPLETION_RESPONSE: dict[str, Any] = {
    "id": "cmpl-1",
    "object": "text_completion",
    "choices": [{"index": 0, "text": "Hi", "finish_reason": "stop"}],
    "usage": USAGE,
}
EMBEDDINGS_RESPONSE: dict[str, Any] = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": RESOLVED,
    "usage": {"prompt_tokens": 11, "total_tokens": 11},
}
CHUNK = (
    b"data: "
    + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
        }
    ).encode()
    + b"\n\n"
)

MESSAGES = [{"role": "user", "content": "Hi"}]
NON_STREAM_ROUTES = [
    pytest.param("/v1/chat/completions", {"model": REQUESTED, "messages": MESSAGES}, id="chat_completions"),
    pytest.param("/v1/completions", {"model": REQUESTED, "prompt": "Hi"}, id="completions"),
    pytest.param("/v1/embeddings", {"model": REQUESTED, "input": "Hi"}, id="embeddings"),
    pytest.param("/generate", {"model": REQUESTED, "prompt": "Hi"}, id="generate"),
    pytest.param("/chat", {"model": REQUESTED, "messages": MESSAGES}, id="chat"),
    pytest.param("/v1/messages", {"model": REQUESTED, "max_tokens": 16, "messages": MESSAGES}, id="messages"),
]
STREAM_ROUTES = [
    pytest.param(
        "/v1/chat/completions", {"model": REQUESTED, "messages": MESSAGES, "stream": True}, id="chat_completions_stream"
    ),
    pytest.param("/v1/completions", {"model": REQUESTED, "prompt": "Hi", "stream": True}, id="completions_stream"),
    pytest.param(
        "/v1/messages",
        {"model": REQUESTED, "max_tokens": 16, "messages": MESSAGES, "stream": True},
        id="messages_stream",
    ),
]
ALL_ROUTES = NON_STREAM_ROUTES + STREAM_ROUTES


def _backend() -> MagicMock:
    def _stream(_request: dict, usage_sink: Any = None) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            yield CHUNK
            if usage_sink is not None:
                usage_sink.prompt_tokens = 11
                usage_sink.completion_tokens = 5

        return gen()

    backend = MagicMock()
    backend.chat_completions = AsyncMock(return_value=dict(CHAT_RESPONSE))
    backend.completions = AsyncMock(return_value=dict(COMPLETION_RESPONSE))
    backend.embeddings = AsyncMock(return_value=dict(EMBEDDINGS_RESPONSE))
    backend.chat_completions_stream = MagicMock(side_effect=_stream)
    backend.completions_stream = MagicMock(side_effect=_stream)
    backend.error_passthrough = False
    return backend


def _app(backend: Any, ledger: CostTracker | None, *, backend_name: str = "frontier", resolved: str = RESOLVED) -> FastAPI:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=backend)
    backend_router.resolve_model = MagicMock(return_value=resolved)
    backend_router.get_backend_name_for_model = MagicMock(return_value=backend_name)
    backend_router.is_tier = MagicMock(return_value=True)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: backend
    app.dependency_overrides[get_backend_router] = lambda: backend_router
    app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
    app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
        max_retries=1, base_delay=0.01, max_delay=0.1, jitter=False
    )
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(default_rate=1000.0, default_burst=1000)
    app.dependency_overrides[is_rate_limit_enabled] = lambda: False
    app.dependency_overrides[get_metrics_collector] = lambda: None
    app.dependency_overrides[get_cost_tracker] = lambda: ledger
    return app


def _rows(db: Path) -> list[tuple[str, str | None]]:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT endpoint, trace_id FROM request_costs ORDER BY id").fetchall()


class TestEveryRecordSiteCarriesTheTrace:
    @pytest.mark.parametrize(("path", "body"), ALL_ROUTES)
    def test_header_value_lands_in_the_row(self, tmp_path: Path, path: str, body: dict[str, Any]) -> None:
        ledger = CostTracker(tmp_path / "costs.db")
        response = TestClient(_app(_backend(), ledger)).post(path, json=body, headers={TRACE_HEADER: TRACE})
        assert response.status_code == 200, response.text
        assert _rows(tmp_path / "costs.db") == [(path, TRACE)]

    @pytest.mark.parametrize(("path", "body"), ALL_ROUTES)
    def test_no_header_is_null_and_not_logged(
        self, tmp_path: Path, path: str, body: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trace_logger = MagicMock()
        monkeypatch.setattr(trace, "logger", trace_logger)
        ledger = CostTracker(tmp_path / "costs.db")
        response = TestClient(_app(_backend(), ledger)).post(path, json=body)
        assert response.status_code == 200, response.text
        assert _rows(tmp_path / "costs.db") == [(path, None)]
        trace_logger.warning.assert_not_called()

    @pytest.mark.parametrize(("path", "body"), ALL_ROUTES)
    def test_invalid_header_is_null_and_still_200(self, tmp_path: Path, path: str, body: dict[str, Any]) -> None:
        ledger = CostTracker(tmp_path / "costs.db")
        response = TestClient(_app(_backend(), ledger)).post(path, json=body, headers={TRACE_HEADER: TRACE.lower()})
        assert response.status_code == 200, response.text
        assert _rows(tmp_path / "costs.db") == [(path, None)]

    @pytest.mark.parametrize(("path", "body"), ALL_ROUTES)
    def test_repeated_header_is_null(
        self, tmp_path: Path, path: str, body: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        routes_logger = MagicMock(wraps=routes.logger)
        monkeypatch.setattr(routes, "logger", routes_logger)
        ledger = CostTracker(tmp_path / "costs.db")
        response = TestClient(_app(_backend(), ledger)).post(
            path, json=body, headers=[(TRACE_HEADER, TRACE), (TRACE_HEADER, TRACE)]
        )
        assert response.status_code == 200, response.text
        assert _rows(tmp_path / "costs.db") == [(path, None)]
        routes_logger.warning.assert_any_call("trace_id_rejected", reason="multiple", count=2)


WRAPPER = "naysayer-fb"


async def _shadow_idle(w: FallbackBackend) -> None:
    if w._shadow_task is not None:
        await w._shadow_task


class TestShadowRowCarriesTheTrace:
    """Record site 10: ``FallbackBackend._shadow_run`` is a task the wrapper
    spawns, so it gets the value through the context the handler set, not as
    an argument. Both the foreground row and the shadow row carry it (§4(a))."""

    @pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
    def test_foreground_and_shadow_rows_share_the_trace(self, tmp_path: Path, stream: bool) -> None:
        codex = make_backend(tmp_path, "ok")
        record_pass(codex)
        w = FallbackBackend(
            name=WRAPPER, primary=codex, fallback=FakeGemini(), fallback_name="gemini", mode="shadow", webhook_url=None
        )
        w.tier_label = "naysayer"
        ledger = CostTracker(tmp_path / "costs.db")
        w.attach_ledger(ledger)
        app = _app(w, ledger, backend_name=WRAPPER, resolved=GEMINI_MODEL)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "naysayer", "messages": [{"role": "user", "content": "review"}], "stream": stream},
                headers={TRACE_HEADER: TRACE},
            )
            assert response.status_code == 200, response.text
            client.portal.call(_shadow_idle, w)
        rows = _rows(tmp_path / "costs.db")
        assert sorted(rows) == sorted([("/v1/chat/completions", TRACE), ("shadow", TRACE)])

    def test_a_later_request_without_header_does_not_inherit(self, tmp_path: Path) -> None:
        """The context is per request: no value leaks into the next one."""
        ledger = CostTracker(tmp_path / "costs.db")
        with TestClient(_app(_backend(), ledger)) as client:
            body = {"model": REQUESTED, "messages": MESSAGES}
            client.post("/v1/chat/completions", json=body, headers={TRACE_HEADER: TRACE})
            client.post("/v1/chat/completions", json=body)
        assert [t for _, t in _rows(tmp_path / "costs.db")] == [TRACE, None]


class TestRecentCostsFilter:
    def _seed(self, tmp_path: Path) -> CostTracker:
        ledger = CostTracker(tmp_path / "costs.db")
        for t in (TRACE, OTHER, None, TRACE):
            ledger.record(model="m", endpoint="/e", tokens_input=1, tokens_output=1, trace_id=t)
        return ledger

    def test_filter_returns_only_matching_rows(self, tmp_path: Path) -> None:
        client = TestClient(_app(_backend(), self._seed(tmp_path)))
        body = client.get("/stats/costs/recent", params={"trace_id": TRACE}).json()
        assert [r["trace_id"] for r in body] == [TRACE, TRACE]

    def test_no_filter_returns_everything(self, tmp_path: Path) -> None:
        client = TestClient(_app(_backend(), self._seed(tmp_path)))
        body = client.get("/stats/costs/recent").json()
        assert [r["trace_id"] for r in body] == [TRACE, None, OTHER, TRACE]

    @pytest.mark.parametrize("bad", [TRACE.lower(), "", "x" * 300, TRACE + "0"], ids=["lower", "empty", "long", "27-chars"])
    def test_invalid_filter_is_422_not_unfiltered(self, tmp_path: Path, bad: str) -> None:
        client = TestClient(_app(_backend(), self._seed(tmp_path)))
        response = client.get("/stats/costs/recent", params={"trace_id": bad})
        assert response.status_code == 422

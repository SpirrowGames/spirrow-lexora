"""B-17: a client that disconnects *mid-stream* must still close the ledger.

`#10` declared the ACTIVE_REQUESTS invariant in `_fail_preflight`'s
docstring and made the last clause of every streaming pre-flight
`except BaseException`. The three `stream_generator()` bodies still
ended at `except Exception`, and `asyncio.CancelledError` is not an
`Exception` -- so a disconnect *after* the first byte skipped
`complete_request()`: the gauge stayed in-flight forever and the attempt
never reached `total_requests` either.

Not a regression from `#10`: the same four cases fail against `develop`
(d4fec9c) with the identical +1.0 per route, and one of them runs with
`error_passthrough=False`, so the leak is neither a frontier nor a
passthrough problem. It is folded in here because `#10` is the change
that stated the invariant.

Why raw ASGI and not `TestClient`: `TestClient` drives the app to
completion and hands back a finished `Response`, so it cannot express
"hang up after the first chunk". The disconnect path only exists in
`StreamingResponse.__call__` for `asgi.spec_version < 2.4`, where the
body pump and `listen_for_disconnect()` share an anyio task group. `2.3`
is what uvicorn's HTTP protocols advertise, so that is what is driven
here -- a fact about the pinned stack (starlette 0.50.0 /
fastapi 0.128.0), not a law about ASGI.

Mutation, so the detector is measured rather than asserted: dropping the
new clause from `chat_completions` reds the two chat cases, from
`completions` reds `completions`, from `messages` reds `messages`; with
all three dropped the rest of the suite is still 469 passed, i.e. it was
blind to this.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from prometheus_client import REGISTRY

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
from lexora.services.metrics import MetricsCollector
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector

# One well-formed OpenAI SSE chunk. `/v1/chat/completions` and
# `/v1/completions` forward the bytes verbatim; `/v1/messages` parses them
# through `anthropic_stream_from_openai`, which is why this is a real
# chunk rather than arbitrary bytes.
FIRST_CHUNK = (
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

CHAT_BODY = {
    "model": "frontier",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
}
COMPLETIONS_BODY = {"model": "frontier", "prompt": "Hi", "stream": True}
MESSAGES_BODY = {
    "model": "frontier",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
}


def _active_requests(endpoint: str) -> float:
    """Current `lexora_active_requests` for one endpoint.

    Read through the public registry and only ever compared as a delta:
    the gauge is a process global that other tests in the same session
    also touch.
    """
    value = REGISTRY.get_sample_value("lexora_active_requests", {"endpoint": endpoint})
    return 0.0 if value is None else value


def _hanging_stream(started: asyncio.Event):
    """A backend stream that emits one chunk and then never finishes.

    The hang is what makes the disconnect land *mid*-stream rather than
    racing the end of the response: the body pump is parked on this
    `await` when the cancellation arrives, so `CancelledError` is thrown
    into `stream_generator()` at its `yield`.
    """

    def factory(_request: dict) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            yield FIRST_CHUNK
            started.set()
            await asyncio.Event().wait()  # never set: park here until cancelled

        return gen()

    return MagicMock(side_effect=factory)


def _build_app(
    backend: MagicMock,
    stats_collector: StatsCollector,
    metrics_collector: MetricsCollector,
) -> FastAPI:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=backend)
    backend_router.resolve_model = MagicMock(return_value="claude-fable-5")
    backend_router.get_backend_name_for_model = MagicMock(return_value="frontier")
    backend_router.is_tier = MagicMock(return_value=True)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: backend
    app.dependency_overrides[get_backend_router] = lambda: backend_router
    app.dependency_overrides[get_stats_collector] = lambda: stats_collector
    app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
        max_retries=1, base_delay=0.01, max_delay=0.1, jitter=False
    )
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
        default_rate=1000.0, default_burst=1000
    )
    app.dependency_overrides[is_rate_limit_enabled] = lambda: True
    app.dependency_overrides[get_metrics_collector] = lambda: metrics_collector
    app.dependency_overrides[get_cost_tracker] = lambda: None
    return app


async def _post_and_disconnect(app: FastAPI, path: str, body: dict) -> list[dict]:
    """Drive the ASGI app and hang up after the first non-empty body chunk.

    Returns the messages the app sent, so the test can assert the stream
    really started (a disconnect *before* any byte would exercise the
    pre-flight instead, which `test_preflight_accounting.py` already
    covers and which is not the hole this file is about).
    """
    payload = json.dumps(body).encode()
    scope: dict[str, Any] = {
        "type": "http",
        # < 2.4 selects the task-group branch of `StreamingResponse`, i.e.
        # the one uvicorn's HTTP protocols actually run.
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    sent: list[dict] = []
    streaming = asyncio.Event()
    request_delivered = False

    async def receive() -> dict:
        nonlocal request_delivered
        if not request_delivered:
            request_delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        # `listen_for_disconnect()` parks here until the client has seen
        # bytes; then the connection drops.
        await streaming.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            streaming.set()

    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return sent


@pytest.mark.parametrize(
    ("endpoint", "stream_attr", "body", "passthrough"),
    [
        ("/v1/chat/completions", "chat_completions_stream", CHAT_BODY, True),
        # ★ passthrough=False on purpose: the leak has nothing to do with
        # the `error_passthrough` machinery `#10` introduced, and burning
        # that into the detector stops a later "only frontier is affected"
        # reading. This case takes the plain `async for chunk in
        # backend...` branch, with no pre-flight at all.
        ("/v1/chat/completions", "chat_completions_stream", CHAT_BODY, False),
        ("/v1/completions", "completions_stream", COMPLETIONS_BODY, True),
        ("/v1/messages", "chat_completions_stream", MESSAGES_BODY, True),
    ],
    ids=["chat", "chat-no-passthrough", "completions", "messages"],
)
async def test_mid_stream_disconnect_closes_the_ledger(
    endpoint: str,
    stream_attr: str,
    body: dict,
    passthrough: bool,
) -> None:
    stats_collector = StatsCollector()
    metrics_collector = MetricsCollector()
    started = asyncio.Event()

    backend = MagicMock()
    backend.chat_completions = AsyncMock()
    backend.completions = AsyncMock()
    backend.error_passthrough = passthrough
    setattr(backend, stream_attr, _hanging_stream(started))

    app = _build_app(backend, stats_collector, metrics_collector)
    before = _active_requests(endpoint)

    sent = await _post_and_disconnect(app, endpoint, body)

    # The disconnect really happened mid-stream: a 200 SSE response had
    # already begun and the backend generator had been resumed past its
    # first `yield`.
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 200
    assert any(m["type"] == "http.response.body" and m.get("body") for m in sent)
    assert started.is_set()

    # ★ the leak detector. `record_request_start` has `inc()`'d the gauge
    # and only `record_request_end` `dec()`s it, so an exit that skips
    # `complete_request` leaves in-flight above where it started forever.
    assert _active_requests(endpoint) == before

    stats = stats_collector.get_stats()
    assert stats["total_requests"] == 1
    assert stats["failed_requests"] == 1

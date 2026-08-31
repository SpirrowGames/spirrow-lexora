"""O-8 at the altitude where it hurts: a 500 that leaks lexora's own types.

The detector for the defect itself lives next to it, in
`tests/backends/test_anthropic.py`. This file pins the harm: a caller who
asked a passthrough tier for a completion, and whose upstream answered an
ordinary `429 {"error": "Too Many Requests"}`, received
`500 {"detail": "'str' object has no attribute 'get'"}` — the status the
feature promises to forward replaced by a sentence about this gateway's
internals.

One route test, not a matrix: the same `_handle_error_response` sits
behind every Anthropic path, so four doors would re-measure one
predicate. Everything below the ASGI layer is the shipped object — a real
`AnthropicBackend`, the real route — with `httpx.MockTransport` standing
in for the network alone.
"""

import json
from unittest.mock import MagicMock

import httpx
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
from lexora.backends.anthropic import AnthropicBackend
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector

UPSTREAM_BODY = {"error": "Too Many Requests"}


@pytest.fixture
def backend() -> AnthropicBackend:
    """A real frontier backend whose upstream answers 429 with a string `error`."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            content=json.dumps(UPSTREAM_BODY).encode(),
            headers={"Retry-After": "30"},
        )

    backend = AnthropicBackend(name="frontier", error_passthrough=True)
    backend._client = httpx.AsyncClient(
        base_url=backend.base_url, transport=httpx.MockTransport(handler)
    )
    return backend


@pytest.fixture
def client(backend: AnthropicBackend) -> TestClient:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=backend)
    backend_router.resolve_model = MagicMock(return_value="claude-fable-5")
    backend_router.get_backend_name_for_model = MagicMock(return_value="frontier")
    backend_router.is_tier = MagicMock(return_value=True)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: backend
    app.dependency_overrides[get_backend_router] = lambda: backend_router
    app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
    app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
        max_retries=1, base_delay=0.01, max_delay=0.1, jitter=False
    )
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
        default_rate=1000.0, default_burst=1000
    )
    app.dependency_overrides[is_rate_limit_enabled] = lambda: True
    app.dependency_overrides[get_metrics_collector] = lambda: None
    app.dependency_overrides[get_cost_tracker] = lambda: None
    return TestClient(app)


def test_string_shaped_upstream_error_is_forwarded_not_500(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "frontier", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 429, response.text
    # The body is a dict, so O-7's predicate forwards it verbatim.
    assert response.json() == UPSTREAM_BODY
    # The failure this pins is not merely "wrong status": it is that the
    # caller was handed a sentence about lexora's own types.
    assert "has no attribute" not in response.text

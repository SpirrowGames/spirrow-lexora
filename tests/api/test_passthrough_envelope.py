"""O-7: the OpenAI-family error envelope must stay a JSON object.

`/v1/messages` has wrapped a non-anthropic-shaped upstream body since D-4′
(`_anthropic_passthrough_body`, "a shape mismatch never becomes data
loss"). The OpenAI-family endpoints were the other half of that decision
and it was never made: all four passthrough call sites handed `e.body`
straight to `JSONResponse`, so a plaintext upstream page (a WAF's HTML
502, a proxy's text/plain) came back as the JSON *string*
`"<html>...</html>"` under `content-type: application/json` — a different
response shape than every other error these endpoints can return.

Reachability is a property of the shipped code, not of these mocks:
`anthropic.py` sets ``body = response.text or None`` when
``response.json()`` raises on the non-streaming path and
``body = error_text or None`` on the streaming path, and `base.py`
documents the field as "raw text (str) when the body was not JSON".

★ The `list` case is the point of this file, not padding. A predicate
written as "is it a str" passes every string test here and still emits a
JSON array as the whole envelope; only "is it a dict" holds. The dict
cases are the other half of the contract: verbatim passthrough is the
behaviour being *protected*, so wrapping a well-formed upstream error
object would itself be the data loss.
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

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
from lexora.backends.base import BackendUpstreamError
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector

PLAINTEXT = "<html><body>502 Bad Gateway</body></html>"


def _upstream_error(body: object) -> BackendUpstreamError:
    return BackendUpstreamError(
        "API error (502): upstream said no",
        status_code=502,
        body=body,
        backend_name="frontier",
    )


def _raising_stream(exc: BaseException):
    """A backend stream whose first `__anext__` raises."""

    def factory(_request: dict) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            raise exc
            yield b""  # pragma: no cover - unreachable, makes this a generator

        return gen()

    return MagicMock(side_effect=factory)


@pytest.fixture
def mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.chat_completions = AsyncMock()
    backend.completions = AsyncMock()
    backend.error_passthrough = True
    return backend


@pytest.fixture
def client(mock_backend: MagicMock) -> TestClient:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=mock_backend)
    backend_router.resolve_model = MagicMock(return_value="claude-fable-5")
    backend_router.get_backend_name_for_model = MagicMock(return_value="frontier")
    backend_router.is_tier = MagicMock(return_value=True)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: mock_backend
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


CHAT_STREAM = {
    "model": "frontier",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
}
CHAT_PLAIN = {"model": "frontier", "messages": [{"role": "user", "content": "Hi"}]}
COMPLETIONS_STREAM = {"model": "frontier", "prompt": "Hi", "stream": True}
COMPLETIONS_PLAIN = {"model": "frontier", "prompt": "Hi"}

# The four call sites the fix replaced, as (id, endpoint, request body,
# name of the backend attribute to arm, streaming?).
CALL_SITES = [
    ("chat-streaming-preflight", "/v1/chat/completions", CHAT_STREAM,
     "chat_completions_stream", True),
    ("chat-non-streaming", "/v1/chat/completions", CHAT_PLAIN,
     "chat_completions", False),
    ("completions-streaming-preflight", "/v1/completions", COMPLETIONS_STREAM,
     "completions_stream", True),
    ("completions-non-streaming", "/v1/completions", COMPLETIONS_PLAIN,
     "completions", False),
]
CALL_SITE_PARAMS = [
    pytest.param(endpoint, body, attr, streaming, id=name)
    for name, endpoint, body, attr, streaming in CALL_SITES
]


def _arm(mock_backend: MagicMock, attr: str, streaming: bool, body: object) -> None:
    error = _upstream_error(body)
    if streaming:
        setattr(mock_backend, attr, _raising_stream(error))
    else:
        setattr(mock_backend, attr, AsyncMock(side_effect=error))


class TestOpenAIPassthroughEnvelope:
    @pytest.mark.parametrize(
        ("endpoint", "body", "attr", "streaming"), CALL_SITE_PARAMS
    )
    def test_plaintext_upstream_body_is_wrapped_in_an_object(
        self,
        client: TestClient,
        mock_backend: MagicMock,
        endpoint: str,
        body: dict,
        attr: str,
        streaming: bool,
    ) -> None:
        """A non-JSON upstream page must not become a bare JSON string."""
        _arm(mock_backend, attr, streaming, PLAINTEXT)

        response = client.post(endpoint, json=body)

        assert response.status_code == 502
        payload = response.json()
        assert isinstance(payload, dict), f"envelope was {type(payload).__name__}"
        assert isinstance(payload["error"], dict)
        assert payload["error"]["type"] == "upstream_error"
        # D-4′: preserved, not dropped.
        assert payload["error"]["upstream"] == PLAINTEXT

    @pytest.mark.parametrize(
        ("endpoint", "body", "attr", "streaming"), CALL_SITE_PARAMS
    )
    def test_dict_upstream_body_is_forwarded_verbatim(
        self,
        client: TestClient,
        mock_backend: MagicMock,
        endpoint: str,
        body: dict,
        attr: str,
        streaming: bool,
    ) -> None:
        """The behaviour being protected: a well-formed body is untouched."""
        upstream = {"error": {"type": "refusal", "message": "declined", "code": 7}}
        _arm(mock_backend, attr, streaming, upstream)

        response = client.post(endpoint, json=body)

        assert response.status_code == 502
        assert response.json() == upstream

    @pytest.mark.parametrize(
        ("endpoint", "body", "attr", "streaming"), CALL_SITE_PARAMS
    )
    def test_list_upstream_body_is_wrapped_in_an_object(
        self,
        client: TestClient,
        mock_backend: MagicMock,
        endpoint: str,
        body: dict,
        attr: str,
        streaming: bool,
    ) -> None:
        """★ Detects a predicate written as "is it a str".

        A JSON body that decodes to a list breaks the envelope exactly like
        a string does, and every string assertion above stays green while
        it happens.
        """
        upstream = [{"message": "declined"}]
        _arm(mock_backend, attr, streaming, upstream)

        response = client.post(endpoint, json=body)

        assert response.status_code == 502
        payload = response.json()
        assert isinstance(payload, dict), f"envelope was {type(payload).__name__}"
        assert payload["error"]["upstream"] == upstream

    @pytest.mark.parametrize(
        ("endpoint", "body", "attr", "streaming"), CALL_SITE_PARAMS
    )
    def test_absent_upstream_body_still_yields_the_envelope(
        self,
        client: TestClient,
        mock_backend: MagicMock,
        endpoint: str,
        body: dict,
        attr: str,
        streaming: bool,
    ) -> None:
        """`body is None`: one envelope, no `error.upstream` key."""
        _arm(mock_backend, attr, streaming, None)

        response = client.post(endpoint, json=body)

        assert response.status_code == 502
        payload = response.json()
        assert isinstance(payload, dict)
        assert payload["error"]["type"] == "upstream_error"
        assert "upstream" not in payload["error"]

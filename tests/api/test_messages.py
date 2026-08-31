"""Tests for the Anthropic Messages API-compatible endpoint (/v1/messages)."""

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexora.api.anthropic_compat import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
)
from lexora.api.routes import (
    get_backend,
    get_backend_router,
    get_metrics_collector,
    get_rate_limiter,
    get_retry_handler,
    get_stats_collector,
    is_rate_limit_enabled,
    router,
)
from lexora.backends.base import BackendError
from lexora.backends.gemini import GeminiGovernanceError
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector


# --- Pure translation unit tests -------------------------------------------------


class TestRequestTranslation:
    """anthropic_to_openai_request."""

    def test_basic_request(self) -> None:
        openai_req = anthropic_to_openai_request(
            {
                "model": "naysayer",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": "Review this."}],
            }
        )
        assert openai_req["model"] == "naysayer"
        assert openai_req["max_tokens"] == 512
        assert openai_req["messages"] == [{"role": "user", "content": "Review this."}]

    def test_system_string_becomes_system_message(self) -> None:
        openai_req = anthropic_to_openai_request(
            {
                "model": "naysayer",
                "max_tokens": 10,
                "system": "You are a skeptic.",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        assert openai_req["messages"][0] == {
            "role": "system",
            "content": "You are a skeptic.",
        }
        assert openai_req["messages"][1]["role"] == "user"

    def test_system_block_list_is_flattened(self) -> None:
        openai_req = anthropic_to_openai_request(
            {
                "model": "naysayer",
                "max_tokens": 10,
                "system": [
                    {"type": "text", "text": "Part A"},
                    {"type": "text", "text": "Part B"},
                ],
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        assert openai_req["messages"][0]["content"] == "Part A\n\nPart B"

    def test_text_content_blocks_converted(self) -> None:
        openai_req = anthropic_to_openai_request(
            {
                "model": "naysayer",
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
            }
        )
        assert openai_req["messages"][0]["content"] == [
            {"type": "text", "text": "hello"}
        ]

    def test_non_text_blocks_forwarded_verbatim(self) -> None:
        """Image/tool blocks must survive so backend gates can reject them."""
        image_block = {"type": "image", "source": {"type": "base64", "data": "..."}}
        openai_req = anthropic_to_openai_request(
            {
                "model": "naysayer",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": [image_block]}],
            }
        )
        assert openai_req["messages"][0]["content"] == [image_block]

    def test_tools_forwarded(self) -> None:
        openai_req = anthropic_to_openai_request(
            {
                "model": "naysayer",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"name": "lookup", "input_schema": {}}],
            }
        )
        assert "tools" in openai_req

    def test_optional_params_mapped(self) -> None:
        openai_req = anthropic_to_openai_request(
            {
                "model": "naysayer",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
                "temperature": 0.3,
                "top_p": 0.8,
                "stop_sequences": ["END"],
            }
        )
        assert openai_req["temperature"] == 0.3
        assert openai_req["top_p"] == 0.8
        assert openai_req["stop"] == ["END"]


class TestResponseTranslation:
    """openai_to_anthropic_response."""

    def test_basic_response(self) -> None:
        anthropic = openai_to_anthropic_response(
            {
                "id": "chatcmpl-abc",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "No."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
            model="naysayer",
        )
        assert anthropic["type"] == "message"
        assert anthropic["role"] == "assistant"
        assert anthropic["model"] == "naysayer"
        assert anthropic["content"] == [{"type": "text", "text": "No."}]
        assert anthropic["stop_reason"] == "end_turn"
        assert anthropic["usage"] == {"input_tokens": 12, "output_tokens": 3}
        assert anthropic["id"].startswith("msg_")

    def test_length_maps_to_max_tokens(self) -> None:
        anthropic = openai_to_anthropic_response(
            {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]},
            model="m",
        )
        assert anthropic["stop_reason"] == "max_tokens"

    def test_content_filter_maps_to_refusal(self) -> None:
        anthropic = openai_to_anthropic_response(
            {
                "choices": [
                    {"message": {"content": ""}, "finish_reason": "content_filter"}
                ]
            },
            model="m",
        )
        assert anthropic["stop_reason"] == "refusal"


# --- Route integration tests -----------------------------------------------------


def _make_async_stream(chunks: list[bytes]) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        for c in chunks:
            yield c

    return gen()


class TestMessagesEndpoint:
    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        backend = MagicMock()
        backend.chat_completions = AsyncMock()
        # Default off — MagicMock auto-creates truthy attributes on demand,
        # so leaving this unset makes routes.py treat every mock backend
        # as an error_passthrough backend (skipping retry + forwarding
        # upstream statuses). Individual tests flip it to True as needed.
        backend.error_passthrough = False
        return backend

    @pytest.fixture
    def mock_backend_router(self, mock_backend: MagicMock) -> MagicMock:
        br = MagicMock()
        br.get_backend_for_model = MagicMock(return_value=mock_backend)
        br.resolve_model = MagicMock(side_effect=lambda m: "gemini-3.1-pro-preview")
        br.get_backend_name_for_model = MagicMock(return_value="gemini")
        return br

    @pytest.fixture
    def client(
        self,
        mock_backend: MagicMock,
        mock_backend_router: MagicMock,
    ) -> TestClient:
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
        app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
            max_retries=1, base_delay=0.01, max_delay=0.1, jitter=False
        )
        app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
            default_rate=1000.0, default_burst=1000
        )
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None
        return TestClient(app)

    def test_non_streaming_success(
        self, client: TestClient, mock_backend: MagicMock, mock_backend_router: MagicMock
    ) -> None:
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Disagree."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }

        resp = client.post(
            "/v1/messages",
            json={
                "model": "naysayer",
                "max_tokens": 1024,
                "system": "Be skeptical.",
                "messages": [{"role": "user", "content": "Is this plan sound?"}],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["model"] == "naysayer"
        assert data["content"][0]["text"] == "Disagree."
        assert data["stop_reason"] == "end_turn"
        assert data["usage"]["input_tokens"] == 20

        # Tier routing: requested tier resolved to the backend model.
        mock_backend_router.get_backend_for_model.assert_called_once_with("naysayer")
        call_arg = mock_backend.chat_completions.call_args[0][0]
        assert call_arg["model"] == "gemini-3.1-pro-preview"
        # System extracted into a leading system message.
        assert call_arg["messages"][0]["role"] == "system"

    def test_governance_refusal_returns_400(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        mock_backend.chat_completions.side_effect = GeminiGovernanceError(
            "data-governance gate: tools not permitted"
        )
        resp = client.post(
            "/v1/messages",
            json={
                "model": "naysayer",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"name": "t", "input_schema": {}}],
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"

    def test_backend_error_returns_502(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        mock_backend.chat_completions.side_effect = BackendError("upstream down")
        resp = client.post(
            "/v1/messages",
            json={
                "model": "naysayer",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert resp.status_code == 502
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "api_error"

    def test_streaming_emits_anthropic_events(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        openai_chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant","content":""},'
            b'"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"No"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock_backend.chat_completions_stream = MagicMock(
            return_value=_make_async_stream(openai_chunks)
        )

        resp = client.post(
            "/v1/messages",
            json={
                "model": "naysayer",
                "max_tokens": 10,
                "stream": True,
                "messages": [{"role": "user", "content": "Review"}],
            },
        )

        assert resp.status_code == 200
        body = resp.text
        # Anthropic event sequence present.
        assert "event: message_start" in body
        assert "event: content_block_start" in body
        assert "event: content_block_delta" in body
        assert "event: content_block_stop" in body
        assert "event: message_delta" in body
        assert "event: message_stop" in body

        # The text delta carries the streamed token.
        deltas = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ") and '"text_delta"' in line
        ]
        assert deltas[0]["delta"]["text"] == "No"

    def test_streaming_governance_refusal_returns_400(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """A governance refusal at stream start surfaces as a clean 400, not SSE."""

        async def boom() -> AsyncIterator[bytes]:
            raise GeminiGovernanceError("tools not permitted")
            yield b""  # pragma: no cover

        mock_backend.chat_completions_stream = MagicMock(return_value=boom())

        resp = client.post(
            "/v1/messages",
            json={
                "model": "naysayer",
                "max_tokens": 10,
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    def test_upstream_error_default_returns_anthropic_502(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """T-frontier-tier D-4′: without passthrough, upstream 4xx still collapses
        to 502 with the Anthropic error envelope (behaviour unchanged for
        naysayer/claude/etc.)."""
        from lexora.backends.base import BackendUpstreamError

        mock_backend.chat_completions.side_effect = BackendUpstreamError(
            "API error (400): declined",
            status_code=400,
            body={"type": "error", "error": {"type": "refusal", "message": "declined"}},
            backend_name="claude",
        )
        resp = client.post(
            "/v1/messages",
            json={
                "model": "naysayer",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        # Non-passthrough backend → collapsed to 502 as before.
        assert resp.status_code == 502
        assert resp.json()["error"]["type"] == "api_error"

    def test_upstream_error_passthrough_forwards_anthropic_body_verbatim(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """T-frontier-tier D-4′: passthrough backends whose upstream body is
        already Anthropic-shaped forward it unchanged, so the SDK parses it
        natively — a Fable/Opus classifier decline reaches the caller with
        the same shape it would have from api.anthropic.com."""
        from lexora.backends.base import BackendUpstreamError

        mock_backend.error_passthrough = True
        body = {"type": "error", "error": {"type": "refusal", "message": "declined"}}
        mock_backend.chat_completions.side_effect = BackendUpstreamError(
            "API error (400): declined",
            status_code=400,
            body=body,
            backend_name="frontier",
        )
        resp = client.post(
            "/v1/messages",
            json={
                "model": "frontier",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert resp.status_code == 400
        assert resp.json() == body

    def test_upstream_error_passthrough_wraps_non_anthropic_body(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """T-frontier-tier D-4′: a passthrough backend whose upstream body is
        NOT Anthropic-shaped (a plaintext error page, an OpenAI-shape body
        from a future non-anthropic passthrough backend) still gets its
        status preserved, and the raw body is carried inside the endpoint's
        envelope rather than dropped."""
        from lexora.backends.base import BackendUpstreamError

        mock_backend.error_passthrough = True
        mock_backend.chat_completions.side_effect = BackendUpstreamError(
            "API error (500): boom",
            status_code=500,
            body="upstream 500 page",
            backend_name="frontier",
        )
        resp = client.post(
            "/v1/messages",
            json={
                "model": "frontier",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "api_error"
        assert body["error"]["upstream"] == "upstream 500 page"

    def test_passthrough_disables_retry_on_messages(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """T-frontier-tier D-4′: /v1/messages retry policy respects
        error_passthrough — one billed call, not four, on a 429."""
        from lexora.backends.base import BackendRateLimitError

        mock_backend.error_passthrough = True
        mock_backend.chat_completions.side_effect = BackendRateLimitError(
            "rate limited", retry_after=1.0, backend_name="frontier"
        )
        resp = client.post(
            "/v1/messages",
            json={
                "model": "frontier",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert resp.status_code == 502  # not passthrough-shaped; but call count matters
        assert mock_backend.chat_completions.await_count == 1

    def test_non_passthrough_messages_still_retries(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Baseline: default /v1/messages still retries rate-limits."""
        from lexora.backends.base import BackendRateLimitError

        mock_backend.chat_completions.side_effect = BackendRateLimitError(
            "rate limited", retry_after=0.001, backend_name="claude"
        )
        resp = client.post(
            "/v1/messages",
            json={
                "model": "naysayer",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert resp.status_code == 502
        # RetryHandler in the fixture uses max_retries=1 → 2 calls total.
        assert mock_backend.chat_completions.await_count == 2

    def test_rate_limit_returns_anthropic_429(
        self, mock_backend: MagicMock, mock_backend_router: MagicMock
    ) -> None:
        app = FastAPI()
        app.include_router(router)
        strict_limiter = RateLimiter(default_rate=1.0, default_burst=1)
        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
        app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
            max_retries=0, base_delay=0.01, jitter=False
        )
        app.dependency_overrides[get_rate_limiter] = lambda: strict_limiter
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None
        rl_client = TestClient(app)

        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-1",
            "choices": [
                {"message": {"content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {},
        }
        payload = {
            "model": "naysayer",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "x"}],
        }
        assert rl_client.post("/v1/messages", json=payload).status_code == 200
        second = rl_client.post("/v1/messages", json=payload)
        assert second.status_code == 429
        assert second.json()["error"]["type"] == "rate_limit_error"
        assert "Retry-After" in second.headers

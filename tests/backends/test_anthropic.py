"""Tests for Anthropic backend."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lexora.backends.anthropic import (
    AnthropicBackend,
    ANTHROPIC_VERSION,
    DEFAULT_MAX_TOKENS,
)
from lexora.backends.base import (
    BackendConnectionError,
    BackendError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnavailableError,
)


class TestAnthropicBackendInit:
    """Tests for AnthropicBackend initialization."""

    def test_default_values(self):
        backend = AnthropicBackend()
        assert backend.base_url == "https://api.anthropic.com"
        assert backend.api_key is None
        assert backend.model_mapping == {}
        assert backend.name is None

    def test_custom_url(self):
        backend = AnthropicBackend(base_url="https://custom.api.com/")
        assert backend.base_url == "https://custom.api.com"

    def test_with_api_key(self):
        backend = AnthropicBackend(api_key="sk-ant-test")
        assert backend.api_key == "sk-ant-test"

    def test_with_model_mapping(self):
        mapping = {"claude-3": "claude-sonnet-4-20250514"}
        backend = AnthropicBackend(model_mapping=mapping)
        assert backend.model_mapping == mapping

    def test_with_name(self):
        backend = AnthropicBackend(name="claude_prod")
        assert backend.name == "claude_prod"

    def test_headers_include_anthropic_version(self):
        backend = AnthropicBackend(api_key="sk-ant-test")
        headers = backend._client.headers
        assert headers["anthropic-version"] == ANTHROPIC_VERSION
        assert headers["x-api-key"] == "sk-ant-test"

    def test_headers_without_api_key(self):
        backend = AnthropicBackend()
        headers = backend._client.headers
        assert headers["anthropic-version"] == ANTHROPIC_VERSION
        assert "x-api-key" not in headers


class TestRequestConversion:
    """Tests for OpenAI → Anthropic request conversion."""

    @pytest.fixture
    def backend(self):
        return AnthropicBackend(name="test")

    def test_basic_conversion(self, backend):
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        result = backend._to_anthropic_request(request)

        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["max_tokens"] == 100

    def test_system_message_extraction(self, backend):
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = backend._to_anthropic_request(request)

        assert result["system"] == "You are helpful."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_multiple_system_messages(self, backend):
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": "Rule 1"},
                {"role": "system", "content": "Rule 2"},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = backend._to_anthropic_request(request)

        assert result["system"] == "Rule 1\n\nRule 2"
        assert len(result["messages"]) == 1

    def test_default_max_tokens(self, backend):
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = backend._to_anthropic_request(request)
        assert result["max_tokens"] == DEFAULT_MAX_TOKENS

    def test_default_max_tokens_configurable(self):
        """T-frontier-tier D-7: BackendSettings.default_max_tokens is honoured.

        A frontier tier that reserves a larger output budget (e.g. reasoning
        models where thinking eats into the completion budget) needs to raise
        this ceiling without touching the module constant.
        """
        backend = AnthropicBackend(default_max_tokens=8000)
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = backend._to_anthropic_request(request)
        assert result["max_tokens"] == 8000

    def test_default_max_tokens_none_falls_back(self):
        backend = AnthropicBackend(default_max_tokens=None)
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = backend._to_anthropic_request(request)
        assert result["max_tokens"] == DEFAULT_MAX_TOKENS

    def test_explicit_max_tokens_wins_over_default(self):
        backend = AnthropicBackend(default_max_tokens=8000)
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        result = backend._to_anthropic_request(request)
        assert result["max_tokens"] == 100

    def test_explicit_none_max_tokens_falls_back_to_default(self):
        """A present-but-None max_tokens must reach the configured default.

        ``request.get("max_tokens", default)`` finds the key and returns None,
        skipping the default and handing Anthropic a non-int it rejects. No
        HTTP route can deliver this None today (the OpenAI-shaped routes
        serialise with exclude_none), but backends are also called with a
        plain dict from inside the process, where nothing strips it.
        """
        backend = AnthropicBackend(default_max_tokens=8000)
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": None,
        }
        result = backend._to_anthropic_request(request)
        assert result["max_tokens"] == 8000

    def test_explicit_zero_max_tokens_is_not_replaced_by_default(self):
        """Regression detector for an `or`-shaped fix of the None case.

        ``request.get("max_tokens") or self.default_max_tokens`` also swallows
        an explicit 0, silently substituting the configured default for a
        limit the caller stated -- a successful request the caller did not
        ask for. 0 is invalid upstream and must stay 0 so it is rejected
        loudly. Missing and invalid are different things.
        """
        backend = AnthropicBackend(default_max_tokens=8000)
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0,
        }
        result = backend._to_anthropic_request(request)
        assert result["max_tokens"] == 0

    def test_optional_params_passed(self, backend):
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["END"],
        }
        result = backend._to_anthropic_request(request)

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stop_sequences"] == ["END"]

    def test_unsupported_params_removed(self, backend):
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "logprobs": True,
        }
        result = backend._to_anthropic_request(request)

        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result
        assert "logprobs" not in result

    def test_model_mapping_applied(self, backend):
        backend.model_mapping = {"claude-3": "claude-sonnet-4-20250514"}
        request = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = backend._to_anthropic_request(request)
        assert result["model"] == "claude-sonnet-4-20250514"

    def test_system_content_blocks(self, backend):
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be helpful."}],
                },
                {"role": "user", "content": "Hello"},
            ],
        }
        result = backend._to_anthropic_request(request)
        assert result["system"] == "Be helpful."

    def test_no_system_message(self, backend):
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = backend._to_anthropic_request(request)
        assert "system" not in result


class TestResponseConversion:
    """Tests for Anthropic → OpenAI response conversion."""

    @pytest.fixture
    def backend(self):
        return AnthropicBackend(name="test")

    def test_basic_conversion(self, backend):
        anthropic_resp = {
            "id": "msg_123",
            "type": "message",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = backend._to_openai_response(anthropic_resp, "claude-sonnet-4-20250514")

        assert result["object"] == "chat.completion"
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["id"].startswith("chatcmpl-")
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_max_tokens_stop_reason(self, backend):
        anthropic_resp = {
            "id": "msg_456",
            "content": [{"type": "text", "text": "Truncated..."}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 100},
        }
        result = backend._to_openai_response(anthropic_resp, "claude-sonnet-4-20250514")
        assert result["choices"][0]["finish_reason"] == "length"

    def test_multiple_content_blocks(self, backend):
        anthropic_resp = {
            "id": "msg_789",
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": " Part 2"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        result = backend._to_openai_response(anthropic_resp, "claude-sonnet-4-20250514")
        assert result["choices"][0]["message"]["content"] == "Part 1 Part 2"

    def test_empty_content(self, backend):
        anthropic_resp = {
            "id": "msg_000",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
        result = backend._to_openai_response(anthropic_resp, "claude-sonnet-4-20250514")
        assert result["choices"][0]["message"]["content"] == ""

    def test_refusal_stop_reason_maps_to_content_filter(self, backend):
        """T-frontier-tier D-4-4: safety classifier refusal is visible.

        Previously the `refusal` stop_reason (emitted by the Anthropic
        safety classifier on Fable/Opus 5-class models) fell through the
        map to `stop`, making a decline indistinguishable from an ordinary
        completion. The frontier tier spec requires the classifier's signal
        to survive translation.
        """
        anthropic_resp = {
            "id": "msg_refuse",
            "content": [{"type": "text", "text": "I can't help with that."}],
            "stop_reason": "refusal",
            "usage": {"input_tokens": 10, "output_tokens": 6},
        }
        result = backend._to_openai_response(anthropic_resp, "claude-sonnet-4-20250514")
        assert result["choices"][0]["finish_reason"] == "content_filter"

    def test_unknown_stop_reason_still_produces_valid_shape(self, backend, caplog):
        """Unmapped stop_reason coerces to `stop` and logs the raw value.

        The OpenAI response shape stays valid, but the raw upstream signal
        is recorded so a future new refusal / safety code is not silently
        rounded off.
        """
        anthropic_resp = {
            "id": "msg_novel",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "some_future_reason",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        result = backend._to_openai_response(anthropic_resp, "claude-sonnet-4-20250514")
        assert result["choices"][0]["finish_reason"] == "stop"


class TestErrorHandling:
    """Tests for error response handling."""

    @pytest.fixture
    def backend(self):
        return AnthropicBackend(name="test")

    def test_429_rate_limit(self, backend):
        response = MagicMock()
        response.status_code = 429
        response.headers = {"Retry-After": "60"}

        with pytest.raises(BackendRateLimitError) as exc_info:
            backend._handle_error_response(response)

        assert exc_info.value.retry_after == 60.0
        assert exc_info.value.backend_name == "test"

    def test_503_unavailable(self, backend):
        response = MagicMock()
        response.status_code = 503
        response.headers = {}

        with pytest.raises(BackendUnavailableError):
            backend._handle_error_response(response)

    def test_529_overloaded(self, backend):
        response = MagicMock()
        response.status_code = 529
        response.headers = {}

        with pytest.raises(BackendUnavailableError):
            backend._handle_error_response(response)

    def test_400_error_with_json(self, backend):
        response = MagicMock()
        response.status_code = 400
        response.headers = {}
        response.json.return_value = {"error": {"message": "Invalid request"}}

        with pytest.raises(BackendError) as exc_info:
            backend._handle_error_response(response)

        assert "Invalid request" in str(exc_info.value)

    def test_400_raises_upstream_error_with_body_and_status(self, backend):
        """T-frontier-tier D-4: upstream 4xx carries status + parsed body."""
        from lexora.backends.base import BackendUpstreamError

        response = MagicMock()
        response.status_code = 400
        response.headers = {}
        response.json.return_value = {"error": {"type": "refusal", "message": "declined"}}

        with pytest.raises(BackendUpstreamError) as exc_info:
            backend._handle_error_response(response)

        assert exc_info.value.status_code == 400
        assert exc_info.value.body == {"error": {"type": "refusal", "message": "declined"}}
        assert exc_info.value.backend_name == "test"

    def test_upstream_error_falls_back_to_text_body(self, backend):
        """A non-JSON error body is preserved as a string, not dropped."""
        from lexora.backends.base import BackendUpstreamError

        response = MagicMock()
        response.status_code = 500
        response.headers = {}
        response.json.side_effect = ValueError("not JSON")
        response.text = "internal server error page"

        with pytest.raises(BackendUpstreamError) as exc_info:
            backend._handle_error_response(response)

        assert exc_info.value.status_code == 500
        assert exc_info.value.body == "internal server error page"

    def test_200_no_error(self, backend):
        response = MagicMock()
        response.status_code = 200
        response.headers = {}

        # Should not raise
        backend._handle_error_response(response)


class TestUnsupportedMethods:
    """Tests for methods not supported by Anthropic API."""

    @pytest.fixture
    def backend(self):
        return AnthropicBackend(name="test")

    @pytest.mark.asyncio
    async def test_completions_raises(self, backend):
        with pytest.raises(BackendError, match="not supported"):
            await backend.completions({"prompt": "Hello"})

    @pytest.mark.asyncio
    async def test_embeddings_raises(self, backend):
        with pytest.raises(BackendError, match="not supported"):
            await backend.embeddings({"input": "Hello"})


class TestChatCompletions:
    """Tests for chat completions endpoint."""

    @pytest.fixture
    def backend(self):
        return AnthropicBackend(
            api_key="sk-ant-test",
            model_mapping={"claude-3": "claude-sonnet-4-20250514"},
            name="test_backend",
        )

    @pytest.mark.asyncio
    async def test_successful_request(self, backend):
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = anthropic_response

        with patch.object(
            backend._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            request = {
                "model": "claude-3",
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 100,
            }
            result = await backend.chat_completions(request)

            assert result["choices"][0]["message"]["content"] == "Hello!"
            assert result["choices"][0]["finish_reason"] == "stop"
            assert result["usage"]["prompt_tokens"] == 10

            # Verify Anthropic format was sent
            call_args = mock_post.call_args
            sent_body = call_args[1]["json"]
            assert sent_body["model"] == "claude-sonnet-4-20250514"
            assert sent_body["system"] == "Be helpful."
            assert len(sent_body["messages"]) == 1

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "30"}

        with patch.object(
            backend._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            request = {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hi"}],
            }

            with pytest.raises(BackendRateLimitError) as exc_info:
                await backend.chat_completions(request)

            assert exc_info.value.retry_after == 30.0


class TestListModels:
    """Tests for list_models."""

    @pytest.mark.asyncio
    async def test_returns_empty_list(self):
        backend = AnthropicBackend()
        result = await backend.list_models()
        assert result == {"object": "list", "data": []}


class TestHealthCheck:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_healthy(self):
        backend = AnthropicBackend(api_key="sk-ant-test")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(
            backend._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response
            result = await backend.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy_server_error(self):
        backend = AnthropicBackend(api_key="sk-ant-test")

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch.object(
            backend._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response
            result = await backend.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_unhealthy_connection_error(self):
        backend = AnthropicBackend()

        with patch.object(
            backend._client, "post", new_callable=AsyncMock
        ) as mock_post:
            import httpx

            mock_post.side_effect = httpx.ConnectError("Connection refused")
            result = await backend.health_check()
            assert result is False


class TestClose:
    """Tests for closing the backend."""

    @pytest.mark.asyncio
    async def test_close(self):
        backend = AnthropicBackend()

        with patch.object(
            backend._client, "aclose", new_callable=AsyncMock
        ) as mock_close:
            await backend.close()
            mock_close.assert_called_once()


class TestParseRetryAfter:
    """Tests for Retry-After header parsing."""

    def test_parse_seconds(self):
        response = MagicMock()
        response.headers = {"Retry-After": "30"}
        assert AnthropicBackend._parse_retry_after(response) == 30.0

    def test_parse_float(self):
        response = MagicMock()
        response.headers = {"Retry-After": "1.5"}
        assert AnthropicBackend._parse_retry_after(response) == 1.5

    def test_parse_missing(self):
        response = MagicMock()
        response.headers = {}
        assert AnthropicBackend._parse_retry_after(response) is None

    def test_parse_invalid(self):
        response = MagicMock()
        response.headers = {"Retry-After": "invalid"}
        assert AnthropicBackend._parse_retry_after(response) is None


# --- T-frontier-tier PR-B fix round: O-4 / O-5 -----------------------------------


class _FakeStreamResponse:
    """Minimal stand-in for the httpx response yielded by `client.stream`."""

    def __init__(
        self,
        status_code: int,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    async def aread(self) -> bytes:
        return self._body

    async def aiter_lines(self):  # pragma: no cover - never reached on error paths
        return
        yield ""


class _FakeStreamCM:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


async def _drain_stream(backend: AnthropicBackend, response: _FakeStreamResponse):
    """Run chat_completions_stream against a canned error response."""
    backend._client.stream = MagicMock(return_value=_FakeStreamCM(response))
    request = {
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 16,
    }
    async for _ in backend.chat_completions_stream(request):
        pass


class TestErrorPassthroughStatusPreservation:
    """O-4: 429 / 503 / 529 must survive as themselves under passthrough.

    The public contract of `error_passthrough` (config.py, base.py,
    README) promises the upstream's own status and body, rate limits
    included. Before this round 429 and 503/529 were converted to
    `BackendRateLimitError` / `BackendUnavailableError` *before* the
    `>= 400` branch; neither is a `BackendUpstreamError`, so the routes'
    generic handler flattened them into 502 — the exact shape the feature
    says it does not emit.

    The fix is deliberately confined to the opted-in backend: the
    exception classes keep their parents (they are in
    `RETRYABLE_EXCEPTIONS`, and re-parenting would move every tier's
    retry semantics — msg-011 D-8 territory). Both halves are pinned
    here, because "non-passthrough is unchanged" is the load-bearing
    half.
    """

    @staticmethod
    def _response(status: int, retry_after: str | None = None) -> MagicMock:
        response = MagicMock()
        response.status_code = status
        response.headers = {} if retry_after is None else {"Retry-After": retry_after}
        response.json.return_value = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "slow down"},
        }
        response.text = "slow down"
        return response

    def test_passthrough_429_becomes_upstream_error_with_retry_after(self):
        from lexora.backends.base import BackendUpstreamError

        backend = AnthropicBackend(name="frontier", error_passthrough=True)

        with pytest.raises(BackendUpstreamError) as exc_info:
            backend._handle_error_response(self._response(429, "30"))

        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 30.0
        assert exc_info.value.body["error"]["type"] == "rate_limit_error"

    def test_non_passthrough_429_still_raises_rate_limit_error(self):
        backend = AnthropicBackend(name="heavy")

        with pytest.raises(BackendRateLimitError) as exc_info:
            backend._handle_error_response(self._response(429, "30"))

        assert exc_info.value.retry_after == 30.0

    @pytest.mark.parametrize("status", [503, 529])
    def test_passthrough_unavailable_becomes_upstream_error(self, status: int):
        from lexora.backends.base import BackendUpstreamError

        backend = AnthropicBackend(name="frontier", error_passthrough=True)

        with pytest.raises(BackendUpstreamError) as exc_info:
            backend._handle_error_response(self._response(status))

        assert exc_info.value.status_code == status

    @pytest.mark.parametrize("status", [503, 529])
    def test_non_passthrough_unavailable_unchanged(self, status: int):
        backend = AnthropicBackend(name="heavy")

        with pytest.raises(BackendUnavailableError):
            backend._handle_error_response(self._response(status))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 503, 529])
    async def test_streaming_passthrough_preserves_status(self, status: int):
        from lexora.backends.base import BackendUpstreamError

        backend = AnthropicBackend(name="frontier", error_passthrough=True)
        response = _FakeStreamResponse(
            status,
            body=json.dumps({"error": {"message": "nope"}}).encode(),
            headers={"Retry-After": "12"},
        )

        with pytest.raises(BackendUpstreamError) as exc_info:
            await _drain_stream(backend, response)

        assert exc_info.value.status_code == status
        assert exc_info.value.retry_after == 12.0

    @pytest.mark.asyncio
    async def test_streaming_non_passthrough_still_rate_limits(self):
        backend = AnthropicBackend(name="heavy")
        response = _FakeStreamResponse(429, headers={"Retry-After": "12"})

        with pytest.raises(BackendRateLimitError):
            await _drain_stream(backend, response)


class TestStreamingErrorBodyDecoding:
    """O-5: an undecodable error body must not crash the async generator.

    Two call sites shared one defect. The `except` branch called
    `bytes.decode()` with no `errors=`, and — less obvious — so did the
    default argument of `body.get("error", {}).get("message", ...)`, which
    Python evaluates eagerly even when the key is present. A UTF-16 body
    reaches that second site precisely because `json.loads` *succeeds* on
    it (RFC 4627 encoding auto-detection) while `bytes.decode()` does not.

    Nothing catches `UnicodeDecodeError` around the generator, so either
    site turned a forwardable upstream answer into a hard 500.
    """

    @pytest.mark.asyncio
    async def test_invalid_utf8_error_body_yields_upstream_error(self):
        from lexora.backends.base import BackendUpstreamError

        backend = AnthropicBackend(name="frontier", error_passthrough=True)
        response = _FakeStreamResponse(
            500, body=b"\xff\xfe\x00binary WAF payload\x80\x81"
        )

        with pytest.raises(BackendUpstreamError) as exc_info:
            await _drain_stream(backend, response)

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_utf16_json_error_body_yields_upstream_error(self):
        """The site the gate missed: valid JSON that plain `.decode()` rejects."""
        from lexora.backends.base import BackendUpstreamError

        backend = AnthropicBackend(name="frontier", error_passthrough=True)
        payload = json.dumps({"error": {"message": "declined"}}).encode("utf-16")
        response = _FakeStreamResponse(400, body=payload)

        with pytest.raises(BackendUpstreamError) as exc_info:
            await _drain_stream(backend, response)

        assert exc_info.value.status_code == 400
        assert "declined" in str(exc_info.value)


def _error_response(status: int, raw: str) -> "httpx.Response":
    """A real `httpx.Response` whose body is exactly `raw`.

    Built from `content=` rather than `json=` on purpose: `.text` has to be
    the literal bytes the upstream sent, because the message this handler
    falls back to *is* `response.text`.
    """
    import httpx

    return httpx.Response(
        status,
        content=raw.encode(),
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


class TestNonStreamingErrorBodyShapes:
    """O-8: the `error` member of an error body may not be an object.

    `isinstance(body, dict)` guards the *outer* payload only. When the
    upstream answers `{"error": "Too Many Requests"}` or `{"error": [...]}` ,
    `body.get("error", {})` returns that str / list and `.get("message")` on
    it raises `AttributeError` — from outside any `try`, so it escapes the
    backend and the route alike and the caller gets a 500 whose body is this
    gateway's own Python internals.

    This is a regression, not an unfinished feature. On `develop` the `.get`
    chain sat *inside* the `try` that wrapped `response.json()`, so
    `AttributeError` was absorbed by `except Exception` and the message
    degraded to `response.text`. Narrowing that `try` to `response.json()`
    alone — needed to carry `body` to the passthrough call sites — moved the
    chain out from under the umbrella.

    The fix restores the `develop` message (`response.text`) rather than
    inventing a better one: the streaming twin 128 lines below already uses
    that exact shape, and one file should not hold two answers to one
    question.
    """

    @pytest.mark.parametrize(
        ("raw", "expected_body", "expected_message"),
        [
            pytest.param(
                '{"error": "Too Many Requests"}',
                {"error": "Too Many Requests"},
                '{"error": "Too Many Requests"}',
                id="error-is-a-string",
            ),
            pytest.param(
                '{"error": ["first", "second"]}',
                {"error": ["first", "second"]},
                '{"error": ["first", "second"]}',
                id="error-is-a-list",
            ),
            pytest.param(
                '{"error": {"message": "declined"}}',
                {"error": {"message": "declined"}},
                "declined",
                id="error-is-an-object",
            ),
            pytest.param(
                '{"detail": "no error key at all"}',
                {"detail": "no error key at all"},
                '{"detail": "no error key at all"}',
                id="no-error-key",
            ),
            pytest.param(
                "<html><body>502 Bad Gateway</body></html>",
                "<html><body>502 Bad Gateway</body></html>",
                "<html><body>502 Bad Gateway</body></html>",
                id="not-json-at-all",
            ),
        ],
    )
    def test_every_body_shape_raises_upstream_error(
        self, raw: str, expected_body: object, expected_message: str
    ) -> None:
        from lexora.backends.base import BackendUpstreamError

        backend = AnthropicBackend(name="frontier", error_passthrough=True)

        with pytest.raises(BackendUpstreamError) as exc_info:
            backend._handle_error_response(_error_response(400, raw))

        assert exc_info.value.status_code == 400
        # O-7 pinned verbatim passthrough; this fix must not move a byte of it.
        assert exc_info.value.body == expected_body
        assert str(exc_info.value) == f"API error (400): {expected_message}"

    def test_non_passthrough_backends_are_affected_too(self) -> None:
        """The blast radius is every Anthropic tier, not just `frontier`.

        `error_passthrough` only gates the 429 / 503 / 529 short-circuit
        above; a plain 400 reaches the same `>= 400` branch either way.
        """
        from lexora.backends.base import BackendUpstreamError

        backend = AnthropicBackend(name="heavy")

        with pytest.raises(BackendUpstreamError) as exc_info:
            backend._handle_error_response(
                _error_response(400, '{"error": "Too Many Requests"}')
            )

        assert exc_info.value.status_code == 400

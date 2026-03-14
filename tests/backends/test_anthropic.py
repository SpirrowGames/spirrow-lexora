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

"""Tests for OpenAI-compatible backend."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from lexora.backends.openai_compatible import OpenAICompatibleBackend
from lexora.backends.base import (
    BackendConnectionError,
    BackendError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnavailableError,
)


class TestOpenAICompatibleBackendInit:
    """Tests for OpenAICompatibleBackend initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        backend = OpenAICompatibleBackend()
        assert backend.base_url == "https://api.openai.com"
        assert backend.api_key is None
        assert backend.model_mapping == {}
        assert backend.name is None

    def test_custom_url(self):
        """Test custom URL initialization."""
        backend = OpenAICompatibleBackend(base_url="https://custom.api.com/v1/")
        assert backend.base_url == "https://custom.api.com/v1"

    def test_with_api_key(self):
        """Test initialization with API key."""
        backend = OpenAICompatibleBackend(api_key="sk-test-key")
        assert backend.api_key == "sk-test-key"

    def test_with_model_mapping(self):
        """Test initialization with model mapping."""
        mapping = {"gpt-4": "gpt-4-0125-preview"}
        backend = OpenAICompatibleBackend(model_mapping=mapping)
        assert backend.model_mapping == mapping

    def test_with_name(self):
        """Test initialization with backend name."""
        backend = OpenAICompatibleBackend(name="openai_prod")
        assert backend.name == "openai_prod"


class TestModelMapping:
    """Tests for model name mapping."""

    def test_map_model_with_mapping(self):
        """Test model mapping when mapping exists."""
        mapping = {"gpt-4": "gpt-4-0125-preview", "gpt-3.5": "gpt-3.5-turbo"}
        backend = OpenAICompatibleBackend(model_mapping=mapping)
        assert backend._map_model("gpt-4") == "gpt-4-0125-preview"
        assert backend._map_model("gpt-3.5") == "gpt-3.5-turbo"

    def test_map_model_without_mapping(self):
        """Test model mapping when no mapping exists (passthrough)."""
        mapping = {"gpt-4": "gpt-4-0125-preview"}
        backend = OpenAICompatibleBackend(model_mapping=mapping)
        assert backend._map_model("gpt-3.5-turbo") == "gpt-3.5-turbo"

    def test_apply_model_mapping(self):
        """Test applying model mapping to request."""
        mapping = {"gpt-4": "gpt-4-0125-preview"}
        backend = OpenAICompatibleBackend(model_mapping=mapping)

        request = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
        mapped = backend._apply_model_mapping(request)

        assert mapped["model"] == "gpt-4-0125-preview"
        assert mapped["messages"] == request["messages"]
        # Original should be unchanged
        assert request["model"] == "gpt-4"


class TestParseRetryAfter:
    """Tests for Retry-After header parsing."""

    def test_parse_retry_after_seconds(self):
        """Test parsing Retry-After as seconds."""
        response = MagicMock()
        response.headers = {"Retry-After": "30"}

        result = OpenAICompatibleBackend._parse_retry_after(response)
        assert result == 30.0

    def test_parse_retry_after_float(self):
        """Test parsing Retry-After as float."""
        response = MagicMock()
        response.headers = {"Retry-After": "1.5"}

        result = OpenAICompatibleBackend._parse_retry_after(response)
        assert result == 1.5

    def test_parse_retry_after_missing(self):
        """Test parsing when Retry-After is missing."""
        response = MagicMock()
        response.headers = {}

        result = OpenAICompatibleBackend._parse_retry_after(response)
        assert result is None

    def test_parse_retry_after_invalid(self):
        """Test parsing invalid Retry-After value."""
        response = MagicMock()
        response.headers = {"Retry-After": "invalid"}

        result = OpenAICompatibleBackend._parse_retry_after(response)
        assert result is None


class TestChatCompletions:
    """Tests for chat completions endpoint."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return OpenAICompatibleBackend(
            base_url="https://api.openai.com",
            api_key="sk-test",
            model_mapping={"gpt-4": "gpt-4-0125-preview"},
            name="test_backend",
        )

    @pytest.mark.asyncio
    async def test_successful_request(self, backend):
        """Test successful chat completion request."""
        expected_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"content": "Hello!"}}],
        }

        with patch.object(backend, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = expected_response

            request = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
            result = await backend.chat_completions(request)

            assert result == expected_response
            # Check that model was mapped
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][1]["model"] == "gpt-4-0125-preview"

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, backend):
        """Test 429 rate limit error handling."""
        with patch.object(backend, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = BackendRateLimitError(
                "Rate limit exceeded",
                retry_after=30.0,
                backend_name="test_backend",
            )

            request = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}

            with pytest.raises(BackendRateLimitError) as exc_info:
                await backend.chat_completions(request)

            assert exc_info.value.retry_after == 30.0
            assert exc_info.value.backend_name == "test_backend"


class TestHandleResponse:
    """Tests for response handling."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return OpenAICompatibleBackend(name="test")

    def test_429_rate_limit(self, backend):
        """Test 429 response raises BackendRateLimitError."""
        response = MagicMock()
        response.status_code = 429
        response.headers = {"Retry-After": "60"}

        with pytest.raises(BackendRateLimitError) as exc_info:
            backend._handle_response(response)

        assert exc_info.value.retry_after == 60.0
        assert exc_info.value.backend_name == "test"

    def test_503_unavailable(self, backend):
        """Test 503 response raises BackendUnavailableError."""
        response = MagicMock()
        response.status_code = 503
        response.headers = {}

        with pytest.raises(BackendUnavailableError):
            backend._handle_response(response)

    def test_400_error_with_json(self, backend):
        """Test 400 response with JSON error body."""
        response = MagicMock()
        response.status_code = 400
        response.headers = {}
        response.json.return_value = {"error": {"message": "Invalid request"}}

        with pytest.raises(BackendError) as exc_info:
            backend._handle_response(response)

        assert "Invalid request" in str(exc_info.value)

    def test_success_response(self, backend):
        """Test successful response."""
        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {"result": "success"}

        result = backend._handle_response(response)
        assert result == {"result": "success"}


class TestHealthCheck:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_healthy(self):
        """Test healthy backend."""
        backend = OpenAICompatibleBackend()

        with patch.object(backend, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"data": []}

            result = await backend.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy(self):
        """Test unhealthy backend."""
        backend = OpenAICompatibleBackend()

        with patch.object(backend, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = BackendError("Connection failed")

            result = await backend.health_check()
            assert result is False


class TestClose:
    """Tests for closing the backend."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the HTTP client."""
        backend = OpenAICompatibleBackend()

        with patch.object(backend._client, "aclose", new_callable=AsyncMock) as mock_close:
            await backend.close()
            mock_close.assert_called_once()

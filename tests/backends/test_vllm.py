"""Tests for vLLM backend."""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from lexora.backends.vllm import VLLMBackend
from lexora.backends.base import (
    BackendConnectionError,
    BackendTimeoutError,
    BackendUnavailableError,
    BackendError,
)


@pytest.fixture
def backend() -> VLLMBackend:
    """Create a vLLM backend instance for testing."""
    return VLLMBackend(base_url="http://test-vllm:8000")


class TestVLLMBackendInit:
    """Tests for VLLMBackend initialization."""

    def test_default_values(self) -> None:
        """Test default initialization values."""
        backend = VLLMBackend()
        assert backend.base_url == "http://localhost:8000"

    def test_custom_url(self) -> None:
        """Test custom URL initialization."""
        backend = VLLMBackend(base_url="http://custom:9000/")
        # Trailing slash should be stripped
        assert backend.base_url == "http://custom:9000"

    def test_custom_timeout(self) -> None:
        """Test custom timeout values."""
        backend = VLLMBackend(timeout=60.0, connect_timeout=10.0)
        assert backend._client.timeout.read == 60.0
        assert backend._client.timeout.connect == 10.0


class TestChatCompletions:
    """Tests for chat_completions method."""

    @pytest.mark.asyncio
    async def test_successful_request(self, backend: VLLMBackend) -> None:
        """Test successful chat completion request."""
        expected_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await backend.chat_completions({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            })

            assert result == expected_response
            mock_post.assert_called_once_with(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]},
            )

    @pytest.mark.asyncio
    async def test_connection_error(self, backend: VLLMBackend) -> None:
        """Test handling of connection errors."""
        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(BackendConnectionError) as exc_info:
                await backend.chat_completions({"model": "test"})

            assert "Failed to connect to vLLM" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_error(self, backend: VLLMBackend) -> None:
        """Test handling of timeout errors."""
        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ReadTimeout("Request timed out")

            with pytest.raises(BackendTimeoutError) as exc_info:
                await backend.chat_completions({"model": "test"})

            assert "timed out" in str(exc_info.value)


class TestCompletions:
    """Tests for completions method."""

    @pytest.mark.asyncio
    async def test_successful_request(self, backend: VLLMBackend) -> None:
        """Test successful completion request."""
        expected_response = {
            "id": "cmpl-123",
            "object": "text_completion",
            "choices": [{"text": "world!"}],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await backend.completions({
                "model": "test-model",
                "prompt": "Hello",
            })

            assert result == expected_response


class TestListModels:
    """Tests for list_models method."""

    @pytest.mark.asyncio
    async def test_successful_request(self, backend: VLLMBackend) -> None:
        """Test successful models list request."""
        expected_response = {
            "object": "list",
            "data": [{"id": "model-1", "object": "model"}],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response

        with patch.object(backend._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await backend.list_models()

            assert result == expected_response
            mock_get.assert_called_once_with("/v1/models")


class TestHealthCheck:
    """Tests for health_check method."""

    @pytest.mark.asyncio
    async def test_healthy(self, backend: VLLMBackend) -> None:
        """Test health check when backend is healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(backend._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await backend.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy(self, backend: VLLMBackend) -> None:
        """Test health check when backend is unhealthy."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch.object(backend._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await backend.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_connection_error_returns_false(self, backend: VLLMBackend) -> None:
        """Test health check returns False on connection error."""
        with patch.object(backend._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            result = await backend.health_check()

            assert result is False


class TestHandleResponse:
    """Tests for _handle_response method."""

    def test_503_unavailable(self, backend: VLLMBackend) -> None:
        """Test 503 status raises BackendUnavailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with pytest.raises(BackendUnavailableError):
            backend._handle_response(mock_response)

    def test_400_error_with_json_body(self, backend: VLLMBackend) -> None:
        """Test 4xx error with JSON error body."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "Invalid request"}}

        with pytest.raises(BackendError) as exc_info:
            backend._handle_response(mock_response)

        assert "Invalid request" in str(exc_info.value)

    def test_500_error_with_text_body(self, backend: VLLMBackend) -> None:
        """Test 5xx error with text body."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_response.text = "Internal server error"

        with pytest.raises(BackendError) as exc_info:
            backend._handle_response(mock_response)

        assert "Internal server error" in str(exc_info.value)


class TestClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_close(self, backend: VLLMBackend) -> None:
        """Test closing the client."""
        with patch.object(backend._client, "aclose", new_callable=AsyncMock) as mock_close:
            await backend.close()
            mock_close.assert_called_once()

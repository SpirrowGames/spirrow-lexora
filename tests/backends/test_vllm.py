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


class TestEmbeddings:
    """Tests for embeddings method."""

    @pytest.mark.asyncio
    async def test_successful_request(self, backend: VLLMBackend) -> None:
        """Test successful embeddings request."""
        expected_response = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await backend.embeddings({
                "model": "text-embedding-ada-002",
                "input": "Hello",
            })

            assert result == expected_response
            mock_post.assert_called_once_with(
                "/v1/embeddings",
                json={"model": "text-embedding-ada-002", "input": "Hello"},
            )


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


class TestChatCompletionsStream:
    """Tests for chat_completions_stream method."""

    @pytest.mark.asyncio
    async def test_successful_stream(self, backend: VLLMBackend) -> None:
        """Test successful streaming chat completion request."""
        chunks = [
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" world"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200

        async def async_iter_bytes():
            for chunk in chunks:
                yield chunk

        mock_response.aiter_bytes = async_iter_bytes
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch.object(backend._client, "stream") as mock_stream:
            mock_stream.return_value = mock_response

            received_chunks = []
            async for chunk in backend.chat_completions_stream({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            }):
                received_chunks.append(chunk)

            assert len(received_chunks) == 3
            assert b"Hello" in received_chunks[0]
            assert b"world" in received_chunks[1]

    @pytest.mark.asyncio
    async def test_stream_connection_error(self, backend: VLLMBackend) -> None:
        """Test handling of connection errors during streaming."""
        with patch.object(backend._client, "stream") as mock_stream:
            mock_stream.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(BackendConnectionError) as exc_info:
                async for _ in backend.chat_completions_stream({"model": "test"}):
                    pass

            assert "Failed to connect to vLLM" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_timeout_error(self, backend: VLLMBackend) -> None:
        """Test handling of timeout errors during streaming."""
        with patch.object(backend._client, "stream") as mock_stream:
            mock_stream.side_effect = httpx.ReadTimeout("Request timed out")

            with pytest.raises(BackendTimeoutError) as exc_info:
                async for _ in backend.chat_completions_stream({"model": "test"}):
                    pass

            assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_503_error(self, backend: VLLMBackend) -> None:
        """Test handling of 503 errors during streaming."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch.object(backend._client, "stream") as mock_stream:
            mock_stream.return_value = mock_response

            with pytest.raises(BackendUnavailableError):
                async for _ in backend.chat_completions_stream({"model": "test"}):
                    pass

    @pytest.mark.asyncio
    async def test_stream_400_error(self, backend: VLLMBackend) -> None:
        """Test handling of 4xx errors during streaming."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.aread = AsyncMock(
            return_value=b'{"error": {"message": "Invalid request"}}'
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch.object(backend._client, "stream") as mock_stream:
            mock_stream.return_value = mock_response

            with pytest.raises(BackendError) as exc_info:
                async for _ in backend.chat_completions_stream({"model": "test"}):
                    pass

            assert "Invalid request" in str(exc_info.value)


class TestCompletionsStream:
    """Tests for completions_stream method."""

    @pytest.mark.asyncio
    async def test_successful_stream(self, backend: VLLMBackend) -> None:
        """Test successful streaming completion request."""
        chunks = [
            b'data: {"id":"cmpl-1","choices":[{"text":"Hello"}]}\n\n',
            b'data: {"id":"cmpl-1","choices":[{"text":" world"}]}\n\n',
            b'data: [DONE]\n\n',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200

        async def async_iter_bytes():
            for chunk in chunks:
                yield chunk

        mock_response.aiter_bytes = async_iter_bytes
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch.object(backend._client, "stream") as mock_stream:
            mock_stream.return_value = mock_response

            received_chunks = []
            async for chunk in backend.completions_stream({
                "model": "test-model",
                "prompt": "Hello",
                "stream": True,
            }):
                received_chunks.append(chunk)

            assert len(received_chunks) == 3


class TestThinkingControls:
    """Tests for _apply_thinking_controls (chat_template_kwargs 方式)."""

    def test_no_thinking_mode_leaves_request_untouched(self) -> None:
        """thinking_mode 未設定なら request をそのまま返す。"""
        backend = VLLMBackend()
        request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        assert backend._apply_thinking_controls(request) is request

    def test_no_think_disables_thinking(self) -> None:
        """no_think は enable_thinking=False を kwargs に載せる。"""
        backend = VLLMBackend(thinking_mode="no_think")
        result = backend._apply_thinking_controls({"messages": []})
        assert result["chat_template_kwargs"] == {"enable_thinking": False}

    def test_no_think_does_not_touch_messages(self) -> None:
        """旧方式の /no_think 文字列注入が残っていないこと。"""
        backend = VLLMBackend(thinking_mode="no_think")
        messages = [{"role": "system", "content": "you are helpful"}]
        result = backend._apply_thinking_controls({"messages": messages})
        assert result["messages"] == messages
        assert "/no_think" not in str(result["messages"])

    def test_think_sets_effort(self) -> None:
        """think + reasoning_effort は両方 kwargs に載る。"""
        backend = VLLMBackend(thinking_mode="think", reasoning_effort="medium")
        result = backend._apply_thinking_controls({"messages": []})
        assert result["chat_template_kwargs"] == {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        }

    def test_think_without_effort_omits_effort(self) -> None:
        """reasoning_effort 未設定ならモデル既定に委ねる。"""
        backend = VLLMBackend(thinking_mode="think")
        result = backend._apply_thinking_controls({"messages": []})
        assert result["chat_template_kwargs"] == {"enable_thinking": True}

    def test_no_think_ignores_effort(self) -> None:
        """no_think のとき reasoning_effort は載せない。"""
        backend = VLLMBackend(thinking_mode="no_think", reasoning_effort="xhigh")
        result = backend._apply_thinking_controls({"messages": []})
        assert result["chat_template_kwargs"] == {"enable_thinking": False}

    def test_caller_kwargs_win(self) -> None:
        """呼び出し側が明示した kwargs を上書きしない。"""
        backend = VLLMBackend(thinking_mode="no_think")
        result = backend._apply_thinking_controls(
            {"messages": [], "chat_template_kwargs": {"enable_thinking": True}}
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_caller_high_effort_remapped_to_xhigh(self) -> None:
        """vLLM の Literal 'high' は Qwen3.5+ template で raise するので xhigh に寄せる。"""
        backend = VLLMBackend(thinking_mode="think")
        result = backend._apply_thinking_controls(
            {"messages": [], "reasoning_effort": "high"}
        )
        assert "reasoning_effort" not in result
        assert result["chat_template_kwargs"]["reasoning_effort"] == "xhigh"

    def test_caller_low_effort_preserved(self) -> None:
        """low/medium は template が解釈できるのでトップレベルのまま通す。"""
        backend = VLLMBackend(thinking_mode="think")
        result = backend._apply_thinking_controls(
            {"messages": [], "reasoning_effort": "low"}
        )
        assert result["reasoning_effort"] == "low"

    def test_original_request_not_mutated(self) -> None:
        """入力 dict を破壊しないこと。"""
        backend = VLLMBackend(thinking_mode="no_think")
        request = {"messages": []}
        backend._apply_thinking_controls(request)
        assert "chat_template_kwargs" not in request

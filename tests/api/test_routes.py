"""Tests for API routes."""

import pytest
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexora import __version__
from lexora.api.routes import (
    router,
    get_backend,
    get_backend_router,
    get_stats_collector,
    get_retry_handler,
    get_rate_limiter,
    is_rate_limit_enabled,
    get_metrics_collector,
)
from lexora.backends.base import BackendError
from lexora.services.stats import StatsCollector
from lexora.services.retry_handler import RetryHandler
from lexora.services.rate_limiter import RateLimiter


class TestAPI:
    """Base test class with shared fixtures."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock backend."""
        backend = MagicMock()
        backend.chat_completions = AsyncMock()
        backend.completions = AsyncMock()
        backend.embeddings = AsyncMock()
        backend.list_models = AsyncMock()
        backend.health_check = AsyncMock()
        backend.close = AsyncMock()
        return backend

    @pytest.fixture
    def mock_backend_router(self, mock_backend: MagicMock) -> MagicMock:
        """Create a mock backend router."""
        router = MagicMock()
        router.get_backend_for_model = MagicMock(return_value=mock_backend)
        router.list_all_models = AsyncMock()
        router.health_check = AsyncMock(return_value={"default": True})
        router.default_backend = mock_backend
        return router

    @pytest.fixture
    def stats_collector(self) -> StatsCollector:
        """Create a stats collector."""
        return StatsCollector()

    @pytest.fixture
    def retry_handler(self) -> RetryHandler:
        """Create a retry handler."""
        return RetryHandler(
            max_retries=1,
            base_delay=0.01,
            max_delay=0.1,
            jitter=False,
        )

    @pytest.fixture
    def rate_limiter(self) -> RateLimiter:
        """Create a rate limiter with high limits for testing."""
        return RateLimiter(default_rate=1000.0, default_burst=1000)

    @pytest.fixture
    def client(
        self,
        mock_backend: MagicMock,
        mock_backend_router: MagicMock,
        stats_collector: StatsCollector,
        retry_handler: RetryHandler,
        rate_limiter: RateLimiter,
    ) -> TestClient:
        """Create test client with dependency overrides."""
        app = FastAPI()
        app.include_router(router)

        # Override dependencies
        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: stats_collector
        app.dependency_overrides[get_retry_handler] = lambda: retry_handler
        app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None

        return TestClient(app)


class TestChatCompletions(TestAPI):
    """Tests for /v1/chat/completions endpoint."""

    def test_successful_request(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test successful chat completion request."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [
                {"message": {"role": "assistant", "content": "Hello!"}, "index": 0}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-123"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    def test_backend_error(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test handling of backend errors."""
        mock_backend.chat_completions.side_effect = BackendError("vLLM error")

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 502

    def test_invalid_request(self, client: TestClient) -> None:
        """Test handling of invalid request."""
        response = client.post(
            "/v1/chat/completions",
            json={"messages": []},  # Missing model
        )

        assert response.status_code == 422


class TestCompletions(TestAPI):
    """Tests for /v1/completions endpoint."""

    def test_successful_request(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test successful completion request."""
        mock_backend.completions.return_value = {
            "id": "cmpl-123",
            "object": "text_completion",
            "choices": [{"text": " world!", "index": 0}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt-3.5",
                "prompt": "Hello",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["text"] == " world!"


class TestEmbeddings(TestAPI):
    """Tests for /v1/embeddings endpoint."""

    def test_successful_request(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test successful embeddings request."""
        mock_backend.embeddings.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0,
                }
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": "Hello, world!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    def test_backend_error(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test handling of backend errors."""
        mock_backend.embeddings.side_effect = BackendError("vLLM error")

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": "Hello",
            },
        )

        assert response.status_code == 502

    def test_batch_input(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test embeddings with batch input."""
        mock_backend.embeddings.return_value = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": ["Hello", "World"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2


class TestListModels(TestAPI):
    """Tests for /v1/models endpoint."""

    def test_successful_request(
        self, client: TestClient, mock_backend_router: MagicMock
    ) -> None:
        """Test successful models list request."""
        mock_backend_router.list_all_models.return_value = {
            "object": "list",
            "data": [
                {"id": "gpt-4", "object": "model", "owned_by": "openai", "backend": "default"},
                {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai", "backend": "default"},
            ],
        }

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2

    def test_backend_error(
        self, client: TestClient, mock_backend_router: MagicMock
    ) -> None:
        """Test handling of backend errors."""
        mock_backend_router.list_all_models.side_effect = BackendError("vLLM unavailable")

        response = client.get("/v1/models")

        assert response.status_code == 502


class TestHealth(TestAPI):
    """Tests for /health endpoint."""

    def test_healthy(self, client: TestClient, mock_backend_router: MagicMock) -> None:
        """Test health check when all backends are healthy."""
        mock_backend_router.health_check.return_value = {"default": True}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["backends"]["default"] == "healthy"
        assert data["vllm_status"] == "healthy"
        assert "version" in data

    def test_unhealthy(self, client: TestClient, mock_backend_router: MagicMock) -> None:
        """Test health check when all backends are unhealthy."""
        mock_backend_router.health_check.return_value = {"default": False}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["backends"]["default"] == "unhealthy"
        assert data["vllm_status"] == "unhealthy"

    def test_degraded(self, client: TestClient, mock_backend_router: MagicMock) -> None:
        """Test health check when some backends are unhealthy."""
        mock_backend_router.health_check.return_value = {"backend1": True, "backend2": False}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["backends"]["backend1"] == "healthy"
        assert data["backends"]["backend2"] == "unhealthy"


class TestStats(TestAPI):
    """Tests for /stats endpoint."""

    def test_empty_stats(self, client: TestClient) -> None:
        """Test stats endpoint with no requests."""
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 0
        assert data["successful_requests"] == 0
        assert data["failed_requests"] == 0
        assert data["success_rate"] == 0.0

    def test_stats_after_requests(
        self,
        client: TestClient,
        mock_backend: MagicMock,
        stats_collector: StatsCollector,
    ) -> None:
        """Test stats endpoint after making requests."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        # Make a request
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 1
        assert data["successful_requests"] == 1
        assert data["total_tokens_input"] == 10
        assert data["total_tokens_output"] == 5


class TestRateLimiting(TestAPI):
    """Tests for rate limiting."""

    @pytest.fixture
    def strict_rate_limiter(self) -> RateLimiter:
        """Create a rate limiter with strict limits for testing."""
        return RateLimiter(default_rate=1.0, default_burst=2)

    @pytest.fixture
    def rate_limited_client(
        self,
        mock_backend: MagicMock,
        mock_backend_router: MagicMock,
        stats_collector: StatsCollector,
        retry_handler: RetryHandler,
        strict_rate_limiter: RateLimiter,
    ) -> TestClient:
        """Create test client with strict rate limiting."""
        app = FastAPI()
        app.include_router(router)

        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: stats_collector
        app.dependency_overrides[get_retry_handler] = lambda: retry_handler
        app.dependency_overrides[get_rate_limiter] = lambda: strict_rate_limiter
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None

        return TestClient(app)

    def test_rate_limit_exceeded(
        self, rate_limited_client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test rate limiting returns 429."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        # First two requests should succeed (burst=2)
        for _ in range(2):
            response = rate_limited_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            )
            assert response.status_code == 200

        # Third request should be rate limited
        response = rate_limited_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_rate_limit_disabled(
        self,
        mock_backend: MagicMock,
        mock_backend_router: MagicMock,
        stats_collector: StatsCollector,
        retry_handler: RetryHandler,
        strict_rate_limiter: RateLimiter,
    ) -> None:
        """Test rate limiting can be disabled."""
        app = FastAPI()
        app.include_router(router)

        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: stats_collector
        app.dependency_overrides[get_retry_handler] = lambda: retry_handler
        app.dependency_overrides[get_rate_limiter] = lambda: strict_rate_limiter
        app.dependency_overrides[is_rate_limit_enabled] = lambda: False  # Disabled
        app.dependency_overrides[get_metrics_collector] = lambda: None

        client = TestClient(app)

        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "usage": {},
        }

        # All requests should succeed when rate limiting disabled
        for _ in range(5):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            )
            assert response.status_code == 200


class TestStreamingChatCompletions(TestAPI):
    """Tests for streaming /v1/chat/completions endpoint."""

    @pytest.fixture
    def streaming_mock_backend(self) -> MagicMock:
        """Create a mock backend with streaming support."""
        backend = MagicMock()
        backend.chat_completions = AsyncMock()
        backend.completions = AsyncMock()
        backend.list_models = AsyncMock()
        backend.health_check = AsyncMock()
        backend.close = AsyncMock()

        # Mock streaming methods
        async def mock_stream() -> AsyncIterator[bytes]:
            chunks = [
                b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}\n\n',
                b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" world"}}]}\n\n',
                b'data: [DONE]\n\n',
            ]
            for chunk in chunks:
                yield chunk

        backend.chat_completions_stream = MagicMock(return_value=mock_stream())
        backend.completions_stream = MagicMock(return_value=mock_stream())

        return backend

    @pytest.fixture
    def streaming_mock_backend_router(self, streaming_mock_backend: MagicMock) -> MagicMock:
        """Create a mock backend router with streaming backend."""
        router = MagicMock()
        router.get_backend_for_model = MagicMock(return_value=streaming_mock_backend)
        router.list_all_models = AsyncMock()
        router.health_check = AsyncMock(return_value={"default": True})
        router.default_backend = streaming_mock_backend
        return router

    @pytest.fixture
    def streaming_client(
        self,
        streaming_mock_backend: MagicMock,
        streaming_mock_backend_router: MagicMock,
        stats_collector: StatsCollector,
        retry_handler: RetryHandler,
        rate_limiter: RateLimiter,
    ) -> TestClient:
        """Create test client with streaming backend."""
        app = FastAPI()
        app.include_router(router)

        app.dependency_overrides[get_backend] = lambda: streaming_mock_backend
        app.dependency_overrides[get_backend_router] = lambda: streaming_mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: stats_collector
        app.dependency_overrides[get_retry_handler] = lambda: retry_handler
        app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None

        return TestClient(app)

    def test_streaming_chat_completion(
        self, streaming_client: TestClient, streaming_mock_backend: MagicMock
    ) -> None:
        """Test streaming chat completion request."""
        response = streaming_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Verify we got streaming data
        content = response.content
        assert b"Hello" in content
        assert b"world" in content

    def test_streaming_completion(
        self, streaming_client: TestClient, streaming_mock_backend: MagicMock
    ) -> None:
        """Test streaming completion request."""
        # Reset the mock to provide fresh generator
        async def mock_stream() -> AsyncIterator[bytes]:
            chunks = [
                b'data: {"id":"cmpl-1","choices":[{"text":"Hello"}]}\n\n',
                b'data: {"id":"cmpl-1","choices":[{"text":" world"}]}\n\n',
                b'data: [DONE]\n\n',
            ]
            for chunk in chunks:
                yield chunk

        streaming_mock_backend.completions_stream = MagicMock(return_value=mock_stream())

        response = streaming_client.post(
            "/v1/completions",
            json={
                "model": "gpt-4",
                "prompt": "Hello",
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_non_streaming_request_still_works(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test that non-streaming requests still work with stream=False."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [
                {"message": {"role": "assistant", "content": "Hello!"}, "index": 0}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-123"

    def test_streaming_rate_limit(
        self,
        streaming_mock_backend: MagicMock,
        streaming_mock_backend_router: MagicMock,
        stats_collector: StatsCollector,
        retry_handler: RetryHandler,
    ) -> None:
        """Test rate limiting applies to streaming requests."""
        strict_limiter = RateLimiter(default_rate=1.0, default_burst=1)

        app = FastAPI()
        app.include_router(router)

        app.dependency_overrides[get_backend] = lambda: streaming_mock_backend
        app.dependency_overrides[get_backend_router] = lambda: streaming_mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: stats_collector
        app.dependency_overrides[get_retry_handler] = lambda: retry_handler
        app.dependency_overrides[get_rate_limiter] = lambda: strict_limiter
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None

        client = TestClient(app)

        # First request should succeed
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200

        # Reset mock for second request
        async def mock_stream() -> AsyncIterator[bytes]:
            yield b'data: {"id":"chatcmpl-1"}\n\n'

        streaming_mock_backend.chat_completions_stream = MagicMock(
            return_value=mock_stream()
        )

        # Second request should be rate limited
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 429

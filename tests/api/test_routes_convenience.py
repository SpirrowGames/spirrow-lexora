"""Tests for convenience API routes (/generate and /chat)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexora.api.routes import (
    router,
    get_backend,
    get_backend_router,
    get_stats_collector,
    get_retry_handler,
    get_rate_limiter,
    is_rate_limit_enabled,
    get_metrics_collector,
    get_model_registry,
)
from lexora.backends.base import BackendError
from lexora.services.stats import StatsCollector
from lexora.services.retry_handler import RetryHandler
from lexora.services.rate_limiter import RateLimiter
from lexora.services.model_registry import ModelRegistry
from lexora.config import RoutingSettings, BackendSettings, ModelInfo


class TestConvenienceAPI:
    """Base test class with shared fixtures for convenience endpoints."""

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
        backend_router = MagicMock()
        backend_router.get_backend_for_model = MagicMock(return_value=mock_backend)
        backend_router.list_all_models = AsyncMock()
        backend_router.health_check = AsyncMock(return_value={"default": True})
        backend_router.default_backend = mock_backend
        return backend_router

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
    def mock_model_registry(self) -> MagicMock:
        """Create a mock model registry with default model."""
        registry = MagicMock(spec=ModelRegistry)
        registry.get_default_model_for_unknown_task.return_value = "default-model"
        return registry

    @pytest.fixture
    def mock_model_registry_no_default(self) -> MagicMock:
        """Create a mock model registry without default model."""
        registry = MagicMock(spec=ModelRegistry)
        registry.get_default_model_for_unknown_task.return_value = None
        return registry

    @pytest.fixture
    def client(
        self,
        mock_backend: MagicMock,
        mock_backend_router: MagicMock,
        stats_collector: StatsCollector,
        retry_handler: RetryHandler,
        rate_limiter: RateLimiter,
        mock_model_registry: MagicMock,
    ) -> TestClient:
        """Create test client with dependency overrides."""
        app = FastAPI()
        app.include_router(router)

        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: stats_collector
        app.dependency_overrides[get_retry_handler] = lambda: retry_handler
        app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None
        app.dependency_overrides[get_model_registry] = lambda: mock_model_registry

        return TestClient(app)

    @pytest.fixture
    def client_no_default_model(
        self,
        mock_backend: MagicMock,
        mock_backend_router: MagicMock,
        stats_collector: StatsCollector,
        retry_handler: RetryHandler,
        rate_limiter: RateLimiter,
        mock_model_registry_no_default: MagicMock,
    ) -> TestClient:
        """Create test client without default model configured."""
        app = FastAPI()
        app.include_router(router)

        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: stats_collector
        app.dependency_overrides[get_retry_handler] = lambda: retry_handler
        app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None
        app.dependency_overrides[get_model_registry] = lambda: mock_model_registry_no_default

        return TestClient(app)


class TestGenerateEndpoint(TestConvenienceAPI):
    """Tests for /generate endpoint."""

    def test_successful_request_with_model(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test successful generation with explicit model."""
        mock_backend.completions.return_value = {
            "id": "cmpl-123",
            "object": "text_completion",
            "choices": [{"text": "Hello, world!", "index": 0}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        response = client.post(
            "/generate",
            json={
                "prompt": "Say hello",
                "model": "test-model",
                "max_tokens": 100,
                "temperature": 0.5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello, world!"

        # Verify backend was called correctly
        mock_backend.completions.assert_called_once()
        call_args = mock_backend.completions.call_args[0][0]
        assert call_args["model"] == "test-model"
        assert call_args["prompt"] == "Say hello"
        assert call_args["max_tokens"] == 100
        assert call_args["temperature"] == 0.5

    def test_successful_request_with_default_model(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test successful generation using default model."""
        mock_backend.completions.return_value = {
            "id": "cmpl-123",
            "choices": [{"text": "Generated text", "index": 0}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }

        response = client.post(
            "/generate",
            json={"prompt": "Test prompt"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Generated text"

        # Verify default model was used
        call_args = mock_backend.completions.call_args[0][0]
        assert call_args["model"] == "default-model"

    def test_no_model_and_no_default(
        self, client_no_default_model: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test error when no model specified and no default configured."""
        response = client_no_default_model.post(
            "/generate",
            json={"prompt": "Test prompt"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "No model specified" in data["detail"]

    def test_backend_error(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test handling of backend errors."""
        mock_backend.completions.side_effect = BackendError("Backend unavailable")

        response = client.post(
            "/generate",
            json={"prompt": "Test", "model": "test-model"},
        )

        assert response.status_code == 502
        assert "Backend unavailable" in response.json()["detail"]

    def test_empty_choices(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test error when backend returns empty choices."""
        mock_backend.completions.return_value = {
            "id": "cmpl-123",
            "choices": [],
            "usage": {},
        }

        response = client.post(
            "/generate",
            json={"prompt": "Test", "model": "test-model"},
        )

        assert response.status_code == 500
        assert "No choices" in response.json()["detail"]

    def test_extra_parameters_passthrough(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test that extra parameters are passed through."""
        mock_backend.completions.return_value = {
            "id": "cmpl-123",
            "choices": [{"text": "Result", "index": 0}],
            "usage": {},
        }

        response = client.post(
            "/generate",
            json={
                "prompt": "Test",
                "model": "test-model",
                "top_p": 0.9,
                "stop": ["\n"],
            },
        )

        assert response.status_code == 200

        # Verify extra params were passed
        call_args = mock_backend.completions.call_args[0][0]
        assert call_args["top_p"] == 0.9
        assert call_args["stop"] == ["\n"]

    def test_default_values(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test that default values are applied."""
        mock_backend.completions.return_value = {
            "id": "cmpl-123",
            "choices": [{"text": "Result", "index": 0}],
            "usage": {},
        }

        response = client.post(
            "/generate",
            json={"prompt": "Test"},
        )

        assert response.status_code == 200

        # Verify defaults were applied
        call_args = mock_backend.completions.call_args[0][0]
        assert call_args["max_tokens"] == 1000
        assert call_args["temperature"] == 0.7


class TestChatEndpoint(TestConvenienceAPI):
    """Tests for /chat endpoint."""

    def test_successful_request_with_model(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test successful chat with explicit model."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [
                {"message": {"role": "assistant", "content": "Hello! How can I help?"}, "index": 0}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

        response = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi there"}],
                "model": "test-model",
                "max_tokens": 200,
                "temperature": 0.8,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Hello! How can I help?"

        # Verify backend was called correctly
        mock_backend.chat_completions.assert_called_once()
        call_args = mock_backend.chat_completions.call_args[0][0]
        assert call_args["model"] == "test-model"
        assert call_args["messages"] == [{"role": "user", "content": "Hi there"}]
        assert call_args["max_tokens"] == 200
        assert call_args["temperature"] == 0.8

    def test_successful_request_with_default_model(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test successful chat using default model."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "choices": [
                {"message": {"role": "assistant", "content": "Response text"}, "index": 0}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }

        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Response text"

        # Verify default model was used
        call_args = mock_backend.chat_completions.call_args[0][0]
        assert call_args["model"] == "default-model"

    def test_no_model_and_no_default(
        self, client_no_default_model: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test error when no model specified and no default configured."""
        response = client_no_default_model.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

        assert response.status_code == 400
        data = response.json()
        assert "No model specified" in data["detail"]

    def test_multi_turn_conversation(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test chat with multiple messages."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "choices": [
                {"message": {"role": "assistant", "content": "Python is great!"}, "index": 0}
            ],
            "usage": {},
        }

        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What's your favorite language?"},
                    {"role": "assistant", "content": "I like Python."},
                    {"role": "user", "content": "Why?"},
                ],
                "model": "test-model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Python is great!"

        # Verify all messages were passed
        call_args = mock_backend.chat_completions.call_args[0][0]
        assert len(call_args["messages"]) == 4

    def test_backend_error(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test handling of backend errors."""
        mock_backend.chat_completions.side_effect = BackendError("Backend error")

        response = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "model": "test-model",
            },
        )

        assert response.status_code == 502
        assert "Backend error" in response.json()["detail"]

    def test_empty_choices(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test error when backend returns empty choices."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "choices": [],
            "usage": {},
        }

        response = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "model": "test-model",
            },
        )

        assert response.status_code == 500
        assert "No choices" in response.json()["detail"]

    def test_extra_parameters_passthrough(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test that extra parameters are passed through."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "choices": [
                {"message": {"role": "assistant", "content": "OK"}, "index": 0}
            ],
            "usage": {},
        }

        response = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "model": "test-model",
                "top_p": 0.95,
                "presence_penalty": 0.5,
            },
        )

        assert response.status_code == 200

        # Verify extra params were passed
        call_args = mock_backend.chat_completions.call_args[0][0]
        assert call_args["top_p"] == 0.95
        assert call_args["presence_penalty"] == 0.5

    def test_default_values(
        self, client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test that default values are applied."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "choices": [
                {"message": {"role": "assistant", "content": "Hi"}, "index": 0}
            ],
            "usage": {},
        }

        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

        assert response.status_code == 200

        # Verify defaults were applied
        call_args = mock_backend.chat_completions.call_args[0][0]
        assert call_args["max_tokens"] == 1000
        assert call_args["temperature"] == 0.7


class TestConvenienceRateLimiting(TestConvenienceAPI):
    """Tests for rate limiting on convenience endpoints."""

    @pytest.fixture
    def strict_rate_limiter(self) -> RateLimiter:
        """Create a rate limiter with strict limits for testing."""
        return RateLimiter(default_rate=1.0, default_burst=1)

    @pytest.fixture
    def rate_limited_client(
        self,
        mock_backend: MagicMock,
        mock_backend_router: MagicMock,
        stats_collector: StatsCollector,
        retry_handler: RetryHandler,
        strict_rate_limiter: RateLimiter,
        mock_model_registry: MagicMock,
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
        app.dependency_overrides[get_model_registry] = lambda: mock_model_registry

        return TestClient(app)

    def test_generate_rate_limited(
        self, rate_limited_client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test rate limiting on /generate endpoint."""
        mock_backend.completions.return_value = {
            "id": "cmpl-123",
            "choices": [{"text": "Hello", "index": 0}],
            "usage": {},
        }

        # First request should succeed
        response = rate_limited_client.post(
            "/generate",
            json={"prompt": "Hi", "model": "test"},
        )
        assert response.status_code == 200

        # Second request should be rate limited
        response = rate_limited_client.post(
            "/generate",
            json={"prompt": "Hi again", "model": "test"},
        )
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_chat_rate_limited(
        self, rate_limited_client: TestClient, mock_backend: MagicMock
    ) -> None:
        """Test rate limiting on /chat endpoint."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "choices": [
                {"message": {"role": "assistant", "content": "Hi"}, "index": 0}
            ],
            "usage": {},
        }

        # First request should succeed
        response = rate_limited_client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "model": "test",
            },
        )
        assert response.status_code == 200

        # Second request should be rate limited
        response = rate_limited_client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test",
            },
        )
        assert response.status_code == 429


class TestConvenienceStats(TestConvenienceAPI):
    """Tests for statistics collection on convenience endpoints."""

    def test_generate_updates_stats(
        self, client: TestClient, mock_backend: MagicMock, stats_collector: StatsCollector
    ) -> None:
        """Test that /generate updates statistics."""
        mock_backend.completions.return_value = {
            "id": "cmpl-123",
            "choices": [{"text": "Generated", "index": 0}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        client.post(
            "/generate",
            json={"prompt": "Test", "model": "test-model"},
        )

        stats = stats_collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["total_tokens_input"] == 10
        assert stats["total_tokens_output"] == 5
        assert "/generate" in stats["requests_per_endpoint"]

    def test_chat_updates_stats(
        self, client: TestClient, mock_backend: MagicMock, stats_collector: StatsCollector
    ) -> None:
        """Test that /chat updates statistics."""
        mock_backend.chat_completions.return_value = {
            "id": "chatcmpl-123",
            "choices": [
                {"message": {"role": "assistant", "content": "Response"}, "index": 0}
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8},
        }

        client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
            },
        )

        stats = stats_collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["total_tokens_input"] == 15
        assert stats["total_tokens_output"] == 8
        assert "/chat" in stats["requests_per_endpoint"]

    def test_failed_request_updates_stats(
        self, client: TestClient, mock_backend: MagicMock, stats_collector: StatsCollector
    ) -> None:
        """Test that failed requests update statistics."""
        mock_backend.completions.side_effect = BackendError("Error")

        client.post(
            "/generate",
            json={"prompt": "Test", "model": "test-model"},
        )

        stats = stats_collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["failed_requests"] == 1
        assert stats["successful_requests"] == 0

"""Tests for model capabilities and task classification API routes."""

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
    get_task_classifier,
)
from lexora.config import (
    BackendSettings,
    ClassifierSettings,
    ModelInfo,
    RoutingSettings,
    VLLMSettings,
)
from lexora.services.model_registry import ModelRegistry
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.router import BackendRouter
from lexora.services.stats import StatsCollector
from lexora.services.task_classifier import (
    AlternativeModel,
    ClassificationResult,
    TaskClassifier,
    TaskClassifierDisabledError,
    TaskClassifierError,
)


class TestModelCapabilities:
    """Tests for /v1/models/capabilities endpoint."""

    @pytest.fixture
    def routing_settings(self) -> RoutingSettings:
        """Create routing settings for tests."""
        return RoutingSettings(
            enabled=True,
            default_backend="heavy",
            default_model_for_unknown_task="coder-model",
            backends={
                "heavy": BackendSettings(
                    type="vllm",
                    url="http://localhost:8000",
                    models=[
                        ModelInfo(
                            name="coder-model",
                            capabilities=["code", "reasoning", "general"],
                            description="Code generation model",
                        ),
                    ],
                ),
                "light": BackendSettings(
                    type="vllm",
                    url="http://localhost:8001",
                    models=[
                        ModelInfo(
                            name="fast-model",
                            capabilities=["summarization", "translation"],
                            description="Fast lightweight model",
                        ),
                    ],
                ),
            },
        )

    @pytest.fixture
    def model_registry(self, routing_settings: RoutingSettings) -> ModelRegistry:
        """Create model registry for tests."""
        return ModelRegistry(routing_settings)

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock backend."""
        backend = MagicMock()
        backend.chat_completions = AsyncMock()
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
    def client(
        self,
        mock_backend: MagicMock,
        mock_backend_router: MagicMock,
        model_registry: ModelRegistry,
    ) -> TestClient:
        """Create test client with dependency overrides."""
        app = FastAPI()
        app.include_router(router)

        stats_collector = StatsCollector()
        retry_handler = RetryHandler(max_retries=1, base_delay=0.01, jitter=False)
        rate_limiter = RateLimiter(default_rate=1000.0, default_burst=1000)

        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: stats_collector
        app.dependency_overrides[get_retry_handler] = lambda: retry_handler
        app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None
        app.dependency_overrides[get_model_registry] = lambda: model_registry
        app.dependency_overrides[get_task_classifier] = lambda: None

        return TestClient(app)

    def test_get_capabilities_success(
        self, client: TestClient, model_registry: ModelRegistry
    ) -> None:
        """Test successful capabilities request."""
        response = client.get("/v1/models/capabilities")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "available_capabilities" in data
        assert "default_model_for_unknown_task" in data

        assert len(data["models"]) == 2

        # Check model details
        model_ids = {m["id"] for m in data["models"]}
        assert model_ids == {"coder-model", "fast-model"}

        # Check capabilities
        capabilities = set(data["available_capabilities"])
        assert "code" in capabilities
        assert "reasoning" in capabilities
        assert "general" in capabilities
        assert "summarization" in capabilities
        assert "translation" in capabilities

        # Check default model
        assert data["default_model_for_unknown_task"] == "coder-model"

    def test_get_capabilities_model_details(self, client: TestClient) -> None:
        """Test that model details are correct."""
        response = client.get("/v1/models/capabilities")

        assert response.status_code == 200
        data = response.json()

        # Find the coder-model
        coder_model = next(m for m in data["models"] if m["id"] == "coder-model")

        assert coder_model["backend"] == "heavy"
        assert coder_model["backend_type"] == "vllm"
        assert coder_model["capabilities"] == ["code", "reasoning", "general"]
        assert coder_model["description"] == "Code generation model"

    def test_get_capabilities_no_registry(self) -> None:
        """Test capabilities endpoint with no registry."""
        app = FastAPI()
        app.include_router(router)

        mock_backend = MagicMock()
        mock_backend_router = MagicMock()
        mock_backend_router.health_check = AsyncMock(return_value={"default": True})
        mock_backend_router.default_backend = mock_backend

        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
        app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
            max_retries=1, base_delay=0.01
        )
        app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
            default_rate=1000.0, default_burst=1000
        )
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None
        app.dependency_overrides[get_model_registry] = lambda: None
        app.dependency_overrides[get_task_classifier] = lambda: None

        client = TestClient(app)

        response = client.get("/v1/models/capabilities")

        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []
        assert data["available_capabilities"] == []
        assert data["default_model_for_unknown_task"] is None


class TestClassifyTask:
    """Tests for /v1/classify-task endpoint."""

    @pytest.fixture
    def routing_settings(self) -> RoutingSettings:
        """Create routing settings for tests."""
        return RoutingSettings(
            enabled=True,
            default_backend="heavy",
            default_model_for_unknown_task="coder-model",
            classifier=ClassifierSettings(
                enabled=True,
                model="classifier-model",
                backend="light",
            ),
            backends={
                "heavy": BackendSettings(
                    type="vllm",
                    url="http://localhost:8000",
                    models=[
                        ModelInfo(
                            name="coder-model",
                            capabilities=["code", "reasoning", "general"],
                        ),
                    ],
                ),
                "light": BackendSettings(
                    type="vllm",
                    url="http://localhost:8001",
                    models=[
                        ModelInfo(
                            name="classifier-model",
                            capabilities=["classification"],
                        ),
                        ModelInfo(
                            name="fast-model",
                            capabilities=["summarization", "translation"],
                        ),
                    ],
                ),
            },
        )

    @pytest.fixture
    def model_registry(self, routing_settings: RoutingSettings) -> ModelRegistry:
        """Create model registry for tests."""
        return ModelRegistry(routing_settings)

    @pytest.fixture
    def mock_task_classifier(self) -> MagicMock:
        """Create a mock task classifier."""
        classifier = MagicMock(spec=TaskClassifier)
        classifier.enabled = True
        classifier.classify = AsyncMock()
        return classifier

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock backend."""
        backend = MagicMock()
        backend.chat_completions = AsyncMock()
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
    def client(
        self,
        mock_backend: MagicMock,
        mock_backend_router: MagicMock,
        model_registry: ModelRegistry,
        mock_task_classifier: MagicMock,
    ) -> TestClient:
        """Create test client with dependency overrides."""
        app = FastAPI()
        app.include_router(router)

        stats_collector = StatsCollector()
        retry_handler = RetryHandler(max_retries=1, base_delay=0.01, jitter=False)
        rate_limiter = RateLimiter(default_rate=1000.0, default_burst=1000)

        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: stats_collector
        app.dependency_overrides[get_retry_handler] = lambda: retry_handler
        app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None
        app.dependency_overrides[get_model_registry] = lambda: model_registry
        app.dependency_overrides[get_task_classifier] = lambda: mock_task_classifier

        return TestClient(app)

    def test_classify_task_success(
        self, client: TestClient, mock_task_classifier: MagicMock
    ) -> None:
        """Test successful task classification."""
        mock_task_classifier.classify.return_value = ClassificationResult(
            recommended_model="coder-model",
            task_type="code",
            confidence=0.95,
            reasoning="Code generation task detected",
            alternatives=[
                AlternativeModel(model="fast-model", score=0.3),
            ],
        )

        response = client.post(
            "/v1/classify-task",
            json={"task_description": "Write a Python function to sort a list"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["recommended_model"] == "coder-model"
        assert data["task_type"] == "code"
        assert data["confidence"] == 0.95
        assert data["reasoning"] == "Code generation task detected"
        assert len(data["alternatives"]) == 1
        assert data["alternatives"][0]["model"] == "fast-model"
        assert data["alternatives"][0]["score"] == 0.3

    def test_classify_task_no_classifier(self) -> None:
        """Test classify task with no classifier configured."""
        app = FastAPI()
        app.include_router(router)

        mock_backend = MagicMock()
        mock_backend_router = MagicMock()
        mock_backend_router.health_check = AsyncMock(return_value={"default": True})
        mock_backend_router.default_backend = mock_backend

        app.dependency_overrides[get_backend] = lambda: mock_backend
        app.dependency_overrides[get_backend_router] = lambda: mock_backend_router
        app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
        app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
            max_retries=1, base_delay=0.01
        )
        app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
            default_rate=1000.0, default_burst=1000
        )
        app.dependency_overrides[is_rate_limit_enabled] = lambda: True
        app.dependency_overrides[get_metrics_collector] = lambda: None
        app.dependency_overrides[get_model_registry] = lambda: None
        app.dependency_overrides[get_task_classifier] = lambda: None

        client = TestClient(app)

        response = client.post(
            "/v1/classify-task",
            json={"task_description": "Test task"},
        )

        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]

    def test_classify_task_classifier_disabled(
        self, client: TestClient, mock_task_classifier: MagicMock
    ) -> None:
        """Test classify task when classifier is disabled."""
        mock_task_classifier.classify.side_effect = TaskClassifierDisabledError(
            "Classifier disabled"
        )

        response = client.post(
            "/v1/classify-task",
            json={"task_description": "Test task"},
        )

        assert response.status_code == 503
        assert "disabled" in response.json()["detail"]

    def test_classify_task_classification_error(
        self, client: TestClient, mock_task_classifier: MagicMock
    ) -> None:
        """Test classify task with classification error."""
        mock_task_classifier.classify.side_effect = TaskClassifierError(
            "Failed to parse response"
        )

        response = client.post(
            "/v1/classify-task",
            json={"task_description": "Test task"},
        )

        assert response.status_code == 500
        assert "Failed to parse" in response.json()["detail"]

    def test_classify_task_empty_description(self, client: TestClient) -> None:
        """Test classify task with empty description."""
        response = client.post(
            "/v1/classify-task",
            json={"task_description": ""},
        )

        # Should fail validation
        assert response.status_code == 422

    def test_classify_task_missing_description(self, client: TestClient) -> None:
        """Test classify task with missing description."""
        response = client.post(
            "/v1/classify-task",
            json={},
        )

        # Should fail validation
        assert response.status_code == 422

    def test_classify_task_unexpected_error(
        self, client: TestClient, mock_task_classifier: MagicMock
    ) -> None:
        """Test classify task with unexpected error."""
        mock_task_classifier.classify.side_effect = RuntimeError("Unexpected error")

        response = client.post(
            "/v1/classify-task",
            json={"task_description": "Test task"},
        )

        assert response.status_code == 500
        assert "Unexpected error" in response.json()["detail"]

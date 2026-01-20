"""Tests for task classifier service."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from lexora.backends.base import BackendError
from lexora.config import (
    BackendSettings,
    ClassifierSettings,
    ModelInfo,
    RoutingSettings,
    VLLMSettings,
)
from lexora.services.model_registry import ModelRegistry
from lexora.services.router import BackendRouter
from lexora.services.task_classifier import (
    ClassificationResult,
    TaskClassifier,
    TaskClassifierDisabledError,
    TaskClassifierError,
)


@pytest.fixture
def routing_settings() -> RoutingSettings:
    """Create routing settings for tests."""
    return RoutingSettings(
        enabled=True,
        default_backend="heavy",
        default_model_for_unknown_task="default-model",
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
                    ModelInfo(
                        name="default-model",
                        capabilities=["general"],
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
def model_registry(routing_settings: RoutingSettings) -> ModelRegistry:
    """Create model registry for tests."""
    return ModelRegistry(routing_settings)


@pytest.fixture
def backend_router(routing_settings: RoutingSettings) -> BackendRouter:
    """Create backend router for tests."""
    vllm_settings = VLLMSettings(url="http://localhost:8000")
    return BackendRouter(
        routing_settings=routing_settings,
        vllm_settings=vllm_settings,
    )


class TestTaskClassifierInitialization:
    """Tests for TaskClassifier initialization."""

    def test_initialization(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test classifier initializes correctly."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        assert classifier.enabled is True

    def test_initialization_disabled(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
    ) -> None:
        """Test classifier initializes as disabled."""
        classifier_settings = ClassifierSettings(enabled=False)
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=classifier_settings,
        )

        assert classifier.enabled is False


class TestTaskClassifierClassify:
    """Tests for TaskClassifier.classify method."""

    @pytest.mark.asyncio
    async def test_classify_disabled_raises_error(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
    ) -> None:
        """Test that classify raises error when disabled."""
        classifier_settings = ClassifierSettings(enabled=False)
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=classifier_settings,
        )

        with pytest.raises(TaskClassifierDisabledError):
            await classifier.classify("test task")

    @pytest.mark.asyncio
    async def test_classify_backend_not_found(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
    ) -> None:
        """Test that classify raises error when backend not found."""
        classifier_settings = ClassifierSettings(
            enabled=True,
            model="test-model",
            backend="nonexistent-backend",
        )
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=classifier_settings,
        )

        with pytest.raises(TaskClassifierError, match="not found"):
            await classifier.classify("test task")

    @pytest.mark.asyncio
    async def test_classify_success(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test successful task classification."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        # Mock the backend
        mock_backend = MagicMock()
        mock_backend.chat_completions = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": '{"task_type": "code", "confidence": 0.95, "reasoning": "Code task"}'
                        }
                    }
                ]
            }
        )
        backend_router._backends["light"] = mock_backend

        result = await classifier.classify("Write a Python function")

        assert isinstance(result, ClassificationResult)
        assert result.task_type == "code"
        assert result.confidence == 0.95
        assert result.reasoning == "Code task"
        assert result.recommended_model == "coder-model"  # Model with 'code' capability

    @pytest.mark.asyncio
    async def test_classify_json_in_markdown(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test classification with JSON in markdown code block."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        mock_backend = MagicMock()
        mock_backend.chat_completions = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"task_type": "summarization", "confidence": 0.85, "reasoning": "Summary task"}\n```'
                        }
                    }
                ]
            }
        )
        backend_router._backends["light"] = mock_backend

        result = await classifier.classify("Summarize this text")

        assert result.task_type == "summarization"
        assert result.confidence == 0.85
        assert result.recommended_model == "fast-model"  # Model with 'summarization' capability

    @pytest.mark.asyncio
    async def test_classify_unknown_task_type(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test classification with unknown task type falls back."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        mock_backend = MagicMock()
        mock_backend.chat_completions = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": '{"task_type": "unknown_capability", "confidence": 0.5, "reasoning": "Unknown"}'
                        }
                    }
                ]
            }
        )
        backend_router._backends["light"] = mock_backend

        result = await classifier.classify("Do something unknown")

        # Should fall back to model with 'general' capability
        assert result.recommended_model in ["coder-model", "default-model"]

    @pytest.mark.asyncio
    async def test_classify_backend_error(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test classification handles backend error."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        mock_backend = MagicMock()
        mock_backend.chat_completions = AsyncMock(
            side_effect=BackendError("Connection failed")
        )
        backend_router._backends["light"] = mock_backend

        with pytest.raises(TaskClassifierError, match="Backend error"):
            await classifier.classify("Test task")

    @pytest.mark.asyncio
    async def test_classify_invalid_response(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test classification handles invalid response."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        mock_backend = MagicMock()
        mock_backend.chat_completions = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "This is not valid JSON"
                        }
                    }
                ]
            }
        )
        backend_router._backends["light"] = mock_backend

        with pytest.raises(TaskClassifierError, match="Failed to parse"):
            await classifier.classify("Test task")

    @pytest.mark.asyncio
    async def test_classify_alternatives(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test that classification includes alternatives."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        mock_backend = MagicMock()
        mock_backend.chat_completions = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": '{"task_type": "code", "confidence": 0.9, "reasoning": "Code task"}'
                        }
                    }
                ]
            }
        )
        backend_router._backends["light"] = mock_backend

        result = await classifier.classify("Write code")

        assert result.alternatives is not None
        assert len(result.alternatives) > 0
        # Alternatives should not include the recommended model
        for alt in result.alternatives:
            assert alt.model != result.recommended_model


class TestTaskClassifierParseResponse:
    """Tests for response parsing."""

    def test_parse_plain_json(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test parsing plain JSON response."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        text = '{"task_type": "code", "confidence": 0.9, "reasoning": "test"}'
        result = classifier._parse_classification_response(text)

        assert result["task_type"] == "code"
        assert result["confidence"] == 0.9

    def test_parse_json_with_whitespace(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test parsing JSON with leading/trailing whitespace."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        text = '  \n{"task_type": "code", "confidence": 0.9, "reasoning": "test"}\n  '
        result = classifier._parse_classification_response(text)

        assert result["task_type"] == "code"

    def test_parse_json_in_code_block(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test parsing JSON in markdown code block."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        text = '```json\n{"task_type": "code", "confidence": 0.9, "reasoning": "test"}\n```'
        result = classifier._parse_classification_response(text)

        assert result["task_type"] == "code"

    def test_parse_json_embedded_in_text(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test parsing JSON embedded in text."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        text = 'Here is my response: {"task_type": "code", "confidence": 0.9, "reasoning": "test"}'
        result = classifier._parse_classification_response(text)

        assert result["task_type"] == "code"

    def test_parse_invalid_json_raises(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test parsing invalid JSON raises error."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        text = "This is not JSON at all"

        with pytest.raises(TaskClassifierError, match="Failed to parse"):
            classifier._parse_classification_response(text)


class TestTaskClassifierAlternatives:
    """Tests for alternative model calculation."""

    def test_calculate_alternatives(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test that alternatives are calculated correctly."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        alternatives = classifier._calculate_alternatives("code", "coder-model")

        # Should have alternatives excluding the recommended model
        assert all(alt.model != "coder-model" for alt in alternatives)

        # Each alternative should have a score
        for alt in alternatives:
            assert 0.0 <= alt.score <= 1.0

    def test_alternatives_scoring(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test that alternatives are scored correctly."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        alternatives = classifier._calculate_alternatives("general", "default-model")

        # coder-model has 'general' capability, should have higher score
        coder_alt = next((a for a in alternatives if a.model == "coder-model"), None)
        assert coder_alt is not None
        assert coder_alt.score >= 0.7  # Higher score for matching capability

    def test_alternatives_sorted_by_score(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        routing_settings: RoutingSettings,
    ) -> None:
        """Test that alternatives are sorted by score descending."""
        classifier = TaskClassifier(
            model_registry=model_registry,
            backend_router=backend_router,
            classifier_settings=routing_settings.classifier,
        )

        alternatives = classifier._calculate_alternatives("code", "coder-model")

        # Should be sorted by score descending
        scores = [alt.score for alt in alternatives]
        assert scores == sorted(scores, reverse=True)

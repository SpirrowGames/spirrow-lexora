"""Tests for model registry service."""

import pytest

from lexora.config import BackendSettings, ModelInfo, RoutingSettings
from lexora.services.model_registry import ModelEntry, ModelRegistry


class TestModelRegistryInitialization:
    """Tests for ModelRegistry initialization."""

    def test_empty_registry_when_routing_disabled(self) -> None:
        """Test that registry is empty when routing is disabled."""
        routing_settings = RoutingSettings(enabled=False)

        registry = ModelRegistry(routing_settings)

        assert registry.model_count == 0
        assert registry.capability_count == 0
        assert registry.get_all_models() == []

    def test_empty_registry_when_no_backends(self) -> None:
        """Test that registry is empty when no backends configured."""
        routing_settings = RoutingSettings(enabled=True, backends={})

        registry = ModelRegistry(routing_settings)

        assert registry.model_count == 0
        assert registry.capability_count == 0

    def test_parses_models_with_capabilities(self) -> None:
        """Test that models with capabilities are parsed correctly."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="heavy",
            backends={
                "heavy": BackendSettings(
                    type="vllm",
                    url="http://localhost:8000",
                    models=[
                        ModelInfo(
                            name="model-a",
                            capabilities=["code", "reasoning"],
                            description="Test model A",
                        ),
                    ],
                ),
            },
        )

        registry = ModelRegistry(routing_settings)

        assert registry.model_count == 1
        models = registry.get_all_models()
        assert len(models) == 1
        assert models[0].id == "model-a"
        assert models[0].backend == "heavy"
        assert models[0].backend_type == "vllm"
        assert models[0].capabilities == ["code", "reasoning"]
        assert models[0].description == "Test model A"

    def test_parses_models_string_format(self) -> None:
        """Test that models in string format get default capabilities."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="backend1",
            backends={
                "backend1": BackendSettings(
                    type="vllm",
                    url="http://localhost:8000",
                    models=["model-a", "model-b"],
                ),
            },
        )

        registry = ModelRegistry(routing_settings)

        assert registry.model_count == 2
        models = registry.get_all_models()
        # String format should get default "general" capability
        for model in models:
            assert model.capabilities == ["general"]

    def test_parses_multiple_backends(self) -> None:
        """Test that models from multiple backends are parsed."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="heavy",
            backends={
                "heavy": BackendSettings(
                    type="vllm",
                    url="http://localhost:8000",
                    models=[
                        ModelInfo(
                            name="heavy-model",
                            capabilities=["code", "reasoning"],
                        ),
                    ],
                ),
                "light": BackendSettings(
                    type="vllm",
                    url="http://localhost:8001",
                    models=[
                        ModelInfo(
                            name="light-model",
                            capabilities=["summarization", "translation"],
                        ),
                    ],
                ),
            },
        )

        registry = ModelRegistry(routing_settings)

        assert registry.model_count == 2
        capabilities = registry.get_available_capabilities()
        assert "code" in capabilities
        assert "reasoning" in capabilities
        assert "summarization" in capabilities
        assert "translation" in capabilities

    def test_stores_default_model_for_unknown_task(self) -> None:
        """Test that default_model_for_unknown_task is stored."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="backend1",
            default_model_for_unknown_task="fallback-model",
            backends={
                "backend1": BackendSettings(
                    url="http://localhost:8000",
                    models=["model-a"],
                ),
            },
        )

        registry = ModelRegistry(routing_settings)

        assert registry.get_default_model_for_unknown_task() == "fallback-model"


class TestModelRegistryGetters:
    """Tests for ModelRegistry getter methods."""

    @pytest.fixture
    def registry(self) -> ModelRegistry:
        """Create a registry with test models."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="heavy",
            default_model_for_unknown_task="default-model",
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
        return ModelRegistry(routing_settings)

    def test_get_model(self, registry: ModelRegistry) -> None:
        """Test getting a specific model by ID."""
        model = registry.get_model("coder-model")

        assert model is not None
        assert model.id == "coder-model"
        assert model.backend == "heavy"

    def test_get_model_not_found(self, registry: ModelRegistry) -> None:
        """Test getting a non-existent model returns None."""
        model = registry.get_model("nonexistent")

        assert model is None

    def test_get_models_by_capability(self, registry: ModelRegistry) -> None:
        """Test getting models by capability."""
        code_models = registry.get_models_by_capability("code")

        assert len(code_models) == 1
        assert code_models[0].id == "coder-model"

    def test_get_models_by_capability_multiple(self, registry: ModelRegistry) -> None:
        """Test getting models by capability shared by multiple models."""
        # Add another model with same capability
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="heavy",
            backends={
                "heavy": BackendSettings(
                    url="http://localhost:8000",
                    models=[
                        ModelInfo(name="model-a", capabilities=["code"]),
                        ModelInfo(name="model-b", capabilities=["code"]),
                    ],
                ),
            },
        )
        registry = ModelRegistry(routing_settings)

        code_models = registry.get_models_by_capability("code")

        assert len(code_models) == 2

    def test_get_models_by_capability_empty(self, registry: ModelRegistry) -> None:
        """Test getting models by non-existent capability."""
        models = registry.get_models_by_capability("nonexistent")

        assert models == []

    def test_get_available_capabilities(self, registry: ModelRegistry) -> None:
        """Test getting all available capabilities."""
        capabilities = registry.get_available_capabilities()

        assert isinstance(capabilities, list)
        assert "code" in capabilities
        assert "reasoning" in capabilities
        assert "general" in capabilities
        assert "summarization" in capabilities
        assert "translation" in capabilities
        # Should be sorted
        assert capabilities == sorted(capabilities)


class TestModelRegistryFindBestModel:
    """Tests for ModelRegistry.find_best_model_for_capability."""

    def test_find_exact_match(self) -> None:
        """Test finding model with exact capability match."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="heavy",
            backends={
                "heavy": BackendSettings(
                    url="http://localhost:8000",
                    models=[
                        ModelInfo(name="coder", capabilities=["code"]),
                        ModelInfo(name="summarizer", capabilities=["summarization"]),
                    ],
                ),
            },
        )
        registry = ModelRegistry(routing_settings)

        best = registry.find_best_model_for_capability("code")

        assert best == "coder"

    def test_find_fallback_to_general(self) -> None:
        """Test fallback to model with 'general' capability."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="heavy",
            backends={
                "heavy": BackendSettings(
                    url="http://localhost:8000",
                    models=[
                        ModelInfo(name="general-model", capabilities=["general"]),
                        ModelInfo(name="specific-model", capabilities=["summarization"]),
                    ],
                ),
            },
        )
        registry = ModelRegistry(routing_settings)

        # Request a capability that doesn't exist
        best = registry.find_best_model_for_capability("nonexistent")

        assert best == "general-model"

    def test_find_fallback_to_default_model(self) -> None:
        """Test fallback to default_model_for_unknown_task."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="heavy",
            default_model_for_unknown_task="fallback-model",
            backends={
                "heavy": BackendSettings(
                    url="http://localhost:8000",
                    models=[
                        # No 'general' capability
                        ModelInfo(name="specific-model", capabilities=["code"]),
                    ],
                ),
            },
        )
        registry = ModelRegistry(routing_settings)

        # Request a capability that doesn't exist and no 'general' model
        best = registry.find_best_model_for_capability("nonexistent")

        assert best == "fallback-model"

    def test_find_fallback_to_first_model(self) -> None:
        """Test fallback to first available model."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="heavy",
            # No default_model_for_unknown_task
            backends={
                "heavy": BackendSettings(
                    url="http://localhost:8000",
                    models=[
                        ModelInfo(name="only-model", capabilities=["specific"]),
                    ],
                ),
            },
        )
        registry = ModelRegistry(routing_settings)

        # Request a capability that doesn't exist
        best = registry.find_best_model_for_capability("nonexistent")

        assert best == "only-model"

    def test_find_returns_none_when_empty(self) -> None:
        """Test that None is returned when registry is empty."""
        routing_settings = RoutingSettings(enabled=False)
        registry = ModelRegistry(routing_settings)

        best = registry.find_best_model_for_capability("any")

        assert best is None

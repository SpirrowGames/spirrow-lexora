"""Registry for model information and capabilities."""

from dataclasses import dataclass

from lexora.config import ModelInfo, RoutingSettings
from lexora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelEntry:
    """Complete model entry with backend information."""

    id: str
    backend: str
    backend_type: str
    capabilities: list[str]
    description: str | None


class ModelRegistry:
    """Registry for model information and capabilities.

    Parses routing settings to build a registry of all models with their
    capabilities and backend information.

    Args:
        routing_settings: Routing configuration containing backend and model info.
    """

    def __init__(self, routing_settings: RoutingSettings) -> None:
        """Initialize the model registry.

        Args:
            routing_settings: Routing configuration.
        """
        self._models: dict[str, ModelEntry] = {}
        self._capabilities: set[str] = set()
        self._default_model_for_unknown_task = routing_settings.default_model_for_unknown_task
        self._routing_settings = routing_settings

        self._parse_settings(routing_settings)

        logger.info(
            "model_registry_initialized",
            model_count=len(self._models),
            capability_count=len(self._capabilities),
            capabilities=sorted(self._capabilities),
        )

    def _parse_settings(self, routing_settings: RoutingSettings) -> None:
        """Parse routing settings to extract model information.

        Args:
            routing_settings: Routing configuration.
        """
        if not routing_settings.enabled or not routing_settings.backends:
            logger.warning(
                "model_registry_no_backends",
                routing_enabled=routing_settings.enabled,
            )
            return

        for backend_name, backend_settings in routing_settings.backends.items():
            for model_info in backend_settings.models:
                entry = ModelEntry(
                    id=model_info.name,
                    backend=backend_name,
                    backend_type=backend_settings.type,
                    capabilities=model_info.capabilities.copy(),
                    description=model_info.description,
                )
                self._models[model_info.name] = entry
                self._capabilities.update(model_info.capabilities)

                logger.debug(
                    "model_registered",
                    model=model_info.name,
                    backend=backend_name,
                    capabilities=model_info.capabilities,
                )

    def get_all_models(self) -> list[ModelEntry]:
        """Get all registered models.

        Returns:
            List of all model entries.
        """
        return list(self._models.values())

    def get_model(self, model_id: str) -> ModelEntry | None:
        """Get a specific model by ID.

        Args:
            model_id: Model identifier.

        Returns:
            ModelEntry or None if not found.
        """
        return self._models.get(model_id)

    def get_models_by_capability(self, capability: str) -> list[ModelEntry]:
        """Get all models that have a specific capability.

        Args:
            capability: Capability tag to search for.

        Returns:
            List of models with the specified capability.
        """
        return [
            model for model in self._models.values()
            if capability in model.capabilities
        ]

    def get_available_capabilities(self) -> list[str]:
        """Get all available capability tags.

        Returns:
            Sorted list of unique capability tags.
        """
        return sorted(self._capabilities)

    def get_default_model_for_unknown_task(self) -> str | None:
        """Get the default model for unknown tasks.

        Returns:
            Model ID or None if not configured.
        """
        return self._default_model_for_unknown_task

    def find_best_model_for_capability(self, capability: str) -> str | None:
        """Find the best model for a given capability.

        First looks for models with the exact capability. If none found,
        falls back to models with 'general' capability, then to the
        default model for unknown tasks.

        Args:
            capability: Capability tag.

        Returns:
            Model ID or None if no suitable model found.
        """
        # First try exact match
        models = self.get_models_by_capability(capability)
        if models:
            return models[0].id

        # Try 'general' capability
        models = self.get_models_by_capability("general")
        if models:
            return models[0].id

        # Fall back to default
        if self._default_model_for_unknown_task:
            return self._default_model_for_unknown_task

        # Last resort: return first available model
        if self._models:
            return next(iter(self._models.keys()))

        return None

    @property
    def model_count(self) -> int:
        """Get the number of registered models."""
        return len(self._models)

    @property
    def capability_count(self) -> int:
        """Get the number of unique capabilities."""
        return len(self._capabilities)

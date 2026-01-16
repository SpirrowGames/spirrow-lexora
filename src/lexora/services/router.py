"""Backend router for multi-model support."""

from typing import Any

from lexora.backends.base import Backend, BackendError
from lexora.backends.vllm import VLLMBackend
from lexora.config import BackendSettings, RoutingSettings, VLLMSettings
from lexora.utils.logging import get_logger

logger = get_logger(__name__)


class BackendRouter:
    """Routes requests to appropriate backends based on model name.

    Supports multi-backend configuration where different models can be served
    by different backend instances.

    Args:
        routing_settings: Routing configuration.
        vllm_settings: Legacy single-backend settings (used when routing disabled).
    """

    def __init__(
        self,
        routing_settings: RoutingSettings,
        vllm_settings: VLLMSettings,
    ) -> None:
        """Initialize the backend router.

        Args:
            routing_settings: Routing configuration.
            vllm_settings: Legacy single-backend settings.
        """
        self._routing_enabled = routing_settings.enabled
        self._default_backend_name = routing_settings.default_backend
        self._backends: dict[str, VLLMBackend] = {}
        self._model_to_backend: dict[str, str] = {}

        if self._routing_enabled and routing_settings.backends:
            # Multi-backend mode
            for name, settings in routing_settings.backends.items():
                self._backends[name] = VLLMBackend(
                    base_url=settings.url,
                    timeout=settings.timeout,
                    connect_timeout=settings.connect_timeout,
                )
                # Map models to this backend
                for model in settings.models:
                    self._model_to_backend[model] = name
                    logger.info(
                        "model_route_registered",
                        model=model,
                        backend=name,
                        url=settings.url,
                    )

            logger.info(
                "multi_backend_routing_enabled",
                backends=list(self._backends.keys()),
                default=self._default_backend_name,
            )
        else:
            # Single backend mode (legacy)
            self._backends["default"] = VLLMBackend(
                base_url=vllm_settings.url,
                timeout=vllm_settings.timeout,
                connect_timeout=vllm_settings.connect_timeout,
            )
            self._default_backend_name = "default"
            logger.info(
                "single_backend_mode",
                url=vllm_settings.url,
            )

    def get_backend_for_model(self, model: str) -> VLLMBackend:
        """Get the appropriate backend for a model.

        Args:
            model: Model name.

        Returns:
            Backend instance for the model.

        Raises:
            BackendError: If no backend is available for the model.
        """
        # Check if model has explicit mapping
        backend_name = self._model_to_backend.get(model)

        if backend_name is None:
            # Use default backend
            backend_name = self._default_backend_name

        backend = self._backends.get(backend_name)
        if backend is None:
            raise BackendError(
                f"No backend available for model '{model}' "
                f"(looked for backend '{backend_name}')"
            )

        logger.debug(
            "routing_request",
            model=model,
            backend=backend_name,
        )
        return backend

    @property
    def default_backend(self) -> VLLMBackend:
        """Get the default backend.

        Returns:
            Default backend instance.
        """
        return self._backends[self._default_backend_name]

    @property
    def backends(self) -> dict[str, VLLMBackend]:
        """Get all backends.

        Returns:
            Dictionary of backend name to instance.
        """
        return self._backends

    @property
    def routing_enabled(self) -> bool:
        """Check if multi-backend routing is enabled.

        Returns:
            True if routing is enabled.
        """
        return self._routing_enabled

    async def health_check(self) -> dict[str, bool]:
        """Check health of all backends.

        Returns:
            Dictionary of backend name to health status.
        """
        health = {}
        for name, backend in self._backends.items():
            health[name] = await backend.health_check()
        return health

    async def list_all_models(self) -> dict[str, Any]:
        """List models from all backends.

        Returns:
            Combined models list in OpenAI format.
        """
        all_models: list[dict[str, Any]] = []

        for name, backend in self._backends.items():
            try:
                models_response = await backend.list_models()
                for model in models_response.get("data", []):
                    # Add backend info to model
                    model["backend"] = name
                    all_models.append(model)
            except BackendError as e:
                logger.warning(
                    "list_models_backend_error",
                    backend=name,
                    error=str(e),
                )

        return {
            "object": "list",
            "data": all_models,
        }

    async def close(self) -> None:
        """Close all backends."""
        for backend in self._backends.values():
            await backend.close()

"""Backend router for multi-model support."""

from typing import Any

from lexora.backends.base import Backend, BackendError
from lexora.backends.factory import create_backend
from lexora.backends.vllm import VLLMBackend
from lexora.config import RoutingSettings, VLLMSettings
from lexora.utils.logging import get_logger

logger = get_logger(__name__)


class BackendRouter:
    """Routes requests to appropriate backends based on model name.

    Supports multi-backend configuration where different models can be served
    by different backend instances. Also supports fallback backends when
    primary backends fail or hit rate limits.

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
        self._backends: dict[str, Backend] = {}
        self._model_to_backend: dict[str, str] = {}
        self._fallback_map: dict[str, list[str]] = {}

        if self._routing_enabled and routing_settings.backends:
            # Multi-backend mode using factory
            for name, settings in routing_settings.backends.items():
                self._backends[name] = create_backend(name, settings)

                # Map models to this backend
                for model_info in settings.models:
                    self._model_to_backend[model_info.name] = name
                    logger.info(
                        "model_route_registered",
                        model=model_info.name,
                        backend=name,
                        url=settings.url,
                        type=settings.type,
                    )

                # Store fallback configuration
                if settings.fallback_backends:
                    self._fallback_map[name] = settings.fallback_backends
                    logger.info(
                        "fallback_registered",
                        backend=name,
                        fallbacks=settings.fallback_backends,
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
                name="default",
            )
            self._default_backend_name = "default"
            logger.info(
                "single_backend_mode",
                url=vllm_settings.url,
            )

    def get_backend_for_model(self, model: str) -> Backend:
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

    def get_backend_name_for_model(self, model: str) -> str:
        """Get the backend name for a model.

        Args:
            model: Model name.

        Returns:
            Backend name.
        """
        backend_name = self._model_to_backend.get(model)
        if backend_name is None:
            backend_name = self._default_backend_name
        return backend_name

    def get_fallback_backends(self, backend_name: str) -> list[Backend]:
        """Get fallback backends for a given backend.

        Args:
            backend_name: Name of the primary backend.

        Returns:
            List of fallback backend instances.
        """
        fallback_names = self._fallback_map.get(backend_name, [])
        fallbacks = []
        for name in fallback_names:
            backend = self._backends.get(name)
            if backend is not None:
                fallbacks.append(backend)
            else:
                logger.warning(
                    "fallback_backend_not_found",
                    primary=backend_name,
                    fallback=name,
                )
        return fallbacks

    def get_backend_by_name(self, name: str) -> Backend | None:
        """Get a backend by its name.

        Args:
            name: Backend name.

        Returns:
            Backend instance or None if not found.
        """
        return self._backends.get(name)

    @property
    def default_backend(self) -> Backend:
        """Get the default backend.

        Returns:
            Default backend instance.
        """
        return self._backends[self._default_backend_name]

    @property
    def backends(self) -> dict[str, Backend]:
        """Get all backends.

        Returns:
            Dictionary of backend name to instance.
        """
        return self._backends

    @property
    def fallback_map(self) -> dict[str, list[str]]:
        """Get the fallback configuration map.

        Returns:
            Dictionary of backend name to list of fallback backend names.
        """
        return self._fallback_map

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

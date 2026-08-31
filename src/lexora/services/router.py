"""Backend router for multi-model support."""

import asyncio
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
        self._backends: dict[str, Backend] = {}
        #: Backends that opted into GET /health. Absent name == checked, so
        #: legacy single-backend mode keeps its probe without saying so.
        self._health_checked: dict[str, bool] = {}
        self._model_to_backend: dict[str, str] = {}

        self._tier_to_backend: dict[str, str] = {}
        self._tier_to_model: dict[str, str] = {}

        if self._routing_enabled and routing_settings.backends:
            # Multi-backend mode using factory
            for name, settings in routing_settings.backends.items():
                self._backends[name] = create_backend(name, settings)
                self._health_checked[name] = settings.health_check
                if not settings.health_check:
                    logger.info("backend_health_check_skipped", backend=name)

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

            # Register tier mappings
            for tier_name, tier_settings in routing_settings.tiers.items():
                if tier_settings.backend in self._backends:
                    self._tier_to_backend[tier_name] = tier_settings.backend
                    # Resolve model name: explicit > backend's first model
                    backend_settings = routing_settings.backends[tier_settings.backend]
                    model_name = tier_settings.model
                    if model_name is None and backend_settings.models:
                        model_name = backend_settings.models[0].name
                    if model_name:
                        self._tier_to_model[tier_name] = model_name
                    logger.info(
                        "tier_registered",
                        tier=tier_name,
                        backend=tier_settings.backend,
                        model=model_name,
                    )
                else:
                    logger.warning(
                        "tier_backend_not_found",
                        tier=tier_name,
                        backend=tier_settings.backend,
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
        # Check tier mapping first, then model mapping, then default
        backend_name = self._tier_to_backend.get(model)
        if backend_name is None:
            backend_name = self._model_to_backend.get(model)
        if backend_name is None:
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
        backend_name = self._tier_to_backend.get(model)
        if backend_name is None:
            backend_name = self._model_to_backend.get(model)
        if backend_name is None:
            backend_name = self._default_backend_name
        return backend_name

    def resolve_model(self, model: str) -> str:
        """Resolve tier name to actual model name.

        If the model is a tier name, returns the configured model name.
        Otherwise returns the model name as-is.

        Args:
            model: Model or tier name.

        Returns:
            Resolved model name.
        """
        return self._tier_to_model.get(model, model)

    def is_tier(self, name: str) -> bool:
        """Return True if ``name`` is a registered tier alias.

        Kept separate from ``resolve_model`` because tier detection cannot be
        derived from ``resolved != requested``: a tier can legitimately share
        its concrete model name (e.g. a tier deliberately named after the
        model it fronts), and the callsite must not confuse "tier alias" with
        "coincidentally-equal name". Consumed by the cost tracker to record
        the tier alongside — and distinct from — the resolved model ID.

        Args:
            name: Candidate name.

        Returns:
            True iff ``name`` is a registered tier.
        """
        return name in self._tier_to_backend

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
    def routing_enabled(self) -> bool:
        """Check if multi-backend routing is enabled.

        Returns:
            True if routing is enabled.
        """
        return self._routing_enabled

    async def health_check(self) -> dict[str, bool | None]:
        """Check health of every backend that opted into being checked.

        ``None`` means the backend was skipped (``health_check: false``), not
        that it is unhealthy or absent -- callers must keep those apart, since
        a skipped backend still serves traffic.

        Backends are probed concurrently. Serially, the total was the sum of
        every probe, so one slow remote API set the latency of the whole
        endpoint: measured 2026-08-11, a single unauthenticated
        ``openai_compatible`` backend stalled ``GET /health`` for 20-40s while
        every other probe stayed at its usual few milliseconds.

        Returns:
            Backend name -> True (healthy) / False (unhealthy) / None (skipped).
        """
        checked = [
            (name, backend)
            for name, backend in self._backends.items()
            if self._health_checked.get(name, True)
        ]

        results = await asyncio.gather(
            *(backend.health_check() for _, backend in checked),
            return_exceptions=True,
        )

        health: dict[str, bool | None] = {
            name: None for name in self._backends if name not in dict(checked)
        }
        for (name, _), result in zip(checked, results):
            if isinstance(result, BaseException):
                # A probe that raised has not said the backend is down; it has
                # said the probe failed. Reporting False is the safe reading
                # for a health endpoint, but log it so the two are separable.
                logger.warning(
                    "backend_health_check_error", backend=name, error=str(result)
                )
                health[name] = False
            else:
                health[name] = bool(result)
        return health

    async def list_all_models(self) -> dict[str, Any]:
        """List models and tier aliases from all backends.

        Every configured tier is emitted as its own entry so callers reading
        ``/v1/models`` can see the tier names the router actually accepts
        (``naysayer``, ``heavy``, ``frontier``, ...) alongside the concrete
        model IDs that back them. Tier entries carry ``resolved_model`` so
        an env override (e.g. ``LEXORA_FRONTIER_MODEL``) is observable here
        rather than needing a separate probe.

        Every entry -- tier aliases included -- carries the four fields the
        OpenAI Model object declares: ``id`` / ``object`` / ``created`` /
        ``owned_by``. Omitting them does not raise in ``openai-python`` (it
        builds responses through a non-validating path), which is worse than
        a crash: the attributes come back as ``None`` on fields declared
        ``int`` / ``str`` and blow up later and elsewhere, in
        ``datetime.fromtimestamp(m.created)`` or a group-by on ``owned_by``.
        Strictly-validating clients (``Model.model_validate``, typed Go/TS
        clients, a schema-checking proxy) do reject the entry outright.

        The two values for a tier alias:

        * ``owned_by: "lexora"`` is a fact, not a placeholder. A tier alias is
          defined by this gateway, not by the upstream vendor that serves the
          resolved model.
        * ``created: 0`` is a sentinel meaning "no creation time exists". A
          tier alias is a config entry, not a published artifact. Mirroring
          the resolved model's ``created`` was considered and rejected: it
          would assert "this alias was created when that model was", which is
          false, and a fabricated-but-plausible timestamp is worse than an
          obvious epoch sentinel.

        The additive keys (``type`` / ``backend`` / ``resolved_model``) are
        kept: extra fields do not fail OpenAI-client validation, only missing
        declared ones do.

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

        # Tier aliases. Emitted after backend models so a listing consumer
        # reading top-down sees concrete IDs first, then the tier names that
        # route to them. `object` is left as "model" for OpenAI-client
        # compatibility, with `type: "tier"` as an additive marker.
        for tier_name, backend_name in self._tier_to_backend.items():
            all_models.append(
                {
                    "id": tier_name,
                    "object": "model",
                    # See the docstring: `owned_by` is a fact, `created: 0` is
                    # a sentinel, and neither is mirrored from the resolved
                    # model.
                    "created": 0,
                    "owned_by": "lexora",
                    "type": "tier",
                    "backend": backend_name,
                    "resolved_model": self._tier_to_model.get(tier_name),
                }
            )

        return {
            "object": "list",
            "data": all_models,
        }

    async def close(self) -> None:
        """Close all backends."""
        for backend in self._backends.values():
            await backend.close()

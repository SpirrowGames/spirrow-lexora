"""Backend router for multi-model support."""

import asyncio
from typing import Any

from lexora.backends.base import Backend, BackendError, ModelNotFoundError
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
        #: Model names declared by more than one backend. T-silent-routing
        #: R-1a: request-time routing for these names is refused (see
        #: ``get_backend_for_model``); at boot we warn once with the full
        #: list of competing backends so an operator does not have to grep
        #: ``model_route_registered`` INFO lines to find the collision.
        self._ambiguous_models: dict[str, list[str]] = {}

        self._tier_to_backend: dict[str, str] = {}
        self._tier_to_model: dict[str, str] = {}

        if self._routing_enabled and routing_settings.backends:
            # Multi-backend mode using factory.
            #
            # First pass: build ``_ambiguous_models`` from the declarations.
            # This has to complete before the second pass registers routes,
            # because the log line for a route needs to know whether the
            # name is ambiguous — otherwise an operator reading the boot
            # log would see three ``model_route_registered`` INFO lines and
            # no cue that only one of the three is reachable.
            declared_by: dict[str, list[str]] = {}
            for name, settings in routing_settings.backends.items():
                # dedupe: a single backend declaring the same model twice
                # is at most one route candidate, not two. It is a config
                # oddity but not the ambiguity R-1a is about.
                for model_name in {m.name for m in settings.models}:
                    declared_by.setdefault(model_name, []).append(name)
            for model_name, backend_names in declared_by.items():
                if len(backend_names) > 1:
                    self._ambiguous_models[model_name] = backend_names
                    logger.warning(
                        "model_declaration_ambiguous",
                        model=model_name,
                        backends=backend_names,
                        note=(
                            "same model name is declared by multiple "
                            "backends; requests naming it directly will be "
                            "refused with 404. Use a tier name instead, or "
                            "rename the model in all but one backend."
                        ),
                    )

            for name, settings in routing_settings.backends.items():
                self._backends[name] = create_backend(name, settings)
                self._health_checked[name] = settings.health_check
                if not settings.health_check:
                    logger.info("backend_health_check_skipped", backend=name)

                # Map models to this backend
                for model_info in settings.models:
                    if model_info.name in self._ambiguous_models:
                        # Do not register a by-name route for an ambiguous
                        # name: the second/third writer would silently
                        # overwrite the first, and the winner would be
                        # decided by YAML declaration order (T-silent-
                        # routing msg-078 §1.1). Leave it absent from the
                        # by-name index so ``get_backend_for_model`` treats
                        # it as "ambiguous, refuse", not "found in one
                        # arbitrary backend".
                        continue
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

        Lookup order (T-silent-routing R-1a-5): the ambiguity check applies
        only to the *requested* name. If ``model`` is a tier alias the tier
        wins, and the concrete model name that the tier resolves to is
        never itself checked for ambiguity — that concrete name is an
        internal detail of the tier's declaration, not a request field, so
        submitting the tier ``light`` must route even though the concrete
        model ``Qwen3.8-27B`` is declared by three backends.

        Args:
            model: Model name.

        Returns:
            Backend instance for the model.

        Raises:
            ModelNotFoundError: When ``model`` is neither a registered tier
                nor a declared model name (R-2: unknown), or when the name
                is declared by two or more backends (R-1a: ambiguous). Both
                cases surface to the caller as HTTP 404.
            BackendError: When the resolved backend name does not exist in
                the router's backend map. This is a configuration defect
                rather than a routing decision — it would only happen when
                the router is asked to look up a backend by a name it did
                not build (e.g., a corrupted lookup table).
        """
        # 1) Tier alias wins (public API surface).
        backend_name = self._tier_to_backend.get(model)
        if backend_name is None:
            # 2) Ambiguous by-name request: refuse rather than pick a
            # winner. This runs before the by-name lookup so an ambiguous
            # name never resolves, even if a later change accidentally
            # registered a route for it.
            if model in self._ambiguous_models:
                declared = self._ambiguous_models[model]
                logger.warning(
                    "model_ambiguous_refused",
                    model=model,
                    backends=declared,
                )
                raise ModelNotFoundError(
                    message=(
                        f"Model '{model}' is declared by multiple backends "
                        f"({', '.join(declared)}); the router cannot pick "
                        f"one without guessing. Use a tier name instead of "
                        f"a raw model name."
                    ),
                    model_name=model,
                    reason="ambiguous",
                )
            # 3) By-name index.
            backend_name = self._model_to_backend.get(model)
        if backend_name is None:
            # R-2 scope: fall-through refusal applies only in
            # multi-backend routing mode. Legacy single-backend mode
            # (``routing.enabled: false``) is the "one backend accepts
            # everything, model name is ignored" arrangement — refusing
            # unknown names there would break every existing caller and
            # is not what this branch is against (the failure this branch
            # is against is *silent* routing decisions between multiple
            # candidates, and single mode has no candidates to pick
            # between).
            if not self._routing_enabled:
                backend_name = self._default_backend_name
            else:
                # Multi-backend mode. R-2: refuse rather than silently
                # fall through to ``default_backend`` — that was the
                # 2026-08 failure ("model=frontier silently answered by
                # heavy under the vLLM name") this branch exists to fix.
                logger.warning(
                    "model_unknown_refused",
                    model=model,
                    default_backend=self._default_backend_name,
                )
                raise ModelNotFoundError(
                    message=(
                        f"Model '{model}' is not registered. See GET "
                        f"/v1/models for the tier aliases and concrete "
                        f"model names this gateway accepts."
                    ),
                    model_name=model,
                    reason="unknown",
                )

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

        Raises the same errors as ``get_backend_for_model`` for the same
        reasons: this is the name-lookup twin of the backend-lookup path,
        used by the stats collector to attribute a request to a backend.
        If the router would refuse the request in one, the answer to "which
        backend does this name go to" is not "the default" — it is "no
        answer", and callers must not silently attribute the request to a
        backend that will never see it.

        Args:
            model: Model name.

        Returns:
            Backend name.

        Raises:
            ModelNotFoundError: Same cases as ``get_backend_for_model``.
        """
        backend_name = self._tier_to_backend.get(model)
        if backend_name is None:
            if model in self._ambiguous_models:
                raise ModelNotFoundError(
                    message=(
                        f"Model '{model}' is declared by multiple backends "
                        f"({', '.join(self._ambiguous_models[model])})."
                    ),
                    model_name=model,
                    reason="ambiguous",
                )
            backend_name = self._model_to_backend.get(model)
        if backend_name is None:
            # Same single/multi-mode split as get_backend_for_model — the
            # stats attribution path must not raise in single mode where
            # get_backend_for_model does not.
            if not self._routing_enabled:
                return self._default_backend_name
            raise ModelNotFoundError(
                message=f"Model '{model}' is not registered.",
                model_name=model,
                reason="unknown",
            )
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

        #: Track which ambiguous names we have already filtered so the
        #: WARNING is issued once per name per listing, not once per copy.
        _filtered_ambiguous: set[str] = set()
        #: Track ids we have already emitted so a name declared by multiple
        #: passthrough backends (three vllm backends all proxying the same
        #: upstream /v1/models) is not returned three times.
        _seen_ids: set[str] = set()

        for name, backend in self._backends.items():
            try:
                models_response = await backend.list_models()
                for model in models_response.get("data", []):
                    model_id = model.get("id")
                    # T-silent-routing W-3: never enumerate a name in
                    # /v1/models that ``get_backend_for_model`` will refuse
                    # with 404. Listing an id the router will not route is
                    # exactly the class of failure this branch is against
                    # ("advertised endpoint that structurally 404s").
                    if isinstance(model_id, str) and model_id in self._ambiguous_models:
                        if model_id not in _filtered_ambiguous:
                            logger.warning(
                                "list_models_ambiguous_filtered",
                                model=model_id,
                                backends=self._ambiguous_models[model_id],
                            )
                            _filtered_ambiguous.add(model_id)
                        continue
                    # Deduplicate by id. Three vllm backends pointing at the
                    # same upstream URL each return an identical row for the
                    # concrete model — carrying all three past this point
                    # gives clients three "same model, different backend"
                    # entries that only differ in the additive ``backend``
                    # field, which is worse than one canonical entry: it
                    # implies a choice of backend that raw-name routing
                    # does not actually support.
                    if isinstance(model_id, str) and model_id in _seen_ids:
                        continue
                    if isinstance(model_id, str):
                        _seen_ids.add(model_id)
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

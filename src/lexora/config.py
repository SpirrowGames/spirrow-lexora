"""Configuration management for Lexora using Pydantic Settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lexora.utils.logging import get_logger

_logger = get_logger(__name__)


class DuplicateYamlKeyError(ValueError):
    """Raised when a YAML mapping declares the same key twice.

    Purpose (T-silent-routing R-5): PyYAML's SafeLoader silently keeps the
    last value when a mapping has duplicate keys, so a config that reads
    ``naysayer: {backend: gemini} / naysayer: {backend: heavy}`` at the
    ``routing.backends`` level parses to a dict Python then sees as
    single-keyed and correct. The disagreement between what the file says
    and what the application sees is exactly the failure this branch is
    against: a config that lies is worse than a config that will not load.

    The message names the key and every line it appears on so an operator
    diffing the file can find the collision without re-parsing.
    """


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that raises on duplicate keys in any mapping.

    Applied to *every* mapping in the file, not just ones the schema knows
    about. R-5 is intentionally wider than R-1a / R-1b: this loader also
    catches ``routing.tiers.frontier`` declared twice, ``models:`` entries
    with duplicated fields, and any future config key. Fixing the class
    of bug at the parser is smaller than adding a validator per field.
    """


def _construct_strict_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    """Build a mapping, but raise when the same key appears twice."""
    loader.flatten_mapping(node)
    seen: dict[Any, tuple[int, int]] = {}
    for key_node, _ in node.value:
        # ``construct_object`` is what SafeLoader would call for each key,
        # so scalar keys become the same Python objects the default loader
        # would produce (``"heavy"`` stays ``"heavy"``, not a ScalarNode).
        key = loader.construct_object(key_node, deep=True)
        try:
            hash(key)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key ({exc})",
                key_node.start_mark,
            ) from exc
        if key in seen:
            first_line, first_col = seen[key]
            raise DuplicateYamlKeyError(
                f"duplicate key {key!r} in YAML mapping: first seen at "
                f"line {first_line}, column {first_col}; repeated at "
                f"line {key_node.start_mark.line + 1}, column "
                f"{key_node.start_mark.column + 1}. A YAML mapping with "
                f"duplicate keys parses to whatever the last one says, so "
                f"the file and the loaded config disagree. Fix the source "
                f"file — do not rely on the loader picking a winner."
            )
        seen[key] = (key_node.start_mark.line + 1, key_node.start_mark.column + 1)
    # Delegate the actual mapping construction to the base implementation so
    # the value nodes are converted through the normal path.
    return loader.construct_mapping(node, deep=deep)  # type: ignore[no-any-return]


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_strict_mapping,
)


#: Backend types whose implementation actually honours ``error_passthrough``.
#: This is a wiring fact, not a policy: ``backends/factory.py`` passes the
#: flag to ``AnthropicBackend`` only, and only the anthropic backend raises
#: ``BackendUpstreamError`` (gemini / vllm / openai_compatible raise plain
#: ``BackendError``, which the routes flatten to 502 by design). Extending
#: passthrough to a second backend means implementing it there and adding
#: the type here — in that order.
ERROR_PASSTHROUGH_TYPES: frozenset[str] = frozenset({"anthropic"})


class ModelInfo(BaseSettings):
    """Individual model information with capabilities."""

    name: str = Field(description="Model name/identifier")
    capabilities: list[str] = Field(
        default_factory=lambda: ["general"],
        description="List of capability tags for this model",
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the model"
    )


def _normalize_model_entry(v: Any) -> ModelInfo:
    """Normalize model entry to ModelInfo - accepts str or dict."""
    if isinstance(v, str):
        return ModelInfo(name=v)
    if isinstance(v, dict):
        return ModelInfo(**v)
    if isinstance(v, ModelInfo):
        return v
    raise ValueError(f"Invalid model entry: {v}")


def _normalize_models_list(v: Any) -> list[ModelInfo]:
    """Normalize models list - accepts list of str or ModelInfo dicts."""
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError("models must be a list")
    return [_normalize_model_entry(item) for item in v]


# Type alias for models field with backward compatibility
ModelsList = Annotated[list[ModelInfo], BeforeValidator(_normalize_models_list)]


class BackendSettings(BaseSettings):
    """Single backend settings."""

    type: Literal[
        "vllm", "openai_compatible", "anthropic", "claude_code", "gemini"
    ] = Field(
        default="vllm",
        description=(
            "Backend type (vllm, openai_compatible, anthropic, claude_code, "
            "or gemini)"
        ),
    )
    url: str = Field(default="http://localhost:8000", description="Backend server URL")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")
    connect_timeout: float = Field(default=5.0, description="Connection timeout in seconds")
    models: ModelsList = Field(
        default_factory=list,
        description="Models served by this backend (str or ModelInfo)",
    )
    api_key: str | None = Field(default=None, description="API key for authentication")
    api_key_env: str | None = Field(
        default=None, description="Environment variable name containing API key"
    )
    model_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Model name mapping (requested_name -> actual_name)",
    )
    thinking_mode: Literal["think", "no_think"] | None = Field(
        default=None,
        description=(
            "Thinking mode for Qwen models: 'think' or 'no_think'. vllm backend "
            "only. chat template の enable_thinking kwarg として送る "
            "(Qwen3 / Qwen3.5+ どちらの template も解釈する)"
        ),
    )
    reasoning_effort: Literal["low", "medium", "xhigh"] | None = Field(
        default=None,
        description=(
            "Thinking depth when thinking_mode='think'. vllm backend only. "
            "Qwen3.5+ の chat template が受け付けるのは low / medium / xhigh "
            "(既定 xhigh)。Qwen3 (2025) 世代の template は解釈せず無視するだけ "
            "なので、旧モデルへ切り戻しても無害。None は model 側の既定に従う"
        ),
    )
    default_max_tokens: int | None = Field(
        default=None,
        description=(
            "Default max output tokens applied when a request omits max_tokens. "
            "Consumed by the gemini and anthropic backends (reasoning models spend "
            "output budget on thinking, so a generous default avoids empty "
            "responses). None falls back to the backend's built-in default. "
            "Effective on /v1/chat/completions, /v1/completions, /generate and "
            "/chat. NOT effective on /v1/messages: the Anthropic-shaped "
            "converter (api/anthropic_compat.py) always sets max_tokens from "
            "its own module constant, so the backend never sees the key "
            "missing and this setting cannot apply. That predates this "
            "setting and is tracked as a separate follow-up; do not read the "
            "line above as covering /v1/messages until it is fixed."
        ),
    )
    paid_key_acknowledged: bool = Field(
        default=False,
        description=(
            "Gemini backend only: operator affirmation that api_key is a "
            "paid/billing-enabled key. Fail-closed when a key is configured "
            "(ADR-2026-05-31-14 D-4 paid-key guarantee)."
        ),
    )
    governance_gate_enabled: bool = Field(
        default=True,
        description=(
            "Gemini backend only: enforce the naysayer data-governance gate "
            "(plain generateContent only; tools / grounding / cached-content / "
            "non-text parts refused — ADR-2026-05-31-14 D-4 / ADR-15 C-2). "
            "Defaults True (fail-closed). Set false to disable the gate and let "
            "tools/non-text surfaces through — this relaxes the ADR data-"
            "governance invariant, so flip it only with explicit owner sign-off."
        ),
    )

    health_check: bool = Field(
        default=True,
        description=(
            "Include this backend in GET /health. Turn it off for backends "
            "whose probe costs more than the answer is worth: the gemini and "
            "anthropic probes each send a real inference request to a remote "
            "API, so every poll bills a call and inherits that provider's "
            "latency, and nothing here can act on the result anyway. A skipped "
            "backend is reported as 'skipped' rather than omitted -- it still "
            "serves traffic, and dropping the name would read as 'not "
            "configured'. Skipping does NOT disable the backend."
        ),
    )
    error_passthrough: bool = Field(
        default=False,
        description=(
            "When true, upstream 4xx/5xx answers are forwarded to the caller "
            "with the upstream status and body preserved, and automatic retry "
            "on rate-limit / connection / timeout is disabled for this "
            "backend. Off by default so existing tiers keep the 502-on-error "
            "behaviour. Turn it on for a tier where the caller has explicitly "
            "chosen this upstream (e.g. frontier) and needs to see the actual "
            "answer — including a safety classifier decline — rather than a "
            "gateway-flattened error message or a silently retried request. "
            "Only implemented for type 'anthropic'; setting it on any other "
            "backend type is rejected at startup rather than accepted and "
            "dropped (see ERROR_PASSTHROUGH_TYPES)."
        ),
    )

    @model_validator(mode="after")
    def _reject_unimplemented_error_passthrough(self) -> "BackendSettings":
        """Fail loud when ``error_passthrough`` is set on a type that ignores it.

        ``error_passthrough`` lives on ``BackendSettings``, i.e. every backend
        type accepts the key — but the factory only forwards it to the
        anthropic backend. Measured on every type: a config saying ``true``
        produced a backend whose ``error_passthrough`` was ``False``, with no
        exception and no warning.

        That is the same defect this branch's sibling PR deleted from the
        fallback machinery: an operator writes a setting, the gateway accepts
        it, and the promised behaviour never happens. It is refused here for
        the same reason it was refused there — a config that lies is worse
        than a config that will not load.
        """
        if self.error_passthrough and self.type not in ERROR_PASSTHROUGH_TYPES:
            supported = ", ".join(sorted(ERROR_PASSTHROUGH_TYPES))
            raise ValueError(
                f"error_passthrough is not implemented for backend type "
                f"'{self.type}' (supported: {supported}). This backend would "
                f"accept the setting and ignore it, so it is refused instead. "
                f"Remove the key from this backend, or point the tier that "
                f"needs verbatim upstream errors at a '{supported}' backend."
            )
        return self

    def get_model_names(self) -> list[str]:
        """Get list of model names for backward compatibility."""
        return [m.name for m in self.models]


class VLLMSettings(BaseSettings):
    """vLLM backend settings (legacy, for single backend)."""

    url: str = Field(default="http://localhost:8000", description="vLLM server URL")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")
    connect_timeout: float = Field(default=5.0, description="Connection timeout in seconds")


class TierSettings(BaseSettings):
    """Tier configuration — maps a tier name to a backend and model."""

    backend: str = Field(description="Backend name this tier routes to")
    model: str | None = Field(
        default=None,
        description="Model name to send to backend (defaults to backend's first model)",
    )
    description: str | None = Field(
        default=None, description="Human-readable tier description"
    )


class ClassifierSettings(BaseSettings):
    """Task classifier settings."""

    enabled: bool = Field(default=False, description="Enable task classification")
    model: str | None = Field(
        default=None, description="Model to use for task classification"
    )
    backend: str | None = Field(
        default=None, description="Backend to use for task classification"
    )


class RoutingSettings(BaseSettings):
    """Model routing settings."""

    enabled: bool = Field(default=False, description="Enable multi-backend routing")
    default_backend: str = Field(default="default", description="Default backend name")
    backends: dict[str, BackendSettings] = Field(
        default_factory=dict,
        description="Backend configurations keyed by name",
    )
    default_model_for_unknown_task: str | None = Field(
        default=None,
        description="Default model to use when task classification fails or returns unknown",
    )
    tiers: dict[str, TierSettings] = Field(
        default_factory=dict,
        description="Tier-to-backend mapping (e.g., light, medium, heavy)",
    )
    classifier: ClassifierSettings = Field(
        default_factory=ClassifierSettings,
        description="Task classifier settings",
    )

    @model_validator(mode="after")
    def _reject_tier_backend_collisions(self) -> "RoutingSettings":
        """Refuse configs where a tier name is also declared as a backend model name.

        T-silent-routing R-1b: tier names are gateway abstractions with a 1:1
        mapping to a backend. If a tier name ``frontier`` is also declared as
        a concrete model name on some backend, ``get_backend_for_model``'s
        tier lookup wins (see ``BackendRouter.get_backend_for_model``), which
        means the model declaration is unreachable but present. Reject rather
        than accept-and-ignore.

        The stronger case — the *same tier* claimed by two different
        backends — cannot reach this validator: ``tiers`` is a ``dict[str,
        TierSettings]`` so duplicate top-level tier keys are already
        collapsed by the YAML loader (SafeLoader) into a single entry, last-
        writer-wins, before Pydantic sees the payload. R-5's strict loader
        catches that class at parse time; this validator catches the
        cross-section collision (tier name vs model name) that R-5 cannot
        see.
        """
        tier_names = set(self.tiers)
        offenders: list[tuple[str, str]] = []
        for backend_name, backend_settings in self.backends.items():
            for model_info in backend_settings.models:
                if model_info.name in tier_names:
                    offenders.append((backend_name, model_info.name))
        if offenders:
            lines = ", ".join(
                f"backend '{bn}' declares model '{mn}'" for bn, mn in offenders
            )
            raise ValueError(
                f"tier/model name collision in routing config: {lines}. "
                f"A tier name is a gateway alias and always wins over a "
                f"concrete model name at request time, so the model "
                f"declaration is unreachable. Rename either side rather "
                f"than shipping a declaration that never routes."
            )
        return self


class ServerSettings(BaseSettings):
    """Server settings."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8001, description="Server port")


class QueueSettings(BaseSettings):
    """Queue settings."""

    max_size: int = Field(default=1000, description="Maximum queue size")
    default_timeout: float = Field(default=60.0, description="Default request timeout in seconds")


class RateLimitSettings(BaseSettings):
    """Rate limit settings."""

    enabled: bool = Field(default=True, description="Enable rate limiting")
    default_rate: float = Field(default=10.0, description="Default requests per second")
    default_burst: int = Field(default=20, description="Default burst size")


class RetrySettings(BaseSettings):
    """Retry settings."""

    max_retries: int = Field(default=3, description="Maximum number of retries")
    base_delay: float = Field(default=1.0, description="Base delay between retries in seconds")
    max_delay: float = Field(default=30.0, description="Maximum delay between retries in seconds")
    exponential_base: float = Field(default=2.0, description="Exponential backoff base")
    respect_retry_after: bool = Field(
        default=True, description="Respect Retry-After header from 429 responses"
    )
    max_retry_after: float = Field(
        default=60.0, description="Maximum Retry-After delay to respect in seconds"
    )


class LoggingSettings(BaseSettings):
    """Logging settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    format: Literal["json", "console"] = Field(default="console", description="Log format")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="LEXORA_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)


def load_yaml_config(config_path: Path | None = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, looks for default locations.

    Returns:
        Dictionary with configuration values.
    """
    if config_path is None:
        default_paths = [
            Path("config/lexora_config.yaml"),
            Path("lexora_config.yaml"),
            Path("/etc/lexora/config.yaml"),
        ]
        for path in default_paths:
            if path.exists():
                config_path = path
                break

    if config_path is None or not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        # Strict loader (R-5): duplicate keys in any mapping are rejected
        # rather than silently overwritten. See ``_StrictSafeLoader``. This
        # runs before Pydantic and is the only layer that can see the raw
        # text — by the time a dict reaches ``RoutingSettings``, a duplicate
        # top-level tier key (``routing.tiers.frontier`` declared twice, for
        # instance) has already been collapsed to whichever appeared last.
        return yaml.load(f, Loader=_StrictSafeLoader) or {}


def _apply_frontier_model_override(
    routing_settings: RoutingSettings, frontier_model: str
) -> None:
    """Apply ``LEXORA_FRONTIER_MODEL`` to every surface that must agree, or refuse to start.

    Two places have to say the same thing about the frontier tier:

      1. ``routing.tiers.frontier.model`` — what the router resolves at
         request time and sends upstream (drives cost tracker keying via
         D-6a).
      2. ``models[0].name`` on the backend that tier points at — what the
         ModelRegistry / ``/v1/models`` / ``/v1/models/capabilities``
         advertise as the frontier tier's concrete model.

    Site 2 is located through ``tiers["frontier"].backend``, never through a
    literal backend name. The tier name ``frontier`` *is* a literal on
    purpose — callers send ``model: "frontier"``, so it is public API — but
    the backend name behind it is an internal label the YAML picks freely
    (``TierSettings.backend`` is a required free-form string). Reading that
    name as a literal made the two updates independent, and PR #11's gate
    measured what it costs: with the tier pointed at a backend named
    ``anthropic_paid``, the tier resolved to Opus 5 while capabilities kept
    advertising Fable 5 — and with an unused backend still named
    ``frontier`` present, the override landed on that one instead,
    advertising an ID no tier could reach.

    **Both or refuse.** The earlier contract said "both or neither", but
    neither is the wrong half here: setting this variable is an operator
    saying "bill me for this model". Applying none of it bills them for the
    YAML default under the name of the model they asked for — the exact loss
    of provenance the frontier tier exists to prevent. So a config where the
    override cannot reach both sites does not start, on the same judgement
    and for the same reason as
    ``BackendSettings._reject_unimplemented_error_passthrough``: a config
    that lies is worse than a config that will not load.

    Raises:
        ValueError: if the frontier tier, its backend, or that backend's
            model list cannot be resolved. Only ever raised when
            ``LEXORA_FRONTIER_MODEL`` is set — an unset variable leaves every
            config loading exactly as before.
    """
    tier = routing_settings.tiers.get("frontier")
    if tier is None:
        known = ", ".join(sorted(routing_settings.tiers)) or "(none)"
        raise ValueError(
            f"LEXORA_FRONTIER_MODEL is set to '{frontier_model}' but this "
            f"config has no 'frontier' tier to apply it to "
            f"(routing.tiers defines: {known}). Applying it anyway could only "
            f"rewrite a backend that no tier resolves to, so startup is "
            f"refused rather than accepting the variable and ignoring it. Add "
            f"a 'frontier' tier under routing.tiers, or unset "
            f"LEXORA_FRONTIER_MODEL."
        )

    backend = routing_settings.backends.get(tier.backend)
    if backend is None:
        known = ", ".join(sorted(routing_settings.backends)) or "(none)"
        raise ValueError(
            f"LEXORA_FRONTIER_MODEL is set to '{frontier_model}' but "
            f"routing.tiers.frontier.backend names '{tier.backend}', which is "
            f"not defined in routing.backends (defined: {known}). The tier "
            f"would resolve to '{frontier_model}' while "
            f"/v1/models/capabilities kept advertising the YAML default, so "
            f"startup is refused rather than shipping that disagreement. Fix "
            f"routing.tiers.frontier.backend, or unset LEXORA_FRONTIER_MODEL."
        )

    if not backend.models:
        raise ValueError(
            f"LEXORA_FRONTIER_MODEL is set to '{frontier_model}' but backend "
            f"'{tier.backend}' (named by routing.tiers.frontier.backend) has "
            f"an empty 'models' list, so there is no advertised model ID to "
            f"swap. The tier would resolve to '{frontier_model}' while "
            f"/v1/models advertised nothing for it, so startup is refused. "
            f"Give routing.backends.{tier.backend}.models at least one entry, "
            f"or unset LEXORA_FRONTIER_MODEL."
        )

    tier.model = frontier_model
    # Preserve capability / description metadata; only swap the ID.
    backend.models[0].name = frontier_model


def create_settings(config_path: Path | None = None) -> Settings:
    """Create settings from YAML config and environment variables.

    Environment variables take precedence over YAML config.

    Args:
        config_path: Optional path to YAML config file.

    Returns:
        Settings instance.
    """
    yaml_config = load_yaml_config(config_path)

    # Build nested settings from YAML
    vllm_config = yaml_config.get("vllm", {})
    server_config = yaml_config.get("server", {})
    queue_config = yaml_config.get("queue", {})
    rate_limit_config = yaml_config.get("rate_limit", {})
    retry_config = yaml_config.get("retry", {})
    logging_config = yaml_config.get("logging", {})
    routing_config = yaml_config.get("routing", {})

    # Parse backends if provided
    routing_settings_kwargs: dict = {}
    if routing_config:
        routing_settings_kwargs["enabled"] = routing_config.get("enabled", False)
        routing_settings_kwargs["default_backend"] = routing_config.get(
            "default_backend", "default"
        )
        backends_config = routing_config.get("backends", {})
        routing_settings_kwargs["backends"] = {
            name: BackendSettings(**cfg) for name, cfg in backends_config.items()
        }
        # New fields for capabilities/classifier
        if "default_model_for_unknown_task" in routing_config:
            routing_settings_kwargs["default_model_for_unknown_task"] = routing_config[
                "default_model_for_unknown_task"
            ]
        tiers_config = routing_config.get("tiers", {})
        if tiers_config:
            routing_settings_kwargs["tiers"] = {
                name: TierSettings(**cfg) for name, cfg in tiers_config.items()
            }
        classifier_config = routing_config.get("classifier", {})
        if classifier_config:
            routing_settings_kwargs["classifier"] = ClassifierSettings(**classifier_config)

    routing_settings = (
        RoutingSettings(**routing_settings_kwargs)
        if routing_settings_kwargs
        else RoutingSettings()
    )

    # Frontier model env override (T-frontier-tier D-2).
    #
    # A verbatim `LEXORA_ROUTING__TIERS__FRONTIER__MODEL` is NOT respected
    # because `create_settings` builds `RoutingSettings(...)` above and
    # passes it as a kwarg to `Settings(...)`; pydantic-settings gives init
    # kwargs precedence over env, so a nested-delimiter env variable is
    # silently ignored (measured 2026-08-31). Rather than restructure the
    # YAML-first loader, expose the one variable the operator actually
    # wants — `LEXORA_FRONTIER_MODEL` — and apply it to both places that
    # need to agree:
    #
    #   1. `routing.tiers.frontier.model` — what the router resolves at
    #      request time (drives cost tracker keying via D-6a).
    #   2. `models[0].name` on the backend `routing.tiers.frontier.backend`
    #      names — what the ModelRegistry / `/v1/models` /
    #      `/v1/models/capabilities` surface as the frontier tier's
    #      concrete model. The backend is reached through the tier, not by
    #      the literal name `frontier`: the backend label is the YAML's to
    #      choose (see `_apply_frontier_model_override`).
    #
    # Doing only one produces an observable lie (the tier resolves to Opus
    # 5 while capabilities keep advertising Fable 5). This does both or
    # refuses to start — see the helper for why "neither" is not the
    # fallback.
    frontier_model_env = os.environ.get("LEXORA_FRONTIER_MODEL")
    if frontier_model_env:
        _apply_frontier_model_override(routing_settings, frontier_model_env)

    return Settings(
        vllm=VLLMSettings(**vllm_config),
        server=ServerSettings(**server_config),
        queue=QueueSettings(**queue_config),
        rate_limit=RateLimitSettings(**rate_limit_config),
        retry=RetrySettings(**retry_config),
        logging=LoggingSettings(**logging_config),
        routing=routing_settings,
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Cached Settings instance.
    """
    return create_settings()

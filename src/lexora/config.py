"""Configuration management for Lexora using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    type: Literal["vllm", "openai_compatible"] = Field(
        default="vllm", description="Backend type (vllm or openai_compatible)"
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
    fallback_backends: list[str] = Field(
        default_factory=list, description="List of fallback backend names"
    )

    def get_model_names(self) -> list[str]:
        """Get list of model names for backward compatibility."""
        return [m.name for m in self.models]


class VLLMSettings(BaseSettings):
    """vLLM backend settings (legacy, for single backend)."""

    url: str = Field(default="http://localhost:8000", description="vLLM server URL")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")
    connect_timeout: float = Field(default=5.0, description="Connection timeout in seconds")


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
    classifier: ClassifierSettings = Field(
        default_factory=ClassifierSettings,
        description="Task classifier settings",
    )


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


class FallbackSettings(BaseSettings):
    """Fallback settings."""

    enabled: bool = Field(default=True, description="Enable fallback to alternative backends")
    on_rate_limit: bool = Field(
        default=True, description="Allow fallback on rate limit (429) errors"
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
    fallback: FallbackSettings = Field(default_factory=FallbackSettings)


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
        return yaml.safe_load(f) or {}


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
    fallback_config = yaml_config.get("fallback", {})

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
        classifier_config = routing_config.get("classifier", {})
        if classifier_config:
            routing_settings_kwargs["classifier"] = ClassifierSettings(**classifier_config)

    return Settings(
        vllm=VLLMSettings(**vllm_config),
        server=ServerSettings(**server_config),
        queue=QueueSettings(**queue_config),
        rate_limit=RateLimitSettings(**rate_limit_config),
        retry=RetrySettings(**retry_config),
        logging=LoggingSettings(**logging_config),
        routing=RoutingSettings(**routing_settings_kwargs) if routing_settings_kwargs else RoutingSettings(),
        fallback=FallbackSettings(**fallback_config),
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Cached Settings instance.
    """
    return create_settings()

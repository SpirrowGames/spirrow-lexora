"""Backend factory for creating backend instances."""

import os

from lexora.backends.anthropic import AnthropicBackend
from lexora.backends.base import Backend
from lexora.backends.claude_code import ClaudeCodeBackend
from lexora.backends.openai_compatible import OpenAICompatibleBackend
from lexora.backends.vllm import VLLMBackend
from lexora.config import BackendSettings
from lexora.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_api_key(settings: BackendSettings) -> str | None:
    """Resolve API key from settings or environment variable.

    Args:
        settings: Backend settings.

    Returns:
        API key string or None if not configured.
    """
    # Direct API key takes precedence
    if settings.api_key:
        return settings.api_key

    # Try environment variable
    if settings.api_key_env:
        api_key = os.environ.get(settings.api_key_env)
        if api_key:
            return api_key
        logger.warning(
            "api_key_env_not_found",
            env_var=settings.api_key_env,
        )

    return None


def create_backend(name: str, settings: BackendSettings) -> Backend:
    """Create a backend instance from settings.

    Args:
        name: Backend name (used for logging and error messages).
        settings: Backend configuration settings.

    Returns:
        Backend instance.

    Raises:
        ValueError: If backend type is unknown.
    """
    if settings.type == "vllm":
        logger.info(
            "creating_vllm_backend",
            name=name,
            url=settings.url,
        )
        return VLLMBackend(
            base_url=settings.url,
            timeout=settings.timeout,
            connect_timeout=settings.connect_timeout,
            name=name,
            thinking_mode=settings.thinking_mode,
        )
    elif settings.type == "openai_compatible":
        api_key = resolve_api_key(settings)
        if api_key is None:
            logger.warning(
                "openai_compatible_no_api_key",
                name=name,
                url=settings.url,
            )
        logger.info(
            "creating_openai_compatible_backend",
            name=name,
            url=settings.url,
            has_api_key=api_key is not None,
            model_mapping_count=len(settings.model_mapping),
        )
        return OpenAICompatibleBackend(
            base_url=settings.url,
            api_key=api_key,
            timeout=settings.timeout,
            connect_timeout=settings.connect_timeout,
            model_mapping=settings.model_mapping,
            name=name,
        )
    elif settings.type == "anthropic":
        api_key = resolve_api_key(settings)
        if api_key is None:
            logger.warning(
                "anthropic_no_api_key",
                name=name,
                url=settings.url,
            )
        logger.info(
            "creating_anthropic_backend",
            name=name,
            url=settings.url,
            has_api_key=api_key is not None,
            model_mapping_count=len(settings.model_mapping),
        )
        return AnthropicBackend(
            base_url=settings.url,
            api_key=api_key,
            timeout=settings.timeout,
            connect_timeout=settings.connect_timeout,
            model_mapping=settings.model_mapping,
            name=name,
        )
    elif settings.type == "claude_code":
        # Extract extra config from model_mapping (factory convention)
        cli_model = settings.model_mapping.get("model", "sonnet")
        allowed_tools_str = settings.model_mapping.get("allowed_tools", "")
        allowed_tools = [t.strip() for t in allowed_tools_str.split(",") if t.strip()] if allowed_tools_str else None
        working_dir = settings.model_mapping.get("working_dir")
        max_turns_str = settings.model_mapping.get("max_turns")
        max_turns = int(max_turns_str) if max_turns_str else None

        logger.info(
            "creating_claude_code_backend",
            name=name,
            cli_model=cli_model,
            has_allowed_tools=allowed_tools is not None,
        )
        return ClaudeCodeBackend(
            model=cli_model,
            timeout=settings.timeout,
            allowed_tools=allowed_tools,
            working_dir=working_dir,
            max_turns=max_turns,
            model_mapping=settings.model_mapping,
            name=name,
        )
    else:
        raise ValueError(f"Unknown backend type: {settings.type}")

"""Backend layer for Lexora."""

from lexora.backends.base import (
    Backend,
    BackendConnectionError,
    BackendError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from lexora.backends.factory import create_backend, resolve_api_key
from lexora.backends.openai_compatible import OpenAICompatibleBackend
from lexora.backends.vllm import VLLMBackend

__all__ = [
    # Base classes and exceptions
    "Backend",
    "BackendError",
    "BackendConnectionError",
    "BackendTimeoutError",
    "BackendUnavailableError",
    "BackendRateLimitError",
    # Backend implementations
    "VLLMBackend",
    "OpenAICompatibleBackend",
    # Factory functions
    "create_backend",
    "resolve_api_key",
]

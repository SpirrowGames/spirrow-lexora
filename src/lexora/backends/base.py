"""Base backend interface and exceptions."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class BackendError(Exception):
    """Base exception for backend errors."""

    pass


class BackendConnectionError(BackendError):
    """Raised when connection to backend fails."""

    pass


class BackendTimeoutError(BackendError):
    """Raised when request to backend times out."""

    pass


class BackendUnavailableError(BackendError):
    """Raised when backend is unavailable (e.g., 503 status)."""

    pass


class BackendRateLimitError(BackendError):
    """Raised when backend returns 429 Too Many Requests."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        backend_name: str | None = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            message: Error message.
            retry_after: Suggested retry delay in seconds from Retry-After header.
            backend_name: Name of the backend that returned the error.
        """
        super().__init__(message)
        self.retry_after = retry_after
        self.backend_name = backend_name


class Backend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send chat completion request to the backend.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            OpenAI-compatible chat completion response.

        Raises:
            BackendError: If the request fails.
        """
        pass

    @abstractmethod
    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send completion request to the backend.

        Args:
            request: OpenAI-compatible completion request.

        Returns:
            OpenAI-compatible completion response.

        Raises:
            BackendError: If the request fails.
        """
        pass

    @abstractmethod
    async def list_models(self) -> dict[str, Any]:
        """List available models.

        Returns:
            OpenAI-compatible models list response.

        Raises:
            BackendError: If the request fails.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy.

        Returns:
            True if backend is healthy, False otherwise.
        """
        pass

    @abstractmethod
    async def chat_completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send streaming chat completion request to the backend.

        Args:
            request: OpenAI-compatible chat completion request.

        Yields:
            SSE data chunks.

        Raises:
            BackendError: If the request fails.
        """
        pass
        yield b""  # Make this an async generator

    @abstractmethod
    async def completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send streaming completion request to the backend.

        Args:
            request: OpenAI-compatible completion request.

        Yields:
            SSE data chunks.

        Raises:
            BackendError: If the request fails.
        """
        pass
        yield b""  # Make this an async generator

    @abstractmethod
    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send embeddings request to the backend.

        Args:
            request: OpenAI-compatible embeddings request.

        Returns:
            OpenAI-compatible embeddings response.

        Raises:
            BackendError: If the request fails.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass

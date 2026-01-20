"""Fallback service for handling backend failures."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

from lexora.backends.base import (
    Backend,
    BackendError,
    BackendRateLimitError,
)
from lexora.config import FallbackSettings
from lexora.services.router import BackendRouter
from lexora.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class FallbackService:
    """Service for executing requests with fallback to alternative backends.

    When a primary backend fails or hits rate limits, this service can
    automatically try fallback backends in order.

    Args:
        router: Backend router instance.
        settings: Fallback configuration settings.
    """

    def __init__(
        self,
        router: BackendRouter,
        settings: FallbackSettings,
    ) -> None:
        """Initialize the fallback service.

        Args:
            router: Backend router instance.
            settings: Fallback configuration settings.
        """
        self._router = router
        self._settings = settings

    @property
    def enabled(self) -> bool:
        """Check if fallback is enabled."""
        return self._settings.enabled

    def should_fallback_on_error(self, error: Exception) -> bool:
        """Determine if fallback should be attempted for an error.

        Args:
            error: The error that occurred.

        Returns:
            True if fallback should be attempted.
        """
        if not self._settings.enabled:
            return False

        # Always fallback on general backend errors
        if isinstance(error, BackendError) and not isinstance(
            error, BackendRateLimitError
        ):
            return True

        # Fallback on rate limit only if configured
        if isinstance(error, BackendRateLimitError):
            return self._settings.on_rate_limit

        return False

    async def execute_with_fallback(
        self,
        model: str,
        operation: Callable[[Backend], Awaitable[T]],
        fallback_exceptions: tuple[type[Exception], ...] = (BackendError,),
    ) -> tuple[T, str]:
        """Execute an operation with fallback to alternative backends.

        Args:
            model: Model name for routing.
            operation: Async function that takes a backend and returns a result.
            fallback_exceptions: Exception types that should trigger fallback.

        Returns:
            Tuple of (result, backend_name) indicating which backend succeeded.

        Raises:
            The last exception if all backends fail.
        """
        primary_backend_name = self._router.get_backend_name_for_model(model)
        primary_backend = self._router.get_backend_for_model(model)

        errors: list[tuple[str, Exception]] = []

        # Try primary backend
        try:
            result = await operation(primary_backend)
            return result, primary_backend_name
        except fallback_exceptions as e:
            errors.append((primary_backend_name, e))
            logger.warning(
                "primary_backend_failed",
                backend=primary_backend_name,
                model=model,
                error=str(e),
                is_rate_limit=isinstance(e, BackendRateLimitError),
            )

            if not self.should_fallback_on_error(e):
                raise

        # Try fallback backends
        fallback_backends = self._router.get_fallback_backends(primary_backend_name)
        for fallback_backend in fallback_backends:
            fallback_name = getattr(fallback_backend, "name", "unknown")
            try:
                logger.info(
                    "trying_fallback_backend",
                    primary=primary_backend_name,
                    fallback=fallback_name,
                    model=model,
                )
                result = await operation(fallback_backend)
                logger.info(
                    "fallback_succeeded",
                    primary=primary_backend_name,
                    fallback=fallback_name,
                    model=model,
                )
                return result, fallback_name
            except fallback_exceptions as e:
                errors.append((fallback_name, e))
                logger.warning(
                    "fallback_backend_failed",
                    backend=fallback_name,
                    model=model,
                    error=str(e),
                    is_rate_limit=isinstance(e, BackendRateLimitError),
                )

        # All backends failed
        logger.error(
            "all_backends_failed",
            model=model,
            primary=primary_backend_name,
            errors=[
                {"backend": name, "error": str(err)} for name, err in errors
            ],
        )

        # Raise the last exception
        if errors:
            raise errors[-1][1]
        raise RuntimeError("No backends available")

    async def execute_stream_with_fallback(
        self,
        model: str,
        operation: Callable[[Backend], AsyncIterator[bytes]],
        fallback_exceptions: tuple[type[Exception], ...] = (BackendError,),
    ) -> tuple[AsyncIterator[bytes], str]:
        """Execute a streaming operation with fallback to alternative backends.

        Note: For streaming, we can only fallback before the stream starts.
        Once streaming begins, errors cannot trigger fallback.

        Args:
            model: Model name for routing.
            operation: Async function that takes a backend and returns an async iterator.
            fallback_exceptions: Exception types that should trigger fallback.

        Returns:
            Tuple of (async iterator, backend_name) indicating which backend succeeded.

        Raises:
            The last exception if all backends fail to start streaming.
        """
        primary_backend_name = self._router.get_backend_name_for_model(model)
        primary_backend = self._router.get_backend_for_model(model)

        errors: list[tuple[str, Exception]] = []

        # Try primary backend
        try:
            stream = operation(primary_backend)
            return stream, primary_backend_name
        except fallback_exceptions as e:
            errors.append((primary_backend_name, e))
            logger.warning(
                "primary_backend_stream_failed",
                backend=primary_backend_name,
                model=model,
                error=str(e),
                is_rate_limit=isinstance(e, BackendRateLimitError),
            )

            if not self.should_fallback_on_error(e):
                raise

        # Try fallback backends
        fallback_backends = self._router.get_fallback_backends(primary_backend_name)
        for fallback_backend in fallback_backends:
            fallback_name = getattr(fallback_backend, "name", "unknown")
            try:
                logger.info(
                    "trying_fallback_backend_stream",
                    primary=primary_backend_name,
                    fallback=fallback_name,
                    model=model,
                )
                stream = operation(fallback_backend)
                logger.info(
                    "fallback_stream_started",
                    primary=primary_backend_name,
                    fallback=fallback_name,
                    model=model,
                )
                return stream, fallback_name
            except fallback_exceptions as e:
                errors.append((fallback_name, e))
                logger.warning(
                    "fallback_backend_stream_failed",
                    backend=fallback_name,
                    model=model,
                    error=str(e),
                    is_rate_limit=isinstance(e, BackendRateLimitError),
                )

        # All backends failed
        logger.error(
            "all_backends_stream_failed",
            model=model,
            primary=primary_backend_name,
            errors=[
                {"backend": name, "error": str(err)} for name, err in errors
            ],
        )

        # Raise the last exception
        if errors:
            raise errors[-1][1]
        raise RuntimeError("No backends available")

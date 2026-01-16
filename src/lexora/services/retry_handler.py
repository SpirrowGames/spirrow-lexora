"""Retry handler with exponential backoff."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from lexora.backends.base import (
    BackendConnectionError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from lexora.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    BackendConnectionError,
    BackendTimeoutError,
    BackendUnavailableError,
)


class RetryHandler:
    """Handles retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff calculation.
        jitter: Whether to add random jitter to delays.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ) -> None:
        """Initialize the retry handler.

        Args:
            max_retries: Maximum number of retry attempts.
            base_delay: Base delay between retries in seconds.
            max_delay: Maximum delay between retries in seconds.
            exponential_base: Base for exponential backoff calculation.
            jitter: Whether to add random jitter to delays.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt.

        Args:
            attempt: The retry attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add up to 25% random jitter
            jitter_amount = delay * random.uniform(0, 0.25)
            delay += jitter_amount

        return delay

    async def execute(
        self,
        func: Callable[[], Awaitable[T]],
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> tuple[T, int]:
        """Execute a function with retry logic.

        Args:
            func: Async function to execute.
            retryable_exceptions: Tuple of exception types that should trigger retry.
                Defaults to RETRYABLE_EXCEPTIONS.

        Returns:
            Tuple of (result, number of retries used).

        Raises:
            The last exception if all retries are exhausted.
        """
        if retryable_exceptions is None:
            retryable_exceptions = RETRYABLE_EXCEPTIONS

        last_exception: Exception | None = None
        retries = 0

        for attempt in range(self.max_retries + 1):
            try:
                result = await func()
                return result, retries
            except retryable_exceptions as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        "retry_attempt",
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
                    retries += 1
                else:
                    logger.error(
                        "retry_exhausted",
                        attempts=self.max_retries + 1,
                        error=str(e),
                    )

        # All retries exhausted
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Unexpected state: no result and no exception")

    def is_retryable(
        self,
        exception: Exception,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> bool:
        """Check if an exception is retryable.

        Args:
            exception: The exception to check.
            retryable_exceptions: Tuple of retryable exception types.

        Returns:
            True if the exception is retryable.
        """
        if retryable_exceptions is None:
            retryable_exceptions = RETRYABLE_EXCEPTIONS
        return isinstance(exception, retryable_exceptions)

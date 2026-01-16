"""Tests for retry handler."""

import pytest

from lexora.backends.base import (
    BackendConnectionError,
    BackendTimeoutError,
    BackendUnavailableError,
    BackendError,
)
from lexora.services.retry_handler import RetryHandler, RETRYABLE_EXCEPTIONS


class TestRetryHandler:
    """Tests for RetryHandler."""

    @pytest.fixture
    def handler(self) -> RetryHandler:
        """Create a retry handler for testing."""
        return RetryHandler(
            max_retries=3,
            base_delay=0.01,  # Fast for testing
            max_delay=0.1,
            exponential_base=2.0,
            jitter=False,  # Disable jitter for predictable tests
        )

    def test_calculate_delay(self, handler: RetryHandler) -> None:
        """Test delay calculation with exponential backoff."""
        # delay = base_delay * (exponential_base ** attempt)
        assert handler.calculate_delay(0) == 0.01  # 0.01 * 2^0 = 0.01
        assert handler.calculate_delay(1) == 0.02  # 0.01 * 2^1 = 0.02
        assert handler.calculate_delay(2) == 0.04  # 0.01 * 2^2 = 0.04

    def test_calculate_delay_max_cap(self) -> None:
        """Test delay is capped at max_delay."""
        handler = RetryHandler(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=False,
        )
        # 1.0 * 2^10 = 1024, but should be capped at 5.0
        assert handler.calculate_delay(10) == 5.0

    def test_calculate_delay_with_jitter(self) -> None:
        """Test delay with jitter enabled."""
        handler = RetryHandler(
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
        )
        # With jitter, delay should be between base and base * 1.25
        delay = handler.calculate_delay(0)
        assert 1.0 <= delay <= 1.25

    @pytest.mark.asyncio
    async def test_execute_success_first_try(self, handler: RetryHandler) -> None:
        """Test successful execution on first try."""
        call_count = 0

        async def successful_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result, retries = await handler.execute(successful_func)

        assert result == "success"
        assert retries == 0
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execute_success_after_retry(self, handler: RetryHandler) -> None:
        """Test successful execution after retries."""
        call_count = 0

        async def failing_then_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise BackendConnectionError("Connection failed")
            return "success"

        result, retries = await handler.execute(failing_then_success)

        assert result == "success"
        assert retries == 2
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_exhausted_retries(self, handler: RetryHandler) -> None:
        """Test that exception is raised after exhausting retries."""
        call_count = 0

        async def always_failing() -> str:
            nonlocal call_count
            call_count += 1
            raise BackendTimeoutError("Timeout")

        with pytest.raises(BackendTimeoutError):
            await handler.execute(always_failing)

        # 1 initial + 3 retries = 4 attempts
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_execute_non_retryable_exception(self, handler: RetryHandler) -> None:
        """Test that non-retryable exceptions are raised immediately."""
        call_count = 0

        async def non_retryable_error() -> str:
            nonlocal call_count
            call_count += 1
            raise BackendError("Generic error")  # Not in RETRYABLE_EXCEPTIONS

        with pytest.raises(BackendError):
            await handler.execute(non_retryable_error)

        # Should fail immediately without retry
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execute_custom_retryable_exceptions(
        self, handler: RetryHandler
    ) -> None:
        """Test execution with custom retryable exceptions."""
        call_count = 0

        async def custom_error() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Custom error")
            return "success"

        result, retries = await handler.execute(
            custom_error, retryable_exceptions=(ValueError,)
        )

        assert result == "success"
        assert retries == 1

    def test_is_retryable_connection_error(self, handler: RetryHandler) -> None:
        """Test is_retryable for connection error."""
        assert handler.is_retryable(BackendConnectionError("test"))

    def test_is_retryable_timeout_error(self, handler: RetryHandler) -> None:
        """Test is_retryable for timeout error."""
        assert handler.is_retryable(BackendTimeoutError("test"))

    def test_is_retryable_unavailable_error(self, handler: RetryHandler) -> None:
        """Test is_retryable for unavailable error."""
        assert handler.is_retryable(BackendUnavailableError("test"))

    def test_is_not_retryable_generic_error(self, handler: RetryHandler) -> None:
        """Test is_retryable for generic error."""
        assert not handler.is_retryable(BackendError("test"))

    def test_is_retryable_custom_exceptions(self, handler: RetryHandler) -> None:
        """Test is_retryable with custom exceptions."""
        assert handler.is_retryable(ValueError("test"), (ValueError, TypeError))
        assert not handler.is_retryable(KeyError("test"), (ValueError, TypeError))


class TestRetryableExceptions:
    """Tests for RETRYABLE_EXCEPTIONS constant."""

    def test_contains_expected_exceptions(self) -> None:
        """Test RETRYABLE_EXCEPTIONS contains expected types."""
        assert BackendConnectionError in RETRYABLE_EXCEPTIONS
        assert BackendTimeoutError in RETRYABLE_EXCEPTIONS
        assert BackendUnavailableError in RETRYABLE_EXCEPTIONS

    def test_does_not_contain_generic_error(self) -> None:
        """Test RETRYABLE_EXCEPTIONS does not contain generic BackendError."""
        assert BackendError not in RETRYABLE_EXCEPTIONS

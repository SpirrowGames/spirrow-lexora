"""Tests for retry handler rate limit support."""

import pytest
from unittest.mock import AsyncMock

from lexora.backends.base import (
    BackendConnectionError,
    BackendRateLimitError,
)
from lexora.services.retry_handler import RetryHandler, RETRYABLE_EXCEPTIONS


class TestRateLimitInRetryableExceptions:
    """Tests for BackendRateLimitError in RETRYABLE_EXCEPTIONS."""

    def test_rate_limit_error_is_retryable(self):
        """Test that BackendRateLimitError is in RETRYABLE_EXCEPTIONS."""
        assert BackendRateLimitError in RETRYABLE_EXCEPTIONS

    def test_is_retryable_rate_limit_error(self):
        """Test is_retryable with BackendRateLimitError."""
        handler = RetryHandler()
        error = BackendRateLimitError("Rate limited", retry_after=30)
        assert handler.is_retryable(error) is True


class TestRetryAfterSupport:
    """Tests for Retry-After header support."""

    def test_calculate_delay_for_exception_without_retry_after(self):
        """Test delay calculation without Retry-After."""
        handler = RetryHandler(base_delay=1.0, jitter=False)
        error = BackendConnectionError("Connection failed")

        delay = handler.calculate_delay_for_exception(0, error)
        assert delay == 1.0

    def test_calculate_delay_for_exception_with_retry_after(self):
        """Test delay calculation respects Retry-After header."""
        handler = RetryHandler(
            base_delay=1.0,
            jitter=False,
            respect_retry_after=True,
            max_retry_after=60.0,
        )
        error = BackendRateLimitError("Rate limited", retry_after=30.0)

        delay = handler.calculate_delay_for_exception(0, error)
        # Retry-After (30) > calculated delay (1), so use Retry-After
        assert delay == 30.0

    def test_calculate_delay_for_exception_with_small_retry_after(self):
        """Test that small Retry-After doesn't reduce calculated delay."""
        handler = RetryHandler(
            base_delay=10.0,
            jitter=False,
            respect_retry_after=True,
        )
        error = BackendRateLimitError("Rate limited", retry_after=2.0)

        delay = handler.calculate_delay_for_exception(0, error)
        # Retry-After (2) < calculated delay (10), so use calculated delay
        assert delay == 10.0

    def test_calculate_delay_with_max_retry_after_limit(self):
        """Test that Retry-After is capped by max_retry_after."""
        handler = RetryHandler(
            base_delay=1.0,
            jitter=False,
            respect_retry_after=True,
            max_retry_after=30.0,
        )
        error = BackendRateLimitError("Rate limited", retry_after=120.0)

        delay = handler.calculate_delay_for_exception(0, error)
        # Retry-After (120) > max (30), so cap at max
        assert delay == 30.0

    def test_calculate_delay_with_respect_retry_after_disabled(self):
        """Test that Retry-After is ignored when disabled."""
        handler = RetryHandler(
            base_delay=1.0,
            jitter=False,
            respect_retry_after=False,
        )
        error = BackendRateLimitError("Rate limited", retry_after=60.0)

        delay = handler.calculate_delay_for_exception(0, error)
        # respect_retry_after=False, so use calculated delay
        assert delay == 1.0

    def test_calculate_delay_with_none_retry_after(self):
        """Test handling of None Retry-After value."""
        handler = RetryHandler(
            base_delay=1.0,
            jitter=False,
            respect_retry_after=True,
        )
        error = BackendRateLimitError("Rate limited", retry_after=None)

        delay = handler.calculate_delay_for_exception(0, error)
        assert delay == 1.0


class TestExecuteWithRateLimitRetry:
    """Tests for execute method with rate limit errors."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_error(self):
        """Test that rate limit errors trigger retry."""
        handler = RetryHandler(max_retries=3, base_delay=0.01, jitter=False)

        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise BackendRateLimitError("Rate limited", retry_after=0.01)
            return "success"

        result, retries = await handler.execute(flaky_func)

        assert result == "success"
        assert retries == 2  # Succeeded on 3rd try
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_respect_retry_after_in_execute(self):
        """Test that execute uses Retry-After value."""
        import time

        handler = RetryHandler(
            max_retries=2,
            base_delay=0.01,
            jitter=False,
            respect_retry_after=True,
            max_retry_after=1.0,
        )

        call_count = 0
        call_times = []

        async def rate_limited_func():
            nonlocal call_count
            call_count += 1
            call_times.append(time.time())
            if call_count == 1:
                raise BackendRateLimitError("Rate limited", retry_after=0.1)
            return "success"

        result, retries = await handler.execute(rate_limited_func)

        assert result == "success"
        # Verify that the delay was approximately 0.1 seconds (Retry-After value)
        if len(call_times) >= 2:
            actual_delay = call_times[1] - call_times[0]
            # Allow some tolerance
            assert actual_delay >= 0.09, f"Delay was {actual_delay}, expected >= 0.09"

    @pytest.mark.asyncio
    async def test_exhaust_retries_on_continuous_rate_limit(self):
        """Test that retries are exhausted on continuous rate limits."""
        handler = RetryHandler(max_retries=2, base_delay=0.01, jitter=False)

        call_count = 0

        async def always_rate_limited():
            nonlocal call_count
            call_count += 1
            raise BackendRateLimitError("Rate limited", retry_after=0.01)

        with pytest.raises(BackendRateLimitError):
            await handler.execute(always_rate_limited)

        # Initial call + 2 retries = 3 total calls
        assert call_count == 3


class TestRetryHandlerInit:
    """Tests for RetryHandler initialization with new parameters."""

    def test_default_retry_after_settings(self):
        """Test default Retry-After settings."""
        handler = RetryHandler()
        assert handler.respect_retry_after is True
        assert handler.max_retry_after == 60.0

    def test_custom_retry_after_settings(self):
        """Test custom Retry-After settings."""
        handler = RetryHandler(
            respect_retry_after=False,
            max_retry_after=120.0,
        )
        assert handler.respect_retry_after is False
        assert handler.max_retry_after == 120.0

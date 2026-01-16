"""Tests for rate limiter."""

import time

import pytest

from lexora.services.rate_limiter import RateLimiter, TokenBucket


class TestTokenBucket:
    """Tests for TokenBucket."""

    def test_initial_state(self) -> None:
        """Test bucket starts full."""
        bucket = TokenBucket(rate=10.0, capacity=20.0)
        assert bucket.available() == 20.0

    def test_consume_success(self) -> None:
        """Test successful token consumption."""
        bucket = TokenBucket(rate=10.0, capacity=20.0)
        assert bucket.consume(5.0) is True
        assert bucket.available() == 15.0

    def test_consume_failure(self) -> None:
        """Test failed consumption when not enough tokens."""
        bucket = TokenBucket(rate=10.0, capacity=5.0)
        assert bucket.consume(10.0) is False
        assert bucket.available() == 5.0  # Tokens not consumed

    def test_refill_over_time(self) -> None:
        """Test tokens refill over time."""
        bucket = TokenBucket(rate=100.0, capacity=20.0)  # 100 tokens/second

        # Consume all tokens
        bucket.consume(20.0)
        assert bucket.available() < 1.0

        # Wait a bit
        time.sleep(0.05)  # 50ms should add ~5 tokens

        available = bucket.available()
        assert available >= 3.0  # Allow some timing variance

    def test_capacity_limit(self) -> None:
        """Test tokens don't exceed capacity."""
        bucket = TokenBucket(rate=1000.0, capacity=10.0)
        time.sleep(0.1)  # Would add 100 tokens if uncapped
        assert bucket.available() == 10.0

    def test_time_until_available(self) -> None:
        """Test time calculation until tokens available."""
        bucket = TokenBucket(rate=10.0, capacity=10.0)

        # When enough tokens, returns 0
        assert bucket.time_until_available(5.0) == 0.0

        # Consume all
        bucket.consume(10.0)

        # Need 5 tokens at 10/sec = 0.5 seconds
        wait_time = bucket.time_until_available(5.0)
        assert 0.4 <= wait_time <= 0.6


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.fixture
    def limiter(self) -> RateLimiter:
        """Create a rate limiter for testing."""
        return RateLimiter(default_rate=10.0, default_burst=20)

    def test_check_allowed(self, limiter: RateLimiter) -> None:
        """Test check returns True when allowed."""
        assert limiter.check("user-1") is True

    def test_consume_allowed(self, limiter: RateLimiter) -> None:
        """Test consume returns True and decrements."""
        assert limiter.consume("user-1", 10.0) is True
        assert limiter.check("user-1", 15.0) is False

    def test_consume_denied(self, limiter: RateLimiter) -> None:
        """Test consume returns False when rate limited."""
        # Exhaust the bucket
        limiter.consume("user-1", 20.0)
        assert limiter.consume("user-1", 5.0) is False

    def test_user_isolation(self, limiter: RateLimiter) -> None:
        """Test users have separate buckets."""
        # Exhaust user-1's bucket
        limiter.consume("user-1", 20.0)

        # user-2 should still be allowed
        assert limiter.consume("user-2", 10.0) is True

    def test_custom_user_limit(self, limiter: RateLimiter) -> None:
        """Test setting custom user limits."""
        limiter.set_user_limit("premium-user", rate=100.0, burst=200)

        info = limiter.get_user_info("premium-user")
        assert info["rate"] == 100.0
        assert info["burst"] == 200
        assert info["is_custom_limit"] is True

    def test_remove_user_limit(self, limiter: RateLimiter) -> None:
        """Test removing custom user limits."""
        limiter.set_user_limit("user-1", rate=100.0, burst=200)
        limiter.remove_user_limit("user-1")

        info = limiter.get_user_info("user-1")
        assert info["rate"] == 10.0  # Back to default
        assert info["is_custom_limit"] is False

    def test_get_user_info(self, limiter: RateLimiter) -> None:
        """Test getting user info."""
        limiter.consume("user-1", 5.0)

        info = limiter.get_user_info("user-1")
        assert info["user_id"] == "user-1"
        assert info["rate"] == 10.0
        assert info["burst"] == 20
        assert info["available_tokens"] == 15.0
        assert info["is_custom_limit"] is False

    def test_time_until_allowed(self, limiter: RateLimiter) -> None:
        """Test time calculation until request allowed."""
        # Exhaust bucket
        limiter.consume("user-1", 20.0)

        wait_time = limiter.time_until_allowed("user-1", 5.0)
        assert wait_time > 0

    def test_reset(self, limiter: RateLimiter) -> None:
        """Test resetting all state."""
        limiter.set_user_limit("user-1", rate=100.0, burst=200)
        limiter.consume("user-2", 10.0)

        limiter.reset()

        # All state should be cleared
        info = limiter.get_user_info("user-1")
        assert info["is_custom_limit"] is False

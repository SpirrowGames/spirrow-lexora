"""Rate limiter using Token Bucket algorithm."""

import time
from dataclasses import dataclass, field
from typing import Any

from lexora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenBucket:
    """Token Bucket for rate limiting.

    Args:
        rate: Tokens added per second.
        capacity: Maximum tokens in the bucket.
    """

    rate: float
    capacity: float
    tokens: float = field(init=False)
    last_update: float = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the bucket with full tokens."""
        self.tokens = self.capacity
        self.last_update = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if not enough tokens.
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def available(self) -> float:
        """Get the number of available tokens.

        Returns:
            Number of available tokens.
        """
        self._refill()
        return self.tokens

    def time_until_available(self, tokens: float = 1.0) -> float:
        """Calculate time until specified tokens are available.

        Args:
            tokens: Number of tokens needed.

        Returns:
            Time in seconds until tokens are available. 0 if already available.
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        needed = tokens - self.tokens
        return needed / self.rate


class RateLimiter:
    """User-based rate limiter using Token Bucket algorithm.

    Args:
        default_rate: Default tokens per second for new users.
        default_burst: Default bucket capacity for new users.
    """

    def __init__(
        self,
        default_rate: float = 10.0,
        default_burst: int = 20,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            default_rate: Default tokens per second for new users.
            default_burst: Default bucket capacity for new users.
        """
        self.default_rate = default_rate
        self.default_burst = default_burst
        self._buckets: dict[str, TokenBucket] = {}
        self._user_limits: dict[str, tuple[float, int]] = {}

    def _get_bucket(self, user_id: str) -> TokenBucket:
        """Get or create a token bucket for a user.

        Args:
            user_id: User identifier.

        Returns:
            TokenBucket for the user.
        """
        if user_id not in self._buckets:
            rate, burst = self._user_limits.get(
                user_id, (self.default_rate, self.default_burst)
            )
            self._buckets[user_id] = TokenBucket(rate=rate, capacity=float(burst))
        return self._buckets[user_id]

    def check(self, user_id: str, tokens: float = 1.0) -> bool:
        """Check if a request is allowed without consuming tokens.

        Args:
            user_id: User identifier.
            tokens: Number of tokens the request would consume.

        Returns:
            True if the request would be allowed.
        """
        bucket = self._get_bucket(user_id)
        return bucket.available() >= tokens

    def consume(self, user_id: str, tokens: float = 1.0) -> bool:
        """Try to consume tokens for a user's request.

        Args:
            user_id: User identifier.
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if rate limited.
        """
        bucket = self._get_bucket(user_id)
        allowed = bucket.consume(tokens)

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                user_id=user_id,
                requested_tokens=tokens,
                available_tokens=bucket.available(),
            )

        return allowed

    def time_until_allowed(self, user_id: str, tokens: float = 1.0) -> float:
        """Get time until a request would be allowed.

        Args:
            user_id: User identifier.
            tokens: Number of tokens the request would consume.

        Returns:
            Time in seconds until the request would be allowed.
        """
        bucket = self._get_bucket(user_id)
        return bucket.time_until_available(tokens)

    def set_user_limit(
        self,
        user_id: str,
        rate: float,
        burst: int,
    ) -> None:
        """Set custom rate limit for a user.

        Args:
            user_id: User identifier.
            rate: Tokens per second.
            burst: Maximum bucket capacity.
        """
        self._user_limits[user_id] = (rate, burst)
        # Reset bucket if it exists
        if user_id in self._buckets:
            self._buckets[user_id] = TokenBucket(rate=rate, capacity=float(burst))

        logger.info(
            "user_rate_limit_set",
            user_id=user_id,
            rate=rate,
            burst=burst,
        )

    def remove_user_limit(self, user_id: str) -> None:
        """Remove custom rate limit for a user (revert to default).

        Args:
            user_id: User identifier.
        """
        self._user_limits.pop(user_id, None)
        self._buckets.pop(user_id, None)

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        """Get rate limit info for a user.

        Args:
            user_id: User identifier.

        Returns:
            Dictionary with rate limit information.
        """
        bucket = self._get_bucket(user_id)
        rate, burst = self._user_limits.get(
            user_id, (self.default_rate, self.default_burst)
        )

        return {
            "user_id": user_id,
            "rate": rate,
            "burst": burst,
            "available_tokens": bucket.available(),
            "is_custom_limit": user_id in self._user_limits,
        }

    def reset(self) -> None:
        """Reset all rate limit state."""
        self._buckets.clear()
        self._user_limits.clear()

"""Statistics collection for Lexora."""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestStats:
    """Statistics for a single request.

    `start_time` and `end_time` are readings off `time.monotonic()`. Only
    their *difference* carries meaning: the reference point of that clock is
    undefined, so neither field is a timestamp and neither may be serialised,
    rendered as a date, or compared against a wall-clock value.

    They are deliberately not `time.time()`. That clock is adjustable
    (`time.get_clock_info("time").adjustable` is True), so an NTP step
    backwards between the two readings makes `duration` negative -- and
    `StatsCollector._record_stats` adds `duration` to a running accumulator
    that is only ever reset wholesale, so one such request skews the reported
    `average_duration_seconds` for the rest of the process lifetime.
    """

    endpoint: str
    model: str
    user_id: str | None
    start_time: float
    end_time: float | None = None
    success: bool = False
    error: str | None = None
    tokens_input: int = 0
    tokens_output: int = 0
    retries: int = 0

    @property
    def duration(self) -> float:
        """Get request duration in seconds.

        Both branches subtract two readings of the same clock. Whoever
        supplies `start_time` therefore has to read the clock named in the
        class docstring: mixing the two epochs does not fail loudly, it
        returns roughly +/- 1.76e9 seconds.
        """
        if self.end_time is None:
            return time.monotonic() - self.start_time
        return self.end_time - self.start_time


@dataclass
class AggregatedStats:
    """Aggregated statistics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_retries: int = 0
    average_duration: float = 0.0
    requests_per_endpoint: dict[str, int] = field(default_factory=dict)
    requests_per_model: dict[str, int] = field(default_factory=dict)
    errors_by_type: dict[str, int] = field(default_factory=dict)


class StatsCollector:
    """Collects and aggregates request statistics.

    Thread-safe statistics collection for monitoring API usage.
    """

    def __init__(self, max_history: int = 10000) -> None:
        """Initialize the stats collector.

        Args:
            max_history: Maximum number of requests to keep in history.
        """
        self._history: list[RequestStats] = []
        self._max_history = max_history
        self._total_duration: float = 0.0
        self._stats = AggregatedStats()

    def start_request(
        self,
        endpoint: str,
        model: str = "",
        user_id: str | None = None,
    ) -> RequestStats:
        """Start tracking a new request.

        Args:
            endpoint: API endpoint being called.
            model: Model being used.
            user_id: Optional user identifier.

        Returns:
            RequestStats instance for tracking the request.
        """
        return RequestStats(
            endpoint=endpoint,
            model=model,
            user_id=user_id,
            start_time=time.monotonic(),
        )

    def complete_request(
        self,
        stats: RequestStats,
        success: bool = True,
        error: str | None = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        retries: int = 0,
    ) -> None:
        """Complete tracking of a request.

        Args:
            stats: The RequestStats instance from start_request.
            success: Whether the request succeeded.
            error: Error message if failed.
            tokens_input: Number of input tokens.
            tokens_output: Number of output tokens.
            retries: Number of retry attempts.
        """
        stats.end_time = time.monotonic()
        stats.success = success
        stats.error = error
        stats.tokens_input = tokens_input
        stats.tokens_output = tokens_output
        stats.retries = retries

        self._record_stats(stats)

    def _record_stats(self, stats: RequestStats) -> None:
        """Record stats to history and update aggregates.

        Args:
            stats: Completed request stats.
        """
        # Add to history
        self._history.append(stats)

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # Update aggregates
        self._stats.total_requests += 1
        if stats.success:
            self._stats.successful_requests += 1
        else:
            self._stats.failed_requests += 1
            if stats.error:
                error_type = stats.error.split(":")[0] if ":" in stats.error else stats.error
                self._stats.errors_by_type[error_type] = (
                    self._stats.errors_by_type.get(error_type, 0) + 1
                )

        self._stats.total_tokens_input += stats.tokens_input
        self._stats.total_tokens_output += stats.tokens_output
        self._stats.total_retries += stats.retries

        # Update duration average
        self._total_duration += stats.duration
        self._stats.average_duration = self._total_duration / self._stats.total_requests

        # Update per-endpoint stats
        self._stats.requests_per_endpoint[stats.endpoint] = (
            self._stats.requests_per_endpoint.get(stats.endpoint, 0) + 1
        )

        # Update per-model stats
        if stats.model:
            self._stats.requests_per_model[stats.model] = (
                self._stats.requests_per_model.get(stats.model, 0) + 1
            )

    def get_stats(self) -> dict[str, Any]:
        """Get current aggregated statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "total_requests": self._stats.total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "success_rate": (
                self._stats.successful_requests / self._stats.total_requests
                if self._stats.total_requests > 0
                else 0.0
            ),
            "total_tokens_input": self._stats.total_tokens_input,
            "total_tokens_output": self._stats.total_tokens_output,
            "total_retries": self._stats.total_retries,
            "average_duration_seconds": round(self._stats.average_duration, 4),
            "requests_per_endpoint": dict(self._stats.requests_per_endpoint),
            "requests_per_model": dict(self._stats.requests_per_model),
            "errors_by_type": dict(self._stats.errors_by_type),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._history.clear()
        self._total_duration = 0.0
        self._stats = AggregatedStats()

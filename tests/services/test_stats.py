"""Tests for statistics collector."""

import time

import pytest

from lexora.services.stats import RequestStats, StatsCollector


class TestRequestStats:
    """Tests for RequestStats dataclass."""

    def test_duration_with_end_time(self) -> None:
        """Test duration calculation with end_time set."""
        stats = RequestStats(
            endpoint="/v1/chat/completions",
            model="test-model",
            user_id="user-1",
            start_time=100.0,
            end_time=105.5,
        )
        assert stats.duration == 5.5

    def test_duration_without_end_time(self) -> None:
        """Test duration calculation without end_time (ongoing request)."""
        start = time.time()
        stats = RequestStats(
            endpoint="/v1/chat/completions",
            model="test-model",
            user_id=None,
            start_time=start,
        )
        # Duration should be close to 0 for a just-created request
        assert 0 <= stats.duration < 1


class TestStatsCollector:
    """Tests for StatsCollector."""

    @pytest.fixture
    def collector(self) -> StatsCollector:
        """Create a stats collector for testing."""
        return StatsCollector()

    def test_start_request(self, collector: StatsCollector) -> None:
        """Test starting a request creates proper stats."""
        stats = collector.start_request(
            endpoint="/v1/chat/completions",
            model="gpt-4",
            user_id="user-123",
        )

        assert stats.endpoint == "/v1/chat/completions"
        assert stats.model == "gpt-4"
        assert stats.user_id == "user-123"
        assert stats.start_time > 0
        assert stats.end_time is None
        assert stats.success is False

    def test_complete_request_success(self, collector: StatsCollector) -> None:
        """Test completing a successful request."""
        stats = collector.start_request("/v1/chat/completions", "gpt-4")
        collector.complete_request(
            stats,
            success=True,
            tokens_input=100,
            tokens_output=50,
        )

        assert stats.success is True
        assert stats.end_time is not None
        assert stats.tokens_input == 100
        assert stats.tokens_output == 50

        aggregated = collector.get_stats()
        assert aggregated["total_requests"] == 1
        assert aggregated["successful_requests"] == 1
        assert aggregated["failed_requests"] == 0

    def test_complete_request_failure(self, collector: StatsCollector) -> None:
        """Test completing a failed request."""
        stats = collector.start_request("/v1/chat/completions", "gpt-4")
        collector.complete_request(
            stats,
            success=False,
            error="ConnectionError: Failed to connect",
        )

        assert stats.success is False
        assert stats.error == "ConnectionError: Failed to connect"

        aggregated = collector.get_stats()
        assert aggregated["total_requests"] == 1
        assert aggregated["failed_requests"] == 1
        assert aggregated["errors_by_type"]["ConnectionError"] == 1

    def test_aggregated_stats(self, collector: StatsCollector) -> None:
        """Test aggregated statistics calculation."""
        # Add multiple requests
        for i in range(5):
            stats = collector.start_request("/v1/chat/completions", "gpt-4")
            collector.complete_request(
                stats,
                success=True,
                tokens_input=100,
                tokens_output=50,
            )

        for i in range(3):
            stats = collector.start_request("/v1/completions", "gpt-3.5")
            collector.complete_request(
                stats,
                success=False,
                error="TimeoutError: Request timed out",
            )

        aggregated = collector.get_stats()
        assert aggregated["total_requests"] == 8
        assert aggregated["successful_requests"] == 5
        assert aggregated["failed_requests"] == 3
        assert aggregated["success_rate"] == 5 / 8
        assert aggregated["total_tokens_input"] == 500
        assert aggregated["total_tokens_output"] == 250
        assert aggregated["requests_per_endpoint"]["/v1/chat/completions"] == 5
        assert aggregated["requests_per_endpoint"]["/v1/completions"] == 3
        assert aggregated["requests_per_model"]["gpt-4"] == 5
        assert aggregated["requests_per_model"]["gpt-3.5"] == 3

    def test_retries_tracking(self, collector: StatsCollector) -> None:
        """Test retry count tracking."""
        stats = collector.start_request("/v1/chat/completions", "gpt-4")
        collector.complete_request(stats, success=True, retries=3)

        aggregated = collector.get_stats()
        assert aggregated["total_retries"] == 3

    def test_max_history_limit(self) -> None:
        """Test that history is limited to max_history."""
        collector = StatsCollector(max_history=10)

        for i in range(20):
            stats = collector.start_request("/v1/chat/completions", "gpt-4")
            collector.complete_request(stats, success=True)

        assert len(collector._history) == 10

    def test_reset(self, collector: StatsCollector) -> None:
        """Test resetting statistics."""
        stats = collector.start_request("/v1/chat/completions", "gpt-4")
        collector.complete_request(stats, success=True)

        collector.reset()

        aggregated = collector.get_stats()
        assert aggregated["total_requests"] == 0
        assert aggregated["successful_requests"] == 0

    def test_empty_stats(self, collector: StatsCollector) -> None:
        """Test getting stats when no requests recorded."""
        aggregated = collector.get_stats()
        assert aggregated["total_requests"] == 0
        assert aggregated["success_rate"] == 0.0
        assert aggregated["average_duration_seconds"] == 0.0

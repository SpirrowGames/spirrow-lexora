"""Tests for Prometheus metrics."""

import pytest
from prometheus_client import REGISTRY

from lexora.services.metrics import (
    MetricsCollector,
    REQUESTS_TOTAL,
    REQUEST_DURATION_SECONDS,
    TOKENS_INPUT_TOTAL,
    TOKENS_OUTPUT_TOTAL,
    RETRIES_TOTAL,
    RATE_LIMIT_REJECTIONS_TOTAL,
    BACKEND_HEALTH,
    ACTIVE_REQUESTS,
    STREAMING_REQUESTS_TOTAL,
)


@pytest.fixture(autouse=True)
def reset_metrics() -> None:
    """Reset metrics before each test."""
    # Note: prometheus_client doesn't provide a clean way to reset metrics
    # In production tests, you might want to use a separate registry per test
    pass


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_init(self) -> None:
        """Test MetricsCollector initialization."""
        collector = MetricsCollector(version="1.0.0")
        assert collector is not None

    def test_record_request_start(self) -> None:
        """Test recording request start."""
        collector = MetricsCollector()
        collector.record_request_start("/v1/chat/completions")
        # Just verify no exceptions raised

    def test_record_request_end_success(self) -> None:
        """Test recording successful request end."""
        collector = MetricsCollector()
        collector.record_request_start("/v1/chat/completions")
        collector.record_request_end(
            endpoint="/v1/chat/completions",
            model="gpt-4",
            status="success",
            duration=1.5,
            tokens_input=100,
            tokens_output=50,
            retries=0,
        )
        # Just verify no exceptions raised

    def test_record_request_end_error(self) -> None:
        """Test recording error request end."""
        collector = MetricsCollector()
        collector.record_request_start("/v1/chat/completions")
        collector.record_request_end(
            endpoint="/v1/chat/completions",
            model="gpt-4",
            status="error",
            duration=0.5,
        )
        # Just verify no exceptions raised

    def test_record_request_streaming(self) -> None:
        """Test recording streaming request."""
        collector = MetricsCollector()
        collector.record_request_start("/v1/chat/completions")
        collector.record_request_end(
            endpoint="/v1/chat/completions",
            model="gpt-4",
            status="success",
            duration=5.0,
            streaming=True,
        )
        # Just verify no exceptions raised

    def test_record_rate_limit_rejection(self) -> None:
        """Test recording rate limit rejection."""
        collector = MetricsCollector()
        collector.record_rate_limit_rejection("user-123")
        # Just verify no exceptions raised

    def test_set_backend_health_healthy(self) -> None:
        """Test setting backend health to healthy."""
        collector = MetricsCollector()
        collector.set_backend_health("vllm", True)
        # Just verify no exceptions raised

    def test_set_backend_health_unhealthy(self) -> None:
        """Test setting backend health to unhealthy."""
        collector = MetricsCollector()
        collector.set_backend_health("vllm", False)
        # Just verify no exceptions raised

    def test_record_request_with_retries(self) -> None:
        """Test recording request with retries."""
        collector = MetricsCollector()
        collector.record_request_start("/v1/completions")
        collector.record_request_end(
            endpoint="/v1/completions",
            model="gpt-3.5",
            status="success",
            duration=2.0,
            retries=3,
        )
        # Just verify no exceptions raised

    def test_multiple_requests(self) -> None:
        """Test recording multiple requests."""
        collector = MetricsCollector()

        for i in range(5):
            collector.record_request_start("/v1/chat/completions")
            collector.record_request_end(
                endpoint="/v1/chat/completions",
                model="gpt-4",
                status="success" if i % 2 == 0 else "error",
                duration=0.5 + i * 0.1,
            )

        # Just verify no exceptions raised

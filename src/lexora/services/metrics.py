"""Prometheus metrics for Lexora."""

from prometheus_client import Counter, Gauge, Histogram, Info

# Application info
APP_INFO = Info("lexora", "Lexora LLM Gateway information")

# Request counters
REQUESTS_TOTAL = Counter(
    "lexora_requests_total",
    "Total number of requests",
    ["endpoint", "model", "status"],
)

# Request duration histogram
REQUEST_DURATION_SECONDS = Histogram(
    "lexora_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# Token counters
TOKENS_INPUT_TOTAL = Counter(
    "lexora_tokens_input_total",
    "Total number of input tokens processed",
    ["model"],
)

TOKENS_OUTPUT_TOTAL = Counter(
    "lexora_tokens_output_total",
    "Total number of output tokens generated",
    ["model"],
)

# Retry counter
RETRIES_TOTAL = Counter(
    "lexora_retries_total",
    "Total number of retry attempts",
    ["endpoint", "model"],
)

# Rate limiting
RATE_LIMIT_REJECTIONS_TOTAL = Counter(
    "lexora_rate_limit_rejections_total",
    "Total number of rate limit rejections",
    ["user_id"],
)

# Backend health
BACKEND_HEALTH = Gauge(
    "lexora_backend_health",
    "Backend health status (1=healthy, 0=unhealthy)",
    ["backend"],
)

# Active requests gauge
ACTIVE_REQUESTS = Gauge(
    "lexora_active_requests",
    "Number of currently active requests",
    ["endpoint"],
)

# Streaming metrics
STREAMING_REQUESTS_TOTAL = Counter(
    "lexora_streaming_requests_total",
    "Total number of streaming requests",
    ["endpoint", "model", "status"],
)


class MetricsCollector:
    """Collector for Prometheus metrics."""

    def __init__(self, version: str = "0.1.0") -> None:
        """Initialize metrics collector.

        Args:
            version: Application version.
        """
        APP_INFO.info({"version": version})

    def record_request_start(self, endpoint: str) -> None:
        """Record the start of a request.

        Args:
            endpoint: API endpoint.
        """
        ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()

    def record_request_end(
        self,
        endpoint: str,
        model: str,
        status: str,
        duration: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        retries: int = 0,
        streaming: bool = False,
    ) -> None:
        """Record the end of a request.

        Args:
            endpoint: API endpoint.
            model: Model name.
            status: Request status (success/error).
            duration: Request duration in seconds.
            tokens_input: Number of input tokens.
            tokens_output: Number of output tokens.
            retries: Number of retry attempts.
            streaming: Whether this was a streaming request.
        """
        ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()

        if streaming:
            STREAMING_REQUESTS_TOTAL.labels(
                endpoint=endpoint, model=model, status=status
            ).inc()
        else:
            REQUESTS_TOTAL.labels(endpoint=endpoint, model=model, status=status).inc()

        REQUEST_DURATION_SECONDS.labels(endpoint=endpoint, model=model).observe(duration)

        if tokens_input > 0:
            TOKENS_INPUT_TOTAL.labels(model=model).inc(tokens_input)
        if tokens_output > 0:
            TOKENS_OUTPUT_TOTAL.labels(model=model).inc(tokens_output)

        if retries > 0:
            RETRIES_TOTAL.labels(endpoint=endpoint, model=model).inc(retries)

    def record_rate_limit_rejection(self, user_id: str) -> None:
        """Record a rate limit rejection.

        Args:
            user_id: User identifier.
        """
        RATE_LIMIT_REJECTIONS_TOTAL.labels(user_id=user_id).inc()

    def set_backend_health(self, backend: str, healthy: bool) -> None:
        """Set backend health status.

        Args:
            backend: Backend name.
            healthy: Whether the backend is healthy.
        """
        BACKEND_HEALTH.labels(backend=backend).set(1 if healthy else 0)

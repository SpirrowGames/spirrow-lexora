"""Main FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from lexora import __version__
from lexora.api.routes import router
from lexora.backends.base import ModelNotFoundError
from lexora.backends.vllm import VLLMBackend
from lexora.config import create_settings, Settings
from lexora.services.metrics import MetricsCollector
from lexora.services.model_registry import ModelRegistry
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.router import BackendRouter
from lexora.services.cost_tracker import CostTracker
from lexora.services.stats import StatsCollector
from lexora.services.task_classifier import TaskClassifier
from lexora.utils.logging import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager.

    Handles startup and shutdown of application resources.

    Args:
        app: FastAPI application instance.

    Yields:
        None
    """
    settings: Settings = app.state.settings
    logger = get_logger(__name__)

    # Startup
    logger.info("lexora_starting", version=__version__)

    # Initialize backend router (supports both single and multi-backend modes)
    app.state.backend_router = BackendRouter(
        routing_settings=settings.routing,
        vllm_settings=settings.vllm,
    )
    # For backward compatibility, also expose default backend as 'backend'
    app.state.backend = app.state.backend_router.default_backend

    # Initialize model registry (for capabilities endpoint)
    app.state.model_registry = ModelRegistry(routing_settings=settings.routing)

    # Initialize task classifier (if enabled)
    app.state.task_classifier = TaskClassifier(
        model_registry=app.state.model_registry,
        backend_router=app.state.backend_router,
        classifier_settings=settings.routing.classifier,
    )

    # Initialize services
    app.state.stats_collector = StatsCollector()
    app.state.cost_tracker = CostTracker()
    app.state.retry_handler = RetryHandler(
        max_retries=settings.retry.max_retries,
        base_delay=settings.retry.base_delay,
        max_delay=settings.retry.max_delay,
        exponential_base=settings.retry.exponential_base,
    )
    app.state.rate_limiter = RateLimiter(
        default_rate=settings.rate_limit.default_rate,
        default_burst=settings.rate_limit.default_burst,
    )
    app.state.rate_limit_enabled = settings.rate_limit.enabled

    # Initialize metrics collector
    app.state.metrics_collector = MetricsCollector(version=__version__)

    logger.info(
        "lexora_started",
        vllm_url=settings.vllm.url,
        host=settings.server.host,
        port=settings.server.port,
    )

    yield

    # Shutdown
    logger.info("lexora_shutting_down")
    await app.state.backend_router.close()
    logger.info("lexora_shutdown_complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings instance. If None, loads from config.

    Returns:
        Configured FastAPI application.
    """
    if settings is None:
        settings = create_settings()

    # Setup logging
    setup_logging(
        level=settings.logging.level,
        format=settings.logging.format,
    )

    app = FastAPI(
        title="Lexora",
        description="LLM Gateway / Router for Spirrow Platform",
        version=__version__,
        lifespan=lifespan,
    )

    # Store settings in app state
    app.state.settings = settings

    # Include API routes
    app.include_router(router)

    # ModelNotFoundError -> 404 in the caller's dialect (T-silent-routing
    # R-1a / R-2). Registered as a global handler so every endpoint that
    # calls into ``BackendRouter.get_backend_for_model`` — the /v1/*, /chat,
    # /generate and /v1/messages families — reports the same status for
    # "unknown model" and "ambiguous model", without a wrapping try/except
    # at each callsite.
    #
    # The body shape follows the endpoint the request landed on:
    #
    # * /v1/messages returns the Anthropic ``{"type": "error", "error":
    #   {...}}`` envelope so the ``anthropic`` SDK parses the failure
    #   natively (the endpoint uses ``anthropic_error_body`` everywhere
    #   else on the error path — a 404 in OpenAI shape would be the one
    #   response that SDK would fail to typecheck).
    # * Everything else returns the OpenAI ``{"error": {"message":,
    #   "type":, "param":, "code":}}`` envelope with ``code:
    #   model_not_found``, matching upstream vLLM / OpenAI for the same
    #   HTTP status.
    #
    # The API code stays ``model_not_found`` for both unknown and ambiguous
    # cases; the router logs the distinct event names
    # (``model_unknown_refused`` / ``model_ambiguous_refused``) so operators
    # can grep the two apart without introducing a client-side branch
    # nobody has (T-silent-routing msg-082 objection B / msg-083 §2).
    from lexora.api.anthropic_compat import anthropic_error_body

    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found_handler(
        request: Request, exc: ModelNotFoundError
    ) -> JSONResponse:
        if request.url.path == "/v1/messages":
            return JSONResponse(
                status_code=404,
                content=anthropic_error_body("not_found_error", str(exc)),
            )
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    # Add metrics endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


# Create default app instance
app = create_app()


def main() -> None:
    """Run the application using uvicorn."""
    settings = create_settings()

    setup_logging(
        level=settings.logging.level,
        format=settings.logging.format,
    )

    uvicorn.run(
        "lexora.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

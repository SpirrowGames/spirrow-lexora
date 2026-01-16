"""Main FastAPI application entry point."""

import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI

from lexora import __version__
from lexora.api.routes import router
from lexora.backends.vllm import VLLMBackend
from lexora.config import create_settings, Settings
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector
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

    # Initialize backend
    app.state.backend = VLLMBackend(
        base_url=settings.vllm.url,
        timeout=settings.vllm.timeout,
        connect_timeout=settings.vllm.connect_timeout,
    )

    # Initialize services
    app.state.stats_collector = StatsCollector()
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

    logger.info(
        "lexora_started",
        vllm_url=settings.vllm.url,
        host=settings.server.host,
        port=settings.server.port,
    )

    yield

    # Shutdown
    logger.info("lexora_shutting_down")
    await app.state.backend.close()
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

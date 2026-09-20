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
from lexora.decide.config import check_typesafe_api_key
from lexora.decide.log import DecisionLog
from lexora.decide.routes import build_default_providers, router as decide_router
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
    decision_log = getattr(app.state, "decision_log", None)
    if decision_log is not None:
        decision_log.close()
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

    # Decision-endpoint startup check (msg-239 / msg-240):
    #
    # This runs before the app object is even built. A config that
    # names ``jev`` for ``primary`` or ``fallback`` while
    # ``TYPESAFE_API_KEY`` is unset does not produce a partially-
    # constructed FastAPI application — it raises during ``create_app``
    # so uvicorn refuses to bind the port, which under
    # ``deploy/lexora.service`` (``Restart=always``) makes the failure
    # observable as a crash loop rather than a silent degradation to
    # NullProvider.
    #
    # The message does NOT carry the value of the env variable, its
    # length, or any prefix (msg-240 §1). Callers do not need those
    # facts to fix the config, and log capture pipelines would
    # otherwise carry them further than intended.
    check_typesafe_api_key(settings.decision)

    app = FastAPI(
        title="Lexora",
        description="LLM Gateway / Router for Spirrow Platform",
        version=__version__,
        lifespan=lifespan,
    )

    # Store settings in app state
    app.state.settings = settings

    # /v1/decide wiring (T-decide-endpoint T02 PR 1). Provider registry
    # is built here so the endpoint has stable state to lean on; PR 1
    # ships only NullProvider (msg-246), but the mount point does not
    # need to change when llm / jev arrive in follow-up PRs.
    app.state.decision_settings = settings.decision
    app.state.decision_providers = build_default_providers()
    # On-disk by default (msg-251 blocking objection): the shadow-mode
    # data-collection story msg-237 requires — "較正曲線とリプレイ評価
    # はここから引く" / "mindwire の 116 判断点リプレイもこの
    # エンドポイント経由で流し、オフライン評価と本番を同一コード
    # パスにする" — is only satisfied if rows survive a process
    # restart. ``settings.decision.log_path`` defaults to
    # ``data/decisions.db``; tests point it at a tmp path or at
    # ``:memory:``. The parent directory is created by DecisionLog
    # itself.
    app.state.decision_log = DecisionLog(path=settings.decision.log_path)

    # Include API routes
    app.include_router(router)
    app.include_router(decide_router)

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
        # Which endpoint the request landed on is answered by the route
        # that matched, not by the request URL.
        #
        # ``request.url.path`` is ``scope["path"]`` verbatim (Starlette
        # 0.50.0, ``URL.__init__``), and ``scope["path"]`` is whatever the
        # ASGI server was handed. A server run with a ``root_path`` behind
        # a proxy that forwards the prefix rather than stripping it hands
        # over ``path="/api/v1/messages"`` with ``root_path="/api"``.
        # Starlette strips the prefix for routing only
        # (``starlette.routing.get_route_path``), so the request matches
        # this endpoint while the URL string does not equal its path — and
        # an Anthropic client would get the OpenAI envelope, the one
        # response shape its SDK cannot typecheck.
        #
        # ``scope["route"]`` is the route the router matched, set by
        # FastAPI's ``APIRoute.matches``. Reading it answers the question
        # instead of re-deriving it from a string the deployment is
        # allowed to rewrite. It is absent only when no route matched, and
        # for those the OpenAI envelope below is the right default.
        matched_route = request.scope.get("route")
        if getattr(matched_route, "path", None) == "/v1/messages":
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

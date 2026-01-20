"""API routes for Lexora."""

import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from lexora import __version__
from lexora.api.models import (
    ChatCompletionRequest,
    ClassifyTaskRequest,
    ClassifyTaskResponse,
    CompletionRequest,
    EmbeddingsRequest,
    ErrorResponse,
    HealthResponse,
    ModelAlternative,
    ModelCapabilitiesResponse,
    ModelCapabilityInfo,
    StatsResponse,
)
from lexora.backends.base import BackendError
from lexora.backends.vllm import VLLMBackend
from lexora.services.metrics import MetricsCollector
from lexora.services.model_registry import ModelRegistry
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.router import BackendRouter
from lexora.services.stats import StatsCollector
from lexora.services.task_classifier import (
    TaskClassifier,
    TaskClassifierDisabledError,
    TaskClassifierError,
)
from lexora.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def get_backend(request: Request) -> VLLMBackend:
    """Get default vLLM backend from app state."""
    return request.app.state.backend


def get_backend_router(request: Request) -> BackendRouter:
    """Get backend router from app state."""
    return request.app.state.backend_router


def get_stats_collector(request: Request) -> StatsCollector:
    """Get stats collector from app state."""
    return request.app.state.stats_collector


def get_retry_handler(request: Request) -> RetryHandler:
    """Get retry handler from app state."""
    return request.app.state.retry_handler


def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter from app state."""
    return request.app.state.rate_limiter


def is_rate_limit_enabled(request: Request) -> bool:
    """Check if rate limiting is enabled."""
    return getattr(request.app.state, "rate_limit_enabled", True)


def get_metrics_collector(request: Request) -> MetricsCollector | None:
    """Get metrics collector from app state."""
    return getattr(request.app.state, "metrics_collector", None)


def get_model_registry(request: Request) -> ModelRegistry | None:
    """Get model registry from app state."""
    return getattr(request.app.state, "model_registry", None)


def get_task_classifier(request: Request) -> TaskClassifier | None:
    """Get task classifier from app state."""
    return getattr(request.app.state, "task_classifier", None)


def check_rate_limit(
    user_id: str | None,
    rate_limiter: RateLimiter,
    enabled: bool,
    metrics_collector: MetricsCollector | None = None,
) -> None:
    """Check rate limit for user.

    Args:
        user_id: User identifier (None uses "anonymous").
        rate_limiter: Rate limiter instance.
        enabled: Whether rate limiting is enabled.
        metrics_collector: Optional metrics collector for recording rejections.

    Raises:
        HTTPException: If rate limited (429).
    """
    if not enabled:
        return

    effective_user_id = user_id or "anonymous"
    if not rate_limiter.consume(effective_user_id):
        wait_time = rate_limiter.time_until_allowed(effective_user_id)

        # Record rate limit rejection in metrics
        if metrics_collector:
            metrics_collector.record_rate_limit_rejection(effective_user_id)

        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {wait_time:.1f} seconds.",
            headers={"Retry-After": str(int(wait_time) + 1)},
        )


@router.post(
    "/v1/chat/completions",
    response_model=None,
    responses={429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def chat_completions(
    request: ChatCompletionRequest,
    backend_router: BackendRouter = Depends(get_backend_router),
    stats_collector: StatsCollector = Depends(get_stats_collector),
    retry_handler: RetryHandler = Depends(get_retry_handler),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    rate_limit_enabled: bool = Depends(is_rate_limit_enabled),
    metrics_collector: MetricsCollector | None = Depends(get_metrics_collector),
) -> dict[str, Any] | StreamingResponse:
    """Proxy chat completion request to vLLM.

    Args:
        request: OpenAI-compatible chat completion request.
        backend_router: Backend router for model routing.
        stats_collector: Statistics collector.
        retry_handler: Retry handler.
        rate_limiter: Rate limiter.
        rate_limit_enabled: Whether rate limiting is enabled.
        metrics_collector: Prometheus metrics collector.

    Returns:
        OpenAI-compatible chat completion response or streaming response.
    """
    endpoint = "/v1/chat/completions"

    # Check rate limit
    check_rate_limit(request.user, rate_limiter, rate_limit_enabled, metrics_collector)

    # Get backend for the requested model
    backend = backend_router.get_backend_for_model(request.model)

    # Convert to dict, excluding None values
    request_dict = request.model_dump(exclude_none=True)

    # Handle streaming request
    if request.stream:
        stats = stats_collector.start_request(
            endpoint=endpoint,
            model=request.model,
            user_id=request.user,
        )
        start_time = time.time()

        # Record metrics start
        if metrics_collector:
            metrics_collector.record_request_start(endpoint)

        async def stream_generator() -> AsyncIterator[bytes]:
            try:
                async for chunk in backend.chat_completions_stream(request_dict):
                    yield chunk

                # Mark as successful on stream completion
                duration = time.time() - start_time
                stats_collector.complete_request(stats, success=True)

                # Record metrics
                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="success",
                        duration=duration,
                        streaming=True,
                    )

                logger.info(
                    "chat_completion_stream_success",
                    model=request.model,
                    duration=duration,
                )
            except BackendError as e:
                duration = time.time() - start_time
                stats_collector.complete_request(stats, success=False, error=str(e))

                # Record error metrics
                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="error",
                        duration=duration,
                        streaming=True,
                    )

                logger.error(
                    "chat_completion_stream_error",
                    model=request.model,
                    error=str(e),
                )
                raise
            except Exception as e:
                duration = time.time() - start_time
                stats_collector.complete_request(stats, success=False, error=str(e))

                # Record error metrics
                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="error",
                        duration=duration,
                        streaming=True,
                    )

                logger.exception(
                    "chat_completion_stream_unexpected_error",
                    model=request.model,
                )
                raise

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming request
    stats = stats_collector.start_request(
        endpoint=endpoint,
        model=request.model,
        user_id=request.user,
    )
    start_time = time.time()

    # Record metrics start
    if metrics_collector:
        metrics_collector.record_request_start(endpoint)

    try:
        async def do_request() -> dict[str, Any]:
            return await backend.chat_completions(request_dict)

        response, retries = await retry_handler.execute(do_request)

        # Extract token counts from response if available
        usage = response.get("usage", {})
        tokens_input = usage.get("prompt_tokens", 0)
        tokens_output = usage.get("completion_tokens", 0)
        duration = time.time() - start_time

        stats_collector.complete_request(
            stats,
            success=True,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            retries=retries,
        )

        # Record metrics
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="success",
                duration=duration,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                retries=retries,
            )

        logger.info(
            "chat_completion_success",
            model=request.model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            retries=retries,
        )

        return response

    except BackendError as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))

        # Record error metrics
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=duration,
            )

        logger.error("chat_completion_error", model=request.model, error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))

        # Record error metrics
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=duration,
            )
        logger.exception("chat_completion_unexpected_error", model=request.model)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/v1/completions",
    response_model=None,
    responses={429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def completions(
    request: CompletionRequest,
    backend_router: BackendRouter = Depends(get_backend_router),
    stats_collector: StatsCollector = Depends(get_stats_collector),
    retry_handler: RetryHandler = Depends(get_retry_handler),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    rate_limit_enabled: bool = Depends(is_rate_limit_enabled),
    metrics_collector: MetricsCollector | None = Depends(get_metrics_collector),
) -> dict[str, Any] | StreamingResponse:
    """Proxy completion request to vLLM.

    Args:
        request: OpenAI-compatible completion request.
        backend_router: Backend router for model routing.
        stats_collector: Statistics collector.
        retry_handler: Retry handler.
        rate_limiter: Rate limiter.
        rate_limit_enabled: Whether rate limiting is enabled.
        metrics_collector: Prometheus metrics collector.

    Returns:
        OpenAI-compatible completion response or streaming response.
    """
    endpoint = "/v1/completions"

    # Check rate limit
    check_rate_limit(request.user, rate_limiter, rate_limit_enabled, metrics_collector)

    # Get backend for the requested model
    backend = backend_router.get_backend_for_model(request.model)

    # Convert to dict, excluding None values
    request_dict = request.model_dump(exclude_none=True)

    # Handle streaming request
    if request.stream:
        stats = stats_collector.start_request(
            endpoint=endpoint,
            model=request.model,
            user_id=request.user,
        )
        start_time = time.time()

        # Record metrics start
        if metrics_collector:
            metrics_collector.record_request_start(endpoint)

        async def stream_generator() -> AsyncIterator[bytes]:
            try:
                async for chunk in backend.completions_stream(request_dict):
                    yield chunk

                # Mark as successful on stream completion
                duration = time.time() - start_time
                stats_collector.complete_request(stats, success=True)

                # Record metrics
                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="success",
                        duration=duration,
                        streaming=True,
                    )

                logger.info(
                    "completion_stream_success",
                    model=request.model,
                    duration=duration,
                )
            except BackendError as e:
                duration = time.time() - start_time
                stats_collector.complete_request(stats, success=False, error=str(e))

                # Record error metrics
                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="error",
                        duration=duration,
                        streaming=True,
                    )

                logger.error(
                    "completion_stream_error",
                    model=request.model,
                    error=str(e),
                )
                raise
            except Exception as e:
                duration = time.time() - start_time
                stats_collector.complete_request(stats, success=False, error=str(e))

                # Record error metrics
                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="error",
                        duration=duration,
                        streaming=True,
                    )

                logger.exception(
                    "completion_stream_unexpected_error",
                    model=request.model,
                )
                raise

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming request
    stats = stats_collector.start_request(
        endpoint=endpoint,
        model=request.model,
        user_id=request.user,
    )
    start_time = time.time()

    # Record metrics start
    if metrics_collector:
        metrics_collector.record_request_start(endpoint)

    try:
        async def do_request() -> dict[str, Any]:
            return await backend.completions(request_dict)

        response, retries = await retry_handler.execute(do_request)

        usage = response.get("usage", {})
        tokens_input = usage.get("prompt_tokens", 0)
        tokens_output = usage.get("completion_tokens", 0)
        duration = time.time() - start_time

        stats_collector.complete_request(
            stats,
            success=True,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            retries=retries,
        )

        # Record metrics
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="success",
                duration=duration,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                retries=retries,
            )

        logger.info(
            "completion_success",
            model=request.model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            retries=retries,
        )

        return response

    except BackendError as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))

        # Record error metrics
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=duration,
            )

        logger.error("completion_error", model=request.model, error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))

        # Record error metrics
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=duration,
            )

        logger.exception("completion_unexpected_error", model=request.model)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/v1/embeddings",
    responses={429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def embeddings(
    request: EmbeddingsRequest,
    backend_router: BackendRouter = Depends(get_backend_router),
    stats_collector: StatsCollector = Depends(get_stats_collector),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    rate_limit_enabled: bool = Depends(is_rate_limit_enabled),
    metrics_collector: MetricsCollector | None = Depends(get_metrics_collector),
) -> dict[str, Any]:
    """Proxy embeddings request to vLLM.

    Args:
        request: OpenAI-compatible embeddings request.
        backend_router: Backend router for model routing.
        stats_collector: Statistics collector.
        rate_limiter: Rate limiter.
        rate_limit_enabled: Whether rate limiting is enabled.
        metrics_collector: Prometheus metrics collector.

    Returns:
        OpenAI-compatible embeddings response.
    """
    endpoint = "/v1/embeddings"

    # Check rate limit
    check_rate_limit(request.user, rate_limiter, rate_limit_enabled, metrics_collector)

    # Get backend for the requested model
    backend = backend_router.get_backend_for_model(request.model)

    stats = stats_collector.start_request(
        endpoint=endpoint,
        model=request.model,
        user_id=request.user,
    )
    start_time = time.time()

    # Record metrics start
    if metrics_collector:
        metrics_collector.record_request_start(endpoint)

    try:
        request_dict = request.model_dump(exclude_none=True)
        response = await backend.embeddings(request_dict)

        # Extract token counts from response if available
        usage = response.get("usage", {})
        tokens_input = usage.get("prompt_tokens", 0)
        tokens_output = 0  # Embeddings don't have output tokens
        duration = time.time() - start_time

        stats_collector.complete_request(
            stats,
            success=True,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )

        # Record metrics
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="success",
                duration=duration,
                tokens_input=tokens_input,
            )

        logger.info(
            "embeddings_success",
            model=request.model,
            tokens_input=tokens_input,
        )

        return response

    except BackendError as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))

        # Record error metrics
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=duration,
            )

        logger.error("embeddings_error", model=request.model, error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))

        # Record error metrics
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=duration,
            )

        logger.exception("embeddings_unexpected_error", model=request.model)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v1/models")
async def list_models(
    backend_router: BackendRouter = Depends(get_backend_router),
) -> dict[str, Any]:
    """List available models from all backends.

    Args:
        backend_router: Backend router.

    Returns:
        OpenAI-compatible models list response.
    """
    try:
        return await backend_router.list_all_models()
    except BackendError as e:
        logger.error("list_models_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/health", response_model=HealthResponse)
async def health(
    backend_router: BackendRouter = Depends(get_backend_router),
    metrics_collector: MetricsCollector | None = Depends(get_metrics_collector),
) -> HealthResponse:
    """Check health status of all backends.

    Args:
        backend_router: Backend router.
        metrics_collector: Prometheus metrics collector.

    Returns:
        Health status response.
    """
    backend_health = await backend_router.health_check()

    # Record backend health in metrics
    if metrics_collector:
        for backend_name, is_healthy in backend_health.items():
            metrics_collector.set_backend_health(backend_name, is_healthy)

    # Convert bool to status strings
    backends_status = {
        name: "healthy" if healthy else "unhealthy"
        for name, healthy in backend_health.items()
    }

    # Determine overall status
    all_healthy = all(backend_health.values())
    any_healthy = any(backend_health.values())

    if all_healthy:
        overall_status = "healthy"
    elif any_healthy:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    # Legacy vllm_status for backward compatibility
    default_backend_healthy = backend_health.get("default", True)
    vllm_status = "healthy" if default_backend_healthy else "unhealthy"

    return HealthResponse(
        status=overall_status,
        backends=backends_status,
        version=__version__,
        vllm_status=vllm_status,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats(
    stats_collector: StatsCollector = Depends(get_stats_collector),
) -> StatsResponse:
    """Get statistics.

    Args:
        stats_collector: Statistics collector.

    Returns:
        Statistics response.
    """
    stats_data = stats_collector.get_stats()
    return StatsResponse(**stats_data)


@router.get("/v1/models/capabilities", response_model=ModelCapabilitiesResponse)
async def get_model_capabilities(
    model_registry: ModelRegistry | None = Depends(get_model_registry),
) -> ModelCapabilitiesResponse:
    """Get all models with their capabilities.

    Returns information about all available models including their
    capabilities and backend information.

    Args:
        model_registry: Model registry instance.

    Returns:
        ModelCapabilitiesResponse with models and capabilities.
    """
    if model_registry is None:
        logger.warning("model_capabilities_request_no_registry")
        return ModelCapabilitiesResponse(
            models=[],
            available_capabilities=[],
            default_model_for_unknown_task=None,
        )

    models = model_registry.get_all_models()
    model_infos = [
        ModelCapabilityInfo(
            id=model.id,
            backend=model.backend,
            backend_type=model.backend_type,
            capabilities=model.capabilities,
            description=model.description,
        )
        for model in models
    ]

    return ModelCapabilitiesResponse(
        models=model_infos,
        available_capabilities=model_registry.get_available_capabilities(),
        default_model_for_unknown_task=model_registry.get_default_model_for_unknown_task(),
    )


@router.post(
    "/v1/classify-task",
    response_model=ClassifyTaskResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Classifier disabled"},
        500: {"model": ErrorResponse, "description": "Classification error"},
    },
)
async def classify_task(
    request: ClassifyTaskRequest,
    task_classifier: TaskClassifier | None = Depends(get_task_classifier),
) -> ClassifyTaskResponse:
    """Classify a task and recommend an appropriate model.

    Uses an LLM to analyze the task description and classify it into
    a capability category, then recommends the most suitable model.

    Args:
        request: ClassifyTaskRequest with task description.
        task_classifier: Task classifier instance.

    Returns:
        ClassifyTaskResponse with recommended model and classification details.

    Raises:
        HTTPException: 503 if classifier is disabled, 500 on classification error.
    """
    if task_classifier is None:
        logger.warning("classify_task_request_no_classifier")
        raise HTTPException(
            status_code=503,
            detail="Task classifier is not configured",
        )

    try:
        result = await task_classifier.classify(request.task_description)

        return ClassifyTaskResponse(
            recommended_model=result.recommended_model,
            task_type=result.task_type,
            confidence=result.confidence,
            reasoning=result.reasoning,
            alternatives=[
                ModelAlternative(model=alt.model, score=alt.score)
                for alt in result.alternatives
            ],
        )

    except TaskClassifierDisabledError:
        logger.warning("classify_task_classifier_disabled")
        raise HTTPException(
            status_code=503,
            detail="Task classifier is disabled",
        )
    except TaskClassifierError as e:
        logger.error("classify_task_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("classify_task_unexpected_error")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during task classification: {e}",
        ) from e

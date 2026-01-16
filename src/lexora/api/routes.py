"""API routes for Lexora."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from lexora import __version__
from lexora.api.models import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    HealthResponse,
    StatsResponse,
)
from lexora.backends.base import BackendError
from lexora.backends.vllm import VLLMBackend
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector
from lexora.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def get_backend(request: Request) -> VLLMBackend:
    """Get vLLM backend from app state."""
    return request.app.state.backend


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


def check_rate_limit(
    user_id: str | None,
    rate_limiter: RateLimiter,
    enabled: bool,
) -> None:
    """Check rate limit for user.

    Args:
        user_id: User identifier (None uses "anonymous").
        rate_limiter: Rate limiter instance.
        enabled: Whether rate limiting is enabled.

    Raises:
        HTTPException: If rate limited (429).
    """
    if not enabled:
        return

    effective_user_id = user_id or "anonymous"
    if not rate_limiter.consume(effective_user_id):
        wait_time = rate_limiter.time_until_allowed(effective_user_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {wait_time:.1f} seconds.",
            headers={"Retry-After": str(int(wait_time) + 1)},
        )


@router.post(
    "/v1/chat/completions",
    responses={429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def chat_completions(
    request: ChatCompletionRequest,
    backend: VLLMBackend = Depends(get_backend),
    stats_collector: StatsCollector = Depends(get_stats_collector),
    retry_handler: RetryHandler = Depends(get_retry_handler),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    rate_limit_enabled: bool = Depends(is_rate_limit_enabled),
) -> dict[str, Any]:
    """Proxy chat completion request to vLLM.

    Args:
        request: OpenAI-compatible chat completion request.
        backend: vLLM backend.
        stats_collector: Statistics collector.
        retry_handler: Retry handler.
        rate_limiter: Rate limiter.
        rate_limit_enabled: Whether rate limiting is enabled.

    Returns:
        OpenAI-compatible chat completion response.
    """
    # Check rate limit
    check_rate_limit(request.user, rate_limiter, rate_limit_enabled)

    stats = stats_collector.start_request(
        endpoint="/v1/chat/completions",
        model=request.model,
        user_id=request.user,
    )

    try:
        # Convert to dict, excluding None values
        request_dict = request.model_dump(exclude_none=True)

        async def do_request() -> dict[str, Any]:
            return await backend.chat_completions(request_dict)

        response, retries = await retry_handler.execute(do_request)

        # Extract token counts from response if available
        usage = response.get("usage", {})
        tokens_input = usage.get("prompt_tokens", 0)
        tokens_output = usage.get("completion_tokens", 0)

        stats_collector.complete_request(
            stats,
            success=True,
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
        stats_collector.complete_request(stats, success=False, error=str(e))
        logger.error("chat_completion_error", model=request.model, error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        stats_collector.complete_request(stats, success=False, error=str(e))
        logger.exception("chat_completion_unexpected_error", model=request.model)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/v1/completions",
    responses={429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def completions(
    request: CompletionRequest,
    backend: VLLMBackend = Depends(get_backend),
    stats_collector: StatsCollector = Depends(get_stats_collector),
    retry_handler: RetryHandler = Depends(get_retry_handler),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    rate_limit_enabled: bool = Depends(is_rate_limit_enabled),
) -> dict[str, Any]:
    """Proxy completion request to vLLM.

    Args:
        request: OpenAI-compatible completion request.
        backend: vLLM backend.
        stats_collector: Statistics collector.
        retry_handler: Retry handler.
        rate_limiter: Rate limiter.
        rate_limit_enabled: Whether rate limiting is enabled.

    Returns:
        OpenAI-compatible completion response.
    """
    # Check rate limit
    check_rate_limit(request.user, rate_limiter, rate_limit_enabled)

    stats = stats_collector.start_request(
        endpoint="/v1/completions",
        model=request.model,
        user_id=request.user,
    )

    try:
        request_dict = request.model_dump(exclude_none=True)

        async def do_request() -> dict[str, Any]:
            return await backend.completions(request_dict)

        response, retries = await retry_handler.execute(do_request)

        usage = response.get("usage", {})
        tokens_input = usage.get("prompt_tokens", 0)
        tokens_output = usage.get("completion_tokens", 0)

        stats_collector.complete_request(
            stats,
            success=True,
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
        stats_collector.complete_request(stats, success=False, error=str(e))
        logger.error("completion_error", model=request.model, error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        stats_collector.complete_request(stats, success=False, error=str(e))
        logger.exception("completion_unexpected_error", model=request.model)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v1/models")
async def list_models(
    backend: VLLMBackend = Depends(get_backend),
) -> dict[str, Any]:
    """List available models from vLLM.

    Args:
        backend: vLLM backend.

    Returns:
        OpenAI-compatible models list response.
    """
    try:
        return await backend.list_models()
    except BackendError as e:
        logger.error("list_models_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/health", response_model=HealthResponse)
async def health(
    backend: VLLMBackend = Depends(get_backend),
) -> HealthResponse:
    """Check health status.

    Args:
        backend: vLLM backend.

    Returns:
        Health status response.
    """
    vllm_healthy = await backend.health_check()

    return HealthResponse(
        status="healthy" if vllm_healthy else "unhealthy",
        vllm_status="healthy" if vllm_healthy else "unhealthy",
        version=__version__,
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

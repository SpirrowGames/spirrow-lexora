"""API routes for Lexora."""

import math
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from lexora import __version__
from lexora.api.anthropic_compat import (
    anthropic_error_body,
    anthropic_stream_from_openai,
    anthropic_to_openai_request,
    extract_user_id,
    openai_to_anthropic_response,
)
from lexora.api.models import (
    ChatCompletionRequest,
    ChatRequest,
    ChatResponse,
    ClassifyTaskRequest,
    ClassifyTaskResponse,
    CompletionRequest,
    EmbeddingsRequest,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MessagesRequest,
    ModelAlternative,
    ModelCapabilitiesResponse,
    ModelCapabilityInfo,
    StatsResponse,
)
from lexora.backends.base import BackendError, BackendUpstreamError
from lexora.backends.gemini import GeminiGovernanceError
from lexora.backends.vllm import VLLMBackend
from lexora.services.metrics import MetricsCollector
from lexora.services.model_registry import ModelRegistry
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.router import BackendRouter
from lexora.services.cost_tracker import CostTracker
from lexora.services.stats import StatsCollector
from lexora.services.task_classifier import (
    TaskClassifier,
    TaskClassifierDisabledError,
    TaskClassifierError,
)
from lexora.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _passthrough_headers(e: BackendUpstreamError) -> dict[str, str] | None:
    """Response headers for a forwarded upstream answer.

    A 429 forwarded without the upstream's ``Retry-After`` is no more
    useful to the caller than the 502 it replaces -- in both cases they
    cannot tell when it is safe to try again. Emitted only when the
    upstream actually sent the header (``retry_after is None`` otherwise),
    and rounded *up* so the value never advertises an earlier retry than
    the upstream allowed.
    """
    if e.retry_after is None:
        return None
    return {"Retry-After": str(max(0, math.ceil(e.retry_after)))}


def _fail_preflight(
    stats_collector: StatsCollector,
    stats: Any,
    metrics_collector: MetricsCollector | None,
    endpoint: str,
    model: str,
    start_time: float,
    error: BaseException,
) -> None:
    """Close a request that failed in the streaming pre-flight.

    **Every exit from a pre-flight block except "the first chunk was
    grabbed" must pass through here** — not merely every early *return*.
    An exception that leaves the handler is such an exit, and it is the
    one the earlier wording missed: the block used to enumerate the
    exception classes it closed the ledger for, and an enumeration says
    nothing about the classes not in it. The last `except` clause of each
    pre-flight is therefore `BaseException` (`asyncio.CancelledError` is
    not an `Exception`), which closes the ledger and re-raises, so the
    invariant is carried by the structure of the `try` rather than by a
    list of types that has to be kept complete.

    Two reasons it must be closed, and the second is the dangerous one:

    1. Without it the request never reaches the stats collector at all, so
       a caller can hammer a tier with rejected streaming requests and the
       ledger shows zero (the bug this closes). The non-streaming branch of
       the same endpoint has always counted its failures; this makes the
       two branches agree rather than inventing a new policy.
    2. ``record_request_start`` has already ``inc()``'d the ACTIVE_REQUESTS
       gauge and only ``record_request_end`` ``dec()``s it. An exit that
       registers the request but skips this call leaks in-flight count
       permanently -- strictly worse than not counting at all.

    ``streaming=True``: the caller asked for a stream, so the attempt
    belongs with the other streaming outcomes even though the response
    that goes back is JSON.
    """
    stats_collector.complete_request(stats, success=False, error=str(error))
    if metrics_collector:
        metrics_collector.record_request_end(
            endpoint=endpoint,
            model=model,
            status="error",
            duration=time.time() - start_time,
            streaming=True,
        )


def _anthropic_passthrough_body(e: BackendUpstreamError) -> dict[str, Any]:
    """Shape a `BackendUpstreamError` body for /v1/messages passthrough.

    D-4′: an anthropic-type backend's error body is already the
    ``{"type": "error", "error": {...}}`` shape the anthropic SDK parses
    natively, so it can be forwarded verbatim. Anything else (a future
    non-anthropic passthrough backend on /v1/messages, or a plaintext
    error page) is wrapped so the endpoint's response envelope stays
    consistent — the upstream body is preserved inside ``error.upstream``
    rather than dropped, so a shape mismatch never becomes data loss.
    """
    body = e.body
    if isinstance(body, dict) and body.get("type") == "error" and "error" in body:
        return body
    envelope: dict[str, Any] = {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": str(e),
        },
    }
    if body is not None:
        envelope["error"]["upstream"] = body
    return envelope


def _openai_passthrough_body(e: BackendUpstreamError) -> dict[str, Any]:
    """Shape a `BackendUpstreamError` body for OpenAI-family passthrough.

    The OpenAI-family counterpart of `_anthropic_passthrough_body`, and it
    exists for the same reason: **the endpoint's error envelope must stay a
    JSON object.** Handing a non-dict body straight to `JSONResponse`
    serialises it *as* JSON — a plaintext upstream page comes back as the
    JSON string `"<html>...</html>"` under `content-type: application/json`,
    which is a different response shape than every other error this endpoint
    can return. `/v1/messages` has been wrapping that case since D-4′; the
    OpenAI-family endpoints were the half of that decision that got missed.

    A non-JSON body is reachable, not hypothetical: `anthropic.py` sets
    ``body = response.text or None`` when ``response.json()`` fails, and
    ``body = error_text or None`` on the streaming path, and `base.py`
    documents the field as "raw text (str) when the body was not JSON".

    The predicate is **"is it a dict"**, not "is it a str": a JSON body may
    decode to a list or a number, and those serialise into a non-object
    envelope exactly like a string does.

    - dict -> forwarded verbatim, byte-for-byte as today. A passthrough
      backend's own error object already carries the ``error.message`` an
      OpenAI-family client reads, so wrapping it would be the data loss.
    - anything else (str / list / number) -> wrapped, with the original
      body preserved under ``error.upstream`` rather than dropped (D-4′:
      a shape mismatch must never become data loss).
    - ``None`` -> the same envelope without ``error.upstream``. This is the
      one deliberate shape change: the previous ``{"error": {"message":
      ...}}`` gains a ``type``, so all four call sites emit one envelope
      instead of two near-identical ones. No test pinned the old shape.
    """
    body = e.body
    if isinstance(body, dict):
        return body
    envelope: dict[str, Any] = {
        "error": {
            "type": "upstream_error",
            "message": str(e),
        },
    }
    if body is not None:
        envelope["error"]["upstream"] = body
    return envelope


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


def get_cost_tracker(request: Request) -> CostTracker | None:
    """Get cost tracker from app state."""
    return getattr(request.app.state, "cost_tracker", None)


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
    cost_tracker: CostTracker | None = Depends(get_cost_tracker),
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
    resolved_model = backend_router.resolve_model(request.model)

    # Convert to dict, excluding None values
    request_dict = request.model_dump(exclude_none=True)
    request_dict["model"] = resolved_model

    # Handle streaming request
    if request.stream:
        # Register the request BEFORE the pre-flight. A pre-flight that
        # rejects still consumed an upstream call, so it must appear in the
        # ledger exactly like the non-streaming branch below; every early
        # return inside the block closes it via `_fail_preflight`.
        stats = stats_collector.start_request(
            endpoint=endpoint,
            model=request.model,
            user_id=request.user,
        )
        start_time = time.time()

        # Record metrics start
        if metrics_collector:
            metrics_collector.record_request_start(endpoint)

        # Pre-flight the first chunk for passthrough backends so an upstream
        # refusal / auth failure surfaces as a proper HTTP status code
        # (matching /v1/messages, routes.py in the anthropic-compat path)
        # instead of a 200 SSE that closes with an error frame. Once bytes
        # have flowed, only SSE-body passthrough is possible — this is a
        # protocol limit, and the frontier docs note it.
        first_chunk: bytes | None = None
        byte_iter = None
        passthrough = getattr(backend, "error_passthrough", False)
        if passthrough:
            byte_iter = backend.chat_completions_stream(request_dict).__aiter__()
            try:
                first_chunk = await byte_iter.__anext__()
            except StopAsyncIteration:
                first_chunk = None
            except BackendUpstreamError as e:
                logger.warning(
                    "chat_completion_stream_upstream_passthrough",
                    model=request.model,
                    upstream_status=e.status_code,
                )
                _fail_preflight(
                    stats_collector,
                    stats,
                    metrics_collector,
                    endpoint,
                    request.model,
                    start_time,
                    e,
                )
                content = _openai_passthrough_body(e)
                return JSONResponse(
                    status_code=e.status_code,
                    content=content,
                    headers=_passthrough_headers(e),
                )
            except BackendError as e:
                logger.error(
                    "chat_completion_stream_preflight_error",
                    model=request.model,
                    error=str(e),
                )
                _fail_preflight(
                    stats_collector,
                    stats,
                    metrics_collector,
                    endpoint,
                    request.model,
                    start_time,
                    e,
                )
                raise HTTPException(status_code=502, detail=str(e)) from e
            # ★ Last clause on purpose. The three above enumerate exception
            # *classes*; this one carries the invariant itself, so a class
            # nobody listed (a bare `RuntimeError` out of `_map_model` /
            # `_to_anthropic_request` / `_to_gemini_request`, an httpx error
            # it does not wrap, or `asyncio.CancelledError` — which is not an
            # `Exception`) still closes the ledger instead of leaking
            # ACTIVE_REQUESTS. `StopAsyncIteration` is an `Exception`, so the
            # clause above it still wins. `raise` re-raises the original:
            # a cancellation is not turned into an HTTP answer, and an
            # `HTTPException` raised inside a sibling `except` clause does not
            # re-enter this `try`, so the ledger is never closed twice.
            except BaseException as e:
                logger.error(
                    "chat_completion_stream_preflight_unhandled",
                    model=request.model,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                _fail_preflight(
                    stats_collector,
                    stats,
                    metrics_collector,
                    endpoint,
                    request.model,
                    start_time,
                    e,
                )
                raise

        async def stream_generator() -> AsyncIterator[bytes]:
            try:
                if passthrough and byte_iter is not None:
                    if first_chunk is not None:
                        yield first_chunk
                    async for chunk in byte_iter:
                        yield chunk
                else:
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
            # ★ Last clause on purpose, and the mirror of the pre-flight's.
            # `asyncio.CancelledError` is a `BaseException`, so a client that
            # hangs up *after* bytes have flowed lands here and nowhere else:
            # the clause above enumerates `Exception` and never sees it, and
            # the ledger would keep the request in-flight forever
            # (`_fail_preflight` states that invariant; this is its other
            # half). `logger.error`, not `logger.exception` -- a disconnect is
            # not a crash and a traceback per hang-up is noise -- and
            # `error_type` because a bare `CancelledError` stringifies to "".
            # `success=False` and the level are deliberately left as they are,
            # to agree with `_fail_preflight` on the same exception class;
            # whether a disconnect is a *failure* is not settled here.
            except BaseException as e:
                duration = time.time() - start_time
                stats_collector.complete_request(stats, success=False, error=str(e))

                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="error",
                        duration=duration,
                        streaming=True,
                    )

                logger.error(
                    "chat_completion_stream_unhandled",
                    model=request.model,
                    error=str(e),
                    error_type=type(e).__name__,
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

        # `error_passthrough` backends (e.g. frontier) must not have their
        # rate-limit / decline retried into extra billed calls: passing an
        # empty tuple to `retryable_exceptions` disables retry without any
        # new machinery — the existing handler simply catches nothing.
        retryable = (
            ()
            if getattr(backend, "error_passthrough", False)
            else None
        )
        response, retries = await retry_handler.execute(
            do_request, retryable_exceptions=retryable
        )

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

        # Record cost. The tier alias (if any) goes to its own column; the
        # `model` column carries the resolved concrete model ID so pricing
        # tracks the actual upstream, not the caller's alias.
        if cost_tracker and (tokens_input > 0 or tokens_output > 0):
            cost_tracker.record(
                model=resolved_model,
                endpoint=endpoint,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                backend=backend_router.get_backend_name_for_model(request.model),
                user_id=request.user,
                duration=duration,
                tier=request.model if backend_router.is_tier(request.model) else None,
            )

        logger.info(
            "chat_completion_success",
            model=request.model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            retries=retries,
        )

        return response

    except BackendUpstreamError as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=duration,
            )
        # Passthrough: forward the upstream status and body verbatim.
        # Non-passthrough backends (existing behaviour) still get 502 with a
        # stringified detail so the pre-frontier response shape is unchanged.
        if getattr(backend, "error_passthrough", False):
            logger.warning(
                "chat_completion_upstream_passthrough",
                model=request.model,
                upstream_status=e.status_code,
            )
            content = _openai_passthrough_body(e)
            return JSONResponse(
                status_code=e.status_code,
                content=content,
                headers=_passthrough_headers(e),
            )
        logger.error("chat_completion_error", model=request.model, error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e
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
    resolved_model = backend_router.resolve_model(request.model)

    # Convert to dict, excluding None values
    request_dict = request.model_dump(exclude_none=True)
    request_dict["model"] = resolved_model

    # Handle streaming request
    if request.stream:
        # Registered before the pre-flight for the same reason as
        # /v1/chat/completions: a rejected pre-flight is still a request.
        stats = stats_collector.start_request(
            endpoint=endpoint,
            model=request.model,
            user_id=request.user,
        )
        start_time = time.time()

        # Record metrics start
        if metrics_collector:
            metrics_collector.record_request_start(endpoint)

        # D-4′: mirror the /v1/chat/completions streaming pre-flight for
        # passthrough backends so a refusal / auth failure yields the real
        # HTTP status instead of a 200 SSE that closes with an error frame.
        first_chunk: bytes | None = None
        byte_iter = None
        passthrough = getattr(backend, "error_passthrough", False)
        if passthrough:
            byte_iter = backend.completions_stream(request_dict).__aiter__()
            try:
                first_chunk = await byte_iter.__anext__()
            except StopAsyncIteration:
                first_chunk = None
            except BackendUpstreamError as e:
                logger.warning(
                    "completion_stream_upstream_passthrough",
                    model=request.model,
                    upstream_status=e.status_code,
                )
                _fail_preflight(
                    stats_collector,
                    stats,
                    metrics_collector,
                    endpoint,
                    request.model,
                    start_time,
                    e,
                )
                content = _openai_passthrough_body(e)
                return JSONResponse(
                    status_code=e.status_code,
                    content=content,
                    headers=_passthrough_headers(e),
                )
            except BackendError as e:
                logger.error(
                    "completion_stream_preflight_error",
                    model=request.model,
                    error=str(e),
                )
                _fail_preflight(
                    stats_collector,
                    stats,
                    metrics_collector,
                    endpoint,
                    request.model,
                    start_time,
                    e,
                )
                raise HTTPException(status_code=502, detail=str(e)) from e
            # ★ Last clause on purpose. The three above enumerate exception
            # *classes*; this one carries the invariant itself, so a class
            # nobody listed (a bare `RuntimeError` out of `_map_model` /
            # `_to_anthropic_request` / `_to_gemini_request`, an httpx error
            # it does not wrap, or `asyncio.CancelledError` — which is not an
            # `Exception`) still closes the ledger instead of leaking
            # ACTIVE_REQUESTS. `StopAsyncIteration` is an `Exception`, so the
            # clause above it still wins. `raise` re-raises the original:
            # a cancellation is not turned into an HTTP answer, and an
            # `HTTPException` raised inside a sibling `except` clause does not
            # re-enter this `try`, so the ledger is never closed twice.
            except BaseException as e:
                logger.error(
                    "completion_stream_preflight_unhandled",
                    model=request.model,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                _fail_preflight(
                    stats_collector,
                    stats,
                    metrics_collector,
                    endpoint,
                    request.model,
                    start_time,
                    e,
                )
                raise

        async def stream_generator() -> AsyncIterator[bytes]:
            try:
                if passthrough and byte_iter is not None:
                    if first_chunk is not None:
                        yield first_chunk
                    async for chunk in byte_iter:
                        yield chunk
                else:
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
            # ★ Last clause on purpose: the mid-stream half of the
            # ACTIVE_REQUESTS invariant. See the same clause in
            # `chat_completions` above for why.
            except BaseException as e:
                duration = time.time() - start_time
                stats_collector.complete_request(stats, success=False, error=str(e))

                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="error",
                        duration=duration,
                        streaming=True,
                    )

                logger.error(
                    "completion_stream_unhandled",
                    model=request.model,
                    error=str(e),
                    error_type=type(e).__name__,
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

        # D-4′: `error_passthrough` is a property of the backend / tier, not
        # of one endpoint. `/v1/completions` respects it identically to
        # `/v1/chat/completions` so a passthrough tier reached via either
        # OpenAI-family surface behaves the same way — no retry on
        # billed classes, and upstream 4xx/5xx forwarded verbatim.
        retryable = (
            ()
            if getattr(backend, "error_passthrough", False)
            else None
        )
        response, retries = await retry_handler.execute(
            do_request, retryable_exceptions=retryable
        )

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

    except BackendUpstreamError as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=duration,
            )
        if getattr(backend, "error_passthrough", False):
            logger.warning(
                "completion_upstream_passthrough",
                model=request.model,
                upstream_status=e.status_code,
            )
            content = _openai_passthrough_body(e)
            return JSONResponse(
                status_code=e.status_code,
                content=content,
                headers=_passthrough_headers(e),
            )
        logger.error("completion_error", model=request.model, error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e
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
    resolved_model = backend_router.resolve_model(request.model)

    stats = stats_collector.start_request(
        endpoint=endpoint,
        model=resolved_model,
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

    # `None` == skipped (`health_check: false`). It is not a health verdict,
    # so it is neither recorded as a metric nor counted toward `status` --
    # folding it into either would turn "we did not ask" into an answer.
    probed = {
        name: healthy
        for name, healthy in backend_health.items()
        if healthy is not None
    }

    if metrics_collector:
        for backend_name, is_healthy in probed.items():
            metrics_collector.set_backend_health(backend_name, is_healthy)

    backends_status = {
        name: "skipped"
        if healthy is None
        else ("healthy" if healthy else "unhealthy")
        for name, healthy in backend_health.items()
    }

    # Determine overall status over the probed subset. `all([])` is True, so
    # the empty case is handled first: with nothing probed there is no
    # evidence for "healthy", and claiming it would be the same mistake as
    # counting a skip as a pass.
    if not probed:
        overall_status = "unknown"
    elif all(probed.values()):
        overall_status = "healthy"
    elif any(probed.values()):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    # Legacy vllm_status for backward compatibility. A skipped or absent
    # default backend reports "unknown" rather than borrowing "healthy".
    default_backend_healthy = backend_health.get("default", True)
    vllm_status = (
        "unknown"
        if default_backend_healthy is None
        else ("healthy" if default_backend_healthy else "unhealthy")
    )

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


def get_default_model(
    model_registry: ModelRegistry | None,
) -> str | None:
    """Get default model from registry.

    Args:
        model_registry: Model registry instance.

    Returns:
        Default model ID or None if not configured.
    """
    if model_registry is None:
        return None
    return model_registry.get_default_model_for_unknown_task()


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "No model available"},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def generate(
    request: GenerateRequest,
    backend_router: BackendRouter = Depends(get_backend_router),
    stats_collector: StatsCollector = Depends(get_stats_collector),
    retry_handler: RetryHandler = Depends(get_retry_handler),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    rate_limit_enabled: bool = Depends(is_rate_limit_enabled),
    metrics_collector: MetricsCollector | None = Depends(get_metrics_collector),
    model_registry: ModelRegistry | None = Depends(get_model_registry),
) -> GenerateResponse:
    """Simple text generation endpoint.

    This is a convenience endpoint that wraps /v1/completions with simpler
    request/response formats. It's designed for magickit integration.

    Args:
        request: Simple generation request with prompt.
        backend_router: Backend router for model routing.
        stats_collector: Statistics collector.
        retry_handler: Retry handler.
        rate_limiter: Rate limiter.
        rate_limit_enabled: Whether rate limiting is enabled.
        metrics_collector: Prometheus metrics collector.
        model_registry: Model registry for default model lookup.

    Returns:
        GenerateResponse with generated text.
    """
    endpoint = "/generate"

    # Check rate limit
    check_rate_limit(request.user, rate_limiter, rate_limit_enabled, metrics_collector)

    # Determine model to use
    model = request.model
    if not model:
        model = get_default_model(model_registry)
    if not model:
        raise HTTPException(
            status_code=400,
            detail="No model specified and no default model configured",
        )

    # Get backend for the model
    backend = backend_router.get_backend_for_model(model)

    # Build completions request
    completion_request = {
        "model": model,
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }

    # Pass through extra fields
    extra_fields = request.model_dump(
        exclude={"prompt", "max_tokens", "temperature", "model", "user"}
    )
    completion_request.update(extra_fields)

    stats = stats_collector.start_request(
        endpoint=endpoint,
        model=model,
        user_id=request.user,
    )
    start_time = time.time()

    if metrics_collector:
        metrics_collector.record_request_start(endpoint)

    try:

        async def do_request() -> dict[str, Any]:
            return await backend.completions(completion_request)

        response, retries = await retry_handler.execute(do_request)

        # Extract generated text from response
        choices = response.get("choices", [])
        if not choices:
            raise HTTPException(
                status_code=500,
                detail="No choices in completion response",
            )
        text = choices[0].get("text", "")

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

        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=model,
                status="success",
                duration=duration,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                retries=retries,
            )

        logger.info(
            "generate_success",
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            retries=retries,
        )

        return GenerateResponse(text=text)

    except BackendError as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))

        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=model,
                status="error",
                duration=duration,
            )

        logger.error("generate_error", model=model, error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))

        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=model,
                status="error",
                duration=duration,
            )

        logger.exception("generate_unexpected_error", model=model)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "No model available"},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def chat(
    request: ChatRequest,
    backend_router: BackendRouter = Depends(get_backend_router),
    stats_collector: StatsCollector = Depends(get_stats_collector),
    retry_handler: RetryHandler = Depends(get_retry_handler),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    rate_limit_enabled: bool = Depends(is_rate_limit_enabled),
    metrics_collector: MetricsCollector | None = Depends(get_metrics_collector),
    model_registry: ModelRegistry | None = Depends(get_model_registry),
) -> ChatResponse:
    """Simple chat endpoint.

    This is a convenience endpoint that wraps /v1/chat/completions with simpler
    request/response formats. It's designed for magickit integration.

    Args:
        request: Simple chat request with messages.
        backend_router: Backend router for model routing.
        stats_collector: Statistics collector.
        retry_handler: Retry handler.
        rate_limiter: Rate limiter.
        rate_limit_enabled: Whether rate limiting is enabled.
        metrics_collector: Prometheus metrics collector.
        model_registry: Model registry for default model lookup.

    Returns:
        ChatResponse with assistant response.
    """
    endpoint = "/chat"

    # Check rate limit
    check_rate_limit(request.user, rate_limiter, rate_limit_enabled, metrics_collector)

    # Determine model to use
    model = request.model
    if not model:
        model = get_default_model(model_registry)
    if not model:
        raise HTTPException(
            status_code=400,
            detail="No model specified and no default model configured",
        )

    # Get backend for the model
    backend = backend_router.get_backend_for_model(model)

    # Build chat completions request
    chat_request = {
        "model": model,
        "messages": [msg.model_dump() for msg in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }

    # Pass through extra fields
    extra_fields = request.model_dump(
        exclude={"messages", "max_tokens", "temperature", "model", "user"}
    )
    chat_request.update(extra_fields)

    stats = stats_collector.start_request(
        endpoint=endpoint,
        model=model,
        user_id=request.user,
    )
    start_time = time.time()

    if metrics_collector:
        metrics_collector.record_request_start(endpoint)

    try:

        async def do_request() -> dict[str, Any]:
            return await backend.chat_completions(chat_request)

        response, retries = await retry_handler.execute(do_request)

        # Extract response text from choices
        choices = response.get("choices", [])
        if not choices:
            raise HTTPException(
                status_code=500,
                detail="No choices in chat completion response",
            )
        message = choices[0].get("message", {})
        response_text = message.get("content", "")

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

        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=model,
                status="success",
                duration=duration,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                retries=retries,
            )

        logger.info(
            "chat_success",
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            retries=retries,
        )

        return ChatResponse(response=response_text)

    except BackendError as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))

        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=model,
                status="error",
                duration=duration,
            )

        logger.error("chat_error", model=model, error=str(e))
        raise HTTPException(status_code=502, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        stats_collector.complete_request(stats, success=False, error=str(e))

        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=model,
                status="error",
                duration=duration,
            )

        logger.exception("chat_unexpected_error", model=model)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/v1/messages",
    response_model=None,
    responses={429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def messages(
    request: MessagesRequest,
    backend_router: BackendRouter = Depends(get_backend_router),
    stats_collector: StatsCollector = Depends(get_stats_collector),
    retry_handler: RetryHandler = Depends(get_retry_handler),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    rate_limit_enabled: bool = Depends(is_rate_limit_enabled),
    metrics_collector: MetricsCollector | None = Depends(get_metrics_collector),
    cost_tracker: CostTracker | None = Depends(get_cost_tracker),
) -> dict[str, Any] | JSONResponse | StreamingResponse:
    """Anthropic Messages API-compatible endpoint.

    Accepts an Anthropic Messages request, translates it to Lexora's internal
    OpenAI-compatible format, routes it through the normal backend router (so
    ``model: "naysayer"`` reaches the Gemini naysayer tier and its
    data-governance gate), and translates the response back to the Anthropic
    Messages shape. Errors are returned in the Anthropic ``{"type": "error",
    ...}`` shape so the ``anthropic`` SDK parses them natively.

    Args:
        request: Anthropic Messages API request.
        backend_router: Backend router for model/tier routing.
        stats_collector: Statistics collector.
        retry_handler: Retry handler.
        rate_limiter: Rate limiter.
        rate_limit_enabled: Whether rate limiting is enabled.
        metrics_collector: Prometheus metrics collector.
        cost_tracker: Cost tracker.

    Returns:
        Anthropic Messages response (dict), an Anthropic-shaped error
        (JSONResponse), or a streaming SSE response.
    """
    endpoint = "/v1/messages"

    req_dict = request.model_dump(exclude_none=True)
    user_id = extract_user_id(req_dict)

    # Rate limiting (Anthropic-shaped 429 so the SDK parses it).
    if rate_limit_enabled:
        effective_user_id = user_id or "anonymous"
        if not rate_limiter.consume(effective_user_id):
            wait_time = rate_limiter.time_until_allowed(effective_user_id)
            if metrics_collector:
                metrics_collector.record_rate_limit_rejection(effective_user_id)
            return JSONResponse(
                status_code=429,
                content=anthropic_error_body(
                    "rate_limit_error",
                    f"Rate limit exceeded. Retry after {wait_time:.1f} seconds.",
                ),
                headers={"Retry-After": str(int(wait_time) + 1)},
            )

    backend = backend_router.get_backend_for_model(request.model)
    resolved_model = backend_router.resolve_model(request.model)

    openai_request = anthropic_to_openai_request(req_dict)
    openai_request["model"] = resolved_model

    passthrough = getattr(backend, "error_passthrough", False)

    # Streaming path: pre-flight the backend stream so governance/connection
    # errors surface as proper HTTP status codes before any SSE headers are sent.
    if request.stream:
        openai_request["stream"] = True
        # Registered before the pre-flight (same rule as the OpenAI-family
        # endpoints). This pre-flight runs for *every* backend, not just
        # passthrough ones, so before this change a governance refusal or a
        # plain BackendError on the streaming path was invisible to stats
        # and metrics too — not only the passthrough case the gate named.
        stats = stats_collector.start_request(
            endpoint=endpoint, model=request.model, user_id=user_id
        )
        if metrics_collector:
            metrics_collector.record_request_start(endpoint)
        start_time = time.time()

        byte_iter = backend.chat_completions_stream(openai_request).__aiter__()
        try:
            first_chunk = await byte_iter.__anext__()
        except StopAsyncIteration:
            first_chunk = None
        except GeminiGovernanceError as e:
            _fail_preflight(
                stats_collector,
                stats,
                metrics_collector,
                endpoint,
                request.model,
                start_time,
                e,
            )
            return JSONResponse(
                status_code=400,
                content=anthropic_error_body("invalid_request_error", str(e)),
            )
        except BackendUpstreamError as e:
            # D-4′: passthrough backends forward the upstream status. For an
            # anthropic-shaped body (Fable/Opus decline), reuse it directly
            # — it is already `{"type": "error", "error": {...}}` and the
            # anthropic SDK will parse it natively. For any other body
            # shape carry it inside the endpoint's error envelope so the
            # shape mismatch never becomes data loss.
            _fail_preflight(
                stats_collector,
                stats,
                metrics_collector,
                endpoint,
                request.model,
                start_time,
                e,
            )
            if passthrough:
                logger.warning(
                    "messages_stream_upstream_passthrough",
                    model=request.model,
                    upstream_status=e.status_code,
                )
                content = _anthropic_passthrough_body(e)
                return JSONResponse(
                    status_code=e.status_code,
                    content=content,
                    headers=_passthrough_headers(e),
                )
            logger.error("messages_stream_error", model=request.model, error=str(e))
            return JSONResponse(
                status_code=502,
                content=anthropic_error_body("api_error", str(e)),
            )
        except BackendError as e:
            _fail_preflight(
                stats_collector,
                stats,
                metrics_collector,
                endpoint,
                request.model,
                start_time,
                e,
            )
            logger.error("messages_stream_error", model=request.model, error=str(e))
            return JSONResponse(
                status_code=502,
                content=anthropic_error_body("api_error", str(e)),
            )

        # ★ Last clause on purpose. The three above enumerate exception
        # *classes*; this one carries the invariant itself, so a class
        # nobody listed (a bare `RuntimeError` out of `_map_model` /
        # `_to_anthropic_request` / `_to_gemini_request`, an httpx error
        # it does not wrap, or `asyncio.CancelledError` — which is not an
        # `Exception`) still closes the ledger instead of leaking
        # ACTIVE_REQUESTS. `StopAsyncIteration` is an `Exception`, so the
        # clause above it still wins. `raise` re-raises the original:
        # a cancellation is not turned into an HTTP answer, and an
        # `HTTPException` raised inside a sibling `except` clause does not
        # re-enter this `try`, so the ledger is never closed twice.
        except BaseException as e:
            logger.error(
                "messages_stream_preflight_unhandled",
                model=request.model,
                error=str(e),
                error_type=type(e).__name__,
            )
            _fail_preflight(
                stats_collector,
                stats,
                metrics_collector,
                endpoint,
                request.model,
                start_time,
                e,
            )
            raise

        async def replayed() -> AsyncIterator[bytes]:
            if first_chunk is not None:
                yield first_chunk
            async for chunk in byte_iter:
                yield chunk

        async def stream_generator() -> AsyncIterator[bytes]:
            try:
                async for event in anthropic_stream_from_openai(
                    replayed(), request.model
                ):
                    yield event
                stats_collector.complete_request(stats, success=True)
                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="success",
                        duration=time.time() - start_time,
                        streaming=True,
                    )
            except Exception as e:  # noqa: BLE001
                stats_collector.complete_request(stats, success=False, error=str(e))
                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="error",
                        duration=time.time() - start_time,
                        streaming=True,
                    )
                logger.exception("messages_stream_unexpected_error", model=request.model)
                raise
            # ★ Last clause on purpose: the mid-stream half of the
            # ACTIVE_REQUESTS invariant. See the same clause in
            # `chat_completions` above for why.
            except BaseException as e:
                stats_collector.complete_request(stats, success=False, error=str(e))
                if metrics_collector:
                    metrics_collector.record_request_end(
                        endpoint=endpoint,
                        model=request.model,
                        status="error",
                        duration=time.time() - start_time,
                        streaming=True,
                    )
                logger.error(
                    "messages_stream_unhandled",
                    model=request.model,
                    error=str(e),
                    error_type=type(e).__name__,
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

    # Non-streaming path.
    stats = stats_collector.start_request(
        endpoint=endpoint, model=request.model, user_id=user_id
    )
    start_time = time.time()
    if metrics_collector:
        metrics_collector.record_request_start(endpoint)

    try:

        async def do_request() -> dict[str, Any]:
            return await backend.chat_completions(openai_request)

        # D-4′: passthrough backends must not retry billed classes on
        # /v1/messages either — the retry policy is a property of the
        # tier/backend, not of the endpoint.
        retryable = () if passthrough else None
        response, retries = await retry_handler.execute(
            do_request, retryable_exceptions=retryable
        )

        anthropic_response = openai_to_anthropic_response(response, request.model)

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

        if cost_tracker and (tokens_input > 0 or tokens_output > 0):
            cost_tracker.record(
                model=resolved_model,
                endpoint=endpoint,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                backend=backend_router.get_backend_name_for_model(request.model),
                user_id=user_id,
                duration=duration,
                tier=request.model if backend_router.is_tier(request.model) else None,
            )

        logger.info(
            "messages_success",
            model=request.model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            retries=retries,
        )
        return anthropic_response

    except GeminiGovernanceError as e:
        stats_collector.complete_request(stats, success=False, error=str(e))
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=time.time() - start_time,
            )
        logger.warning("messages_governance_refused", model=request.model, error=str(e))
        return JSONResponse(
            status_code=400,
            content=anthropic_error_body("invalid_request_error", str(e)),
        )
    except BackendUpstreamError as e:
        stats_collector.complete_request(stats, success=False, error=str(e))
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=time.time() - start_time,
            )
        if passthrough:
            logger.warning(
                "messages_upstream_passthrough",
                model=request.model,
                upstream_status=e.status_code,
            )
            content = _anthropic_passthrough_body(e)
            return JSONResponse(
                status_code=e.status_code,
                content=content,
                headers=_passthrough_headers(e),
            )
        logger.error("messages_error", model=request.model, error=str(e))
        return JSONResponse(
            status_code=502,
            content=anthropic_error_body("api_error", str(e)),
        )
    except BackendError as e:
        stats_collector.complete_request(stats, success=False, error=str(e))
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=time.time() - start_time,
            )
        logger.error("messages_error", model=request.model, error=str(e))
        return JSONResponse(
            status_code=502,
            content=anthropic_error_body("api_error", str(e)),
        )
    except Exception as e:
        stats_collector.complete_request(stats, success=False, error=str(e))
        if metrics_collector:
            metrics_collector.record_request_end(
                endpoint=endpoint,
                model=request.model,
                status="error",
                duration=time.time() - start_time,
            )
        logger.exception("messages_unexpected_error", model=request.model)
        return JSONResponse(
            status_code=500,
            content=anthropic_error_body("api_error", str(e)),
        )


# --- Cost tracking endpoints ---


@router.get("/stats/costs")
async def get_costs(
    period: str = "today",
    model: str | None = None,
    user_id: str | None = None,
    backend: str | None = None,
    tier: str | None = None,
    cost_tracker: CostTracker | None = Depends(get_cost_tracker),
) -> dict[str, Any]:
    """Get aggregated API costs.

    Args:
        period: "today", "month", "all", or ISO date "YYYY-MM-DD".
        model: Filter by resolved model name (concrete upstream ID).
        user_id: Filter by user.
        backend: Filter by backend name.
        tier: Filter by tier alias (``frontier``, ``naysayer``, ...). Since
            2026-08-31 the ledger stores tier aliases in a dedicated
            column, so ``?model=frontier`` no longer matches; use
            ``?tier=frontier`` instead. Requests sent as concrete model
            IDs are never matched by a tier filter.

    Returns:
        Aggregated cost data with summary, per-model breakdown, per-tier
        breakdown, daily totals, and any models seen without a known price.
    """
    if cost_tracker is None:
        raise HTTPException(status_code=503, detail="Cost tracking not available")
    return cost_tracker.get_costs(
        period=period, model=model, user_id=user_id, backend=backend, tier=tier
    )


@router.get("/stats/costs/recent")
async def get_recent_costs(
    limit: int = 50,
    cost_tracker: CostTracker | None = Depends(get_cost_tracker),
) -> list[dict[str, Any]]:
    """Get recent request cost records.

    Args:
        limit: Maximum number of records to return.

    Returns:
        List of recent request cost records.
    """
    if cost_tracker is None:
        raise HTTPException(status_code=503, detail="Cost tracking not available")
    return cost_tracker.get_recent(limit=limit)

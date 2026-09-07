"""Base backend interface and exceptions."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


class BackendError(Exception):
    """Base exception for backend errors."""

    pass


class ModelNotFoundError(BackendError):
    """Raised when a requested model name cannot be routed.

    Used by ``BackendRouter`` for two distinct configuration situations that
    share one API-facing shape (T-silent-routing R-1a / R-2):

    * ``unknown`` — the requested name is neither a registered tier nor a
      model declared by any backend, and the router refuses to silently
      fall through to ``default_backend``.
    * ``ambiguous`` — the requested name is declared as a model by two or
      more backends, so no single backend can be picked without guessing.

    Both surface to the caller as HTTP 404 with OpenAI's standard
    ``model_not_found`` code. The distinction stays server-side: the router
    logs a different event name for each case (``model_unknown_refused`` vs
    ``model_ambiguous_refused``) and the human-readable ``message`` here
    explains the specific situation. Clients that want to programmatically
    distinguish should not — the remedy is identical (change the ``model``
    string).
    """

    def __init__(
        self,
        message: str,
        model_name: str,
        reason: str,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable explanation. Forwarded verbatim as the
                HTTP response's ``message`` field.
            model_name: The name the caller requested. Forwarded as the
                ``param`` value in the OpenAI error envelope.
            reason: Machine-readable classifier for server-side logging /
                metrics. One of ``"unknown"`` or ``"ambiguous"``. Not
                exposed on the API — the API code stays ``model_not_found``
                regardless.
        """
        super().__init__(message)
        self.model_name = model_name
        self.reason = reason


class BackendConnectionError(BackendError):
    """Raised when connection to backend fails."""

    pass


class BackendTimeoutError(BackendError):
    """Raised when request to backend times out."""

    pass


class BackendUnavailableError(BackendError):
    """Raised when backend is unavailable (e.g., 503 status)."""

    pass


class BackendRateLimitError(BackendError):
    """Raised when backend returns 429 Too Many Requests."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        backend_name: str | None = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            message: Error message.
            retry_after: Suggested retry delay in seconds from Retry-After header.
            backend_name: Name of the backend that returned the error.
        """
        super().__init__(message)
        self.retry_after = retry_after
        self.backend_name = backend_name


class BackendUpstreamError(BackendError):
    """An answer from the upstream that the caller should see verbatim.

    Distinguished from ``BackendConnectionError`` / ``BackendTimeoutError``:
    those mean *we* failed to reach the upstream (Lexora's own failure,
    502 is honest). ``BackendUpstreamError`` means the upstream *did*
    answer — the answer just happens to be a 4xx / 5xx and, for backends
    that opt into ``error_passthrough``, its status code and body are
    what the caller must receive.

    Frontier tier (msg-011 D-4) is the first user: a Fable / Opus safety
    classifier decline arrives as HTTP 400 with a structured body, and
    collapsing it into a 502 with a stringified message drops both the
    status and the machine-readable body. Everything besides the passthrough
    path continues to see this as a plain ``BackendError``, so existing
    tiers keep the 502-on-error behaviour.
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        body: object | None = None,
        retry_after: float | None = None,
        backend_name: str | None = None,
    ) -> None:
        """Initialize upstream error.

        Args:
            message: Human-readable message for logs and non-passthrough paths.
            status_code: HTTP status from the upstream.
            body: Parsed JSON body from the upstream (dict) or raw text
                (str) when the body was not JSON. ``None`` when the
                upstream sent no body.
            retry_after: Retry-After header value in seconds, if any.
            backend_name: Name of the backend that returned the error.
        """
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        self.backend_name = backend_name


@dataclass
class UsageSink:
    """Per-request carrier for the token counts a streaming call observed.

    ``*_stream`` is an ``AsyncIterator[bytes]`` and cannot return a value, so a
    backend that already parses upstream usage writes it here and the handler
    reads it after the relay terminates.

    **Per request, never as state on the backend.** Backend instances are
    created once (``services/router.py``) and held on ``app.state``, so one
    instance serves concurrent requests: an attribute would let two
    overlapping streams overwrite each other's count -- a number landing on
    the wrong bill.

    Left at zero by any backend that does not fill it, and the handler's guard
    (``tokens_input > 0 or tokens_output > 0``, the predicate the six
    non-streaming sites use) then opens no row. That is why the verbatim-relay
    backends, whose bytes do not carry the number, need no special case.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0


class Backend(ABC):
    """Abstract base class for LLM backends."""

    #: When True, request handlers forward upstream 4xx/5xx status codes and
    #: bodies verbatim (via ``BackendUpstreamError``) and skip the retry
    #: handler so a 429/decline is not retried into extra billed calls. Off
    #: by default so every existing backend keeps its 502-on-error shape.
    #: Set by subclasses whose config carries ``error_passthrough: true``.
    error_passthrough: bool = False

    #: When True, this backend parses the upstream's token counts itself and
    #: writes them into the per-request ``UsageSink`` it is handed, so a
    #: stream of its that reaches normal completion is *expected* to leave a
    #: non-empty sink.
    #:
    #: **A claim about the backend, never about the wire format.** It says
    #: only "this implementation fills the sink"; it names no upstream, no
    #: field and no frame, so a handler that reads it learns nothing about
    #: what the upstream sends and the router still holds no upstream format.
    #: That is the whole reason it is a bit here rather than a format hint.
    #:
    #: Off by default, so a backend that relays bytes verbatim -- and
    #: therefore holds no parsed count -- needs no special case, exactly as
    #: with the sink itself. Set True on ``anthropic``, ``claude_code`` and
    #: ``gemini``, the three that write the sink; left False on
    #: ``openai_compatible`` and ``vllm``, whose ``*_stream`` docstrings say
    #: "accepted and never filled" and point at this attribute so the prose
    #: and the value sit together.
    #:
    #: NOT ``error_passthrough``, and the two must never be confused: that
    #: one is an error-shape flag and it is set on ``anthropic``
    #: (``config.ERROR_PASSTHROUGH_TYPES`` is ``{"anthropic"}``), which is
    #: one of the three backends that *do* fill the sink. As a stand-in for
    #: this bit it is not merely different, it is anti-correlated.
    fills_usage_sink: bool = False

    @abstractmethod
    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send chat completion request to the backend.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            OpenAI-compatible chat completion response.

        Raises:
            BackendError: If the request fails.
        """
        pass

    @abstractmethod
    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send completion request to the backend.

        Args:
            request: OpenAI-compatible completion request.

        Returns:
            OpenAI-compatible completion response.

        Raises:
            BackendError: If the request fails.
        """
        pass

    @abstractmethod
    async def list_models(self) -> dict[str, Any]:
        """List available models.

        Returns:
            OpenAI-compatible models list response.

        Raises:
            BackendError: If the request fails.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy.

        Returns:
            True if backend is healthy, False otherwise.
        """
        pass

    @abstractmethod
    async def chat_completions_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None = None
    ) -> AsyncIterator[bytes]:
        """Send streaming chat completion request to the backend.

        Args:
            request: OpenAI-compatible chat completion request.
            usage_sink: Optional per-request carrier the backend fills with
                the upstream token counts if it has them. Defaults to None so
                every existing caller keeps working unchanged. A backend that
                leaves it untouched bills nothing; see ``UsageSink``.

        Yields:
            SSE data chunks.

        Raises:
            BackendError: If the request fails.
        """
        pass
        yield b""  # Make this an async generator

    @abstractmethod
    async def completions_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None = None
    ) -> AsyncIterator[bytes]:
        """Send streaming completion request to the backend.

        Args:
            request: OpenAI-compatible completion request.
            usage_sink: Optional per-request carrier; see
                ``chat_completions_stream``.

        Yields:
            SSE data chunks.

        Raises:
            BackendError: If the request fails.
        """
        pass
        yield b""  # Make this an async generator

    @abstractmethod
    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send embeddings request to the backend.

        Args:
            request: OpenAI-compatible embeddings request.

        Returns:
            OpenAI-compatible embeddings response.

        Raises:
            BackendError: If the request fails.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass

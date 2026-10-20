"""vLLM backend implementation using httpx."""

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from lexora.backends.base import (
    Backend,
    BackendConnectionError,
    BackendError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnavailableError,
    UsageSink,
)
from lexora.utils.logging import get_logger

logger = get_logger(__name__)

#: Any SSE event delimiter. The SSE spec allows CRLF, LF or CR as a line end,
#: so a blank line is one of these three pairs. One alternation, searched
#: leftmost, so ``\r\n\r\n`` is taken whole.
_SSE_EVENT_DELIMITER = re.compile(rb"\r\n\r\n|\n\n|\r\r")
_SSE_LINE_END = re.compile(rb"\r\n|\n|\r")

#: Cap on the usage reader's side buffer (T-streaming-ledger-row msg-877 §3).
#: The usage event measured on vLLM 0.17.1 is about 150 bytes, so only bytes
#: that are not the measured shape can reach this; past it the reader gives up
#: for the rest of the request and the sink stays at zero.
USAGE_PARSE_BUFFER_CAP = 64 * 1024


def _with_usage_requested(request: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``request`` whose ``stream_options.include_usage`` is True.

    **Unconditional, whatever the caller sent** (T-streaming-ledger-row
    msg-879): honouring a caller's ``include_usage: false`` would let it take
    a full generation and leave no ledger row, and the ledger is a
    measurement, not a client preference. Every other ``stream_options`` key
    the caller sent is kept. Neither the caller's dict nor its
    ``stream_options`` dict is mutated: both are copied before the key is set.
    """
    existing = request.get("stream_options")
    options = dict(existing) if isinstance(existing, dict) else {}
    options["include_usage"] = True
    return {**request, "stream_options": options}


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class _StreamUsageReader:
    """Reads the final usage event out of a *side copy* of the relayed bytes.

    The relay never waits on this: ``chat_completions_stream`` yields each
    upstream chunk first and only then passes the same bytes to ``feed``.
    ``feed`` catches every exception raised while parsing, and after one the
    reader stops for the rest of the request. **There is no path from this
    class to the client stream**: the worst it can do is leave the sink at
    zero, which opens no row, exactly as before this reader existed.

    The usage event is a ``data`` payload that is not ``[DONE]``, parses as a
    JSON object, and has ``choices == []`` and a ``usage`` object whose
    ``prompt_tokens`` / ``completion_tokens`` are non-negative ints. That is
    the shape measured on vLLM 0.17.1 (msg-874): once, immediately before
    ``data: [DONE]``, with no ``usage`` key on any earlier event. If more than
    one event matches, the last wins. An incomplete event at end of stream is
    discarded, as the SSE spec says.
    """

    def __init__(self, sink: UsageSink | None) -> None:
        self._sink = sink
        self._buffer = b""
        self._stopped = sink is None

    def feed(self, chunk: bytes) -> None:
        if self._stopped:
            return
        try:
            self._feed(chunk)
        except Exception as exc:  # fail open, by design: see the class doc
            self._stop("vllm_stream_usage_reader_error", error=repr(exc))

    def _stop(self, event: str, **fields: Any) -> None:
        self._stopped = True
        self._buffer = b""
        logger.debug(event, **fields)

    def _feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while (match := _SSE_EVENT_DELIMITER.search(self._buffer)) is not None:
            event = self._buffer[: match.start()]
            self._buffer = self._buffer[match.end() :]
            self._read_event(event)
        if len(self._buffer) > USAGE_PARSE_BUFFER_CAP:
            self._stop(
                "vllm_stream_usage_buffer_cap",
                cap=USAGE_PARSE_BUFFER_CAP,
                buffered=len(self._buffer),
            )

    def _read_event(self, event: bytes) -> None:
        data_lines = []
        for line in _SSE_LINE_END.split(event):
            if line.startswith(b"data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(b" ") else value)
        if not data_lines:
            return
        payload = b"\n".join(data_lines)
        if payload == b"[DONE]":
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            return
        if not isinstance(obj, dict) or obj.get("choices") != []:
            return
        usage = obj.get("usage")
        if not isinstance(usage, dict):
            return
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if not (_is_count(prompt) and _is_count(completion)):
            return
        if self._sink is not None:
            self._sink.prompt_tokens = prompt
            self._sink.completion_tokens = completion


class VLLMBackend(Backend):
    """vLLM backend implementation using httpx async client.

    Args:
        base_url: Base URL of the vLLM server.
        timeout: Request timeout in seconds.
        connect_timeout: Connection timeout in seconds.
        name: Optional backend name for error messages.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
        name: str | None = None,
        thinking_mode: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialize the vLLM backend.

        Args:
            base_url: Base URL of the vLLM server.
            timeout: Request timeout in seconds.
            connect_timeout: Connection timeout in seconds.
            name: Optional backend name for error messages.
            thinking_mode: Thinking mode directive ('think' or 'no_think').
            reasoning_effort: Thinking depth when thinking_mode is 'think'
                ('low', 'medium' or 'xhigh'). Ignored for 'no_think'.
        """
        self.base_url = base_url.rstrip("/")
        self.name = name
        self._thinking_mode = thinking_mode
        self._reasoning_effort = reasoning_effort
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
        )

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        """Parse Retry-After header from response.

        Args:
            response: HTTP response.

        Returns:
            Retry delay in seconds, or None if header not present or invalid.
        """
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return None

        try:
            return float(retry_after)
        except ValueError:
            return None

    def _apply_thinking_controls(self, request: dict[str, Any]) -> dict[str, Any]:
        """Apply the thinking-mode controls via chat template kwargs.

        Qwen3 (2025) 世代は system メッセージ先頭の ``/no_think`` 文字列で
        thinking を切る方式だったが、Qwen3.5 以降は thinking が既定 ON になり、
        制御は chat template の ``enable_thinking`` / ``reasoning_effort``
        kwargs に移った。文字列 directive は黙って無視される (= light ティアが
        気付かないまま thinking で走る) ため、kwargs 方式に統一する。
        ``enable_thinking`` は Qwen3-32B の chat template も解釈するので、
        旧モデルへ切り戻しても同じ経路で効く。

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            Modified request with chat_template_kwargs populated.
        """
        if not self._thinking_mode:
            return request

        kwargs = dict(request.get("chat_template_kwargs") or {})

        if self._thinking_mode == "no_think":
            kwargs.setdefault("enable_thinking", False)
        else:
            kwargs.setdefault("enable_thinking", True)
            if self._reasoning_effort:
                kwargs.setdefault("reasoning_effort", self._reasoning_effort)

        request = {**request, "chat_template_kwargs": kwargs}

        # vLLM のトップレベル reasoning_effort は Literal["low","medium","high"]
        # だが、Qwen3.5+ の chat template が受け付けるのは low/medium/xhigh で、
        # "high" は raise_exception → 400 になる。トップレベル値は
        # chat_template_kwargs より優先されるので、ここで吸収しておく。
        caller_effort = request.get("reasoning_effort")
        if caller_effort == "high":
            request.pop("reasoning_effort")
            kwargs["reasoning_effort"] = "xhigh"
            logger.debug(
                "vllm_reasoning_effort_remapped", requested="high", applied="xhigh"
            )

        return request

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send chat completion request to vLLM.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            OpenAI-compatible chat completion response.

        Raises:
            BackendError: If the request fails.
        """
        return await self._post("/v1/chat/completions", self._apply_thinking_controls(request))

    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send completion request to vLLM.

        Args:
            request: OpenAI-compatible completion request.

        Returns:
            OpenAI-compatible completion response.

        Raises:
            BackendError: If the request fails.
        """
        return await self._post("/v1/completions", request)

    async def list_models(self) -> dict[str, Any]:
        """List available models from vLLM.

        Returns:
            OpenAI-compatible models list response.

        Raises:
            BackendError: If the request fails.
        """
        return await self._get("/v1/models")

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send embeddings request to vLLM.

        Args:
            request: OpenAI-compatible embeddings request.

        Returns:
            OpenAI-compatible embeddings response.

        Raises:
            BackendError: If the request fails.
        """
        return await self._post("/v1/embeddings", request)

    async def health_check(self) -> bool:
        """Check if vLLM is healthy.

        Returns:
            True if vLLM is healthy, False otherwise.
        """
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def chat_completions_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None = None
    ) -> AsyncIterator[bytes]:
        """Send streaming chat completion request to vLLM.

        Fills ``usage_sink`` (T-streaming-ledger-row PR-C, msg-879/msg-880):

        * **The upstream is always asked for usage.** ``stream_options.
          include_usage: true`` is written into a copy of the request whatever
          the caller sent -- including an explicit ``false``, which would
          otherwise buy a full generation with no ledger row. Other
          ``stream_options`` keys survive; the caller's dict is not mutated.
        * **Client-visible effect, deliberate:** the relay is verbatim, so a
          caller who did not ask now also receives vLLM's final usage event,
          ``"choices": []`` plus ``usage``, immediately before
          ``data: [DONE]``. Evidence that this is safe: msg-874 (one probe
          against vLLM 0.17.1 measured that exact event, and that no earlier
          event carries ``usage``) and msg-760 (31 days of production logs:
          no client streams from a vLLM tier, and both known readers skip
          empty-``choices`` events).
        * **The relay does not depend on the parser.** Each upstream chunk is
          yielded byte-for-byte *before* a side copy goes to
          ``_StreamUsageReader``; that reader catches its own exceptions and
          caps its buffer at ``USAGE_PARSE_BUFFER_CAP``. No parse failure,
          delimiter style or oversize event can delay, change or end what the
          client receives -- the worst case is a zero sink and no row.
        * Billed as ``prompt_tokens`` / ``completion_tokens`` as reported;
          measured to sum to ``total_tokens`` (msg-874 (iv)). vLLM counts
          reasoning tokens inside ``completion_tokens`` and reports no split,
          so ``thinking_tokens`` stays ``None`` -- nothing is inferred.
        * A client that disconnects before the final event bills zero, as on
          every backend: the count is in the last event.

        ``Backend.fills_usage_sink`` nevertheless stays ``False`` on this
        class: the flag is per class and ``completions_stream`` below does not
        fill the sink, so declaring would count every streamed
        ``/v1/completions`` as the defect. The cost is that
        ``lexora_stream_usage_missing_total`` does not watch this path;
        ``tests/backends/test_vllm_stream_usage.py`` pins it instead.

        Args:
            request: OpenAI-compatible chat completion request.
            usage_sink: Per-request carrier, filled from the usage event.

        Yields:
            SSE data chunks, exactly as vLLM sent them.

        Raises:
            BackendError: If the request fails.
        """
        reader = _StreamUsageReader(usage_sink)
        upstream_request = self._apply_thinking_controls(_with_usage_requested(request))
        async for chunk in self._post_stream("/v1/chat/completions", upstream_request):
            # Yield first, read after: the relay never waits on the reader.
            yield chunk
            reader.feed(chunk)

    async def completions_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None = None
    ) -> AsyncIterator[bytes]:
        """Send streaming completion request to vLLM.

        Args:
            request: OpenAI-compatible completion request.
            usage_sink: Accepted and never filled. Unlike
                ``chat_completions_stream`` this path sends no
                ``stream_options``: only ``/v1/chat/completions`` was probed
                (msg-874), and an upstream that rejected the field here would
                break the path, so it stays unasked and unfilled. Zero means
                no row -- "we cannot say", not "free". A caller can therefore
                pick this endpoint to leave no row; that gap is known and
                stated (coverage table on ``api/routes.py``'s
                ``get_cost_tracker``), not accidental.

        Yields:
            SSE data chunks.

        Raises:
            BackendError: If the request fails.
        """
        async for chunk in self._post_stream("/v1/completions", request):
            yield chunk

    async def _post_stream(
        self, path: str, data: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send streaming POST request to vLLM.

        Args:
            path: API path.
            data: Request body.

        Yields:
            SSE data chunks.

        Raises:
            BackendError: If the request fails.
        """
        try:
            logger.debug("vllm_stream_request", path=path, model=data.get("model"))
            async with self._client.stream("POST", path, json=data) as response:
                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response)
                    raise BackendRateLimitError(
                        "vLLM rate limit exceeded (429)",
                        retry_after=retry_after,
                        backend_name=self.name,
                    )

                if response.status_code == 503:
                    raise BackendUnavailableError("vLLM is temporarily unavailable")

                if response.status_code >= 400:
                    # Read error body for non-streaming error response
                    error_body = await response.aread()
                    # Decode exactly once, with replacement, BEFORE the
                    # `try`. The pre-fix shape called `.decode()` (no
                    # `errors=`) at two sites — the `.get(..., default)`
                    # fallback (Python evaluates arguments eagerly, so
                    # the default ran whether or not the key was
                    # present) and again in the bare `except`. On a
                    # non-UTF-8 upstream body (vLLM ingress error page,
                    # mislabelled UTF-16, …) something inside the `try`
                    # always fails; which line goes first depends on
                    # the byte pattern and does not matter for the
                    # shape of the bug. What matters is that repeating
                    # the same decode in `except` turns a diagnosable
                    # `BackendError` (carrying the upstream status and
                    # body) into a bare `UnicodeDecodeError` — the
                    # upstream status is lost and the generator dies.
                    # Decoding once up front means neither branch below
                    # can throw a decoding error, and the trap cannot
                    # come back by someone editing one branch in
                    # isolation. Mirrors `anthropic.py` §"Decode
                    # exactly once, with replacement" / `gemini.py`.
                    import json
                    error_text = (
                        error_body.decode(errors="replace") if error_body else ""
                    )
                    try:
                        error_json = json.loads(error_body)
                        error_message = error_json.get("error", {}).get(
                            "message", error_text
                        )
                    except Exception:
                        error_message = error_text
                    raise BackendError(
                        f"vLLM error ({response.status_code}): {error_message}"
                    )

                async for chunk in response.aiter_bytes():
                    yield chunk

        except httpx.ConnectError as e:
            logger.error("vllm_stream_connection_error", path=path, error=str(e))
            raise BackendConnectionError(f"Failed to connect to vLLM: {e}") from e
        except httpx.TimeoutException as e:
            logger.error("vllm_stream_timeout", path=path, error=str(e))
            raise BackendTimeoutError(f"vLLM request timed out: {e}") from e
        except (BackendError, BackendUnavailableError, BackendRateLimitError):
            raise
        except httpx.HTTPError as e:
            logger.error("vllm_stream_http_error", path=path, error=str(e))
            raise BackendError(f"vLLM request failed: {e}") from e

    async def _get(self, path: str) -> dict[str, Any]:
        """Send GET request to vLLM.

        Args:
            path: API path.

        Returns:
            JSON response.

        Raises:
            BackendError: If the request fails.
        """
        try:
            logger.debug("vllm_get_request", path=path)
            response = await self._client.get(path)
            return self._handle_response(response)
        except httpx.ConnectError as e:
            logger.error("vllm_connection_error", path=path, error=str(e))
            raise BackendConnectionError(f"Failed to connect to vLLM: {e}") from e
        except httpx.TimeoutException as e:
            logger.error("vllm_timeout", path=path, error=str(e))
            raise BackendTimeoutError(f"vLLM request timed out: {e}") from e
        except httpx.HTTPError as e:
            logger.error("vllm_http_error", path=path, error=str(e))
            raise BackendError(f"vLLM request failed: {e}") from e

    async def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Send POST request to vLLM.

        Args:
            path: API path.
            data: Request body.

        Returns:
            JSON response.

        Raises:
            BackendError: If the request fails.
        """
        try:
            logger.debug("vllm_post_request", path=path, model=data.get("model"))
            response = await self._client.post(path, json=data)
            return self._handle_response(response)
        except httpx.ConnectError as e:
            logger.error("vllm_connection_error", path=path, error=str(e))
            raise BackendConnectionError(f"Failed to connect to vLLM: {e}") from e
        except httpx.TimeoutException as e:
            logger.error("vllm_timeout", path=path, error=str(e))
            raise BackendTimeoutError(f"vLLM request timed out: {e}") from e
        except httpx.HTTPError as e:
            logger.error("vllm_http_error", path=path, error=str(e))
            raise BackendError(f"vLLM request failed: {e}") from e

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle HTTP response from vLLM.

        Args:
            response: HTTP response.

        Returns:
            JSON response body.

        Raises:
            BackendError: If the response indicates an error.
        """
        if response.status_code == 429:
            retry_after = self._parse_retry_after(response)
            raise BackendRateLimitError(
                "vLLM rate limit exceeded (429)",
                retry_after=retry_after,
                backend_name=self.name,
            )

        if response.status_code == 503:
            raise BackendUnavailableError("vLLM is temporarily unavailable")

        if response.status_code >= 400:
            try:
                error_body = response.json()
                error_message = error_body.get("error", {}).get("message", response.text)
            except Exception:
                error_message = response.text
            raise BackendError(f"vLLM error ({response.status_code}): {error_message}")

        return response.json()

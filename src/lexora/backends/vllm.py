"""vLLM backend implementation using httpx."""

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
)
from lexora.utils.logging import get_logger

logger = get_logger(__name__)


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
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send streaming chat completion request to vLLM.

        Args:
            request: OpenAI-compatible chat completion request.

        Yields:
            SSE data chunks.

        Raises:
            BackendError: If the request fails.
        """
        async for chunk in self._post_stream("/v1/chat/completions", self._apply_thinking_controls(request)):
            yield chunk

    async def completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send streaming completion request to vLLM.

        Args:
            request: OpenAI-compatible completion request.

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

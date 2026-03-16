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
    ) -> None:
        """Initialize the vLLM backend.

        Args:
            base_url: Base URL of the vLLM server.
            timeout: Request timeout in seconds.
            connect_timeout: Connection timeout in seconds.
            name: Optional backend name for error messages.
            thinking_mode: Thinking mode directive ('think' or 'no_think').
        """
        self.base_url = base_url.rstrip("/")
        self.name = name
        self._thinking_mode = thinking_mode
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

    def _inject_thinking_mode(self, request: dict[str, Any]) -> dict[str, Any]:
        """Inject thinking mode directive into chat messages.

        Prepends /think or /no_think to the first system message,
        or adds a new system message if none exists.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            Modified request with thinking directive injected.
        """
        if not self._thinking_mode:
            return request

        directive = f"/{self._thinking_mode}"
        request = {**request, "messages": list(request.get("messages", []))}

        messages = request["messages"]
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                **messages[0],
                "content": f"{directive}\n{messages[0].get('content', '')}",
            }
        else:
            messages.insert(0, {"role": "system", "content": directive})

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
        return await self._post("/v1/chat/completions", self._inject_thinking_mode(request))

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
        async for chunk in self._post_stream("/v1/chat/completions", self._inject_thinking_mode(request)):
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
                    try:
                        import json
                        error_json = json.loads(error_body)
                        error_message = error_json.get("error", {}).get(
                            "message", error_body.decode()
                        )
                    except Exception:
                        error_message = error_body.decode()
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

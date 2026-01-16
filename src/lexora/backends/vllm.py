"""vLLM backend implementation using httpx."""

from typing import Any

import httpx

from lexora.backends.base import (
    Backend,
    BackendConnectionError,
    BackendError,
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
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
    ) -> None:
        """Initialize the vLLM backend.

        Args:
            base_url: Base URL of the vLLM server.
            timeout: Request timeout in seconds.
            connect_timeout: Connection timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
        )

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send chat completion request to vLLM.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            OpenAI-compatible chat completion response.

        Raises:
            BackendError: If the request fails.
        """
        return await self._post("/v1/chat/completions", request)

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

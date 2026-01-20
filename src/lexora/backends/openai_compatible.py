"""OpenAI-compatible backend implementation using httpx."""

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


class OpenAICompatibleBackend(Backend):
    """OpenAI-compatible API backend implementation.

    Supports any API that implements the OpenAI API specification,
    including OpenAI itself, Azure OpenAI, and other compatible services.

    Args:
        base_url: Base URL of the API server.
        api_key: API key for authentication.
        timeout: Request timeout in seconds.
        connect_timeout: Connection timeout in seconds.
        model_mapping: Optional mapping from requested model names to actual names.
        name: Optional backend name for error messages.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com",
        api_key: str | None = None,
        timeout: float = 60.0,
        connect_timeout: float = 5.0,
        model_mapping: dict[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the OpenAI-compatible backend.

        Args:
            base_url: Base URL of the API server.
            api_key: API key for authentication.
            timeout: Request timeout in seconds.
            connect_timeout: Connection timeout in seconds.
            model_mapping: Optional mapping from requested model names to actual names.
            name: Optional backend name for error messages.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_mapping = model_mapping or {}
        self.name = name

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            headers=headers,
        )

    def _map_model(self, model: str) -> str:
        """Map requested model name to actual model name.

        Args:
            model: Requested model name.

        Returns:
            Actual model name to use.
        """
        return self.model_mapping.get(model, model)

    def _apply_model_mapping(self, request: dict[str, Any]) -> dict[str, Any]:
        """Apply model mapping to request.

        Args:
            request: Original request dictionary.

        Returns:
            Request with mapped model name.
        """
        if "model" in request:
            mapped_request = request.copy()
            mapped_request["model"] = self._map_model(request["model"])
            return mapped_request
        return request

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
            # Try parsing as seconds (integer)
            return float(retry_after)
        except ValueError:
            pass

        # Could also try parsing HTTP-date format, but most APIs use seconds
        return None

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send chat completion request.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            OpenAI-compatible chat completion response.

        Raises:
            BackendError: If the request fails.
        """
        mapped_request = self._apply_model_mapping(request)
        return await self._post("/v1/chat/completions", mapped_request)

    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send completion request.

        Args:
            request: OpenAI-compatible completion request.

        Returns:
            OpenAI-compatible completion response.

        Raises:
            BackendError: If the request fails.
        """
        mapped_request = self._apply_model_mapping(request)
        return await self._post("/v1/completions", mapped_request)

    async def list_models(self) -> dict[str, Any]:
        """List available models.

        Returns:
            OpenAI-compatible models list response.

        Raises:
            BackendError: If the request fails.
        """
        return await self._get("/v1/models")

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send embeddings request.

        Args:
            request: OpenAI-compatible embeddings request.

        Returns:
            OpenAI-compatible embeddings response.

        Raises:
            BackendError: If the request fails.
        """
        mapped_request = self._apply_model_mapping(request)
        return await self._post("/v1/embeddings", mapped_request)

    async def health_check(self) -> bool:
        """Check if the API is healthy.

        Returns:
            True if API is healthy, False otherwise.
        """
        try:
            # Try to list models as a health check
            await self._get("/v1/models")
            return True
        except BackendError:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def chat_completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send streaming chat completion request.

        Args:
            request: OpenAI-compatible chat completion request.

        Yields:
            SSE data chunks.

        Raises:
            BackendError: If the request fails.
        """
        mapped_request = self._apply_model_mapping(request)
        async for chunk in self._post_stream("/v1/chat/completions", mapped_request):
            yield chunk

    async def completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send streaming completion request.

        Args:
            request: OpenAI-compatible completion request.

        Yields:
            SSE data chunks.

        Raises:
            BackendError: If the request fails.
        """
        mapped_request = self._apply_model_mapping(request)
        async for chunk in self._post_stream("/v1/completions", mapped_request):
            yield chunk

    async def _post_stream(
        self, path: str, data: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send streaming POST request.

        Args:
            path: API path.
            data: Request body.

        Yields:
            SSE data chunks.

        Raises:
            BackendError: If the request fails.
        """
        try:
            logger.debug(
                "openai_stream_request",
                path=path,
                model=data.get("model"),
                backend=self.name,
            )
            async with self._client.stream("POST", path, json=data) as response:
                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response)
                    raise BackendRateLimitError(
                        f"Rate limit exceeded (429)",
                        retry_after=retry_after,
                        backend_name=self.name,
                    )

                if response.status_code == 503:
                    raise BackendUnavailableError(
                        f"Backend is temporarily unavailable"
                    )

                if response.status_code >= 400:
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
                        f"API error ({response.status_code}): {error_message}"
                    )

                async for chunk in response.aiter_bytes():
                    yield chunk

        except httpx.ConnectError as e:
            logger.error(
                "openai_stream_connection_error",
                path=path,
                error=str(e),
                backend=self.name,
            )
            raise BackendConnectionError(f"Failed to connect to API: {e}") from e
        except httpx.TimeoutException as e:
            logger.error(
                "openai_stream_timeout",
                path=path,
                error=str(e),
                backend=self.name,
            )
            raise BackendTimeoutError(f"API request timed out: {e}") from e
        except (BackendError, BackendUnavailableError, BackendRateLimitError):
            raise
        except httpx.HTTPError as e:
            logger.error(
                "openai_stream_http_error",
                path=path,
                error=str(e),
                backend=self.name,
            )
            raise BackendError(f"API request failed: {e}") from e

    async def _get(self, path: str) -> dict[str, Any]:
        """Send GET request.

        Args:
            path: API path.

        Returns:
            JSON response.

        Raises:
            BackendError: If the request fails.
        """
        try:
            logger.debug("openai_get_request", path=path, backend=self.name)
            response = await self._client.get(path)
            return self._handle_response(response)
        except httpx.ConnectError as e:
            logger.error(
                "openai_connection_error",
                path=path,
                error=str(e),
                backend=self.name,
            )
            raise BackendConnectionError(f"Failed to connect to API: {e}") from e
        except httpx.TimeoutException as e:
            logger.error(
                "openai_timeout",
                path=path,
                error=str(e),
                backend=self.name,
            )
            raise BackendTimeoutError(f"API request timed out: {e}") from e
        except httpx.HTTPError as e:
            logger.error(
                "openai_http_error",
                path=path,
                error=str(e),
                backend=self.name,
            )
            raise BackendError(f"API request failed: {e}") from e

    async def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Send POST request.

        Args:
            path: API path.
            data: Request body.

        Returns:
            JSON response.

        Raises:
            BackendError: If the request fails.
        """
        try:
            logger.debug(
                "openai_post_request",
                path=path,
                model=data.get("model"),
                backend=self.name,
            )
            response = await self._client.post(path, json=data)
            return self._handle_response(response)
        except httpx.ConnectError as e:
            logger.error(
                "openai_connection_error",
                path=path,
                error=str(e),
                backend=self.name,
            )
            raise BackendConnectionError(f"Failed to connect to API: {e}") from e
        except httpx.TimeoutException as e:
            logger.error(
                "openai_timeout",
                path=path,
                error=str(e),
                backend=self.name,
            )
            raise BackendTimeoutError(f"API request timed out: {e}") from e
        except httpx.HTTPError as e:
            logger.error(
                "openai_http_error",
                path=path,
                error=str(e),
                backend=self.name,
            )
            raise BackendError(f"API request failed: {e}") from e

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle HTTP response.

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
                f"Rate limit exceeded (429)",
                retry_after=retry_after,
                backend_name=self.name,
            )

        if response.status_code == 503:
            raise BackendUnavailableError("Backend is temporarily unavailable")

        if response.status_code >= 400:
            try:
                error_body = response.json()
                error_message = error_body.get("error", {}).get("message", response.text)
            except Exception:
                error_message = response.text
            raise BackendError(f"API error ({response.status_code}): {error_message}")

        return response.json()

"""Anthropic Claude API backend implementation."""

import json
import time
import uuid
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

# Anthropic API version
ANTHROPIC_VERSION = "2023-06-01"

# Default max_tokens (required by Anthropic API)
DEFAULT_MAX_TOKENS = 4096

# OpenAI parameters not supported by Anthropic
_UNSUPPORTED_PARAMS = {
    "frequency_penalty",
    "presence_penalty",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "n",
    "seed",
    "user",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "function_call",
    "functions",
}

# stop_reason mapping: Anthropic → OpenAI
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
}


class AnthropicBackend(Backend):
    """Anthropic Claude API backend.

    Translates OpenAI-compatible requests/responses to/from
    the Anthropic Messages API format.

    Args:
        base_url: Base URL of the Anthropic API.
        api_key: Anthropic API key.
        timeout: Request timeout in seconds.
        connect_timeout: Connection timeout in seconds.
        model_mapping: Optional mapping from requested model names to actual names.
        name: Optional backend name for error messages.
    """

    def __init__(
        self,
        base_url: str = "https://api.anthropic.com",
        api_key: str | None = None,
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
        model_mapping: dict[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_mapping = model_mapping or {}
        self.name = name

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if api_key:
            headers["x-api-key"] = api_key

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            headers=headers,
        )

    def _map_model(self, model: str) -> str:
        """Map requested model name to actual model name."""
        return self.model_mapping.get(model, model)

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        """Parse Retry-After header from response."""
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except ValueError:
            return None

    def _to_anthropic_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI chat completion request to Anthropic Messages API format.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            Anthropic Messages API request.
        """
        anthropic_req: dict[str, Any] = {}

        # Model mapping
        if "model" in request:
            anthropic_req["model"] = self._map_model(request["model"])

        # Extract system messages from messages list
        messages = request.get("messages", [])
        system_parts: list[str] = []
        non_system_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    # Handle content blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block["text"])
                        elif isinstance(block, str):
                            system_parts.append(block)
            else:
                non_system_messages.append(msg)

        if system_parts:
            anthropic_req["system"] = "\n\n".join(system_parts)

        anthropic_req["messages"] = non_system_messages

        # max_tokens is required by Anthropic
        anthropic_req["max_tokens"] = request.get("max_tokens", DEFAULT_MAX_TOKENS)

        # Pass through supported parameters
        if "temperature" in request:
            anthropic_req["temperature"] = request["temperature"]
        if "top_p" in request:
            anthropic_req["top_p"] = request["top_p"]
        if "stop" in request:
            anthropic_req["stop_sequences"] = request["stop"]
        if "stream" in request:
            anthropic_req["stream"] = request["stream"]

        return anthropic_req

    def _to_openai_response(
        self, anthropic_resp: dict[str, Any], model: str
    ) -> dict[str, Any]:
        """Convert Anthropic Messages API response to OpenAI chat completion format.

        Args:
            anthropic_resp: Anthropic API response.
            model: Model name used in the request.

        Returns:
            OpenAI-compatible chat completion response.
        """
        # Extract text content from content blocks
        content_parts: list[str] = []
        for block in anthropic_resp.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                content_parts.append(block.get("text", ""))

        content = "".join(content_parts)

        # Map stop_reason
        stop_reason = anthropic_resp.get("stop_reason", "end_turn")
        finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")

        # Map usage
        anthropic_usage = anthropic_resp.get("usage", {})
        prompt_tokens = anthropic_usage.get("input_tokens", 0)
        completion_tokens = anthropic_usage.get("output_tokens", 0)

        return {
            "id": f"chatcmpl-{anthropic_resp.get('id', uuid.uuid4().hex[:24])}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error HTTP response from Anthropic API.

        Args:
            response: HTTP response.

        Raises:
            BackendRateLimitError: On 429 status.
            BackendUnavailableError: On 529 (overloaded) or 503 status.
            BackendError: On other error statuses.
        """
        if response.status_code == 429:
            retry_after = self._parse_retry_after(response)
            raise BackendRateLimitError(
                "Rate limit exceeded (429)",
                retry_after=retry_after,
                backend_name=self.name,
            )

        if response.status_code in (503, 529):
            raise BackendUnavailableError("Backend is temporarily unavailable")

        if response.status_code >= 400:
            try:
                error_body = response.json()
                error_message = error_body.get("error", {}).get(
                    "message", response.text
                )
            except Exception:
                error_message = response.text
            raise BackendError(
                f"API error ({response.status_code}): {error_message}"
            )

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send chat completion request via Anthropic Messages API.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            OpenAI-compatible chat completion response.
        """
        model = self._map_model(request.get("model", ""))
        anthropic_req = self._to_anthropic_request(request)

        try:
            logger.debug(
                "anthropic_request",
                model=model,
                backend=self.name,
            )
            response = await self._client.post("/v1/messages", json=anthropic_req)
            self._handle_error_response(response)
            anthropic_resp = response.json()
            return self._to_openai_response(anthropic_resp, model)

        except httpx.ConnectError as e:
            logger.error(
                "anthropic_connection_error", error=str(e), backend=self.name
            )
            raise BackendConnectionError(
                f"Failed to connect to Anthropic API: {e}"
            ) from e
        except httpx.TimeoutException as e:
            logger.error("anthropic_timeout", error=str(e), backend=self.name)
            raise BackendTimeoutError(
                f"Anthropic API request timed out: {e}"
            ) from e
        except (BackendError, BackendUnavailableError, BackendRateLimitError):
            raise
        except httpx.HTTPError as e:
            logger.error(
                "anthropic_http_error", error=str(e), backend=self.name
            )
            raise BackendError(f"Anthropic API request failed: {e}") from e

    async def chat_completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send streaming chat completion request via Anthropic Messages API.

        Translates Anthropic SSE events to OpenAI SSE format.

        Args:
            request: OpenAI-compatible chat completion request.

        Yields:
            SSE data chunks in OpenAI format.
        """
        model = self._map_model(request.get("model", ""))
        anthropic_req = self._to_anthropic_request(request)
        anthropic_req["stream"] = True

        try:
            logger.debug(
                "anthropic_stream_request",
                model=model,
                backend=self.name,
            )
            async with self._client.stream(
                "POST", "/v1/messages", json=anthropic_req
            ) as response:
                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response)
                    raise BackendRateLimitError(
                        "Rate limit exceeded (429)",
                        retry_after=retry_after,
                        backend_name=self.name,
                    )

                if response.status_code in (503, 529):
                    raise BackendUnavailableError(
                        "Backend is temporarily unavailable"
                    )

                if response.status_code >= 400:
                    error_body = await response.aread()
                    try:
                        error_json = json.loads(error_body)
                        error_message = error_json.get("error", {}).get(
                            "message", error_body.decode()
                        )
                    except Exception:
                        error_message = error_body.decode()
                    raise BackendError(
                        f"API error ({response.status_code}): {error_message}"
                    )

                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                created = int(time.time())

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:].strip()
                    if not data_str:
                        continue

                    try:
                        event_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event_data.get("type", "")

                    if event_type == "message_start":
                        # Initial chunk with role
                        openai_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n".encode()

                    elif event_type == "content_block_delta":
                        delta = event_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            openai_chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n".encode()

                    elif event_type == "message_delta":
                        delta = event_data.get("delta", {})
                        stop_reason = delta.get("stop_reason", "end_turn")
                        finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")
                        openai_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n".encode()

                    elif event_type == "message_stop":
                        yield b"data: [DONE]\n\n"

        except httpx.ConnectError as e:
            logger.error(
                "anthropic_stream_connection_error",
                error=str(e),
                backend=self.name,
            )
            raise BackendConnectionError(
                f"Failed to connect to Anthropic API: {e}"
            ) from e
        except httpx.TimeoutException as e:
            logger.error(
                "anthropic_stream_timeout", error=str(e), backend=self.name
            )
            raise BackendTimeoutError(
                f"Anthropic API request timed out: {e}"
            ) from e
        except (BackendError, BackendUnavailableError, BackendRateLimitError):
            raise
        except httpx.HTTPError as e:
            logger.error(
                "anthropic_stream_http_error", error=str(e), backend=self.name
            )
            raise BackendError(f"Anthropic API request failed: {e}") from e

    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Not supported by Anthropic API."""
        raise BackendError("Text completions are not supported by Anthropic API")

    async def completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Not supported by Anthropic API."""
        raise BackendError("Text completions are not supported by Anthropic API")
        yield b""  # Make this an async generator

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Not supported by Anthropic API."""
        raise BackendError("Embeddings are not supported by Anthropic API")

    async def list_models(self) -> dict[str, Any]:
        """Return configured models in OpenAI format.

        Anthropic doesn't have a direct /v1/models equivalent,
        so we return the models from our configuration.
        """
        # Return an empty model list; actual models are managed by config
        return {"object": "list", "data": []}

    async def health_check(self) -> bool:
        """Check if the Anthropic API is reachable.

        Sends a minimal request to verify connectivity and authentication.
        """
        try:
            response = await self._client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            # Any response that's not a connection error means the API is reachable
            # 401 means bad key but API is up; 200 means all good
            return response.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

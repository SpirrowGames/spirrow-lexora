"""Google Gemini API backend implementation (native generateContent).

This backend translates OpenAI-compatible requests/responses to/from the
Gemini ``generateContent`` REST surface. It is purpose-built for the
**naysayer** role of the Spirrow trilateral (ADR-2026-05-31-14 /
ADR-2026-05-31-15): an independent, different-training-distribution reviewer
reached over a second vendor boundary.

Because crossing that boundary moves SpirrowGames design information outside
the Anthropic boundary (ADR-15 C-2), this adapter enforces a **data-governance
gate**: only the plain ``generateContent`` surface is reachable. Grounding
(google_search / retrieval), function-calling / tools, the File API, explicit
context caching, the Live API and the Interactions API are all refused at the
adapter boundary so that a training-excluded paid key stays on the one surface
the data-governance invariant permits (ADR-14 D-4).

The required invariant is a **paid key** (free keys are training-included and
forbidden). ZDR is *recommended*, not required, as of the
T-zdr-invariant-downgrade decision (Takahito, 2026-06-01): it is re-required
per project only when someone else's personal data goes on the LLM path. The
gate is independent of ZDR and stays as-is either way.

Context assembly (the N-1 "context-bundle builder" that injects ADR/diff/thread
text for the naysayer) is intentionally NOT done here: Lexora is a gateway and
leaves prompt/context assembly to the orchestration layer (Lexora CLAUDE.md,
DESIGN.md). This adapter only transports an already-assembled messages array.
"""

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

# Gemini REST API version segment.
GEMINI_API_VERSION = "v1beta"

# Default output token cap when the caller does not specify max_tokens.
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# Data-governance gate (ADR-2026-05-31-14 D-4 / ADR-2026-05-31-15 C-2):
# OpenAI-compatible request keys whose presence implies a Gemini surface
# beyond plain generateContent (function-calling / grounding / cached content).
# Their presence is refused rather than silently dropped, so a violation is
# visible instead of quietly downgraded.
_FORBIDDEN_REQUEST_KEYS = {
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "parallel_tool_calls",
    "cached_content",
    "cachedContent",
}

# Message-content part types that are not plain text. The naysayer surface is
# text-only; file/image/audio parts would pull in the File API or multimodal
# ingestion paths the gate exists to exclude.
_ALLOWED_CONTENT_PART_TYPES = {"text"}

# Gemini finishReason -> OpenAI finish_reason.
_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
}


class GeminiGovernanceError(BackendError):
    """Raised when a request would use a Gemini surface the gate forbids.

    Distinct subclass so callers/tests can assert specifically on a
    data-governance refusal versus a generic backend failure.
    """

    pass


class GeminiBackend(Backend):
    """Google Gemini API backend over native ``generateContent``.

    Translates OpenAI-compatible chat requests/responses to/from the Gemini
    REST API and enforces the naysayer data-governance gate (plain
    ``generateContent`` only).

    Args:
        base_url: Base URL of the Gemini API (without version/model path).
        api_key: Gemini API key (paid key; see ADR-14 D-4).
        timeout: Request timeout in seconds.
        connect_timeout: Connection timeout in seconds.
        model_mapping: Optional mapping from requested model names to actual names.
        name: Optional backend name for error messages.
        health_check_model: Model name health_check probes. Should be supplied
            from config (e.g. the backend's first configured model) so the
            health probe tracks the served model instead of a hard-coded name.
        paid_key_acknowledged: Operator affirmation that ``api_key`` is a
            paid/billing-enabled key. Required (fail-closed) when an api_key is
            configured — see ADR-2026-05-31-14 D-4 paid-key guarantee.

    Raises:
        GeminiGovernanceError: If an api_key is configured without
            paid_key_acknowledged.
    """

    def __init__(
        self,
        base_url: str = "https://generativelanguage.googleapis.com",
        api_key: str | None = None,
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
        model_mapping: dict[str, str] | None = None,
        name: str | None = None,
        health_check_model: str | None = None,
        paid_key_acknowledged: bool = False,
    ) -> None:
        # Paid-key structural guarantee (ADR-2026-05-31-14 D-4): with ZDR
        # downgraded to recommended, the paid key is the last line of defense
        # against training use of naysayer content. Gemini keys do not encode
        # their billing tier in the string, so we cannot probe paid/free at
        # runtime; instead we fail closed unless the operator has explicitly
        # affirmed (in config) that the configured key is a paid/billing-enabled
        # key. A key without that affirmation is refused — no accidental free-key
        # use via defaults.
        if api_key and not paid_key_acknowledged:
            raise GeminiGovernanceError(
                "paid-key guarantee: a GEMINI_API_KEY is configured but "
                "paid_key_acknowledged is not set. Set paid_key_acknowledged: "
                "true in the gemini backend config only after confirming the "
                "key belongs to a billing-enabled (paid) Google project. Free "
                "keys are training-included and forbidden for the naysayer "
                "route (ADR-2026-05-31-14 D-4)."
            )

        self.base_url = self._normalize_base_url(base_url)
        self.api_key = api_key
        self.model_mapping = model_mapping or {}
        self.name = name
        self.paid_key_acknowledged = paid_key_acknowledged
        # Model used by health_check. Defaults to a known model but should be
        # supplied from config so health probes the same model the backend
        # actually serves (rather than a hard-coded name).
        self.health_check_model = health_check_model or "gemini-2.5-flash"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["x-goog-api-key"] = api_key

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            headers=headers,
        )

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Strip trailing slash and any OpenAI-compat suffix.

        The shared config historically pointed Gemini at the OpenAI-compat
        shim (``.../v1beta/openai``). This native adapter builds its own
        ``/v1beta/models/...`` paths, so any such suffix is removed to leave
        the bare host.

        Args:
            base_url: Configured backend URL.

        Returns:
            Host base URL with no version/compat path and no trailing slash.
        """
        url = base_url.rstrip("/")
        for suffix in ("/v1beta/openai", "/v1/openai", "/openai"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        return url.rstrip("/")

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

    def _enforce_governance_gate(self, request: dict[str, Any]) -> None:
        """Refuse requests that would leave the plain generateContent surface.

        Enforces the ADR-2026-05-31-14 D-4 invariant: the naysayer key may only
        reach plain ``generateContent``. Function-calling/grounding/cached-content
        request keys and non-text message parts are rejected.

        Args:
            request: OpenAI-compatible chat completion request.

        Raises:
            GeminiGovernanceError: If a forbidden surface is requested.
        """
        present_forbidden = _FORBIDDEN_REQUEST_KEYS.intersection(request)
        if present_forbidden:
            raise GeminiGovernanceError(
                "data-governance gate: keys "
                f"{sorted(present_forbidden)} are not permitted for the "
                "naysayer surface (plain generateContent only; no tools / "
                "grounding / cached content)"
            )

        for msg in request.get("messages", []):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type not in _ALLOWED_CONTENT_PART_TYPES:
                    raise GeminiGovernanceError(
                        "data-governance gate: message content part of type "
                        f"{block_type!r} is not permitted for the naysayer "
                        "surface (text-only; no File API / multimodal ingestion)"
                    )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Flatten an OpenAI message ``content`` field to plain text.

        Args:
            content: A string or a list of content blocks.

        Returns:
            Concatenated text. Non-text blocks are dropped here; the gate has
            already refused them upstream.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return ""

    def _to_gemini_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI chat completion request to a Gemini request body.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            Gemini ``generateContent`` request body.
        """
        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []

        for msg in request.get("messages", []):
            role = msg.get("role")
            text = self._content_to_text(msg.get("content", ""))
            if role == "system":
                if text:
                    system_parts.append(text)
            else:
                # Gemini uses "model" for assistant turns; everything else maps
                # to "user".
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": text}]})

        gemini_req: dict[str, Any] = {"contents": contents}

        if system_parts:
            gemini_req["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }

        generation_config: dict[str, Any] = {
            "maxOutputTokens": request.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        }
        if "temperature" in request:
            generation_config["temperature"] = request["temperature"]
        if "top_p" in request:
            generation_config["topP"] = request["top_p"]
        if "stop" in request:
            stop = request["stop"]
            generation_config["stopSequences"] = (
                stop if isinstance(stop, list) else [stop]
            )
        gemini_req["generationConfig"] = generation_config

        return gemini_req

    def _to_openai_response(
        self, gemini_resp: dict[str, Any], model: str
    ) -> dict[str, Any]:
        """Convert a Gemini response to OpenAI chat completion format.

        Args:
            gemini_resp: Gemini API response.
            model: Model name used in the request.

        Returns:
            OpenAI-compatible chat completion response.
        """
        candidates = gemini_resp.get("candidates", [])
        content_text = ""
        finish_reason = "stop"
        if candidates:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            content_text = "".join(
                part.get("text", "") for part in parts if isinstance(part, dict)
            )
            finish_reason = _FINISH_REASON_MAP.get(
                candidate.get("finishReason", "STOP"), "stop"
            )
        elif gemini_resp.get("promptFeedback", {}).get("blockReason"):
            # Prompt blocked before any candidate was produced (safety/block).
            # Surface as content_filter so a block is observable symmetrically
            # with the streaming path, not silently reported as "stop".
            finish_reason = "content_filter"

        usage = gemini_resp.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get(
            "totalTokenCount", prompt_tokens + completion_tokens
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content_text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    def _model_path(self, model: str, method: str) -> str:
        """Build the REST path for a model method (generateContent etc.)."""
        return f"/{GEMINI_API_VERSION}/models/{model}:{method}"

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle an error HTTP response from the Gemini API.

        Args:
            response: HTTP response.

        Raises:
            BackendRateLimitError: On 429 status.
            BackendUnavailableError: On 503 status.
            BackendError: On other error statuses.
        """
        if response.status_code == 429:
            retry_after = self._parse_retry_after(response)
            raise BackendRateLimitError(
                "Rate limit exceeded (429)",
                retry_after=retry_after,
                backend_name=self.name,
            )

        if response.status_code == 503:
            raise BackendUnavailableError("Backend is temporarily unavailable")

        if response.status_code >= 400:
            try:
                error_body = response.json()
                error_message = error_body.get("error", {}).get(
                    "message", response.text
                )
            except Exception:
                error_message = response.text
            raise BackendError(f"API error ({response.status_code}): {error_message}")

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a chat completion request via Gemini ``generateContent``.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            OpenAI-compatible chat completion response.

        Raises:
            GeminiGovernanceError: If the request leaves the permitted surface.
            BackendError: If the request fails.
        """
        self._enforce_governance_gate(request)
        model = self._map_model(request.get("model", ""))
        gemini_req = self._to_gemini_request(request)

        try:
            logger.debug("gemini_request", model=model, backend=self.name)
            response = await self._client.post(
                self._model_path(model, "generateContent"), json=gemini_req
            )
            self._handle_error_response(response)
            return self._to_openai_response(response.json(), model)

        except httpx.ConnectError as e:
            logger.error("gemini_connection_error", error=str(e), backend=self.name)
            raise BackendConnectionError(
                f"Failed to connect to Gemini API: {e}"
            ) from e
        except httpx.TimeoutException as e:
            logger.error("gemini_timeout", error=str(e), backend=self.name)
            raise BackendTimeoutError(f"Gemini API request timed out: {e}") from e
        except (BackendError, BackendUnavailableError, BackendRateLimitError):
            raise
        except httpx.HTTPError as e:
            logger.error("gemini_http_error", error=str(e), backend=self.name)
            raise BackendError(f"Gemini API request failed: {e}") from e

    async def chat_completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send a streaming chat completion via ``streamGenerateContent``.

        ``streamGenerateContent`` is the streaming variant of the same plain
        generateContent surface (not the Live API), so it stays within the
        data-governance gate.

        Args:
            request: OpenAI-compatible chat completion request.

        Yields:
            SSE data chunks in OpenAI format.

        Raises:
            GeminiGovernanceError: If the request leaves the permitted surface.
            BackendError: If the request fails.
        """
        self._enforce_governance_gate(request)
        model = self._map_model(request.get("model", ""))
        gemini_req = self._to_gemini_request(request)

        try:
            logger.debug("gemini_stream_request", model=model, backend=self.name)
            async with self._client.stream(
                "POST",
                self._model_path(model, "streamGenerateContent"),
                params={"alt": "sse"},
                json=gemini_req,
            ) as response:
                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response)
                    raise BackendRateLimitError(
                        "Rate limit exceeded (429)",
                        retry_after=retry_after,
                        backend_name=self.name,
                    )
                if response.status_code == 503:
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
                role_sent = False

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if not data_str:
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    candidates = event.get("candidates", [])
                    if not candidates:
                        # Prompt-level block (no candidate): surface as
                        # content_filter so a block is observable symmetrically
                        # with the non-streaming path.
                        block = event.get("promptFeedback", {}).get("blockReason")
                        if block:
                            if not role_sent:
                                yield self._sse_chunk(
                                    chunk_id,
                                    created,
                                    model,
                                    {"role": "assistant", "content": ""},
                                    None,
                                )
                                role_sent = True
                            yield self._sse_chunk(
                                chunk_id, created, model, {}, "content_filter"
                            )
                        continue
                    candidate = candidates[0]
                    parts = candidate.get("content", {}).get("parts", [])
                    text = "".join(
                        p.get("text", "") for p in parts if isinstance(p, dict)
                    )

                    if not role_sent:
                        yield self._sse_chunk(
                            chunk_id,
                            created,
                            model,
                            {"role": "assistant", "content": ""},
                            None,
                        )
                        role_sent = True

                    if text:
                        yield self._sse_chunk(
                            chunk_id, created, model, {"content": text}, None
                        )

                    raw_finish = candidate.get("finishReason")
                    if raw_finish:
                        finish_reason = _FINISH_REASON_MAP.get(raw_finish, "stop")
                        yield self._sse_chunk(
                            chunk_id, created, model, {}, finish_reason
                        )

                yield b"data: [DONE]\n\n"

        except httpx.ConnectError as e:
            logger.error(
                "gemini_stream_connection_error", error=str(e), backend=self.name
            )
            raise BackendConnectionError(
                f"Failed to connect to Gemini API: {e}"
            ) from e
        except httpx.TimeoutException as e:
            logger.error("gemini_stream_timeout", error=str(e), backend=self.name)
            raise BackendTimeoutError(f"Gemini API request timed out: {e}") from e
        except (BackendError, BackendUnavailableError, BackendRateLimitError):
            raise
        except httpx.HTTPError as e:
            logger.error("gemini_stream_http_error", error=str(e), backend=self.name)
            raise BackendError(f"Gemini API request failed: {e}") from e

    @staticmethod
    def _sse_chunk(
        chunk_id: str,
        created: int,
        model: str,
        delta: dict[str, Any],
        finish_reason: str | None,
    ) -> bytes:
        """Build one OpenAI-format SSE chat-completion chunk."""
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n".encode()

    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Not supported: the naysayer surface is chat-only."""
        raise GeminiGovernanceError(
            "data-governance gate: legacy text completions are not part of the "
            "naysayer surface (use chat_completions / generateContent)"
        )

    async def completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Not supported: the naysayer surface is chat-only."""
        raise GeminiGovernanceError(
            "data-governance gate: legacy text completions are not part of the "
            "naysayer surface (use chat_completions / generateContent)"
        )
        yield b""  # pragma: no cover  # async-generator marker; unreachable after raise

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Not supported: embedContent is a separate, gated surface."""
        raise GeminiGovernanceError(
            "data-governance gate: embeddings (embedContent) is a separate "
            "Gemini surface and is not permitted for the naysayer key "
            "(plain generateContent only)"
        )

    async def list_models(self) -> dict[str, Any]:
        """Return configured models in OpenAI format.

        Models are managed by configuration; we do not call the Gemini
        ListModels surface from the gated naysayer key.
        """
        return {"object": "list", "data": []}

    async def health_check(self) -> bool:
        """Check if the Gemini API is reachable and the key authenticates.

        Sends a minimal generateContent request.
        """
        try:
            model = self._map_model(self.health_check_model)
            response = await self._client.post(
                self._model_path(model, "generateContent"),
                json={
                    "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                    "generationConfig": {"maxOutputTokens": 1},
                },
            )
            # Any non-5xx means the API is up (401/403 = bad key but reachable).
            return response.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

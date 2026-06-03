"""Anthropic Messages API compatibility layer.

Lexora's internal canonical format is the OpenAI ChatCompletion shape; every
backend translates between that and its own native surface. This module adds an
*inbound* adapter so clients using the Anthropic Messages API (e.g. the
``anthropic`` SDK pointed at Lexora via ``base_url``) can reach any routed model
— most notably the **naysayer** tier (Gemini, ADR-2026-05-31-15) — through the
same ``/v1/messages`` surface they already use for Claude.

It is the mirror image of ``backends/anthropic.py`` (which translates the other
direction, OpenAI -> Anthropic, when calling *out* to Anthropic). Here we
translate Anthropic -> OpenAI on the way in and OpenAI -> Anthropic on the way
out, then let the existing router / governance gate / stats / retry machinery do
its job unchanged.

Governance note: non-text content parts and tool/grounding fields are forwarded
*faithfully* into the internal request (rather than silently dropped) so that a
backend gate — e.g. the Gemini naysayer data-governance gate — still fires on
them instead of being bypassed at this layer.
"""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

# Anthropic stop_reason <- OpenAI finish_reason.
# content_filter maps to "refusal": the closest Anthropic semantic for a
# safety/blocklist stop (e.g. Gemini SAFETY/BLOCKLIST surfaced as content_filter).
_FINISH_TO_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}

# Default output cap when the Anthropic request omits max_tokens. The Anthropic
# API requires max_tokens, but we stay lenient at the gateway boundary.
DEFAULT_MAX_TOKENS = 4096


def _system_to_text(system: Any) -> str:
    """Flatten an Anthropic ``system`` field (str or block list) to plain text."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n\n".join(p for p in parts if p)
    return ""


def _convert_message_content(content: Any) -> Any:
    """Convert an Anthropic message ``content`` to OpenAI message content.

    A plain string passes through. A block list is converted block-by-block:
    ``text`` blocks become OpenAI ``{"type": "text", "text": ...}`` parts; any
    other block type (image, document, tool_use, tool_result, thinking, ...) is
    forwarded unchanged so a downstream gate can see and reject it rather than
    having it silently stripped here.

    Args:
        content: Anthropic message content (str or list of blocks).

    Returns:
        OpenAI-compatible message content (str or list of parts).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            else:
                # Forward non-text / tool blocks verbatim so backend gates fire.
                parts.append(block)
        return parts
    return content


def anthropic_to_openai_request(req: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic Messages request to an internal OpenAI chat request.

    Args:
        req: Anthropic Messages API request body.

    Returns:
        OpenAI-compatible chat completion request dict. The ``model`` field is
        copied through as-is; tier resolution is the router's responsibility.
    """
    messages: list[dict[str, Any]] = []

    system_text = _system_to_text(req.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for msg in req.get("messages", []):
        role = msg.get("role", "user")
        messages.append(
            {"role": role, "content": _convert_message_content(msg.get("content", ""))}
        )

    openai_req: dict[str, Any] = {
        "model": req.get("model", ""),
        "messages": messages,
        "max_tokens": req.get("max_tokens", DEFAULT_MAX_TOKENS),
    }

    if "temperature" in req:
        openai_req["temperature"] = req["temperature"]
    if "top_p" in req:
        openai_req["top_p"] = req["top_p"]
    if "top_k" in req:
        openai_req["top_k"] = req["top_k"]
    if req.get("stop_sequences"):
        openai_req["stop"] = req["stop_sequences"]
    if req.get("stream"):
        openai_req["stream"] = True

    # Forward tool / grounding fields verbatim so backend gates (e.g. the Gemini
    # naysayer gate) reject them instead of this layer silently dropping them.
    if "tools" in req:
        openai_req["tools"] = req["tools"]
    if "tool_choice" in req:
        openai_req["tool_choice"] = req["tool_choice"]

    return openai_req


def extract_user_id(req: dict[str, Any]) -> str | None:
    """Extract a rate-limit user id from the Anthropic ``metadata.user_id`` field."""
    metadata = req.get("metadata")
    if isinstance(metadata, dict):
        user_id = metadata.get("user_id")
        if isinstance(user_id, str):
            return user_id
    return None


def openai_to_anthropic_response(
    resp: dict[str, Any], model: str
) -> dict[str, Any]:
    """Convert an OpenAI chat completion response to an Anthropic Messages response.

    Args:
        resp: OpenAI-compatible chat completion response.
        model: Model (or tier) name to echo back to the client.

    Returns:
        Anthropic Messages API response body.
    """
    choices = resp.get("choices", [])
    content_text = ""
    finish_reason = "stop"
    if choices:
        message = choices[0].get("message", {})
        content_text = message.get("content") or ""
        finish_reason = choices[0].get("finish_reason") or "stop"

    stop_reason = _FINISH_TO_STOP_REASON.get(finish_reason, "end_turn")

    usage = resp.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    message_id = resp.get("id") or f"msg_{uuid.uuid4().hex[:24]}"
    if not message_id.startswith("msg_"):
        message_id = f"msg_{message_id}"

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content_text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def anthropic_error_body(error_type: str, message: str) -> dict[str, Any]:
    """Build an Anthropic-shaped error body ``{"type": "error", "error": {...}}``."""
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _sse(event: str, data: dict[str, Any]) -> bytes:
    """Encode one Anthropic-style SSE event (``event:`` + ``data:`` lines)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def anthropic_stream_from_openai(
    openai_chunks: AsyncIterator[bytes],
    model: str,
    message_id: str | None = None,
) -> AsyncIterator[bytes]:
    """Translate an OpenAI SSE chunk stream into Anthropic Messages SSE events.

    Consumes the OpenAI-format ``data: {...}`` chunk stream produced by a backend
    (``chat_completions_stream``) and emits the Anthropic event sequence:
    ``message_start`` -> ``content_block_start`` -> ``ping`` ->
    ``content_block_delta``* -> ``content_block_stop`` -> ``message_delta`` ->
    ``message_stop``.

    Token counts are not available in the OpenAI streaming chunks emitted by the
    gateway, so usage fields are reported as 0 in the streamed events (the
    non-streaming path reports real counts).

    Args:
        openai_chunks: Async iterator of OpenAI SSE byte chunks.
        model: Model (or tier) name to echo in ``message_start``.
        message_id: Optional message id; generated if omitted.

    Yields:
        Anthropic-format SSE byte chunks.
    """
    msg_id = message_id or f"msg_{uuid.uuid4().hex[:24]}"

    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    yield _sse("ping", {"type": "ping"})

    finish_reason: str | None = None
    try:
        async for raw in openai_chunks:
            for line in raw.decode("utf-8", errors="replace").splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text:
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    except Exception as exc:  # noqa: BLE001 - surface mid-stream failures to client
        # Anthropic streams report mid-stream failures via an ``error`` event.
        yield _sse(
            "error",
            anthropic_error_body("api_error", f"Upstream stream error: {exc}"),
        )
        return

    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    stop_reason = _FINISH_TO_STOP_REASON.get(finish_reason or "stop", "end_turn")
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})

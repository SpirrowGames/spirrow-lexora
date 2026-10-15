"""Ingress gate: redaction and the N-3 route declaration (ADR-14 §7.3).

Source: spirrow-mindwire#482 (ADR-2026-05-31-14 §7.3, read at head
``0a4197d``); Lexora-side design T-naysayer-codex-backend msg-800 §4-§6,
revised in msg-802 (``unrouted`` on the tier, S-1..S-4).

Every route handler that sends a tier request to a backend calls
``apply_ingress`` after building the request it will send and before the
backend is called. For a tier with neither ``redact: true`` nor codex in its
chain it returns the backend unchanged, so every other tier behaves exactly
as before.

* **N-3 route.** For a tier whose chain contains codex, a request without
  ``X-Mindwire-N3-Route: v1`` (missing, malformed or another version) never
  reaches codex: the tier's ``unrouted`` backend answers, ``n3_route=absent``
  is logged at INFO with the tier, and a present-but-wrong value is logged
  as ``n3_route_value`` (first 32 characters). The ledger row's ``backend``
  and ``answered_by`` name the backend that answered. The header only
  *permits* codex on a tier that already uses it; on a tier without codex it
  is not read at all.
* **Redaction.** For a ``redact: true`` tier every text of the request --
  system, every message, every content block (``tool_use`` input,
  ``thinking`` and unknown block types included; binary media excepted),
  OpenAI ``tool_calls`` / ``function_call`` arguments, a completion prompt,
  an embeddings input -- is redacted in place. If the scanner raises, the
  request fails and no backend is called (ADR-14 §7.3 失敗したとき).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from lexora.backends.answer_route import AnswerRoute, set_answer_route
from lexora.ingress.context import set_redactions
from lexora.ingress.redact import ASSIGNMENT_MIN_LEN, Redactor, format_redactions, is_key_name
from lexora.utils.logging import get_logger

logger = get_logger(__name__)

N3_ROUTE_HEADER = "X-Mindwire-N3-Route"
N3_ROUTE_V1 = "v1"
#: How much of a wrong header value the log keeps (msg-802 §5).
N3_ROUTE_VALUE_LOG_CHARS = 32


class RedactionFailed(Exception):
    """The scanner raised; the request must not reach a backend."""


@dataclass(frozen=True)
class TierIngress:
    """A tier's ingress settings, computed once when the router is built."""

    redact: bool
    codex_chain: bool
    unrouted: str | None

    @property
    def active(self) -> bool:
        return self.redact or self.codex_chain


#: Fields that only name or identify things (roles, block types, ids that a
#: later block must match, signatures); never scanned.
_STRUCTURAL_KEYS = frozenset(
    {"type", "role", "id", "tool_use_id", "tool_call_id", "name", "media_type", "signature", "cache_control"}
)
#: Binary parts: their payload is base64 media, not text, and redacting it
#: would only corrupt the media. Passed through.
_BINARY_PART_TYPES = frozenset({"image", "image_url", "input_audio", "redacted_thinking"})


def _redact_json(
    value: Any, redactor: Redactor, *, sensitive: bool = False, skip: frozenset[str] = frozenset()
) -> Any:
    """Redact structured data (an Anthropic ``tool_use`` input, parsed
    OpenAI tool-call arguments, an unknown block).

    Every string is scanned as text. A JSON member is D-1's key line in
    another syntax (``"password": ...``), so a member whose name matches the
    D-1 key names has its value redacted under the same rule -- 8+ characters
    as serialized, quotes included -- and when that value is an object or a
    list, every scalar in it is redacted (the window's "redact more when
    unsure", without a line count to apply). ``skip`` names structural
    keys left alone at this level only: deeper down they are user data."""
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key in skip and not sensitive:
                out[key] = item
                continue
            key_sensitive = sensitive or (isinstance(key, str) and is_key_name(key))
            out[key] = _redact_json(item, redactor, sensitive=key_sensitive)
        return out
    if isinstance(value, list):
        return [_redact_json(item, redactor, sensitive=sensitive) for item in value]
    if sensitive and value is not None and not isinstance(value, bool):
        serialized = json.dumps(value, ensure_ascii=False)
        if len(serialized) >= ASSIGNMENT_MIN_LEN:
            return redactor.label_assignment(serialized)
    if isinstance(value, str):
        return redactor.redact(value)
    return value


def _redact_arguments(arguments: Any, redactor: Redactor) -> Any:
    """OpenAI ``function.arguments``: a JSON string. Valid JSON is redacted
    structurally and serialized back, so it stays valid JSON; anything else
    is redacted as text."""
    if not isinstance(arguments, str):
        return _redact_json(arguments, redactor)
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return redactor.redact(arguments)
    before = sum(redactor.counts.values())
    redacted = _redact_json(parsed, redactor)
    if sum(redactor.counts.values()) == before:
        return arguments  # nothing found: keep the caller's exact bytes
    return json.dumps(redacted, ensure_ascii=False)


def _redact_part(part: dict[str, Any], redactor: Redactor) -> dict[str, Any]:
    kind = part.get("type")
    if kind == "text" and isinstance(part.get("text"), str):
        return {**part, "text": redactor.redact(part["text"])}
    if kind == "tool_result" and "content" in part:
        rest = _redact_json({k: v for k, v in part.items() if k != "content"}, redactor, skip=_STRUCTURAL_KEYS)
        return {**rest, "content": _redact_content(part["content"], redactor)}
    if kind in _BINARY_PART_TYPES:
        return part
    # tool_use (``input``), thinking, document, and any block type this gate
    # does not know: every string in it is scanned (fail closed).
    return _redact_json(part, redactor, skip=_STRUCTURAL_KEYS)  # type: ignore[no-any-return]


def _redact_content(content: Any, redactor: Redactor) -> Any:
    """Redact an OpenAI/Anthropic message ``content`` (str or part list)."""
    if isinstance(content, str):
        return redactor.redact(content)
    if isinstance(content, list):
        out: list[Any] = []
        for part in content:
            if isinstance(part, str):
                out.append(redactor.redact(part))
            elif isinstance(part, dict):
                out.append(_redact_part(part, redactor))
            else:
                out.append(part)
        return out
    return content


def _redact_message(message: dict[str, Any], redactor: Redactor) -> None:
    """Redact one message in place (see ``redact_request``)."""
    for key, value in list(message.items()):
        if key in _STRUCTURAL_KEYS:
            continue
        if key == "content":
            message[key] = _redact_content(value, redactor)
        elif key == "tool_calls" and isinstance(value, list):
            message[key] = [_redact_tool_call(call, redactor) for call in value]
        elif key == "function_call" and isinstance(value, dict):
            message[key] = _redact_tool_call({"function": value}, redactor)["function"]
        else:
            message[key] = _redact_json(value, redactor)


def _redact_tool_call(call: Any, redactor: Redactor) -> Any:
    if not isinstance(call, dict):
        return _redact_json(call, redactor)
    function = call.get("function")
    rest = _redact_json({k: v for k, v in call.items() if k != "function"}, redactor, skip=_STRUCTURAL_KEYS)
    if not isinstance(function, dict):
        return {**rest, **({"function": _redact_json(function, redactor)} if "function" in call else {})}
    fn_rest = _redact_json({k: v for k, v in function.items() if k != "arguments"}, redactor, skip=_STRUCTURAL_KEYS)
    if "arguments" in function:
        fn_rest["arguments"] = _redact_arguments(function["arguments"], redactor)
    return {**rest, "function": fn_rest}


def redact_request(request: dict[str, Any]) -> Counter[str]:
    """Redact every text of ``request`` in place; return counts by kind.

    Covers every message (which carries the system prompt: the
    ``/v1/messages`` adapter turns ``system`` into the first message) --
    its ``content`` blocks of any type but binary media, OpenAI
    ``tool_calls`` / ``function_call`` arguments and its other text fields
    -- plus ``system`` if still present, ``prompt`` and ``input``."""
    redactor = Redactor()
    messages = request.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                _redact_message(message, redactor)
    for key in ("system", "prompt", "input"):
        if key in request:
            request[key] = _redact_content(request[key], redactor)
    return redactor.counts


class IngressBackend:
    """Wraps the backend a gated request goes to.

    Stamps the answer route (unrouted requests) and the redaction counts from
    inside each call, the same way ``FallbackBackend`` stamps its route, so
    the handler's ``cost_tracker.record`` -- which runs in that context --
    sees them. Everything else is delegated to the wrapped backend."""

    def __init__(self, inner: Any, *, route: AnswerRoute | None, redactions: str | None) -> None:
        self._inner = inner
        self._route = route
        self._redactions = redactions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def inner(self) -> Any:
        return self._inner

    def _stamp(self) -> None:
        if self._route is not None:
            set_answer_route(self._route)
        set_redactions(self._redactions)

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        self._stamp()
        return await self._inner.chat_completions(request)

    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        self._stamp()
        return await self._inner.completions(request)

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        self._stamp()
        return await self._inner.embeddings(request)

    async def chat_completions_stream(self, request: dict[str, Any], **kwargs: Any) -> AsyncIterator[bytes]:
        self._stamp()
        async for chunk in self._inner.chat_completions_stream(request, **kwargs):
            yield chunk

    async def completions_stream(self, request: dict[str, Any], **kwargs: Any) -> AsyncIterator[bytes]:
        self._stamp()
        async for chunk in self._inner.completions_stream(request, **kwargs):
            yield chunk


def apply_ingress(
    backend_router: Any,
    requested: str,
    n3_route: str | None,
    request: dict[str, Any],
    backend: Any,
) -> Any:
    """Gate one request. Returns the backend to call.

    ``requested`` is the name the caller sent (tier alias or model);
    ``request`` is the dict about to be sent, redacted here in place;
    ``backend`` is what the router resolved for ``requested``.

    Raises ``RedactionFailed`` when the scanner raises."""
    ingress = backend_router.tier_ingress(requested)
    # ``BackendRouter.tier_ingress`` returns ``TierIngress | None`` and
    # nothing else. The isinstance test exists for the route tests that hand
    # the handlers a ``MagicMock`` router (whose attributes are mocks, not
    # tier settings); a real router never takes this branch with a tier.
    if not isinstance(ingress, TierIngress) or not ingress.active:
        return backend

    route: AnswerRoute | None = None
    if ingress.codex_chain and n3_route != N3_ROUTE_V1:
        # S-2/S-3 guarantee a configured, codex-free ``unrouted`` here.
        unrouted_name = ingress.unrouted
        target = backend_router.get_backend_by_name(unrouted_name) if unrouted_name else None
        if target is None:  # unreachable after the startup checks
            raise RuntimeError(f"tier '{requested}' has no usable 'unrouted' backend")
        fields: dict[str, Any] = {"tier": requested, "unrouted": unrouted_name}
        if n3_route is not None:
            fields["n3_route_value"] = n3_route[:N3_ROUTE_VALUE_LOG_CHARS]
        logger.info("n3_route_unrouted", n3_route="absent", **fields)
        route = AnswerRoute(backend=unrouted_name, answered_by=unrouted_name)  # type: ignore[arg-type]
        backend = target

    redactions: str | None = None
    if ingress.redact:
        try:
            counts = redact_request(request)
        except Exception as exc:  # noqa: BLE001 - any scanner failure blocks the request
            logger.error("ingress_redaction_failed", tier=requested, error_type=type(exc).__name__)
            raise RedactionFailed(f"redaction failed for tier '{requested}'") from exc
        redactions = format_redactions(counts)
        if redactions:
            logger.info("ingress_redacted", tier=requested, redactions=redactions)

    return IngressBackend(backend, route=route, redactions=redactions)

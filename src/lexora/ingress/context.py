"""Per-request ledger value for redactions (``redactions=<kind:count,...>``).

Same mechanism, and the same safety argument, as
``lexora.backends.answer_route``: the ingress wrapper (``gate.IngressBackend``)
stamps the value from inside the backend call it wraps, in the context the
handler's ``cost_tracker.record`` runs in, and ``CostTracker.record`` reads
it. Every request runs in its own context, so a value never reaches another
request. Requests that are not redacted never set it, so their rows keep
``redactions`` NULL.
"""

from __future__ import annotations

from contextvars import ContextVar

_CURRENT: ContextVar[str | None] = ContextVar("lexora_redactions", default=None)


def set_redactions(value: str | None) -> None:
    _CURRENT.set(value)


def current_redactions() -> str | None:
    return _CURRENT.get()

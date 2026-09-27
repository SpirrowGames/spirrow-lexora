"""Which backend actually answered this request (T-naysayer-codex-backend B-2).

A ``type: fallback`` backend (``backends/fallback.py``) sits behind the
naysayer tier and hands each request to codex or to Gemini. The ledger has to
record the backend that *answered*, not the wrapper the tier names, and which
route it took (``answered_by``: ``codex`` / ``gemini-fallback`` /
``codex-shadow``, msg-448 B-2).

The ten ``cost_tracker.record`` sites in ``api/routes.py`` all pass
``backend=<the tier's backend name>``. Rather than thread a new argument
through each of them, the wrapper stamps an ``AnswerRoute`` into this context
variable and ``CostTracker.record`` reads it. Why that is safe:

* The wrapper sets it from inside its own ``chat_completions`` (awaited
  directly by the handler -- ``RetryHandler.execute`` awaits, it does not
  spawn a task) or from inside its stream generator (whose body runs in the
  context of the handler's stream loop, which is also where that handler
  writes its ledger row). A value set there is visible to the ``record`` call
  that follows in the same context.
* Every request runs in its own task with its own copy of the context, so a
  value never reaches another request.
* A background task (the shadow run) copies the context when it is created;
  it sets its own ``AnswerRoute`` before recording, so it never inherits the
  foreground's.

Backends other than the wrapper never set it, so every other row keeps
``answered_by`` NULL and the ``backend`` its handler passed.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

#: ``answered_by`` values (msg-448 B-2). NULL for every row not routed by a
#: fallback wrapper, and for the Gemini answer in shadow mode (Gemini is the
#: primary there, not a fallback).
ANSWERED_BY_CODEX = "codex"
ANSWERED_BY_GEMINI_FALLBACK = "gemini-fallback"
ANSWERED_BY_CODEX_SHADOW = "codex-shadow"

#: Rows answered by the subscription CLI: cost 0, pricing known (B-2).
SUBSCRIPTION_ANSWERS = frozenset({ANSWERED_BY_CODEX, ANSWERED_BY_CODEX_SHADOW})


@dataclass(frozen=True)
class AnswerRoute:
    """What the ledger row for this request should say about who answered.

    ``backend``: config name of the backend that answered. ``model``: the
    concrete model it served, when that differs from the tier's model (codex
    serves its own model, not the tier's Gemini one); None keeps the caller's.
    ``answered_by``: one of the three values above, or None.
    """

    backend: str
    answered_by: str | None
    model: str | None = None


_CURRENT: ContextVar[AnswerRoute | None] = ContextVar("lexora_answer_route", default=None)


def set_answer_route(route: AnswerRoute) -> None:
    _CURRENT.set(route)


def current_answer_route() -> AnswerRoute | None:
    return _CURRENT.get()

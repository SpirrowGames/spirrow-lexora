"""Per-request trace id from ``X-Mindwire-Trace`` (T-cost-row-trace-id).

mindwire stamps every naysayer turn with a ULID ``turn_id`` and sends it as
``X-Mindwire-Trace``. Lexora stores it in the nullable ``trace_id`` column of
the cost row, and ``/stats/costs/recent?trace_id=`` returns only the rows of
that turn, so mindwire can attest which backend answered the turn.

Validation (msg-627 §2a, amended by msg-629):

* No header -> ``None``, silently. That is an ordinary request.
* A header that is not a canonical ULID (26 chars of upper-case Crockford
  base32, first char ``0-7`` so the value fits in 128 bits) -> ``None`` plus a
  ``trace_id_rejected`` warning carrying the length only, never the value.
  Empty is invalid, not absent.
* No normalisation (no upper-casing, no trimming). mindwire selects rows by
  exact match, so a rewritten value would silently stop matching.
* An invalid value never fails the request: it is just not recorded.

The handlers pass the parsed value to ``CostTracker.record`` explicitly. The
one record site that cannot receive an argument from a handler -- the shadow
run in ``backends/fallback.py``, a task the wrapper spawns -- reads it from
``current_trace_id()`` instead. The handler body sets it before calling the
backend, and ``loop.create_task`` copies the context at creation, so the
shadow task sees the value. It is set in the handler body and NOT in the
dependency, because a sync dependency runs in a threadpool and a value set
there never reaches the handler's context.
"""

from __future__ import annotations

import re
from contextvars import ContextVar

from lexora.utils.logging import get_logger

logger = get_logger(__name__)

TRACE_HEADER = "X-Mindwire-Trace"

#: Canonical ULID: 26 chars, upper-case Crockford base32 (no I/L/O/U), first
#: char 0-7 so the value is at most 128 bits.
ULID_RE = re.compile(r"^[0-7][0-9A-HJKMNP-TV-Z]{25}$")

ULID_LENGTH = 26

_CURRENT_TRACE: ContextVar[str | None] = ContextVar("lexora_trace_id", default=None)


def parse_trace_id(raw: str | None) -> str | None:
    """Return ``raw`` if it is a canonical ULID, else ``None``.

    Args:
        raw: The header value, or None when the header was not sent.

    Returns:
        The value unchanged when valid; None when absent or invalid.
    """
    if raw is None:
        return None
    if len(raw) != ULID_LENGTH or not ULID_RE.fullmatch(raw):
        logger.warning("trace_id_rejected", length=len(raw))
        return None
    return raw


def set_current_trace_id(trace_id: str | None) -> None:
    """Make ``trace_id`` visible to record sites the handler cannot reach."""
    _CURRENT_TRACE.set(trace_id)


def current_trace_id() -> str | None:
    """The trace id set by the handler of the current request, if any."""
    return _CURRENT_TRACE.get()

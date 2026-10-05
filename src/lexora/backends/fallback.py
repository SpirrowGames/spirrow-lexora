"""``type: fallback`` -- codex first, Gemini when codex cannot answer.

T-naysayer-codex-backend PR-2b, specified by msg-448 (B-1 to B-6) and
msg-450 (B-7), endorsed by Einstein after msg-450.

The naysayer tier points at this backend; it holds a ``primary`` (a codex
backend) and a ``fallback`` (a Gemini backend), both ordinary configured
backends, and a ``mode``:

* ``fallback`` (B-1): evaluate ``primary.codex_availability()`` once. If it
  is closed, or codex runs and ends in quota / launch failure / a closed gate
  / timeout / a latch, the request is answered by Gemini. Output of a latched
  run is never returned (D-1d). **Gemini is used only while no byte has been
  sent to the client**: codex produces its whole answer before its stream
  yields anything, so the switch is decided on the first ``__anext__`` of
  the codex stream; once a codex byte is out, any later error propagates.
* ``shadow`` (B-5): Gemini always answers. At most one codex run goes on in
  the background on the same request (through ``_run_gated``, so latch,
  hold and write-ahead all apply); if one is already running the request is
  skipped and counted in ``shadow_skipped``. A comparison row -- verdicts and
  timings, never text -- goes to ``shadow_comparisons`` in the cost DB.

**Time budget (msg-687 C).** Each request's deadline is its arrival at
the wrapper plus ``caller_budget_s`` (900s, ``LEXORA_BACKEND_TIMEOUT_S``:
the backend limit mindwire assumes under ADR-14; msg-702).
Codex waits at most ``slot_wait_s`` for its single slot and runs at most
``codex_timeout_s``; the fallback call is cut at ``min(Gemini's own
timeout, deadline - now)`` and then raises ``FallbackDeadlineExceeded``
(a plain ``BackendError``: never retried). ``slot_wait_s + codex_timeout_s
+ fallback_floor_s <= caller_budget_s`` is checked at start-up, so Gemini
always keeps at least ``fallback_floor_s``. Shadow mode's Gemini answer is
not cut: it is today's naysayer call and must stay as it is (msg-687 E).

**Ledger row of a fallback answer (msg-687 D).** ``backend`` and
``answered_by`` are both the constant ``gemini-fallback``, whatever the
``fallback`` setting names: that setting only chooses which backend is
called.

``CodexAuthError`` falls back too (msg-498 B-1'): an expired or revoked
device login is a known, recoverable state, and the "started" notice carries
``reason=auth`` so the operator learns a re-login is needed. Errors that do
NOT fall back and reach the caller: ``CodexFailed`` (unclassified -- an
unknown failure is not hidden behind Gemini) and
``CodexUnsupportedInputError`` (a caller error).

**Notifications (B-3), producer declaration.** Intended reader: the human
operator (Takahito). Surface: a Discord webhook named by the
``LEXORA_FALLBACK_WEBHOOK_URL`` environment variable, posted directly by
Lexora -- not a chatroom thread, not mindwire's notifier, not the daily
digest (msg-267 scope 4). If the variable is unset, one WARNING is logged at
construction and nothing is sent; a failed post is logged at WARNING and
retried on the next 10-minute cycle; neither ever blocks a request. Sent
once when fallback starts (STARTED), every 6 hours while it lasts
(CONTINUING), and once when codex is back (ENDED). Body: fixed format, at
most 500 characters, carrying only the event, the reason,
``fallback_since``, the Gemini calls and their cost in that period
(counted from the ledger), and ``quota_hold_until``.

ENDED (msg-685 / msg-687 F) is sent by ``notice_tick``, not by the codex
answer, when all three hold: a fallback period is open, codex has answered
at least once since the last fallback (``_answered_since_fallback``), and
the last fallback (``_last_fallback_at``) is at least 15 minutes old. A
codex answer only sets the flag and never moves the time, so a steady run
of codex answers cannot postpone ENDED; a fallback inside the 15 minutes
clears the flag and the period simply goes on (no new STARTED); a period
with no requests at all never ends by itself (codex has not answered).

**Shadow mode sends no state notice (msg-679 / msg-687 F).** In shadow
mode Gemini always answers, so a codex failure there is not a routing
change: ``_falling_back`` and ``_codex_answered`` return at once and
touch no state. The codex latches (``quota_hold``, ``slot_hold``) live
in ``CodexBackend`` and apply to both modes alike (msg-681).

**Data controls (A-15-2b, msg-535).** Codex closes with
``reason=data_controls_unverified`` when the human check of the ChatGPT
account's data controls is missing or older than 30 days
(``codex_data_controls.py``); that is an ordinary fallback, so it gets the
STARTED / CONTINUING / ENDED notices above, and those notices then also
carry ``runbook=deploy/RUNBOOK.md#8``. Before that, from 7 days ahead of
expiry, the notice loop posts one EXPIRING notice per UTC calendar day
("codex stops in N days"), same webhook, same 500-character cap. Its
state is the UTC date of the last EXPIRING notice sent; a failed post
leaves the date unset, so the next 10-minute tick retries. The notice loop
runs from start-up (``start``), not only during a fallback, so the warning
goes out even while codex is answering everything, and in shadow mode too:
it is a real warning, not a state change. Its body names ``mode=<mode>``,
so in shadow mode it is not read as "production routing will switch". Re-verifying (updating
the file) needs no restart: the next request re-reads it.

State (``fallback_since``, ``_last_fallback_at``,
``_answered_since_fallback``, the last notification time, ``shadow_skipped``,
the date of the last EXPIRING notice) is in memory. That is system-wide
because B-7 guarantees one process (``services/process_lock.py``). A
restart during fallback re-sends "start"; accepted in msg-448 B-3. A
restart inside the warning window may send a second EXPIRING notice that
day; the same trade-off.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Callable
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import httpx

from lexora.backends.answer_route import (
    ANSWERED_BY_CODEX,
    ANSWERED_BY_CODEX_SHADOW,
    ANSWERED_BY_GEMINI_FALLBACK,
    AnswerRoute,
    set_answer_route,
)
from lexora.backends.base import Backend, BackendError, UsageSink
from lexora.backends.codex import (
    CodexAuthError,
    CodexAvailability,
    CodexBackend,
    CodexError,
    CodexLaunchError,
    CodexNotVerifiedError,
    CodexQuotaError,
    CodexSlotTimeout,
    CodexTimeout,
    CodexToolUseViolation,
    CodexUnverifiableRun,
)
from lexora.backends.codex_data_controls import (
    REASON_DATA_CONTROLS_UNVERIFIED,
    RUNBOOK_POINTER,
)
from lexora.config import LEXORA_BACKEND_TIMEOUT_S, check_fallback_budget
from lexora.services.trace import current_trace_id
from lexora.utils.logging import get_logger

if TYPE_CHECKING:
    from lexora.services.cost_tracker import CostTracker

logger = get_logger(__name__)

#: Environment variable holding the Discord webhook URL (B-3).
WEBHOOK_ENV = "LEXORA_FALLBACK_WEBHOOK_URL"
NOTICE_MAX_CHARS = 500
CHECK_INTERVAL_S = 600.0
REMIND_AFTER = timedelta(hours=6)
#: ENDED waits until the last fallback is this old (msg-685).
RECOVER_AFTER = timedelta(minutes=15)
#: The ledger's ``backend`` for a fallback answer, whatever the ``fallback``
#: setting names (msg-687 D); equals ``answered_by``.
BACKEND_GEMINI_FALLBACK = ANSWERED_BY_GEMINI_FALLBACK
#: ``shadow_comparisons.codex_verdict`` when the gate was closed and codex
#: did not run (B-5'); ``codex_reason`` then holds the gate's reason.
SHADOW_NOT_RUN = "not_run"

#: The B-1 list plus ``CodexAuthError`` (B-1'): these fall back to Gemini.
#: ``CodexUnverifiableRun`` is a ``CodexToolUseViolation`` (a latch).
FALLBACK_ERRORS: tuple[type[CodexError], ...] = (
    CodexNotVerifiedError,
    CodexQuotaError,
    CodexAuthError,
    CodexLaunchError,
    CodexTimeout,
    CodexSlotTimeout,
    CodexToolUseViolation,
)


class FallbackDeadlineExceeded(BackendError):
    """The fallback answer did not finish by the request's deadline
    (msg-687 C). A plain ``BackendError``: the retry handler never retries
    it, since a retry would run past the caller's budget anyway."""


def fallback_reason(exc: CodexError) -> str:
    """Short reason for a codex failure that fell back (status / notice)."""
    if isinstance(exc, CodexNotVerifiedError):
        return exc.reason
    if isinstance(exc, CodexQuotaError):
        return "quota"
    if isinstance(exc, CodexAuthError):
        return "auth"
    if isinstance(exc, CodexLaunchError):
        return "launch_failed"
    if isinstance(exc, CodexTimeout):
        return "timeout"
    if isinstance(exc, CodexSlotTimeout):
        return "slot_wait_timeout"
    if isinstance(exc, CodexUnverifiableRun):
        return "unverifiable_run"
    if isinstance(exc, CodexToolUseViolation):
        return "tool_use_violation"
    return type(exc).__name__


# --------------------------------------------------------------------------
# Verdict extraction (B-5)
# --------------------------------------------------------------------------

_PR_GATE_VERDICT = re.compile(r"VERDICT:\s*(APPROVE|REQUEST_CHANGES)\b")
_BLOCKS_TRUE = re.compile(r"\bblocks\**:\**\s*true\b", re.IGNORECASE)
#: What marks an answer as design-time shaped when it has no ``blocks: true``:
#: the section headers or a ``blocks:`` field at all.
_DESIGN_TIME = re.compile(
    r"\*\*(Endorsements|Objections)\*\*|\bblocks\**:\**\s*(true|false)\b", re.IGNORECASE
)


def extract_verdict(text: str | None) -> str:
    """``APPROVE`` / ``REQUEST_CHANGES`` (PR-gate; the last ``VERDICT:``
    line wins), else ``blocking`` / ``non_blocking`` (design-time: whether
    ``blocks: true`` appears), else ``unparsed``."""
    if not text:
        return "unparsed"
    found = _PR_GATE_VERDICT.findall(text)
    if found:
        return found[-1]
    if _BLOCKS_TRUE.search(text):
        return "blocking"
    if _DESIGN_TIME.search(text):
        return "non_blocking"
    return "unparsed"


def _response_text(response: dict[str, Any]) -> str | None:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


def _sse_text(chunks: list[bytes]) -> str:
    """Concatenate ``choices[0].delta.content`` of OpenAI SSE chunks."""
    parts: list[str] = []
    for line in b"".join(chunks).decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            delta = json.loads(line[6:])["choices"][0]["delta"]
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


class FallbackBackend(Backend):
    """See the module docstring."""

    fills_usage_sink: bool = True

    def __init__(
        self,
        *,
        name: str,
        primary: CodexBackend,
        fallback: Backend,
        fallback_name: str,
        mode: str,
        webhook_url: str | None = None,
        check_interval_s: float = CHECK_INTERVAL_S,
        remind_after: timedelta = REMIND_AFTER,
        recover_after: timedelta = RECOVER_AFTER,
        clock: Callable[[], datetime] = _utcnow,
        caller_budget_s: float = LEXORA_BACKEND_TIMEOUT_S,
        slot_wait_s: float = 30.0,
        codex_timeout_s: float = 270.0,
        fallback_floor_s: float = 600.0,
        fallback_timeout_s: float | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in ("fallback", "shadow"):
            raise ValueError(f"fallback backend '{name}': mode must be 'fallback' or 'shadow', not {mode!r}")
        try:
            check_fallback_budget(caller_budget_s, slot_wait_s, codex_timeout_s, fallback_floor_s)
        except ValueError as exc:
            raise ValueError(f"fallback backend '{name}': {exc}") from exc
        self.name = name
        self.primary = primary
        self.fallback = fallback
        self.fallback_name = fallback_name
        self.mode = mode
        self.webhook_url = webhook_url
        self.check_interval_s = check_interval_s
        self.remind_after = remind_after
        self.recover_after = recover_after
        self._clock = clock
        # msg-687 C.
        self.caller_budget_s = caller_budget_s
        self.slot_wait_s = slot_wait_s
        self.codex_timeout_s = codex_timeout_s
        self.fallback_floor_s = fallback_floor_s
        #: Gemini's own configured timeout (the router passes it); ``None``
        #: when unknown, and then only the deadline bounds the call.
        self.fallback_timeout_s = fallback_timeout_s
        self._monotonic = monotonic
        #: Attached by ``main.lifespan`` (``attach_ledger``); None in tests
        #: that do not need counts, and then counts read as unknown.
        self.ledger: CostTracker | None = None
        #: Tier name(s) routed here, set by the router; the shadow rows' tier.
        self.tier_label: str | None = None
        # Fallback / notification state (B-3), in memory (B-7).
        self._fallback_since: datetime | None = None
        self._last_reason: str | None = None
        self._last_hold: datetime | None = None
        self._last_notice_at: datetime | None = None
        self._pending_notice: str | None = None
        #: Bumped each time a state notice (STARTED / ENDED) starts its post.
        #: A post writes ``_pending_notice`` afterwards only if no newer one
        #: started meanwhile. Identity checks on the parked value were ABA-prone
        #: (PR-gate on lexora#78: None -> None let a stale ENDED park).
        self._notice_seq = 0
        # msg-685: the two inputs of ENDED. Codex answers set the flag only.
        self._last_fallback_at: datetime | None = None
        self._answered_since_fallback = False
        self._loop_task: asyncio.Task[None] | None = None
        self._send_tasks: set[asyncio.Task[None]] = set()
        # A-15-2b pre-expiry warning: UTC date of the last EXPIRING notice
        # delivered (or skipped for want of a webhook). In memory (B-7).
        self._expiry_warned_on: date | None = None
        # Shadow state (B-5).
        self._shadow_task: asyncio.Task[None] | None = None
        self.shadow_skipped = 0
        if not webhook_url:
            logger.warning(
                "fallback_webhook_unset",
                backend=name,
                env=WEBHOOK_ENV,
                note="fallback notifications will not be sent",
            )

    def attach_ledger(self, ledger: CostTracker) -> None:
        self.ledger = ledger

    def start(self) -> None:
        """Start the notice loop (called from ``main.lifespan``, inside the
        running event loop). Idempotent; requests also ensure it."""
        self._ensure_loop()

    # ---- helpers --------------------------------------------------------

    def _codex_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """The request as codex gets it: without ``model``, which carries the
        tier's Gemini model (routes resolve it); codex serves its own."""
        return {k: v for k, v in request.items() if k != "model"}

    def _codex_model(self) -> str:
        return self.primary.resolve_model(None)

    def _route_codex(self, answered_by: str = ANSWERED_BY_CODEX) -> None:
        set_answer_route(AnswerRoute(self.primary.name, answered_by, self._codex_model()))

    def _route_gemini(self, answered_by: str | None) -> None:
        # msg-687 D: a fallback answer is recorded under the constant name;
        # the Gemini answer of shadow mode keeps the Gemini backend's name,
        # exactly as a direct naysayer -> gemini row does today (msg-687 E).
        backend = BACKEND_GEMINI_FALLBACK if answered_by == ANSWERED_BY_GEMINI_FALLBACK else self.fallback_name
        set_answer_route(AnswerRoute(backend, answered_by))

    def _deadline(self) -> float:
        """This request's deadline on the monotonic clock (msg-687 C)."""
        return self._monotonic() + self.caller_budget_s

    def fallback_budget(self, deadline: float) -> float:
        """Seconds the fallback call may take: ``min(Gemini's timeout,
        deadline - now)``, never below 0 (msg-687 C)."""
        left = deadline - self._monotonic()
        if self.fallback_timeout_s is not None:
            left = min(self.fallback_timeout_s, left)
        return max(left, 0.0)

    def _deadline_error(self, budget: float) -> FallbackDeadlineExceeded:
        logger.warning("naysayer_fallback_deadline", backend=self.name, budget_s=round(budget, 1))
        return FallbackDeadlineExceeded(
            f"fallback answer did not finish within the request budget ({budget:.0f}s left of "
            f"caller_budget_s={self.caller_budget_s:.0f})"
        )

    async def _bounded_fallback_chat(self, request: dict[str, Any], deadline: float) -> dict[str, Any]:
        budget = self.fallback_budget(deadline)
        try:
            async with asyncio.timeout(budget):
                return await self.fallback.chat_completions(request)
        except TimeoutError as exc:
            raise self._deadline_error(budget) from exc

    async def _bounded_fallback_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None, deadline: float
    ) -> AsyncIterator[bytes]:
        """The Gemini stream, cut when the budget runs out. The timeout
        covers each ``__anext__`` only, never a ``yield``, so it cannot fire
        while the caller is busy with a chunk."""
        budget = self.fallback_budget(deadline)
        ends = self._monotonic() + budget
        stream = self.fallback.chat_completions_stream(request, usage_sink)
        try:
            while True:
                try:
                    async with asyncio.timeout(max(ends - self._monotonic(), 0.0)):
                        chunk = await stream.__anext__()
                except StopAsyncIteration:
                    return
                except TimeoutError as exc:
                    raise self._deadline_error(budget) from exc
                yield chunk
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

    # ---- fallback mode (B-1) ------------------------------------------------

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loop()
        if self.mode == "shadow":
            return await self._shadow_chat(request)
        deadline = self._deadline()
        availability = await self.primary.codex_availability()
        self._last_hold = availability.quota_hold_until
        reason = availability.reason
        if availability.open:
            try:
                response = await self.primary.chat_completions(
                    self._codex_request(request),
                    availability,
                    timeout=self.codex_timeout_s,
                    slot_wait_s=self.slot_wait_s,
                )
            except FALLBACK_ERRORS as exc:
                reason = fallback_reason(exc)
                self._note_quota(exc)
            else:
                self._route_codex()
                self._codex_answered()
                return response
        self._falling_back(reason)
        self._route_gemini(ANSWERED_BY_GEMINI_FALLBACK)
        return await self._bounded_fallback_chat(request, deadline)

    async def chat_completions_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None = None
    ) -> AsyncIterator[bytes]:
        self._ensure_loop()
        if self.mode == "shadow":
            async for chunk in self._shadow_stream(request, usage_sink):
                yield chunk
            return
        deadline = self._deadline()
        availability = await self.primary.codex_availability()
        self._last_hold = availability.quota_hold_until
        reason = availability.reason
        if availability.open:
            codex_stream = self.primary.chat_completions_stream(
                self._codex_request(request),
                usage_sink,
                availability,
                timeout=self.codex_timeout_s,
                slot_wait_s=self.slot_wait_s,
            )
            try:
                # codex finishes its run before the first chunk: every
                # failure surfaces here, while nothing has been sent (B-1).
                first = await codex_stream.__anext__()
            except FALLBACK_ERRORS as exc:
                reason = fallback_reason(exc)
                self._note_quota(exc)
            else:
                self._route_codex()
                self._codex_answered()
                yield first
                # A byte is out: from here on nothing may switch to Gemini.
                async for chunk in codex_stream:
                    yield chunk
                return
        self._falling_back(reason)
        self._route_gemini(ANSWERED_BY_GEMINI_FALLBACK)
        async for chunk in self._bounded_fallback_stream(request, usage_sink, deadline):
            yield chunk

    def _note_quota(self, exc: CodexError) -> None:
        if isinstance(exc, CodexQuotaError) and exc.reset_at is not None:
            self._last_hold = exc.reset_at

    # ---- notifications (B-3) --------------------------------------------------

    def _falling_back(self, reason: str | None) -> None:
        """A request is about to be answered by Gemini as a fallback.
        Shadow mode: nothing (msg-679 / msg-687 F)."""
        if self.mode == "shadow":
            return
        now = self._clock()
        self._last_reason = reason
        self._last_fallback_at = now
        self._answered_since_fallback = False
        if self._fallback_since is None:
            self._fallback_since = now
            self._last_notice_at = now
            logger.warning("naysayer_fallback_started", backend=self.name, reason=reason)
            self._notify(self._notice_text("STARTED"))
        self._ensure_loop()

    def _codex_answered(self) -> None:
        """Codex answered. Only marks the period as recoverable; ENDED is
        ``notice_tick``'s (msg-685). Shadow mode: nothing (msg-679)."""
        if self.mode == "shadow" or self._fallback_since is None:
            return
        self._answered_since_fallback = True

    def _recovered(self, now: datetime) -> bool:
        """The three ENDED conditions of msg-685 / msg-687 F."""
        return (
            self._fallback_since is not None
            and self._answered_since_fallback
            and self._last_fallback_at is not None
            and now - self._last_fallback_at >= self.recover_after
        )

    async def _end_fallback(self) -> None:
        assert self._fallback_since is not None
        text = self._notice_text("ENDED")
        logger.info("naysayer_fallback_ended", backend=self.name, since=self._fallback_since.isoformat())
        self._fallback_since = None
        self._last_notice_at = None
        self._last_fallback_at = None
        self._answered_since_fallback = False
        if self.webhook_url:
            await self._deliver_state(text)

    def _totals(self) -> tuple[int, float] | None:
        if self.ledger is None or self._fallback_since is None:
            return None
        try:
            return self.ledger.fallback_totals(self._fallback_since)
        except Exception as exc:  # noqa: BLE001 - a notice must never raise
            logger.warning("fallback_totals_unreadable", backend=self.name, error=str(exc))
            return None

    def _notice_text(self, event: str) -> str:
        totals = self._totals()
        calls, cost = (str(totals[0]), f"{totals[1]:.4f}") if totals else ("unknown", "unknown")
        since = self._fallback_since.isoformat() if self._fallback_since else "none"
        hold = self._last_hold.isoformat() if self._last_hold else "unknown"
        text = (
            f"[Lexora naysayer] fallback {event}: reason={self._last_reason or 'unknown'} "
            f"fallback_since={since} gemini_calls={calls} gemini_cost_usd={cost} "
            f"quota_hold_until={hold}"
        )
        if self._last_reason == REASON_DATA_CONTROLS_UNVERIFIED:
            text += f" runbook={RUNBOOK_POINTER}"
        return text[:NOTICE_MAX_CHARS]

    def _notify(self, text: str) -> None:
        """Send in the background; a failure leaves it for the next cycle."""
        if not self.webhook_url:
            return
        task = asyncio.get_running_loop().create_task(self._deliver_or_park(text))
        self._send_tasks.add(task)
        task.add_done_callback(self._send_tasks.discard)

    async def _deliver_or_park(self, text: str) -> None:
        if not await self._deliver_state(text):
            self._ensure_loop()

    async def _deliver_state(self, text: str) -> bool:
        """Post a state notice (STARTED / ENDED); the newest state wins.

        A parked notice is an older state, so it is obsolete once this one
        is delivered: success clears it, failure replaces it (PR-gate on
        lexora#76: a parked STARTED must not follow a delivered ENDED).
        Either write happens only if no newer state notice started while
        this post was in flight (``_notice_seq``, a counter, so a parked
        value that changed and changed back still counts as newer)."""
        self._notice_seq += 1
        seq = self._notice_seq
        delivered = await self._post(text)
        if self._notice_seq == seq:
            self._pending_notice = None if delivered else text
        return delivered

    async def _post(self, text: str) -> bool:
        if not self.webhook_url:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.webhook_url, json={"content": text[:NOTICE_MAX_CHARS]})
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - never block a request
            # The URL is a credential: log the error class and status only.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning("fallback_notice_failed", backend=self.name, error=type(exc).__name__, status=status)
            return False
        return True

    def _ensure_loop(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.get_running_loop().create_task(self._notice_loop())

    async def _notice_loop(self) -> None:
        while True:
            await asyncio.sleep(self.check_interval_s)
            await self.notice_tick()

    async def notice_tick(self) -> None:
        """One 10-minute check: retry a parked notice, end the fallback
        period if codex is back (msg-685), else remind if due, then the
        daily data-controls EXPIRING notice if due."""
        if self._pending_notice is not None:
            text = self._pending_notice
            seq = self._notice_seq
            if await self._post(text):
                if self._notice_seq == seq and self._pending_notice == text:
                    self._pending_notice = None
        now = self._clock()
        if self._recovered(now):
            await self._end_fallback()
        elif self._fallback_since is not None and self._last_notice_at is not None:
            if now - self._last_notice_at >= self.remind_after:
                parked = self._pending_notice
                seq = self._notice_seq
                if not self.webhook_url or await self._post(self._notice_text("CONTINUING")):
                    self._last_notice_at = now
                    # CONTINUING restates the open period: a parked notice
                    # (its STARTED) would only arrive after it, out of order.
                    if self.webhook_url and self._notice_seq == seq and self._pending_notice is parked:
                        self._pending_notice = None
        await self._expiry_tick()

    def _expiry_text(self, days_left: int, expires_at: datetime, verified_at: datetime) -> str:
        text = (
            f"[Lexora naysayer] data controls EXPIRING (mode={self.mode}): codex stops in {days_left} day(s) "
            f"at expires_at={expires_at.isoformat()} (verified_at={verified_at.isoformat()}); "
            f"then reason={REASON_DATA_CONTROLS_UNVERIFIED} and Gemini answers. "
            f"Re-verify per runbook={RUNBOOK_POINTER}; no restart needed."
        )
        return text[:NOTICE_MAX_CHARS]

    async def _expiry_tick(self) -> None:
        """A-15-2b: at most one EXPIRING notice per UTC calendar day while
        the record is valid and within 7 days of expiry. Expired / missing
        records are not warned here: codex is closed then, and the fallback
        notices (``reason=data_controls_unverified``) take over."""
        try:
            state = self.primary.data_controls_state()
        except Exception as exc:  # noqa: BLE001 - a notice must never raise
            logger.warning("data_controls_state_failed", backend=self.name, error=type(exc).__name__)
            return
        now = self._clock()
        if not state.warn_due(now) or state.expires_at is None or state.verified_at is None:
            return
        today = now.astimezone(timezone.utc).date()
        if self._expiry_warned_on == today:
            return
        days_left = state.days_left(now)
        logger.warning(
            "codex_data_controls_expiring",
            backend=self.name,
            days_left=days_left,
            expires_at=state.expires_at.isoformat(),
        )
        text = self._expiry_text(days_left, state.expires_at, state.verified_at)
        if not self.webhook_url or await self._post(text):
            self._expiry_warned_on = today

    # ---- shadow mode (B-5) -------------------------------------------------

    def _start_shadow(
        self, request: dict[str, Any]
    ) -> tuple[asyncio.Future[tuple[str | None, float | None]], asyncio.Task[None]] | None:
        if self._shadow_task is not None and not self._shadow_task.done():
            self.shadow_skipped += 1
            logger.info("naysayer_shadow_skipped", backend=self.name, skipped=self.shadow_skipped)
            return None
        loop = asyncio.get_running_loop()
        gemini_done: asyncio.Future[tuple[str | None, float | None]] = loop.create_future()
        task = loop.create_task(self._shadow_run(self._codex_request(request), gemini_done))
        self._shadow_task = task
        return gemini_done, task

    def _cancel_shadow(self, task: asyncio.Task[None] | None) -> None:
        """The caller went away (PR-gate on lexora#78): stop this request's
        shadow run so it does not keep codex's single slot for up to
        ``codex_timeout_s``. Cancelling kills and reaps the codex process and
        gives the slot back (``CodexBackend._execute``). No comparison row is
        written for it: Gemini never finished either, so there is no pair.
        Only a cancel/close calls this -- a Gemini *error* still yields a
        comparison row (gemini_verdict="error"), so that shadow run is kept."""
        if task is not None and not task.done():
            logger.info("naysayer_shadow_cancelled", backend=self.name, reason="caller_gone")
            task.cancel()

    async def _shadow_chat(self, request: dict[str, Any]) -> dict[str, Any]:
        started_shadow = self._start_shadow(request)
        gemini_done, shadow_task = started_shadow if started_shadow is not None else (None, None)
        self._route_gemini(None)
        started = time.monotonic()
        text: str | None = None
        try:
            response = await self.fallback.chat_completions(request)
            text = _response_text(response)
            return response
        except asyncio.CancelledError:
            self._cancel_shadow(shadow_task)
            raise
        # An ordinary Gemini error (5xx, timeout, ReadError) deliberately does
        # NOT cancel the shadow run (PR-gate on lexora#79 considered): its
        # result is not discarded. `finally` hands it (None, None) and it
        # writes a row with gemini_verdict="error" and codex's own verdict --
        # what codex did on a request Gemini failed, which is exactly the
        # evidence the fallback-mode switch is judged on. It holds the slot no
        # longer than any other shadow run (codex_timeout_s).
        finally:
            if gemini_done is not None and not gemini_done.done():
                gemini_done.set_result((text, time.monotonic() - started if text is not None else None))

    async def _shadow_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None
    ) -> AsyncIterator[bytes]:
        started_shadow = self._start_shadow(request)
        gemini_done, shadow_task = started_shadow if started_shadow is not None else (None, None)
        self._route_gemini(None)
        started = time.monotonic()
        chunks: list[bytes] = []
        completed = False
        try:
            async for chunk in self.fallback.chat_completions_stream(request, usage_sink):
                chunks.append(chunk)
                yield chunk
            completed = True
        except (asyncio.CancelledError, GeneratorExit):
            # Cancelled mid-await, or closed at a `yield` (the consumer
            # stopped reading): either way the caller is gone.
            self._cancel_shadow(shadow_task)
            raise
        # A Gemini stream error does not cancel it either: see _shadow_chat.
        finally:
            if gemini_done is not None and not gemini_done.done():
                result = (_sse_text(chunks), time.monotonic() - started) if completed else (None, None)
                gemini_done.set_result(result)

    async def _shadow_run(
        self,
        request: dict[str, Any],
        gemini_done: asyncio.Future[tuple[str | None, float | None]],
    ) -> None:
        # Own context (copied at task creation): stamp the shadow route so
        # this task's ledger row never inherits the foreground's.
        self._route_codex(ANSWERED_BY_CODEX_SHADOW)
        availability = await self.primary.codex_availability()
        if not availability.open:
            # B-5: a closed gate (a quota hold included) means no shadow run.
            # B-5' (msg-498): still write a row -- `not_run` with the gate's
            # reason -- so a latch that stops codex mid-way through the
            # comparison period shows up in `shadow_report` instead of the
            # rows silently ceasing to grow. One row per request, no text.
            logger.info("naysayer_shadow_not_run", backend=self.name, reason=availability.reason)
            gemini_text, gemini_seconds = await gemini_done
            self._write_comparison(
                gemini_text=gemini_text,
                gemini_seconds=gemini_seconds,
                codex_verdict=SHADOW_NOT_RUN,
                codex_reason=availability.reason or "unknown",
                codex_seconds=None,
            )
            return
        started = time.monotonic()
        codex_text: str | None = None
        codex_reason: str | None = None
        try:
            # Same codex limits as fallback mode, so the shadow period
            # measures what production would do (msg-680/681).
            response = await self.primary.chat_completions(
                request, availability, timeout=self.codex_timeout_s, slot_wait_s=self.slot_wait_s
            )
        except CodexError as exc:
            codex_reason = fallback_reason(exc)
        else:
            codex_text = _response_text(response)
            usage = response.get("usage") or {}
            details = usage.get("prompt_tokens_details")
            cached = details.get("cached_tokens") if isinstance(details, dict) else None
            thinking = usage.get("lexora_thinking_tokens")
            if self.ledger is not None:
                self.ledger.record(
                    model=self._codex_model(),
                    endpoint="shadow",
                    tokens_input=int(usage.get("prompt_tokens", 0)),
                    tokens_output=int(usage.get("completion_tokens", 0)),
                    # msg-687 H-2: NULL when the CLI did not report them.
                    tokens_thinking=None if thinking is None else int(thinking),
                    tokens_cached_input=None if cached is None else int(cached),
                    tier=self.tier_label,
                    # Set by the handler before it called this wrapper; the
                    # task copied that context when `_start_shadow` made it.
                    trace_id=current_trace_id(),
                )
        codex_seconds = time.monotonic() - started
        gemini_text, gemini_seconds = await gemini_done
        self._write_comparison(
            gemini_text=gemini_text,
            gemini_seconds=gemini_seconds,
            codex_verdict="error" if codex_reason is not None else extract_verdict(codex_text),
            codex_reason=codex_reason,
            codex_seconds=codex_seconds,
        )

    def _write_comparison(
        self,
        *,
        gemini_text: str | None,
        gemini_seconds: float | None,
        codex_verdict: str,
        codex_reason: str | None,
        codex_seconds: float | None,
    ) -> None:
        if self.ledger is None:
            return
        try:
            self.ledger.record_shadow_comparison(
                tier=self.tier_label,
                gemini_verdict="error" if gemini_text is None else extract_verdict(gemini_text),
                codex_verdict=codex_verdict,
                codex_reason=codex_reason,
                gemini_seconds=gemini_seconds,
                codex_seconds=codex_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - a comparison must never break serving
            logger.warning("shadow_comparison_unwritable", backend=self.name, error=str(exc))

    # ---- status (B-6) --------------------------------------------------------

    def status_fields(self, availability: CodexAvailability) -> dict[str, Any]:
        """The status endpoint's fields outside the ``codex`` block (B-6).

        ``mode`` is what the NEXT request would do, from the same
        ``availability`` the endpoint reports: ``shadow``, else ``codex``
        when codex is open, else ``fallback``. So a closed codex reads
        ``fallback`` before any request has fallen back (B-4's preflight
        reads this). ``fallback_since`` / the counts describe the period
        the notifications describe: from the first fallback answer to the
        ENDED notice (msg-685: codex answered, and no fallback for 15
        minutes).
        """
        if self.mode == "shadow":
            mode = "shadow"
        else:
            mode = "codex" if availability.open else "fallback"
        totals = self._totals()
        return {
            "mode": mode,
            "fallback_since": self._fallback_since.isoformat() if self._fallback_since else None,
            "fallback_calls": totals[0] if totals else None,
            "fallback_cost_usd": totals[1] if totals else None,
            "shadow_skipped": self.shadow_skipped,
        }

    # ---- the rest of the Backend interface -----------------------------------

    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Text completions: codex has none, so Gemini answers (routed as
        a fallback answer in fallback mode, as the primary in shadow mode)."""
        self._route_gemini(ANSWERED_BY_GEMINI_FALLBACK if self.mode == "fallback" else None)
        return await self.fallback.completions(request)

    async def completions_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None = None
    ) -> AsyncIterator[bytes]:
        self._route_gemini(ANSWERED_BY_GEMINI_FALLBACK if self.mode == "fallback" else None)
        async for chunk in self.fallback.completions_stream(request, usage_sink):
            yield chunk

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self.fallback.embeddings(request)

    async def list_models(self) -> dict[str, Any]:
        return {"object": "list", "data": []}

    async def health_check(self) -> bool:
        """Healthy when either side can answer. Starts no model call on the
        codex side; the Gemini side's probe is whatever its backend does."""
        if (await self.primary.codex_availability()).open:
            return True
        return await self.fallback.health_check()

    async def close(self) -> None:
        """Cancel the notice loop, pending posts and a running shadow run.
        The wrapped backends are closed by the router like any other."""
        tasks = [t for t in (self._loop_task, self._shadow_task, *self._send_tasks) if t is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

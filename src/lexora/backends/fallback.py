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

Errors that do NOT fall back and reach the caller: ``CodexAuthError``,
``CodexFailed``, ``CodexUnsupportedInputError`` (not in B-1's list).

**Notifications (B-3), producer declaration.** Intended reader: the human
operator (Takahito). Surface: a Discord webhook named by the
``LEXORA_FALLBACK_WEBHOOK_URL`` environment variable, posted directly by
Lexora -- not a chatroom thread, not mindwire's notifier, not the daily
digest (msg-267 scope 4). If the variable is unset, one WARNING is logged at
construction and nothing is sent; a failed post is logged at WARNING and
retried on the next 10-minute cycle; neither ever blocks a request. Sent
once when fallback starts, every 6 hours while it lasts, once when codex
answers again. Body: fixed format, at most 500 characters, carrying only
the event, the reason, ``fallback_since``, the Gemini calls and their cost
in that period (counted from the ledger), and ``quota_hold_until``.

State (``fallback_since``, the last notification time, ``shadow_skipped``)
is in memory. That is system-wide because B-7 guarantees one process
(``services/process_lock.py``). A restart during fallback re-sends "start";
accepted in msg-448 B-3.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import httpx

from lexora.backends.answer_route import (
    ANSWERED_BY_CODEX,
    ANSWERED_BY_CODEX_SHADOW,
    ANSWERED_BY_GEMINI_FALLBACK,
    AnswerRoute,
    set_answer_route,
)
from lexora.backends.base import Backend, UsageSink
from lexora.backends.codex import (
    CodexAvailability,
    CodexBackend,
    CodexError,
    CodexLaunchError,
    CodexNotVerifiedError,
    CodexQuotaError,
    CodexTimeout,
    CodexToolUseViolation,
    CodexUnverifiableRun,
)
from lexora.utils.logging import get_logger

if TYPE_CHECKING:
    from lexora.services.cost_tracker import CostTracker

logger = get_logger(__name__)

#: Environment variable holding the Discord webhook URL (B-3).
WEBHOOK_ENV = "LEXORA_FALLBACK_WEBHOOK_URL"
NOTICE_MAX_CHARS = 500
CHECK_INTERVAL_S = 600.0
REMIND_AFTER = timedelta(hours=6)

#: The B-1 list: these fall back to Gemini. ``CodexUnverifiableRun`` is a
#: ``CodexToolUseViolation`` (a latch).
FALLBACK_ERRORS: tuple[type[CodexError], ...] = (
    CodexNotVerifiedError,
    CodexQuotaError,
    CodexLaunchError,
    CodexTimeout,
    CodexToolUseViolation,
)


def fallback_reason(exc: CodexError) -> str:
    """Short reason for a codex failure that fell back (status / notice)."""
    if isinstance(exc, CodexNotVerifiedError):
        return exc.reason
    if isinstance(exc, CodexQuotaError):
        return "quota"
    if isinstance(exc, CodexLaunchError):
        return "launch_failed"
    if isinstance(exc, CodexTimeout):
        return "timeout"
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
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        if mode not in ("fallback", "shadow"):
            raise ValueError(f"fallback backend '{name}': mode must be 'fallback' or 'shadow', not {mode!r}")
        self.name = name
        self.primary = primary
        self.fallback = fallback
        self.fallback_name = fallback_name
        self.mode = mode
        self.webhook_url = webhook_url
        self.check_interval_s = check_interval_s
        self.remind_after = remind_after
        self._clock = clock
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
        self._loop_task: asyncio.Task[None] | None = None
        self._send_tasks: set[asyncio.Task[None]] = set()
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
        set_answer_route(AnswerRoute(self.fallback_name, answered_by))

    # ---- fallback mode (B-1) ------------------------------------------------

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "shadow":
            return await self._shadow_chat(request)
        availability = await self.primary.codex_availability()
        self._last_hold = availability.quota_hold_until
        reason = availability.reason
        if availability.open:
            try:
                response = await self.primary.chat_completions(self._codex_request(request), availability)
            except FALLBACK_ERRORS as exc:
                reason = fallback_reason(exc)
                self._note_quota(exc)
            else:
                self._route_codex()
                self._codex_answered()
                return response
        self._falling_back(reason)
        self._route_gemini(ANSWERED_BY_GEMINI_FALLBACK)
        return await self.fallback.chat_completions(request)

    async def chat_completions_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None = None
    ) -> AsyncIterator[bytes]:
        if self.mode == "shadow":
            async for chunk in self._shadow_stream(request, usage_sink):
                yield chunk
            return
        availability = await self.primary.codex_availability()
        self._last_hold = availability.quota_hold_until
        reason = availability.reason
        if availability.open:
            codex_stream = self.primary.chat_completions_stream(
                self._codex_request(request), usage_sink, availability
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
        async for chunk in self.fallback.chat_completions_stream(request, usage_sink):
            yield chunk

    def _note_quota(self, exc: CodexError) -> None:
        if isinstance(exc, CodexQuotaError) and exc.reset_at is not None:
            self._last_hold = exc.reset_at

    # ---- notifications (B-3) --------------------------------------------------

    def _falling_back(self, reason: str | None) -> None:
        self._last_reason = reason
        if self._fallback_since is None:
            now = self._clock()
            self._fallback_since = now
            self._last_notice_at = now
            logger.warning("naysayer_fallback_started", backend=self.name, reason=reason)
            self._notify(self._notice_text("STARTED"))
        self._ensure_loop()

    def _codex_answered(self) -> None:
        if self._fallback_since is None:
            return
        text = self._notice_text("ENDED")
        logger.info("naysayer_fallback_ended", backend=self.name, since=self._fallback_since.isoformat())
        self._fallback_since = None
        self._last_notice_at = None
        self._notify(text)

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
        return text[:NOTICE_MAX_CHARS]

    def _notify(self, text: str) -> None:
        """Send in the background; a failure leaves it for the next cycle."""
        if not self.webhook_url:
            return
        task = asyncio.get_running_loop().create_task(self._deliver_or_park(text))
        self._send_tasks.add(task)
        task.add_done_callback(self._send_tasks.discard)

    async def _deliver_or_park(self, text: str) -> None:
        if not await self._post(text):
            self._pending_notice = text
            self._ensure_loop()

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
        """One 10-minute check: retry a parked notice, then remind if due."""
        if self._pending_notice is not None:
            text = self._pending_notice
            if await self._post(text):
                if self._pending_notice == text:
                    self._pending_notice = None
        if self._fallback_since is None or self._last_notice_at is None:
            return
        now = self._clock()
        if now - self._last_notice_at >= self.remind_after:
            if not self.webhook_url or await self._post(self._notice_text("CONTINUING")):
                self._last_notice_at = now

    # ---- shadow mode (B-5) -------------------------------------------------

    def _start_shadow(self, request: dict[str, Any]) -> asyncio.Future[tuple[str | None, float | None]] | None:
        if self._shadow_task is not None and not self._shadow_task.done():
            self.shadow_skipped += 1
            logger.info("naysayer_shadow_skipped", backend=self.name, skipped=self.shadow_skipped)
            return None
        loop = asyncio.get_running_loop()
        gemini_done: asyncio.Future[tuple[str | None, float | None]] = loop.create_future()
        self._shadow_task = loop.create_task(self._shadow_run(self._codex_request(request), gemini_done))
        return gemini_done

    async def _shadow_chat(self, request: dict[str, Any]) -> dict[str, Any]:
        gemini_done = self._start_shadow(request)
        self._route_gemini(None)
        started = time.monotonic()
        text: str | None = None
        try:
            response = await self.fallback.chat_completions(request)
            text = _response_text(response)
            return response
        finally:
            if gemini_done is not None and not gemini_done.done():
                gemini_done.set_result((text, time.monotonic() - started if text is not None else None))

    async def _shadow_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None
    ) -> AsyncIterator[bytes]:
        gemini_done = self._start_shadow(request)
        self._route_gemini(None)
        started = time.monotonic()
        chunks: list[bytes] = []
        completed = False
        try:
            async for chunk in self.fallback.chat_completions_stream(request, usage_sink):
                chunks.append(chunk)
                yield chunk
            completed = True
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
            logger.info("naysayer_shadow_not_run", backend=self.name, reason=availability.reason)
            return
        started = time.monotonic()
        codex_text: str | None = None
        codex_reason: str | None = None
        try:
            response = await self.primary.chat_completions(request, availability)
        except CodexError as exc:
            codex_reason = fallback_reason(exc)
        else:
            codex_text = _response_text(response)
            usage = response.get("usage") or {}
            if self.ledger is not None:
                self.ledger.record(
                    model=self._codex_model(),
                    endpoint="shadow",
                    tokens_input=int(usage.get("prompt_tokens", 0)),
                    tokens_output=int(usage.get("completion_tokens", 0)),
                    tier=self.tier_label,
                )
        codex_seconds = time.monotonic() - started
        gemini_text, gemini_seconds = await gemini_done
        if self.ledger is None:
            return
        try:
            self.ledger.record_shadow_comparison(
                tier=self.tier_label,
                gemini_verdict="error" if gemini_text is None else extract_verdict(gemini_text),
                codex_verdict="error" if codex_reason is not None else extract_verdict(codex_text),
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
        next codex answer.
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

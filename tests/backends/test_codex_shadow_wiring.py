"""T-naysayer-codex-backend msg-687 A-H: concurrency 1, ``slot_hold``, the
time budget, the constant ``gemini-fallback`` row, shadow wiring, the
two-state ENDED debounce, the model pin and codex's cached / thinking
tokens. Codex is the fake CLI from ``test_codex``; Gemini is
``test_fallback.FakeGemini``."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lexora.api.routes import _ledger_token_extras as thinking_cached
from lexora.backends.answer_route import current_answer_route
from lexora.backends.codex import (
    CODEX_MODEL_PIN,
    CodexAuthError,
    CodexBackend,
    CodexLaunchError,
    CodexSlotTimeout,
    CodexUsage,
    usage_from_events,
)
from lexora.backends.codex_verification import CodexStateStore
from lexora.backends.factory import create_backend
from lexora.backends.fallback import (
    FALLBACK_ERRORS,
    SHADOW_NOT_RUN,
    FallbackBackend,
    FallbackDeadlineExceeded,
    fallback_reason,
)
from lexora.config import BackendSettings, CodexSettings, FallbackSettings
from lexora.services.cost_tracker import CostTracker
from tests.backends.test_codex import make_backend, record_pass
from tests.backends.test_fallback import (  # noqa: F401 - fixture
    GEMINI_MODEL,
    GEMINI_TEXT,
    REQUEST,
    Clock,
    FakeGemini,
    _capture_posts,
    _drain,
    _sends,
    _text,
    wrappers,
)

SHORT_WAIT = 0.2


class Mono:
    """A settable monotonic clock for the deadline arithmetic."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _build(
    wrappers: list[FallbackBackend],
    tmp_path: Path,
    scenario: str = "ok",
    *,
    mode: str = "fallback",
    verified: bool = True,
    gemini: FakeGemini | None = None,
    fallback_name: str = "gemini",
    clock: Clock | None = None,
    **kwargs: Any,
) -> FallbackBackend:
    codex = make_backend(tmp_path, scenario)
    if verified:
        record_pass(codex)
    kwargs.setdefault("slot_wait_s", SHORT_WAIT)
    kwargs.setdefault("codex_timeout_s", 30.0)
    w = FallbackBackend(
        name="naysayer-codex",
        primary=codex,
        fallback=gemini or FakeGemini(),
        fallback_name=fallback_name,
        mode=mode,
        webhook_url="https://example.invalid/hook",
        clock=clock or Clock(),
        **kwargs,
    )
    w.tier_label = "naysayer"
    w.attach_ledger(CostTracker(tmp_path / "costs.db"))
    wrappers.append(w)
    return w


def _comparisons(db: Path) -> list[tuple[str, str | None]]:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT codex_verdict, codex_reason FROM shadow_comparisons ORDER BY id").fetchall()


def _cost_rows(db: Path) -> list[sqlite3.Row]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM request_costs ORDER BY id").fetchall()


async def _shadow_settled(w: FallbackBackend) -> None:
    if w._shadow_task is not None:
        await w._shadow_task


async def _take_slot(backend: CodexBackend) -> None:
    """Stand in for a run that holds the only slot."""
    await backend._semaphore.acquire()


async def _slot_timeout(backend: CodexBackend) -> None:
    """One request that gives up waiting while the slot is taken."""
    with pytest.raises(CodexSlotTimeout):
        await backend.chat_completions(REQUEST, slot_wait_s=SHORT_WAIT)


# --------------------------------------------------------------------------
# A: concurrency fixed at 1
# --------------------------------------------------------------------------


class TestConcurrencyIsOne:
    def test_config_default_is_one_and_two_is_refused(self) -> None:
        assert CodexSettings(codex_home="/srv/codex").max_concurrency == 1
        with pytest.raises(ValidationError):
            CodexSettings(codex_home="/srv/codex", max_concurrency=2)

    def test_config_with_two_does_not_start(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            BackendSettings(
                type="codex",
                models=[CODEX_MODEL_PIN.model],
                codex={"codex_home": str(tmp_path / "h"), "max_concurrency": 2},
            )

    def test_direct_construction_with_two_raises(self, tmp_path: Path) -> None:
        store = CodexStateStore(tmp_path / "s.db")
        assert CodexBackend(codex_home="/srv/codex", state_store=store)._semaphore._value == 1
        with pytest.raises(ValueError, match="max_concurrency must be 1"):
            CodexBackend(codex_home="/srv/codex", state_store=store, max_concurrency=2)


# --------------------------------------------------------------------------
# B: slot wait and slot_hold
# --------------------------------------------------------------------------


class TestSlotHold:
    async def test_slot_timeout_sets_slot_hold_and_never_latches(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        await _take_slot(backend)
        await _slot_timeout(backend)
        availability = await backend.codex_availability()
        assert availability.reason == "slot_hold"
        assert isinstance(availability.error, CodexSlotTimeout)
        # Codex never ran: the run is finished, nothing latched.
        assert backend.state_store.unfinished_runs() == []
        assert backend.state_store.uncleared_violations() == []
        assert backend.inflight_runs() == 0

    async def test_no_hold_when_the_slot_is_free_at_the_timeout(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        await _take_slot(backend)
        backend._semaphore.release()  # the holder let go just in time
        resp = await backend.chat_completions(REQUEST, slot_wait_s=SHORT_WAIT)
        assert _text(resp).startswith("REVIEW: ")
        assert backend._slot_hold is None

    def test_slot_timeout_falls_back_with_its_own_reason(self) -> None:
        assert CodexSlotTimeout in FALLBACK_ERRORS
        assert fallback_reason(CodexSlotTimeout("x")) == "slot_wait_timeout"

    @pytest.mark.parametrize("mode", ["fallback", "shadow"])
    async def test_both_modes_read_the_same_hold(self, tmp_path: Path, wrappers: list, mode: str) -> None:
        w = _build(wrappers, tmp_path, mode=mode)
        posted = _capture_posts(w)
        await _take_slot(w.primary)
        await _slot_timeout(w.primary)
        assert (await w.primary.codex_availability()).reason == "slot_hold"
        started = time.monotonic()
        assert _text(await w.chat_completions(REQUEST)) == GEMINI_TEXT
        assert time.monotonic() - started < SHORT_WAIT  # no second wait: the hold fast-fails
        await _shadow_settled(w)
        await _sends(w)
        if mode == "shadow":
            assert _comparisons(tmp_path / "costs.db") == [(SHADOW_NOT_RUN, "slot_hold")]
            assert posted == []
        else:
            assert w._last_reason == "slot_hold"
            assert [p.split(":")[0] for p in posted] == ["[Lexora naysayer] fallback STARTED"]

    @pytest.mark.parametrize("mode", ["fallback", "shadow"])
    async def test_hold_clears_when_the_running_run_ends(self, tmp_path: Path, wrappers: list, mode: str) -> None:
        w = _build(wrappers, tmp_path, "slow_ok", mode=mode)
        holder = asyncio.create_task(w.primary.chat_completions(REQUEST))
        while not w.primary._semaphore.locked():
            await asyncio.sleep(0.01)
        await _slot_timeout(w.primary)
        assert (await w.primary.codex_availability()).reason == "slot_hold"
        await holder
        assert w.primary._slot_hold is None
        assert (await w.primary.codex_availability()).open
        # The next request uses codex again, in both modes.
        (Path(w.primary.codex_home) / "slow_s").write_text("0")
        resp = await w.chat_completions(REQUEST)
        await _shadow_settled(w)
        if mode == "fallback":
            assert _text(resp).startswith("REVIEW: ")
        else:
            assert _comparisons(tmp_path / "costs.db")[-1][0] != SHADOW_NOT_RUN

    async def test_hold_clears_when_the_running_run_times_out(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "sleep")
        record_pass(backend)
        holder = asyncio.create_task(backend.chat_completions(REQUEST, timeout=1.0))
        while not backend._semaphore.locked():
            await asyncio.sleep(0.01)
        await _slot_timeout(backend)
        assert backend._slot_hold is not None
        with pytest.raises(Exception):
            await holder
        assert backend._slot_hold is None

    async def test_hold_clears_when_the_running_run_raises(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        backend._wrap = lambda inner, workdir: [str(tmp_path / "no-such-bwrap")]  # type: ignore[method-assign]
        availability = await backend.codex_availability()
        assert availability.open
        # A hold set by a waiter while this run holds the slot; the run then
        # fails at the spawn, and its `finally` still clears the hold.
        backend._slot_hold = (time.monotonic(), 330.0)
        with pytest.raises(CodexLaunchError):
            await backend.chat_completions(REQUEST, availability)
        assert backend._slot_hold is None

    async def test_stale_hold_is_ignored_with_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import lexora.backends.codex as codex_module

        warned: list[str] = []

        class Recorder:
            def warning(self, event: str, **_kw: Any) -> None:
                warned.append(event)

        monkeypatch.setattr(codex_module, "logger", Recorder())
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        backend._slot_hold = (time.monotonic() - 331.0, 330.0)  # codex_timeout_s + slot_wait_s
        assert (await backend.codex_availability()).open
        assert warned == ["codex_slot_hold_stale_ignored"]
        backend._slot_hold = (time.monotonic() - 329.0, 330.0)
        assert (await backend.codex_availability()).reason == "slot_hold"

    async def test_wrapper_sets_the_bound_from_its_limits(self, tmp_path: Path, wrappers: list) -> None:
        w = _build(wrappers, tmp_path, codex_timeout_s=300.0, slot_wait_s=SHORT_WAIT)
        await _take_slot(w.primary)
        await w.chat_completions(REQUEST)
        assert w.primary._slot_hold is not None
        assert w.primary._slot_hold[1] == pytest.approx(300.0 + SHORT_WAIT)


# --------------------------------------------------------------------------
# C: the time budget
# --------------------------------------------------------------------------


class TestBudget:
    def test_defaults_are_the_agreed_values(self) -> None:
        s = FallbackSettings(primary="codex", fallback="gemini")
        assert (s.caller_budget_s, s.slot_wait_s, s.codex_timeout_s, s.fallback_floor_s) == (930, 30, 300, 600)

    @pytest.mark.parametrize(
        ("values", "ok"),
        [
            ({}, True),
            ({"caller_budget_s": 930, "slot_wait_s": 30, "codex_timeout_s": 300, "fallback_floor_s": 600}, True),
            ({"caller_budget_s": 929}, False),
            ({"codex_timeout_s": 301}, False),
            ({"fallback_floor_s": 900}, False),
            ({"caller_budget_s": 2000, "codex_timeout_s": 600, "fallback_floor_s": 900}, True),
        ],
    )
    def test_inequality_is_checked_at_load(self, values: dict[str, float], ok: bool) -> None:
        if ok:
            FallbackSettings(primary="codex", fallback="gemini", **values)
        else:
            with pytest.raises(ValidationError, match="fallback budget does not hold"):
                FallbackSettings(primary="codex", fallback="gemini", **values)

    def test_wrapper_refuses_a_broken_budget(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="fallback budget does not hold"):
            FallbackBackend(
                name="w",
                primary=make_backend(tmp_path),
                fallback=FakeGemini(),
                fallback_name="gemini",
                mode="fallback",
                codex_timeout_s=400.0,
            )

    async def test_gemini_keeps_nearly_its_whole_timeout_after_a_quick_codex_failure(
        self, tmp_path: Path, wrappers: list
    ) -> None:
        mono = Mono()
        w = _build(wrappers, tmp_path, verified=False, fallback_timeout_s=900.0, monotonic=mono)
        budgets: list[float] = []
        real = w.fallback_budget

        def spy(deadline: float) -> float:
            mono.now += 0.5  # codex's quick failure took half a second
            budgets.append(real(deadline))
            return budgets[-1]

        w.fallback_budget = spy  # type: ignore[method-assign]
        assert _text(await w.chat_completions(REQUEST)) == GEMINI_TEXT
        assert budgets == [900.0]  # min(900, 930 - 0.5)

    def test_budget_after_codex_used_its_whole_share_is_the_floor(self, tmp_path: Path) -> None:
        mono = Mono()
        w = FallbackBackend(
            name="w",
            primary=make_backend(tmp_path),
            fallback=FakeGemini(),
            fallback_name="gemini",
            mode="fallback",
            fallback_timeout_s=900.0,
            monotonic=mono,
        )
        deadline = w._deadline()
        mono.now += 30.0 + 300.0
        assert w.fallback_budget(deadline) == pytest.approx(600.0)
        mono.now += 1000.0
        assert w.fallback_budget(deadline) == 0.0

    async def test_fallback_is_cut_at_the_deadline(self, tmp_path: Path, wrappers: list) -> None:
        w = _build(wrappers, tmp_path, verified=False, gemini=FakeGemini(delay=5.0), fallback_timeout_s=0.2)
        with pytest.raises(FallbackDeadlineExceeded):
            await w.chat_completions(REQUEST)

    async def test_fallback_stream_is_cut_at_the_deadline(self, tmp_path: Path, wrappers: list) -> None:
        w = _build(wrappers, tmp_path, verified=False, gemini=FakeGemini(delay=5.0), fallback_timeout_s=0.2)
        with pytest.raises(FallbackDeadlineExceeded):
            await _drain(w.chat_completions_stream(REQUEST))

    async def test_fallback_stream_within_budget_is_whole(self, tmp_path: Path, wrappers: list) -> None:
        w = _build(wrappers, tmp_path, verified=False, fallback_timeout_s=5.0)
        chunks = await _drain(w.chat_completions_stream(REQUEST))
        assert chunks[-1] == b"data: [DONE]\n\n"

    async def test_shadow_gemini_is_not_cut(self, tmp_path: Path, wrappers: list) -> None:
        """Shadow mode's Gemini answer is today's naysayer call (msg-687 E)."""
        w = _build(wrappers, tmp_path, mode="shadow", gemini=FakeGemini(delay=0.4), fallback_timeout_s=0.1)
        assert _text(await w.chat_completions(REQUEST)) == GEMINI_TEXT

    async def test_codex_runs_with_the_wrapper_limits(self, tmp_path: Path, wrappers: list) -> None:
        w = _build(wrappers, tmp_path, codex_timeout_s=123.0, slot_wait_s=7.0)
        seen: dict[str, Any] = {}

        async def spy(request: dict[str, Any], availability: Any = None, **kw: Any) -> dict[str, Any]:
            seen.update(kw)
            raise CodexAuthError("x")

        w.primary.chat_completions = spy  # type: ignore[method-assign]
        await w.chat_completions(REQUEST)
        assert seen == {"timeout": 123.0, "slot_wait_s": 7.0}


# --------------------------------------------------------------------------
# D / E: ledger rows
# --------------------------------------------------------------------------


class TestLedgerRows:
    async def test_fallback_row_is_gemini_fallback_whatever_the_setting(self, tmp_path: Path, wrappers: list) -> None:
        w = _build(wrappers, tmp_path, verified=False, fallback_name="gemini-paid-2")
        await w.chat_completions(REQUEST)
        route = current_answer_route()
        assert route is not None and (route.backend, route.answered_by) == ("gemini-fallback", "gemini-fallback")
        w.ledger.record(model=GEMINI_MODEL, endpoint="/v1/chat/completions", tokens_input=10, tokens_output=1, backend="naysayer-codex", tier="naysayer")
        row = _cost_rows(tmp_path / "costs.db")[-1]
        assert (row["backend"], row["answered_by"]) == ("gemini-fallback", "gemini-fallback")

    async def test_shadow_gemini_row_equals_todays_naysayer_row(self, tmp_path: Path, wrappers: list) -> None:
        """mindwire still attests ``expected=gemini`` against this row."""
        today = CostTracker(tmp_path / "today.db")
        today.record(model=GEMINI_MODEL, endpoint="/v1/chat/completions", tokens_input=100, tokens_output=10, backend="gemini", tier="naysayer")
        w = _build(wrappers, tmp_path, mode="shadow", verified=False)

        async def handler() -> None:  # a request's own context, as in the routes
            await w.chat_completions(REQUEST)
            w.ledger.record(model=GEMINI_MODEL, endpoint="/v1/chat/completions", tokens_input=100, tokens_output=10, backend="naysayer-codex", tier="naysayer")

        await asyncio.create_task(handler())
        await _shadow_settled(w)
        keys = ("backend", "answered_by", "model", "tier", "cost_usd", "pricing_known", "tokens_input", "tokens_output")
        [before] = _cost_rows(tmp_path / "today.db")
        after = [r for r in _cost_rows(tmp_path / "costs.db") if r["endpoint"] != "shadow"]
        assert [tuple(r[k] for k in keys) for r in after] == [tuple(before[k] for k in keys)]
        assert (before["backend"], before["answered_by"]) == ("gemini", None)


# --------------------------------------------------------------------------
# F: notices
# --------------------------------------------------------------------------


async def _quota_with_reset(w: FallbackBackend) -> None:
    w.primary._scenario = "quota"  # type: ignore[attr-defined]


async def _auth(w: FallbackBackend) -> None:
    async def fails(*_a: Any, **_k: Any) -> Any:
        raise CodexAuthError("not logged in")

    w.primary.chat_completions = fails  # type: ignore[method-assign]


async def _gate_closed(w: FallbackBackend) -> None:
    w.primary.data_controls = None  # no record: closed as data_controls_unverified


async def _slot(w: FallbackBackend) -> None:
    await _take_slot(w.primary)


FOUR_WAYS = [
    pytest.param(_slot, "slot_wait_timeout", id="slot-wait-timeout"),
    pytest.param(_quota_with_reset, "quota", id="quota-with-reset"),
    pytest.param(_auth, "auth", id="auth"),
    pytest.param(_gate_closed, "data_controls_unverified", id="gate-closed"),
]


class TestShadowSendsNothing:
    @pytest.mark.parametrize(("setup", "reason"), FOUR_WAYS)
    async def test_shadow_posts_nothing_and_records_the_reason(
        self, tmp_path: Path, wrappers: list, setup: Any, reason: str
    ) -> None:
        w = _build(wrappers, tmp_path, mode="shadow")
        posted = _capture_posts(w)
        await setup(w)
        assert _text(await w.chat_completions(REQUEST)) == GEMINI_TEXT
        await _shadow_settled(w)
        await w.notice_tick()
        await _sends(w)
        assert posted == []
        assert w._fallback_since is None and w._last_fallback_at is None
        [(verdict, codex_reason)] = _comparisons(tmp_path / "costs.db")
        assert codex_reason == reason
        assert verdict in ("error", SHADOW_NOT_RUN)

    @pytest.mark.parametrize(("setup", "reason"), FOUR_WAYS)
    async def test_fallback_mode_starts_once(self, tmp_path: Path, wrappers: list, setup: Any, reason: str) -> None:
        w = _build(wrappers, tmp_path)
        posted = _capture_posts(w)
        await setup(w)
        for _ in range(3):
            assert _text(await w.chat_completions(REQUEST)) == GEMINI_TEXT
        await _sends(w)
        assert [p.split(":")[0] for p in posted] == ["[Lexora naysayer] fallback STARTED"]
        assert f"reason={reason} " in posted[0]

    def test_guards_touch_no_state_in_shadow(self, tmp_path: Path) -> None:
        w = FallbackBackend(
            name="w", primary=make_backend(tmp_path), fallback=FakeGemini(), fallback_name="gemini", mode="shadow"
        )
        w._falling_back("quota")
        w._codex_answered()
        assert (w._fallback_since, w._last_fallback_at, w._answered_since_fallback, w._last_reason) == (
            None,
            None,
            False,
            None,
        )


class Scripted:
    """Codex that fails with a slot timeout or answers, as scripted."""

    def __init__(self, w: FallbackBackend) -> None:
        self.fail = False
        real = w.primary.chat_completions

        async def run(request: dict[str, Any], availability: Any = None, **kw: Any) -> dict[str, Any]:
            if self.fail:
                raise CodexSlotTimeout("busy")
            return await real(request, availability, **kw)

        w.primary.chat_completions = run  # type: ignore[method-assign]


class TestEndedDebounce:
    def _events(self, posted: list[str]) -> list[str]:
        return [p.split(":")[0].removeprefix("[Lexora naysayer] fallback ") for p in posted]

    async def _setup(self, tmp_path: Path, wrappers: list) -> tuple[FallbackBackend, list[str], Clock, Scripted]:
        clock = Clock()
        w = _build(wrappers, tmp_path, clock=clock)
        posted = _capture_posts(w)
        return w, posted, clock, Scripted(w)

    async def test_steady_codex_answers_end_the_period_once(self, tmp_path: Path, wrappers: list) -> None:
        w, posted, clock, codex = await self._setup(tmp_path, wrappers)
        codex.fail = True
        await w.chat_completions(REQUEST)
        codex.fail = False
        t0 = clock.now
        for minute in range(1, 31):
            clock.now = t0 + timedelta(minutes=minute)
            assert _text(await w.chat_completions(REQUEST)).startswith("REVIEW: ")
            await w.notice_tick()
            await _sends(w)
            expected = ["STARTED"] if minute < 15 else ["STARTED", "ENDED"]
            assert self._events(posted) == expected, minute
        for hour in range(1, 8):
            clock.now = t0 + timedelta(hours=hour, minutes=31)
            await w.notice_tick()
        assert self._events(posted) == ["STARTED", "ENDED"]  # no CONTINUING after the end
        assert w._fallback_since is None

    async def test_no_requests_never_ends_and_keeps_reminding(self, tmp_path: Path, wrappers: list) -> None:
        w, posted, clock, codex = await self._setup(tmp_path, wrappers)
        codex.fail = True
        await w.chat_completions(REQUEST)
        await _sends(w)
        t0 = clock.now
        for tick in range(1, 43):  # 7 hours of 10-minute ticks
            clock.now = t0 + timedelta(minutes=10 * tick)
            await w.notice_tick()
        events = self._events(posted)
        assert "ENDED" not in events
        assert events[0] == "STARTED" and events.count("CONTINUING") == 1
        assert w._fallback_since is not None

    async def test_alternating_sends_one_start_then_one_end(self, tmp_path: Path, wrappers: list) -> None:
        w, posted, clock, codex = await self._setup(tmp_path, wrappers)
        t0 = clock.now
        for minute in range(20):
            clock.now = t0 + timedelta(minutes=minute)
            codex.fail = minute % 2 == 0
            await w.chat_completions(REQUEST)
            await w.notice_tick()
            await _sends(w)
        assert self._events(posted) == ["STARTED"]
        last_fallback = t0 + timedelta(minutes=18)
        clock.now = last_fallback + timedelta(minutes=14, seconds=59)
        await w.notice_tick()
        assert self._events(posted) == ["STARTED"]
        clock.now = last_fallback + timedelta(minutes=15)
        await w.notice_tick()
        assert self._events(posted) == ["STARTED", "ENDED"]

    async def test_boundary_is_fifteen_minutes(self, tmp_path: Path, wrappers: list) -> None:
        w, posted, clock, codex = await self._setup(tmp_path, wrappers)
        codex.fail = True
        await w.chat_completions(REQUEST)
        fell_back = clock.now
        codex.fail = False
        await w.chat_completions(REQUEST)
        clock.now = fell_back + timedelta(minutes=14, seconds=59)
        await w.notice_tick()
        assert self._events(posted) == ["STARTED"]
        clock.now = fell_back + timedelta(minutes=15)
        await w.notice_tick()
        assert self._events(posted) == ["STARTED", "ENDED"]

    async def test_a_fallback_inside_the_window_restarts_the_wait_without_a_new_start(
        self, tmp_path: Path, wrappers: list
    ) -> None:
        w, posted, clock, codex = await self._setup(tmp_path, wrappers)
        codex.fail = True
        await w.chat_completions(REQUEST)
        since = w._fallback_since
        codex.fail = False
        await w.chat_completions(REQUEST)
        clock.now += timedelta(minutes=10)
        codex.fail = True
        await w.chat_completions(REQUEST)
        clock.now += timedelta(minutes=10)
        await w.notice_tick()
        await _sends(w)
        assert self._events(posted) == ["STARTED"]
        assert w._fallback_since == since and not w._answered_since_fallback


class TestExpiringNamesTheMode:
    @pytest.mark.parametrize("mode", ["fallback", "shadow"])
    def test_text_carries_mode(self, tmp_path: Path, mode: str) -> None:
        w = FallbackBackend(
            name="w", primary=make_backend(tmp_path), fallback=FakeGemini(), fallback_name="gemini", mode=mode
        )
        clock = Clock()
        text = w._expiry_text(3, clock.now, clock.now)
        assert text.startswith(f"[Lexora naysayer] data controls EXPIRING (mode={mode}): ")


# --------------------------------------------------------------------------
# G: the model pin
# --------------------------------------------------------------------------


class TestModelPin:
    def test_pin_values(self) -> None:
        assert CODEX_MODEL_PIN.model == "gpt-6.1-sol"
        assert (CODEX_MODEL_PIN.context_window, CODEX_MODEL_PIN.max_context_window) == (272000, 872000)
        assert CODEX_MODEL_PIN.max_output_tokens is None  # not in the catalog; not guessed

    def _settings(self, tmp_path: Path, model: str) -> BackendSettings:
        return BackendSettings(
            type="codex",
            models=[model],
            codex={"codex_home": str(tmp_path / "home"), "state_db_path": str(tmp_path / "codex.db")},
        )

    def test_factory_accepts_the_pinned_model(self, tmp_path: Path) -> None:
        assert isinstance(create_backend("codex", self._settings(tmp_path, CODEX_MODEL_PIN.model)), CodexBackend)

    def test_factory_refuses_another_model(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is not the pinned model"):
            create_backend("codex", self._settings(tmp_path, "gpt-6-astra"))


# --------------------------------------------------------------------------
# H-2: cached and thinking tokens
# --------------------------------------------------------------------------


class TestCodexTokens:
    def test_reasoning_is_taken_out_of_completion(self) -> None:
        events = [
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 300, "reasoning_output_tokens": 250},
            }
        ]
        assert usage_from_events(events) == CodexUsage(1000, 50, 400, 250)

    def test_missing_fields_stay_none(self) -> None:
        events = [{"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}]
        assert usage_from_events(events) == CodexUsage(5, 2, None, None)

    async def test_response_and_ledger_kwargs(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok_reasoning")
        record_pass(backend)
        resp = await backend.chat_completions(REQUEST)
        assert resp["usage"]["completion_tokens"] == 50
        assert thinking_cached(resp["usage"]) == {"tokens_thinking": 250, "tokens_cached_input": 400}

    async def test_unreported_thinking_is_null_not_zero(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        resp = await backend.chat_completions(REQUEST)
        assert thinking_cached(resp["usage"]) == {"tokens_thinking": None, "tokens_cached_input": 20}

    async def test_stream_fills_the_sink(self, tmp_path: Path) -> None:
        from lexora.backends.base import UsageSink

        backend = make_backend(tmp_path, "ok_reasoning")
        record_pass(backend)
        sink = UsageSink()
        await _drain(backend.chat_completions_stream(REQUEST, usage_sink=sink))
        assert (sink.prompt_tokens, sink.completion_tokens, sink.thinking_tokens, sink.cached_input_tokens) == (1000, 50, 250, 400)

    async def test_shadow_row_carries_both(self, tmp_path: Path, wrappers: list) -> None:
        w = _build(wrappers, tmp_path, "ok_reasoning", mode="shadow")
        await w.chat_completions(REQUEST)
        await _shadow_settled(w)
        [row] = [r for r in _cost_rows(tmp_path / "costs.db") if r["endpoint"] == "shadow"]
        assert (row["tokens_input"], row["tokens_output"], row["tokens_thinking"], row["tokens_cached_input"]) == (1000, 50, 250, 400)
        assert (row["answered_by"], row["cost_usd"]) == ("codex-shadow", 0.0)

    def test_gemini_rows_are_unchanged(self) -> None:
        gemini_usage = {"prompt_tokens": 9, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 3}, "lexora_thinking_tokens": 4}
        assert thinking_cached(gemini_usage) == {"tokens_thinking": 4, "tokens_cached_input": 3}
        assert thinking_cached({"lexora_thinking_tokens": 0}) == {"tokens_thinking": 0, "tokens_cached_input": 0}
        assert thinking_cached({"prompt_tokens": 1}) == {"tokens_thinking": None, "tokens_cached_input": None}

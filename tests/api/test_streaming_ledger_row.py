"""T-streaming-ledger-row PR-A: a streamed request opens a ledger row.

R-12 closed the four non-streaming handlers that computed token counts and
dropped them. The three streaming handlers were left out over a reason written
into `get_cost_tracker`'s docstring, and the reason was wrong: it said a count
there would have to be read back out of relayed SSE bytes. Only two of the five
backends relay. `anthropic`, `claude_code` and `gemini` decode the upstream
themselves and *build* the OpenAI frames from dicts they have already parsed,
so in those the number is a live Python object one scope from the `yield`.
There was no parser to place.

The shipped shape, and what fences each half: the count is produced inside
`backend.*_stream` (`test_anthropic_stream_usage.py` measures that half); it
travels in a per-request `UsageSink` passed into the call, never on the backend
instance, which is cached and shared; and the row is opened inside
`stream_generator` in a `finally`, under the same
`tokens_input > 0 or tokens_output > 0` guard the six non-streaming sites use.

This file measures the router half, at the route, for the reason
`test_ledger_coverage.py` gives: R-11 shipped five green unit detectors over a
token parser while the ledger stayed wrong, because nothing asserted the parser
was reached. The requirement is "a row is opened", so that is what is driven.

The fake backend fills the sink itself rather than being an `anthropic`. The
router must not know any upstream format, so reaching for a real backend here
would assert the opposite of the design.

That last clause used to read "and these detectors keep working when PR-B adds
`gemini` and `claude_code`" -- a prediction. PR-B has landed both, and the
prediction is now a measurement: this file's 16 cases stayed 16 green under
every single-site mutation in `test_gemini_stream_usage.py`'s and
`test_claude_code_stream_usage.py`'s tables, and not one line of this file or
of `stream_generator` changed to admit either backend. That is the design
property -- the router learned no upstream format -- observed rather than
asserted.

Mutation, so the detectors are measured rather than asserted. Counts are this
file alone, 16 cases, run on win32 / CPython 3.12.

Red before green, measured by reverting `src/` alone to `c8a97dc` and leaving
every test file in place: **10 red / 6 green here, and 617 passed across the
rest of the suite.** Red = the three `test_streamed_request_records_exactly
_one_row` cases, the three `test_duration_tracks_the_call_and_not_a_constant`
cases, the two `TestThePassthroughBranchIsWiredToo` cases,
`test_an_error_mid_stream_still_bills_what_arrived` and
`test_exactly_one_row_per_streamed_request`. Green against `develop`, and
therefore fences rather than detectors, = all three
`TestNothingIsBilledThatWasNotObserved` cases and all three
`TestExistingBehaviourIsUnchanged` cases -- a streamed request opened no row
before this change either, so "no row when nothing was observed" was already
true and only has to stay true. The 617 is the point of that run: the suite
was blind to a whole class of unbilled traffic.

Each row below is a single-site edit against the finished tree, restored
before the next. The backend file (`tests/backends/test_anthropic_stream_usage
.py`, 7 cases) was run under every one of them and stayed 7 green throughout,
which is what shows these detectors are measuring the router's wiring and not
the backend's parsing:

- Dropping `usage_sink=usage_sink` from the *non*-passthrough call in
  `chat_completions` only: 4 red / 12 green -- both `[chat_completions]`
  cases, `test_an_error_mid_stream_still_bills_what_arrived` and
  `test_exactly_one_row_per_streamed_request`, which all drive that branch.
  `[completions]` and `[messages]` stay green, so the three handlers are three
  detectors and not one assertion counted three times; both passthrough cases
  stay green, so the two call sites inside a single handler are separated too.
- Dropping `usage_sink=usage_sink` from the *passthrough* pre-flight call in
  `chat_completions` only: 1 red / 15 green -- `[chat_completions]` of
  `TestThePassthroughBranchIsWiredToo`, and nothing else. The exact mirror of
  the row above, and the reason both branches are driven: one call site each,
  and either can be missed on its own while the other stays green.
- Deleting the whole `finally:` block from `messages`: 2 red / 14 green --
  both `[messages]` cases. Bound to that handler's wiring, not to a block
  existing somewhere in the file.
- Weakening the guard to a bare `if cost_tracker:` at all three sites:
  3 red / 13 green -- exactly the three
  `TestNothingIsBilledThatWasNotObserved` cases, and not one detector moves.
  That disjointness is what shows the guard fence reaches the guard and only
  the guard.
- Replacing `user_id=` with a constant at all three sites: 3 red / 13 green,
  the three `test_streamed_request_records_exactly_one_row` cases. R-13's
  lesson applied forward: the column that says whose bill a row lands on is
  asserted in the same commit that first writes it, not two rounds later.
- Replacing `duration=` with `0.0` at all three sites: 3 red / 13 green, the
  three `test_duration_tracks_the_call_and_not_a_constant` cases.
- Replacing `duration=` with `0.05` at all three sites -- i.e. `SLOW` itself,
  the constant a `SLOW - slack <= d <= wall + slack` bracket admits by
  construction: also 3 red / 13 green, the same three cases. That is the
  measurement behind the claim made at `SLOW` above, and the reason the
  bracket was replaced rather than merely widened.
- Turning the `chat_completions` `finally:` into an `else:`, i.e. billing only
  a stream that ended cleanly: 1 red / 15 green --
  `test_an_error_mid_stream_still_bills_what_arrived`. That single case is the
  entire difference between "covers every terminal exit" and "covers the happy
  one", which is why it exists as its own case rather than as an extra
  assertion on one of the others.

EVERY LINE ABOVE THAT NAMES `test_duration_tracks_the_call_and_not_a_constant`
IS HISTORY, NOT THIS FILE. That case has been REPLACED, and not because a run
was unlucky. It drove the route twice and asserted `slow > fast` across two
DISJOINT wall-clock intervals, so its real premise was "no pause of at least
`SLOW` lands inside the second one" -- a bound on the execution environment,
measured nowhere. It turned `origin/develop` RED at `21637ea`
(`assert 0.051075674 > 0.069269409`, and `assert 0.050877193 > 0.053628683`
three seconds later, both `[completions]`, same sha: ONE disturbance observed
twice, not two independent failures). The rows above are kept because they are
measurements that were really taken against the file as it then stood; they are
not claims about the file as it stands. `SLOW`'s comment carries the cause and
why the premise was removed rather than tuned.

Its replacement is `test_duration_is_the_handlers_own_interval_and_not_a
_constant`, which installs `_SteppingClock` into `routes`' globals and asserts
EQUALITY against the double's own script. Receipts, same discipline as the
table above, single-site edits against the finished tree, restored byte-
identically before the next (`filecmp.cmp(shallow=False)` against a pre-edit
snapshot taken in this worktree, `git status` clean after each). The mutated
pattern is the `duration=` argument of the ledger `cost_tracker.record(...)`
call in all three streaming handlers, anchored on the `user_id=` line above it:
pattern found 3, expected 3, replaced 3, each time.

- `duration=0.0` -- the constant the old case existed to reject: 3 red /
  13 green, exactly the three new cases (-78 bytes).
- `duration=0.05`, i.e. `SLOW` itself: 3 red / 13 green, the same three
  (-75 bytes).
- `duration=4.75`, i.e. `ADVANCE_A` -- a constant equal to one of the two
  scripted advances, which is the shape a single-advance version could not
  see: 3 red / 13 green, the same three (-75 bytes).

THE THIRD ROW IS THE ONE THAT EARNS THE SECOND ADVANCE, and it was measured as
a counterfactual rather than argued: with the case cut down to
`for advance in (ADVANCE_A,)` / `assert recorded == [ADVANCE_A]` -- two
single-site edits, both restored byte-identically -- the SAME `duration=4.75`
mutation is **16 passed, 0 failed**. One advance is satisfied by the constant
equal to it. Two are not. That is the same "a constant inside the band" the
`SLOW` comment records against a bracket, one level up.

AND THE TRAP, checked first-hand before the case was written rather than taken
on trust, and then measured as red/green rather than argued. `msg-230` F-1
states it: reaching for the nearest existing double, `_FrozenClock`, yields
`duration == 0.0` by construction -- which is the very value `duration=0.0`
produces -- so a frozen double is GREEN under the mutation the case exists to
reject. Setting BOTH advances to `0.0` turns `_SteppingClock` into exactly that
frozen clock (one edit, pattern found 1, expected 1, -2 bytes), and this case
alone then runs:

  frozen double, unmutated tree                       3 passed
  frozen double  + `duration=0.0`                     3 passed  <- BLIND
  stepping double + `duration=0.0`                    3 failed  <- the fix

Both files restored byte-identically afterwards. `_SteppingClock`'s docstring
records why per-read stepping was rejected in turn, which is the other double
this case could have reached for and the other way it could have gone wrong.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexora.api import routes
from lexora.api.routes import (
    get_backend,
    get_backend_router,
    get_cost_tracker,
    get_metrics_collector,
    get_rate_limiter,
    get_retry_handler,
    get_stats_collector,
    is_rate_limit_enabled,
    router,
)
from lexora.backends.base import BackendError
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector

# Same tier/model split as `test_ledger_coverage.py`: the caller says `heavy`,
# the router resolves `claude-fable-5`. Keeping them distinct is what lets the
# assertions tell the `model` column (concrete) from the `tier` column (alias).
REQUESTED = "heavy"
RESOLVED = "claude-fable-5"
BACKEND_NAME = "frontier"
USER_ID = "user-42"

PROMPT_TOKENS = 137
COMPLETION_TOKENS = 42

# Real seconds the backend stream is held open, so `duration` has something to
# measure rather than a tick of zero. Same role as `SLOW` in
# `test_ledger_coverage.py`.
#
# That file's bracket is deliberately NOT reused, and the reason is measured.
# (It carried a `DURATION_SLACK` constant when this was written; that constant
# has since been deleted and both of its ends are now derived. The reason
# below is unaffected -- it was never about the size of the slack.) Written
# here as `SLOW - slack <= d <= wall + slack` with
# one tick of slack, `[chat_completions]` failed twice on this runner at 0.0780
# recorded against a 0.0778 bound. Probing says why: the reported
# `monotonic` resolution of 0.015625 is nominal -- the observed step is 15 or
# 16 ms and never 15.625 (128 transitions over 2 s: 48 at 15 ms, 80 at 16 ms)
# -- and against a true 50 ms delay the recorded value takes exactly the four
# values 0.047 / 0.062 / 0.063 / 0.078, i.e. 3, 4 or 5 whole ticks. The
# recorded number therefore sits up to two ticks above the interval it
# measures, so a one-tick bound cannot hold against anything.
#
# Widening to two ticks would be a number picked to fit the failure. That half
# of the decision stands. What replaced the bracket did not.
#
# THIS PARAGRAPH USED TO SAY that `duration` is "fenced below by comparison
# instead -- a delayed call against an undelayed one -- which rests on no
# resolution premise". The comparison shipped and the claim was false, in the
# way the sentence itself hid: `assert slow > fast` reads the clock in TWO
# DISJOINT INTERVALS, so what it rests on is not a resolution premise but a
# SCHEDULING one -- "no pause of at least 0.05 s lands inside the second
# interval". That is a bound on the execution environment, and it was measured
# nowhere. Denominating the margin in ticks of a 15..16 ms clock is precisely
# what made a scheduling question read as a resolution question, which is how
# an unmeasured platform premise passed review here.
#
# It failed. `origin/develop` `21637ea`, CI red twice on `[completions]`:
# `assert 0.051075674 > 0.069269409` and `assert 0.050877193 > 0.053628683`.
# It is NOT a property of the CI platform. Holding the platform fixed and
# rebinding `monotonic` to `perf_counter`, the UNDELAYED call costs 0.0003 s
# over 42 pairs per route with 0 inversions -- a finer clock reveals a SMALLER
# reading, not a larger one, so "the fine Linux clock exposes the real cost of
# the undelayed call" is refuted rather than untested. What does invert it is
# a pause inside the measured window: injecting one on win32 gives PASS at
# 30 ms, PASS by 4.9 ms at 55 ms, and FAIL at 70 ms with
# `slow=0.047608 / fast=0.070551` -- the same shape and the same magnitudes as
# the two CI samples, which straddle that boundary. What produced the pause on
# the runner was not identified, and is deliberately not named here: the fix
# does not need the cause, because it removes the premise the cause acts on.
#
# The premise is therefore REMOVED and not tuned. `duration` is fenced below by
# `test_duration_is_the_handlers_own_interval_and_not_a_constant`, which
# installs a stepping clock double and asserts EQUALITY against the double's
# own script. That is an identity over
# `duration = routes.time.monotonic() - start_time` with both reads taken from
# the double, so it rests on nothing about resolution, platform, scheduling or
# load, and two different scripted advances are what kill a constant INSIDE a
# plausible band -- no single constant satisfies both.
#
# WHAT THAT FENCE DOES NOT BUY, written here rather than left to be discovered
# from a green run: it cannot show the handler reads a REAL clock. Under a
# scripted double every reading is scripted, by construction. That half is
# already held, by identity and not by margin, in `test_ledger_coverage.py` --
# `test_route_records_exactly_one_row`'s `backend_delta <= duration <= wall`,
# whose ends are read through `read_outer_clock` off the handler's own clock,
# and `test_the_lower_end_is_read_off_the_handlers_own_clock`'s frozen-clock
# provenance fence -- with the premises of both guarded by
# `test_interval_clock.py`. THE TWO TOGETHER ARE WHAT `assert slow > fast` WAS
# TRYING TO BUY IN ONE MOVE, and the premise is what one move cost. No single
# assertion buys both.
#
# Two corrections to the paragraphs above, neither of which changes the
# decision they record.
#
# The `SLOW - slack <= d <= wall + slack` shape quoted above described
# `test_ledger_coverage.py` as it stood when this file was written. That file's
# upper end no longer carries slack: it reads `wall` off the handler's own
# clock, which makes `d <= wall` hold identically and removes the resolution
# premise from that end rather than widening the budget for it
# (T-duration-slack-underestimates-the-tick). Its LOWER end still carries one
# tick, so the reasoning above still applies to the half this file was
# comparing itself against, and the decision here is unchanged.
#
# And the observation above -- "the reported `monotonic` resolution of 0.015625
# is nominal" -- was right, was recorded here first, and was left sitting in a
# comment while the file it indicts went on shipping a one-tick upper bound
# that then flaked about one full-suite run in eight for another round. Under
# load the step reaches 31 and 32 ms, not merely 15 or 16. A measurement that
# falsifies another file's premise is not filed by writing it down next to the
# code that already worked around it.
#
# F-4, checked rather than assumed. After the replacement above, `SLOW`'s only
# remaining reader is `test_streamed_request_records_exactly_one_row`, which
# holds the stream open so that `duration` is a real interval for the type and
# sign assertion rather than a tick of zero. It is KEPT for that one reader,
# and this sentence is the reason it was not deleted: a constant whose last
# reader goes away is exactly what `DURATION_SLACK` was.
SLOW = 0.05

# One well-formed OpenAI SSE chunk. `/v1/chat/completions` and `/v1/completions`
# forward it verbatim; `/v1/messages` parses it through
# `anthropic_stream_from_openai`, which is why it is a real chunk and not
# arbitrary bytes.
CHUNK = (
    b"data: "
    + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}
            ],
        }
    ).encode()
    + b"\n\n"
)

CHAT_BODY = {
    "model": REQUESTED,
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
    "user": USER_ID,
}
COMPLETIONS_BODY = {
    "model": REQUESTED,
    "prompt": "Hi",
    "stream": True,
    "user": USER_ID,
}
# `/v1/messages` takes its user id from `metadata.user_id`, matching that
# endpoint's own non-streaming record site, not from a top-level `user`.
MESSAGES_BODY = {
    "model": REQUESTED,
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
    "metadata": {"user_id": USER_ID},
}

ROUTES = [
    ("/v1/chat/completions", CHAT_BODY),
    ("/v1/completions", COMPLETIONS_BODY),
    ("/v1/messages", MESSAGES_BODY),
]
ROUTE_IDS = ["chat_completions", "completions", "messages"]

# The two scripted advances the stepping double below is driven with. They are
# not durations of anything: no call in this file takes four or eight seconds,
# and that is the point -- a value the machine could not have produced is the
# cleanest possible evidence that the recorded number came off the clock the
# handler was given.
#
# TWO of them, not one, and DIFFERENT. One advance is satisfied by the constant
# equal to it, which is the failure mode `SLOW` above records against a bracket
# ("a constant inside the band"). No single constant satisfies both.
#
# Neither equals any constant in this tree, checked and not assumed: not `0.0`,
# not `SLOW`, not `SLOW` plus or minus a tick, and `grep -rn -- 4.75 src tests`
# = 0 and `grep -rn -- 8.25 src tests` = 0 at `21637ea`. Both are exactly
# representable in binary floating point, so the literal here and the double's
# reading are the same value and `==` needs no tolerance to be honest.
ADVANCE_A = 4.75
ADVANCE_B = 8.25


class _SteppingClock:
    """A stand-in for the `time` module whose `monotonic()` steps ONCE, on cue.

    THE FIFTH CLOCK DOUBLE IN THIS SUITE, and the second outside
    `test_interval_clock.py` -- whose inventory comment and
    `test_ledger_coverage.py`'s `_FrozenClock` docstring are the other two ends
    of that register and were corrected in the same commit that added this.

    **Only `monotonic()` is faked**, and `__getattr__` delegates the rest, in
    the same shape as `_BackwardsWallClock`, `_TickSkippingClock` and
    `_ClockAttributeRecorder` in `test_interval_clock.py` and `_FrozenClock` in
    `test_ledger_coverage.py`.

    WHY STEPPING AND NOT FROZEN, stated first because reaching for the nearest
    existing double is the trap here. `_FrozenClock` makes the handler's
    interval `0.0` by construction, and `0.0` is indistinguishable from the
    `duration=0.0` constant the case below exists to reject -- so a frozen
    double produces a case that the very mutation it targets passes GREEN.
    Measured, not predicted, and measured BEFORE this class was written: the
    TRAP paragraph at the end of this module's docstring carries the reading,
    and the `duration=0.0` row above it carries the 3 red this double gets
    where a frozen one would get 0.

    READ-COUNT INVARIANT, which is the property `_FrozenClock`'s docstring
    argues for and the reason this does not advance per read. `monotonic()`
    returns `self._reading`; `step()` ASSIGNS rather than accumulates. Every
    read before the cue returns `0.0` and every read after returns `advance`,
    no matter how many reads there are and no matter how many times the cue
    fires. A double that moved per read would make this case's verdict depend
    on how often `routes.py` happens to call `time.monotonic()`, i.e. on an
    implementation detail of the code under test.

    EXACT, and not merely close. The base reading is `0.0`, so the handler's
    `time.monotonic() - start_time` is `advance - 0.0`, which is `advance`
    itself in IEEE 754 with no rounding to absorb. That is why the assertion
    below can be `==` with no tolerance rather than a bracket, and a bracket is
    the shape that needed a premise.

    The cue is pulled by the backend double (`_backend(clock=...)`) rather than
    by counting reads, for `_TickSkippingClock`'s reason: it then lands
    strictly between the handler's two endpoints -- `start_time` is read before
    the response is consumed, the generator body runs after it, and the ledger
    row's `duration` is computed in the `finally` after that.

    ONE INSTANCE PER REQUEST. The step is one-way, so a second request served
    by the same instance reads `advance` at BOTH of its endpoints and records
    `0.0`. Measured out of tree rather than reasoned about: reusing one
    instance gives `duration=4.75` then `duration=0.0`, `steps` 1 then 2. The
    case below builds a fresh one per (route, advance) pair, which is what
    makes the `>= 1` step guard sufficient; a reset knob is deliberately not
    offered, because an instance that can be rewound is one a future case can
    silently reuse into the `0.0` above.
    """

    def __init__(self, advance: float) -> None:
        self._advance = advance
        self._reading = 0.0
        self.steps = 0

    def step(self) -> None:
        self._reading = self._advance
        self.steps += 1

    def monotonic(self) -> float:
        return self._reading

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _backend(
    *,
    fills: bool = True,
    passthrough: bool = False,
    delay: float = 0.0,
    raise_after_first_chunk: bool = False,
    clock: _SteppingClock | None = None,
) -> MagicMock:
    """A backend whose streams optionally fill the sink they are handed.

    `fills=False` is the two verbatim-relay backends' behaviour in miniature:
    the stream runs to completion and writes nothing, which must leave the
    ledger untouched rather than record a zero.

    `raise_after_first_chunk` fills the sink and *then* fails, which is the
    only shape that separates "the row is opened at every terminal exit" from
    "the row is opened when the stream ends cleanly".

    The signature carries `usage_sink=None` by default on purpose: a fake that
    *required* the argument would red on `TypeError` rather than on the
    assertion, and a `TypeError` red says nothing about whether a row was
    opened.

    `clock` is the cue for `_SteppingClock`. The generator's body runs only
    once the response is consumed, which is after the handler read
    `start_time` and before its `finally` computes `duration`, so stepping
    from in here lands the advance strictly inside the interval being
    measured -- without the double having to count the handler's reads. See
    `_SteppingClock` for why that independence is the point.
    """

    def _factory(_request: dict, usage_sink: Any = None) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            if clock is not None:
                clock.step()
            if delay:
                await asyncio.sleep(delay)
            yield CHUNK
            if fills and usage_sink is not None:
                usage_sink.prompt_tokens = PROMPT_TOKENS
                usage_sink.completion_tokens = COMPLETION_TOKENS
            if raise_after_first_chunk:
                raise BackendError("upstream went away mid-stream")

        return gen()

    backend = MagicMock()
    backend.chat_completions_stream = MagicMock(side_effect=_factory)
    backend.completions_stream = MagicMock(side_effect=_factory)
    backend.error_passthrough = passthrough
    return backend


def _client(backend: MagicMock, cost_tracker: Any) -> TestClient:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=backend)
    backend_router.resolve_model = MagicMock(return_value=RESOLVED)
    backend_router.get_backend_name_for_model = MagicMock(return_value=BACKEND_NAME)
    backend_router.is_tier = MagicMock(return_value=True)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: backend
    app.dependency_overrides[get_backend_router] = lambda: backend_router
    app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
    app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
        max_retries=1, base_delay=0.01, max_delay=0.1, jitter=False
    )
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
        default_rate=1000.0, default_burst=1000
    )
    app.dependency_overrides[is_rate_limit_enabled] = lambda: False
    app.dependency_overrides[get_metrics_collector] = lambda: None
    app.dependency_overrides[get_cost_tracker] = lambda: cost_tracker
    return TestClient(app)


class TestStreamingRoutesOpenARow:
    """D-2b: each streaming handler records the count its backend supplied."""

    @pytest.mark.parametrize(("endpoint", "body"), ROUTES, ids=ROUTE_IDS)
    def test_streamed_request_records_exactly_one_row(
        self, endpoint: str, body: dict
    ) -> None:
        tracker = MagicMock()
        response = _client(_backend(delay=SLOW), tracker).post(endpoint, json=body)

        assert response.status_code == 200
        assert tracker.record.call_count == 1
        kwargs = tracker.record.call_args.kwargs
        assert kwargs["model"] == RESOLVED
        assert kwargs["endpoint"] == endpoint
        assert kwargs["tokens_input"] == PROMPT_TOKENS
        assert kwargs["tokens_output"] == COMPLETION_TOKENS
        assert kwargs["backend"] == BACKEND_NAME
        # The alias goes to its own column, never into `model`.
        assert kwargs["tier"] == REQUESTED
        # The column that says whose bill this lands on, asserted against a
        # value the request actually carried -- `None` staying `None` is what
        # a hard-coded column would also produce. R-13 is the reason this is
        # here in the first commit rather than two rounds later.
        assert kwargs["user_id"] == USER_ID
        # Type and sign only. The magnitude is not asserted here -- see `SLOW`
        # for why a bracket cannot be honestly written against this clock --
        # and is fenced instead by `test_duration_is_the_handlers_own_interval
        # _and_not_a_constant`. A negative duration needs no clock premise to
        # reject, so it is rejected here.
        assert isinstance(kwargs["duration"], float)
        assert kwargs["duration"] >= 0.0

    @pytest.mark.parametrize(("endpoint", "body"), ROUTES, ids=ROUTE_IDS)
    def test_duration_is_the_handlers_own_interval_and_not_a_constant(
        self, endpoint: str, body: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same route twice, under a clock whose step this test writes.

        `routes.time` is replaced by `_SteppingClock`, which reads `0.0` until
        the backend double pulls its cue and `advance` from then on. The
        handler computes `duration = time.monotonic() - start_time` with BOTH
        reads off that double, and the cue lands strictly between them, so the
        recorded number IS the scripted advance -- an identity, asserted with
        `==` and no tolerance.

        Driven with TWO different advances. One would be satisfied by the
        constant equal to it; the pair cannot be, which is the case a bracket
        cannot reach and the reason `ADVANCE_A != ADVANCE_B`.

        WHAT THIS REPLACED, and why the replacement is not a tuning. This case
        used to be `test_duration_tracks_the_call_and_not_a_constant`: two
        calls, one delayed by `SLOW` and one not, `assert slow > fast`. Those
        are two DISJOINT wall-clock intervals, so the assertion's real premise
        was "no pause of at least `SLOW` lands inside the second one" -- a
        bound on the execution environment that was never measured. It turned
        `develop` red at `21637ea`. The premise is gone here rather than
        loosened: no resolution, platform, scheduling or load property has to
        hold for an identity over a scripted double.

        WHAT THIS DOES NOT BUY. It cannot show the handler reads a real clock;
        under a scripted double every reading is scripted. `SLOW`'s comment
        above carries the full statement and names the two cases in
        `test_ledger_coverage.py` that hold that half by identity. The two
        together are what `assert slow > fast` was trying to buy in one move.
        """
        recorded = []
        for advance in (ADVANCE_A, ADVANCE_B):
            clock = _SteppingClock(advance)
            monkeypatch.setattr(routes, "time", clock)
            tracker = MagicMock()
            response = _client(_backend(clock=clock), tracker).post(
                endpoint, json=body
            )

            assert response.status_code == 200
            # Vacuity guard, first, so a double that never fired reds with the
            # diagnosis that names it rather than as a bare `0.0 != 4.75`.
            # `>= 1` and not `== 1` on purpose: `step()` assigns, so the
            # reading is invariant under repeats, and pinning the count would
            # make this case depend on how many times `routes.py` builds the
            # stream -- the read-count coupling `_SteppingClock` exists to
            # avoid.
            assert clock.steps >= 1, (
                f"the clock double never stepped for {endpoint} -- the "
                f"advance asserted below was never placed inside the "
                f"handler's interval"
            )
            assert tracker.record.call_count == 1
            recorded.append(tracker.record.call_args.kwargs["duration"])

        # Exact, both of them, in one assertion so the failure message carries
        # the pair rather than only the first half of it.
        assert recorded == [ADVANCE_A, ADVANCE_B]


class TestThePassthroughBranchIsWiredToo:
    """Two call sites per handler, and either can be missed on its own.

    `/v1/chat/completions` and `/v1/completions` pre-flight the first chunk for
    passthrough backends, and that pre-flight -- not the loop inside
    `stream_generator` -- is the call that creates the iterator the whole
    response is then served from. Passing the sink to only one of the two
    leaves half the traffic unbilled and the other half green.

    `/v1/messages` has one call site for both branches, so it is covered by
    `TestStreamingRoutesOpenARow` and is not repeated here.
    """

    @pytest.mark.parametrize(
        ("endpoint", "body"), ROUTES[:2], ids=ROUTE_IDS[:2]
    )
    def test_passthrough_stream_records_a_row(
        self, endpoint: str, body: dict
    ) -> None:
        tracker = MagicMock()
        response = _client(_backend(passthrough=True), tracker).post(
            endpoint, json=body
        )

        assert response.status_code == 200
        assert tracker.record.call_count == 1
        assert tracker.record.call_args.kwargs["tokens_input"] == PROMPT_TOKENS


class TestNothingIsBilledThatWasNotObserved:
    """The guard, driven from both sides of the thing it guards."""

    @pytest.mark.parametrize(("endpoint", "body"), ROUTES, ids=ROUTE_IDS)
    def test_a_backend_that_fills_nothing_opens_no_row(
        self, endpoint: str, body: dict
    ) -> None:
        """`openai_compatible` / `vllm` in miniature.

        Their bytes do not carry a count, so their sink stays at zero. Zero
        must mean "no row", not "a row saying this was free": a zero row would
        be indistinguishable in `/stats/costs` from a genuinely free request,
        and would let the relay backends quietly certify themselves as costing
        nothing.

        This is also the disconnect case, in the only form that matters to the
        ledger: usage arrives in the upstream's final frame, so a stream cut
        before that frame leaves the sink at zero and lands exactly here.
        """
        tracker = MagicMock()
        response = _client(_backend(fills=False), tracker).post(endpoint, json=body)

        assert response.status_code == 200
        assert tracker.record.call_count == 0


class TestEveryTerminalExitIsCovered:
    """D-2b: `finally`, not the success path."""

    def test_an_error_mid_stream_still_bills_what_arrived(self) -> None:
        """The upstream sent a count and then the stream failed.

        The tokens were spent whether or not the client got a clean ending, so
        the row is owed. A row write placed on the success path alone is green
        on every other case in this file and red only here.
        """
        tracker = MagicMock()
        with _client(_backend(raise_after_first_chunk=True), tracker) as client:
            with pytest.raises(BackendError):
                client.post("/v1/chat/completions", json=CHAT_BODY)

        assert tracker.record.call_count == 1
        assert tracker.record.call_args.kwargs["tokens_input"] == PROMPT_TOKENS
        assert tracker.record.call_args.kwargs["tokens_output"] == COMPLETION_TOKENS

    def test_exactly_one_row_per_streamed_request(self) -> None:
        """`finally` runs once per generator, so it cannot double-count.

        Stated as a measurement rather than a deduction: the sink is read in a
        `finally` inside the same generator that yields the bytes, and a second
        row would mean the generator was driven twice.
        """
        tracker = MagicMock()
        client = _client(_backend(), tracker)
        client.post("/v1/chat/completions", json=CHAT_BODY)
        client.post("/v1/chat/completions", json=CHAT_BODY)

        assert tracker.record.call_count == 2


class TestExistingBehaviourIsUnchanged:
    """Fences: the sink must not have changed what anyone receives."""

    @pytest.mark.parametrize(("endpoint", "body"), ROUTES, ids=ROUTE_IDS)
    def test_the_stream_still_reaches_the_client(
        self, endpoint: str, body: dict
    ) -> None:
        """The bytes, not just the status.

        `/v1/chat/completions` and `/v1/completions` relay `CHUNK` verbatim;
        `/v1/messages` translates it, so it is checked for the translated
        marker instead. Either way the assertion is that content came out --
        a change that broke the relay while still opening a row would pass
        every other case here.
        """
        response = _client(_backend(), MagicMock()).post(endpoint, json=body)

        assert response.status_code == 200
        if endpoint == "/v1/messages":
            assert "event: message_start" in response.text
            assert "Hi" in response.text
        else:
            assert CHUNK.decode() in response.text

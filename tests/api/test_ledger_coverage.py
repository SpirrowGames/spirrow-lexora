"""R-12: an endpoint that computes token counts must open a ledger row.

At `develop` (`00eafd5`) `cost_tracker.record` had exactly two call sites in
the whole tree -- `routes.py:591` (`/v1/chat/completions`) and
`routes.py:1989` (`/v1/messages`) -- while `usage.get("prompt_tokens", 0)` had
six. The four handlers in between computed the numbers the ledger wants and
dropped them:

    endpoint              handler        computes tokens   records a row
    /v1/completions       completions    routes.py:938     no
    /v1/embeddings        embeddings     routes.py:1079    no
    /generate             generate       routes.py:1468    no
    /chat                 chat           routes.py:1632    no

All four are live `@router.post` routes and all four call the backend directly
rather than delegating to `chat_completions`, so "recorded upstream" was not
the reason. None carried a comment explaining the omission, and none even took
`cost_tracker` as a dependency. `/stats/costs` and `/stats/costs/recent` served
the sum of the two recorded routes without saying so anywhere.

Not covered after this file: the three streaming paths (`stream_generator` at
`00eafd5`'s `routes.py:428` / `:812` / `:1883`). That was recorded here as
deliberate, over a reason that turned out to be false -- that those handlers
relay the bytes `backend.*_stream` yields without decoding them, so a count
would have to be parsed back out mid-relay. Only two of the five backends
relay. The gap is closed by T-streaming-ledger-row and its detectors live in
`test_streaming_ledger_row.py`; the four routes below are unaffected and this
file is still only about them.

Why these detectors drive the routes instead of calling a helper: R-11 shipped
five green unit detectors over the token parser while the ledger itself stayed
wrong, because nothing asserted the parser was reached. The requirement here is
"a row is opened", so that is what is measured -- at the route.

Mutation, so the detectors are measured rather than asserted. Counts are this
file alone, 16 cases:

- Against `develop`: 6 red / 10 green. Red = the four
  `TestLedgerCoversEveryNonStreamingRoute` cases (`record` never called) plus
  the two `TestUnpricedModelSaysSo` cases that read the row back out of SQLite.
  Green, and therefore fences rather than detectors, = both
  `TestExistingRoutesAreUnchanged` cases, all seven `TestGuard` cases, and
  `test_no_embedding_model_is_priced_as_chat`. The other 566 tests are green
  against `develop` too: the suite was blind to this.
- Half-fix, `/v1/completions` alone (block and dependency added there and
  nowhere else): 5 red / 11 green. `[completions]` goes green while
  `[embeddings]`, `[generate]` and `[chat]` stay red -- four independent
  detectors, not one assertion counted four times.
- Removing only `generate`'s `Depends(get_cost_tracker)`, leaving its `record`
  block: 3 red / 13 green. Of the four detectors only `[generate]` reds, on
  `NameError` -- so it is bound to that handler's wiring, not to the block
  existing somewhere in the file. The other two reds drive the same broken
  route.
- Weakening the guard to a bare `if cost_tracker:` at the four new call sites
  only: 4 red / 12 green. All four detectors stay green and only `TestGuard`'s
  four new-route cases red; its two existing-route cases stay green, which is
  what shows the mutation reached only the new sites.

R-13 (2026-09-06) extended the four detectors from four columns to six. As
shipped, `user_id` and `duration` were written by the four new call sites and
asserted nowhere: replacing both with constants at all four sites left the
whole suite at 582 passed, exit 0 -- not one test noticed. `user_id` is the
column that says whose bill a row lands on. The added mutations, this file
alone, 16 cases:

- `user_id` replaced by a constant at all four new sites, `duration`
  untouched: 4 red / 12 green -- all four detectors, so the column is read on
  every route and not just one.
- `duration` replaced by `-999.0` at all four new sites, `user_id` untouched:
  4 red / 12 green. Disjoint from nothing, but applied separately from the
  above, so each of the two assertions reds on its own rather than the pair
  being carried by one of them.
- `duration` replaced by `999.0` instead: also 4 red / 12 green. The bracket
  is closed at both ends, so a placeholder fails whichever direction it is
  wrong in, not merely if it is negative.
- `user_id` replaced by a constant at `/generate` alone: 1 red / 15 green, and
  the red is `[generate]`. Per-site, not one assertion counted four times.

The `duration` half of that shipped first as `SLOW <= duration <= wall` with
`wall` read off `time.monotonic()`, and that version was flaky-red on its own
unmodified tree. It was reported green on three consecutive local runs, a
green CI gate and a 590-passed suite; re-measured on the same commit with
nothing changed, the four detectors red 10 runs in 15, on a different route id
each time. All of those greens were luck. The mechanism and the tolerance that
removes it are recorded under "WHY THE UPPER END CARRIES NO TOLERANCE" below,
along with the later change that removed the tolerance from the other end too.
What the episode is worth keeping for is the general shape: repeating a run is
not evidence about an
assertion that reads a clock, and neither the gate nor CI can supply that
evidence, because each runs the suite exactly once. This is the timing form of
R-11's lesson -- there, a fence existed without working; here, a fence worked
without staying working.

What that episode owes to `T-unchecked-comment-claims` is stated here because
this is the file that keeps wanting to break it. A rate offered as a property
of the tree is inadmissible however carefully it was measured -- three ten-run
measurements of one unchanged tree came back 10, 9 and 7, which is not error
bars but the absence of a quantity to measure.
"""

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
from lexora.services.cost_tracker import DEFAULT_PRICING, CostTracker
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector

# The caller always says `heavy` (a tier alias); the router always resolves it
# to `claude-fable-5`. Keeping the two distinct is what lets the assertions
# below tell the `model` column (concrete) from the `tier` column (alias).
REQUESTED = "heavy"
RESOLVED = "claude-fable-5"
BACKEND_NAME = "frontier"

# Sent in the body of every new-route request below, so `user_id` is measured
# as "the caller's value arrived" rather than "None stayed None" -- the latter
# is what a hard-coded column would also produce.
USER_ID = "user-42"

# Real seconds each backend call is made to take, so `duration` has something
# to measure. Without it every one of the four routes records exactly 0.0
# (measured on this runner: `time.get_clock_info("monotonic").resolution` is
# 15.625 ms and the handler finishes inside one tick), and asserting
# "duration > 0" would be asserting the clock rather than the column. With
# the delay the recorded value was 0.0528..0.0597 across the four routes.
SLOW = 0.05

# The duration bracket below is
#
#     backend_delta  <=  duration  <=  wall
#
# and neither end carries a tolerance: both are exact, and both are the same
# nesting identity read at a different depth. It reached that shape in two
# steps, upper end first, and BOTH steps are written out here -- because each
# replaced something that looked more careful than what replaced it and was
# wrong anyway, and that is the transferable part.
#
# `routes.py` measures `duration` with `time.monotonic()` at every record site
# -- 46 occurrences, zero `time.time()`, zero `time.perf_counter()`, counted
# rather than assumed.
#
# WHY THE UPPER END CARRIES NO TOLERANCE: IT DOES NOT NEED ONE.
#
# Until `703e545` `wall` was read off `time.perf_counter()` and the upper end
# was `duration <= wall + DURATION_SLACK` -- one tick of the handler's clock,
# budgeted for that clock's quantum. That bound is FALSIFIED, not merely
# unproven at load. Measured on this runner (win32) while the full suite ran
# in another process, i.e. under the load the failure needs:
#
#     reported monotonic resolution     0.015625
#     observed monotonic steps (ms)     {15: 475, 16: 791, 31: 5, 32: 2}
#     largest observed step             0.032         <- 2.05x the report
#
# `time.monotonic()` here is `GetTickCount64()`, advanced by the system timer
# interrupt. When that interrupt is delayed the clock does not advance late,
# it advances by TWO ticks at once, while `time.get_clock_info("monotonic")`
# goes on reporting 0.015625 -- that field is a nominal figure, not an
# observation. So the error of a `monotonic` delta is NOT bounded by the
# reported resolution, and a bound that budgets one reported tick is unsound
# by roughly a factor of two.
#
# It lost accordingly. Instrumented so that every run yields samples instead
# of waiting for a rare traceback -- 8 full-suite runs, 32 samples, one red:
#
#     endpoint    duration              wall (perf_counter)   upper margin
#     /generate   0.0779999999795109    0.06230009999126196   -0.0000749
#
# over an old-upper-margin distribution of min -0.0000749 / median +0.016530 /
# max +0.032192. One negative in 32 samples is the one-run-in-eight full-suite
# red that had been carried as a known flake.
#
# Reading `wall` off the handler's own clock removes the error term rather
# than widening the budget for it. Let `m` be that clock. The outer window
# strictly contains the inner one -- checked against `routes.py` and against
# the code below, not against a comment: `started` is read before the request
# is dispatched and `wall` after the response returns, while `start_time` and
# `duration` are both read inside the handler. So in real time
#
#     t_outer_start < t_inner_start <= t_inner_end < t_outer_end
#
# and `m` is non-decreasing, hence
#
#     m(outer_start) <= m(inner_start) <= m(inner_end) <= m(outer_end)
#     => duration = m(inner_end) - m(inner_start)
#                <= m(outer_end) - m(outer_start) = wall
#
# No quantum appears anywhere in that. It holds for a clock of ANY coarseness
# and it holds for a clock that SKIPS ticks, which is the whole point: a skip
# moves both readings along one shared timeline, and the outer difference
# contains the inner one term by term. Measured twice, and the two are
# different measurements rather than one repeated:
#
#     on the 32 samples above, paired -- the monotonic window read strictly
#     INSIDE the perf_counter one, the pairing that cannot flatter this bound:
#     min +0.000000 / median +0.000000 / max +0.016000, zero negatives, and on
#     the very sample that reddened the old bound it was exactly 0.0, `wall`
#     and `duration` having landed on the same tick;
#
#     on 32 fresh samples of the shape actually shipped below, 8 more
#     full-suite runs under load: min +0.000000 / median +0.015000 /
#     max +0.016000, zero negatives.
#
# A margin of zero is the expected reading and not a near miss: this bound is
# an identity between two readings of one coarse clock, not a tolerance that
# happens to hold.
#
# Hence no slack on the upper end. Adding one to a bound that holds
# identically only admits wrong values. What that costs and gains, measured on
# the same samples rather than asserted: `(wall_perf + DURATION_SLACK) -
# wall_monotonic` ran min -0.005198 / median +0.014921 / max +0.028292. The
# new bound is typically ~15 ms tighter and is occasionally up to 5.2 ms
# looser. It is not uniformly tighter and is not claimed to be.
#
# The premise the upper end DOES rest on is that both ends read one clock.
# That is a fact about two files agreeing, which is the exact shape this suite
# keeps finding unfenced, so it is not left to this comment:
# `test_interval_clock.py::test_a_skipped_tick_cannot_break_the_upper_bound`
# drives a real request through a clock that skips forward mid-request and
# asserts both halves -- that the bound below survives it, and that the
# superseded `perf_counter`-plus-one-tick bound is RED on the very same
# request. It reads `read_outer_clock` from this module, so reverting the
# outer clock reddens it deterministically instead of one run in eight.
#
# WHY THE LOWER END NO LONGER CARRIES ONE EITHER.
#
# It used to. `DURATION_SLACK = max(time.get_clock_info("monotonic").resolution,
# 0.001)` stood here and widened the lower end to `SLOW - DURATION_SLACK`, one
# reported tick of the handler's clock with a 1 ms floor. The measurement above
# falsifies that end's premise exactly as it falsifies the old upper one: a
# skipped tick makes a delta under-read as well as over-read, and 32 ms of skip
# is not covered by a 15.625 ms tolerance. It survived on a measured margin
# (min +0.011625 / median +0.027625 over 32 samples) rather than on its
# derivation, and a margin is not a proof.
#
# It is now gone rather than widened, and the same move that fixed the upper
# end fixed this one: REMOVE THE PREMISE INSTEAD OF BUDGETING FOR IT. The
# backend double records its own readings around its sleep, off the same
# clock, and the lower end became `backend_delta <= duration` -- a fourth
# nesting level on the chain derived above, exact by the identical argument
# and containing no constant at all. Deriving a tolerance from a better
# measurement would have been strictly worse than not needing one.
#
# ∴ NO CONSTANT REMAINS IN EITHER END OF THE BRACKET, and the two checks that
# read the deleted one went with it in different ways, which is the part worth
# recording:
#
#   - `test_interval_clock.py::test_the_lower_bound_slack_keeps_its_derivation
#     _and_its_floor` was a pure VALUE fence over the constant. Its entire
#     subject was the constant, so when the constant went the check had no
#     subject and was DELETED. A fence standing over a premise nothing rests
#     on is the dead-premise defect that already cost this suite
#     `test_the_outer_clock_is_finer_than_the_slack`; keeping it "just in
#     case" would have been the same mistake a third time.
#   - `test_interval_clock.py::test_a_skipped_tick_cannot_break_the_upper
#     _bound` also read it, but NOT as a fence -- it reconstructs the
#     SUPERSEDED bound to prove its own reproduction still bites. That is a
#     negative control and it SURVIVES, on an inlined local. The invariant is
#     zero constants in any bound that still SHIPS, not zero constants in the
#     tree: a description of a historical shape may carry one where a live
#     bound may not.
#
# The old constant's value is still recoverable from git history if the
# arithmetic above ever needs re-deriving; it is not kept here as a variable
# nothing reads.


def read_outer_clock() -> float:
    """Read the bracket's outer end off the handler's own clock.

    `routes.time.monotonic` and not a local `time.monotonic`. The two are the
    identical function object today, and the indirection is the point: the
    upper bound is exact only while ONE clock reads both ends, and a local
    reference would be value-identical while decoupling silently the day
    `routes` changed clocks. This file has already been bitten by that exact
    shape -- deriving the since-deleted `DURATION_SLACK` off `time` instead
    of `monotonic` was value-invariant on both platforms this project runs on,
    and no value assertion could see it. It is also why the backend double
    reads its own interval through this function rather than off a local
    `time.monotonic`: the LOWER end has exactly the same premise.

    Resolving through `routes.time` also makes the premise testable instead of
    merely stated. `test_interval_clock.py` installs a clock that skips
    forward into `routes.time`, and because this function resolves there the
    skip reaches BOTH ends of the bracket, which is what a real skipped tick
    does and what a locally-held clock reference would have hidden.
    """
    return routes.time.monotonic()

USAGE = {"prompt_tokens": 11, "completion_tokens": 5}

COMPLETION_RESPONSE: dict[str, Any] = {
    "id": "cmpl-1",
    "object": "text_completion",
    "choices": [{"index": 0, "text": "Hi", "finish_reason": "stop"}],
    "usage": USAGE,
}
CHAT_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": USAGE,
}
EMBEDDINGS_RESPONSE: dict[str, Any] = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": RESOLVED,
    "usage": {"prompt_tokens": 11, "total_tokens": 11},
}

COMPLETIONS_BODY = {"model": REQUESTED, "prompt": "Hi", "user": USER_ID}
EMBEDDINGS_BODY = {"model": REQUESTED, "input": "Hi", "user": USER_ID}
GENERATE_BODY = {"model": REQUESTED, "prompt": "Hi", "user": USER_ID}
CHAT_BODY = {
    "model": REQUESTED,
    "messages": [{"role": "user", "content": "Hi"}],
    "user": USER_ID,
}
CHAT_COMPLETIONS_BODY = {
    "model": REQUESTED,
    "messages": [{"role": "user", "content": "Hi"}],
}
MESSAGES_BODY = {
    "model": REQUESTED,
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hi"}],
}

# The four routes R-12 adds, and the two that already recorded. `tokens_output`
# is 0 for embeddings because the handler hard-codes it (embeddings have no
# completion side); that 0 is the reason the guard is `input > 0 OR output > 0`
# and not `and`.
NEW_ROUTES = [
    ("/v1/completions", COMPLETIONS_BODY, 11, 5),
    ("/v1/embeddings", EMBEDDINGS_BODY, 11, 0),
    ("/generate", GENERATE_BODY, 11, 5),
    ("/chat", CHAT_BODY, 11, 5),
]
EXISTING_ROUTES = [
    ("/v1/chat/completions", CHAT_COMPLETIONS_BODY, 11, 5),
    ("/v1/messages", MESSAGES_BODY, 11, 5),
]


def _backend(usage: dict[str, int] | None = USAGE, delay: float = 0.0) -> MagicMock:
    """A backend whose three non-streaming methods return `usage`.

    `usage=None` strips the block entirely, which is what the guard fence
    drives; the response stays otherwise well-formed so the handlers still
    reach their recording site rather than erroring out earlier.

    `delay` holds each call open for that many real seconds. See `SLOW`: it is
    what gives the `duration` column a value distinguishable from a constant.

    Each call appends its own elapsed reading to `backend.intervals`. Those
    readings are the bracket's LOWER end and the reason it carries no constant
    -- the derivation is at the bracket itself. Two properties of the three
    lines below are load-bearing and neither is incidental:

    - they are read through `read_outer_clock`, i.e. `routes.time.monotonic`,
      the same clock object the handler subtracts. This is REQUIRED for the
      lower bound to be exact: the nesting identity holds only while one clock
      reads all four points, and a local `time.monotonic` would be
      value-identical today while decoupling silently the day `routes` changed
      clocks -- the exact shape this file was already bitten by once, recorded
      in `read_outer_clock`'s own docstring.

      AND IT IS FENCED, by `test_the_lower_end_is_read_off_the_handlers_own_
      clock` immediately below the bracket: it drives a request through a
      FROZEN `routes.time` and asserts these readings come back exactly 0.0,
      so swapping the two reads below for a local reference reds all four of
      its params with the measured value in the message.

      It was not always, and the sequence is the point rather than trivia.
      That fence exists because the swap to `__import__("time").monotonic()`
      -- the same function object, so it isolates provenance alone -- was
      measured GREEN across EVERY case of this file and
      `test_interval_clock.py`. That is 32 cases, re-counted at `5bfd428`
      rather than carried forward: this docstring shipped the figure as "48",
      which is the count for those two files PLUS
      `test_streaming_ledger_row.py`. The measurement was sound and its
      stated scope was not, so the scope is corrected here rather than the
      number quietly reused.

      The same docstring also said "reverting this to a local reddens it",
      and that was FALSE when written -- nothing in the suite could see the
      difference, which is the defect `read_outer_clock`'s docstring records
      one level up. With the fence in place the swap now reds exactly its
      four params and leaves the other 32 cases green, measured the same way
      it was measured false.

      Why the fence that covers the UPPER end does not reach down here, which
      is why the lower end needed one of its own:
      `test_a_skipped_tick_cannot_break_the_upper_bound` catches the
      equivalent decoupling by making `routes.time` skip FORWARD, and a
      forward skip inflates the handler's `duration` -- which pushes
      `backend_delta <= duration` further from failing, not closer. Detecting
      a decoupled lower end takes a clock that advances SLOWER than real
      time; the frozen one is the read-count-invariant limiting case of that,
      and `_FrozenClock`'s docstring records why the read-count property is
      what decided the instrument;
    - the reading is taken whether or not `delay` is truthy, so every call
      records. The `len(...) == 1` guard at the assertion site is a real
      check rather than a coincidence of which double was constructed.
    """
    intervals: list[float] = []

    def _with(payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        if usage is None:
            body.pop("usage", None)
        else:
            body["usage"] = usage
        return body

    def _responder(payload: dict[str, Any]) -> Any:
        body = _with(payload)

        async def _respond(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            entered = read_outer_clock()
            if delay:
                await asyncio.sleep(delay)
            intervals.append(read_outer_clock() - entered)
            return body

        return _respond

    backend = MagicMock()
    backend.completions = AsyncMock(side_effect=_responder(COMPLETION_RESPONSE))
    backend.chat_completions = AsyncMock(side_effect=_responder(CHAT_RESPONSE))
    backend.embeddings = AsyncMock(side_effect=_responder(EMBEDDINGS_RESPONSE))
    backend.error_passthrough = False
    backend.intervals = intervals
    return backend


def _client(
    backend: MagicMock, cost_tracker: Any, resolved: str = RESOLVED
) -> TestClient:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=backend)
    backend_router.resolve_model = MagicMock(return_value=resolved)
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


class _FrozenClock:
    """A stand-in for the `time` module whose `monotonic()` never advances.

    Installed into `routes`' globals by the fence below, which is the only
    thing that uses it. It is the fourth clock double in this suite and the
    only one outside `test_interval_clock.py`; the other three are named from
    there and the reason this one is not with them is recorded at both ends.

    **Only `monotonic()` is faked**, and `__getattr__` delegates the rest, in
    the same shape as `_BackwardsWallClock`, `_TickSkippingClock` and
    `_ClockAttributeRecorder` over in `test_interval_clock.py`.

    WHY FROZEN AND NOT MERELY SLOW. What the fence below has to detect is a
    lower end whose two readings stop coming off the handler's clock, and the
    general instrument for that is a clock running SLOWER than real time --
    of which a decrementing or per-read-decelerating clock is the obvious
    shape. It is the wrong one, and `_TickSkippingClock`'s docstring already
    says why in as many words: a clock that moves per read makes the verdict
    depend on HOW MANY TIMES `routes.py` reads it, i.e. on an implementation
    detail of the code under test. A test whose meaning moves when a handler
    gains or loses a `time.monotonic()` call is not a fence.

    A frozen clock is the limiting case of "slower than real time", at rate
    zero, and it is READ-COUNT-INVARIANT: every read returns the same value
    no matter how many there are. So there is no trigger to place, no
    interleaving to reason about, and no dependence on `routes.py`'s internals
    -- the same properties `_TickSkippingClock` had to work for by tying its
    jump to the backend double rather than to a read count.
    """

    def __init__(self) -> None:
        self._frozen = time.monotonic()

    def monotonic(self) -> float:
        return self._frozen

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class TestLedgerCoversEveryNonStreamingRoute:
    """D-1..D-4: the four routes that computed tokens now open a row."""

    @pytest.mark.parametrize(
        ("endpoint", "body", "tokens_input", "tokens_output"),
        NEW_ROUTES,
        ids=["completions", "embeddings", "generate", "chat"],
    )
    def test_route_records_exactly_one_row(
        self, endpoint: str, body: dict, tokens_input: int, tokens_output: int
    ) -> None:
        tracker = MagicMock()
        # Read before the request is dispatched and after the response
        # returns, off the handler's own clock -- see `read_outer_clock`. The
        # containment this relies on is visible in these three lines and
        # nowhere else, which is why they are kept adjacent.
        backend = _backend(delay=SLOW)
        started = read_outer_clock()
        response = _client(backend, tracker).post(endpoint, json=body)
        wall = read_outer_clock() - started

        assert response.status_code == 200
        assert tracker.record.call_count == 1
        kwargs = tracker.record.call_args.kwargs
        assert kwargs["model"] == RESOLVED
        assert kwargs["endpoint"] == endpoint
        assert kwargs["tokens_input"] == tokens_input
        assert kwargs["tokens_output"] == tokens_output
        assert kwargs["backend"] == BACKEND_NAME
        # The alias goes to its own column, never into `model` -- the contract
        # `CostTracker.record` states in its docstring.
        assert kwargs["tier"] == REQUESTED
        # `user_id` is the column that says who to bill, so it is measured
        # against a value the request actually carried. Until R-13 nothing in
        # the suite asserted it: replacing it with a constant at all four call
        # sites left 582 tests passing.
        assert kwargs["user_id"] == USER_ID
        # `duration` cannot be pinned to a number, so it is bracketed instead:
        # at least the interval the backend double measured for itself, at
        # most the elapsed of the whole call measured from out here. The two
        # ends are now SYMMETRIC, and that symmetry is the content of
        # `T-duration-slack-underestimates-the-tick`:
        #
        #   upper   `duration <= wall`           exact, no tolerance.
        #   lower   `backend_delta <= duration`  exact, no tolerance.
        #
        # Both are the same identity read at different depths. Four readings
        # of ONE non-decreasing clock, nested --
        #
        #     m(started) <= m(entered) <= m(left) <= m(wall)
        #
        # where the inner pair is taken by the double inside `_respond` and
        # the middle pair by the handler either side of the backend call.
        # `started`/`wall` are read out here through `read_outer_clock`,
        # `entered`/`left` inside the double through the same function, and
        # `start_time`/`duration` inside the handler off `routes.time`. From
        # `a <= b <= c <= d` follows `c - b <= d - a` for a clock of ANY
        # coarseness, including one that skips ticks: a skip moves every
        # reading along one shared timeline. Neither end contains a constant,
        # so neither can be falsified by a measurement of the clock -- which
        # is exactly what happened to the tolerance both ends used to carry.
        # The lower end was `SLOW - DURATION_SLACK <= duration` until that
        # constant was deleted, and it survived on a measured margin over a
        # premise measured FALSE on this runner (the reported 15.625 ms
        # resolution against observed 31-32 ms steps). It no longer rests on
        # anything that can be false.
        #
        # WHAT THAT COSTS. `backend_delta` fences the interval's PROVENANCE,
        # not its MAGNITUDE against `SLOW`: nothing below says the double was
        # held open for 50 ms rather than 50 us. Re-measured before the old
        # end was removed rather than argued -- with both ends present and the
        # new one placed first, `duration` replaced by `0.0` and by `-999.0`
        # at all four new call sites each reddened all four params ON THIS
        # ASSERTION, the old end never being reached (`0.062000 <= 0.0`).
        # `999.0` still reds all four on the upper end. So the constants this
        # bracket exists to reject are rejected by the derived end, three
        # orders of magnitude clear, without a constant doing it.
        #
        # What it still does not catch is a value *inside* the band, and one
        # always exists: `SLOW` itself clears both ends, `wall` being an outer
        # reading of a call the backend holds open for `SLOW`. The vacuity
        # guard below is what keeps the lower end from becoming a third such
        # value -- an empty `intervals` is the shape in which this bound would
        # stop measuring while still looking like a bound.
        assert isinstance(kwargs["duration"], float)
        # Without this the lower bound is vacuous in the one way that matters:
        # an un-instrumented double would leave `intervals` empty and an
        # `IndexError` is a worse red than a named one, while a retried call
        # would leave two and the bound would silently read the wrong one.
        assert len(backend.intervals) == 1, (
            f"the double recorded {len(backend.intervals)} intervals, not one "
            f"-- the lower bound below would be reading the wrong call or no "
            f"call at all"
        )
        backend_delta = backend.intervals[0]
        assert backend_delta <= kwargs["duration"] <= wall

    @pytest.mark.parametrize(
        ("endpoint", "body"),
        [(e, b) for e, b, _, _ in NEW_ROUTES],
        ids=["completions", "embeddings", "generate", "chat"],
    )
    def test_the_lower_end_is_read_off_the_handlers_own_clock(
        self,
        endpoint: str,
        body: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The bracket's lower end reads the clock the handler reads.

        The third leg of the nesting identity, and until this test the only
        one with no fence. The handler's half is held by
        `test_interval_clock.py::test_the_handler_measures_its_interval_off_
        monotonic_alone`, the outer half by `::test_a_skipped_tick_cannot_
        break_the_upper_bound`. The double's half was bare, and bare in a way
        no value assertion could see: swapping both of `_backend`'s readings
        for `__import__("time").monotonic()` -- the same function object, so
        the swap isolates provenance alone -- was measured GREEN across every
        case in this file and `test_interval_clock.py`.

        That is not a hypothetical hazard, it is one this file has already
        been bitten by once: `read_outer_clock`'s docstring records deriving
        the since-deleted `DURATION_SLACK` off `time` instead of `monotonic`,
        value-invariant on both platforms, invisible to every assertion in the
        suite. Same shape, one level down.

        The instrument is a FROZEN `routes.time.monotonic` -- see
        `_FrozenClock` for why frozen rather than merely slow, which is a
        question about read-count dependence and not a matter of taste. Under
        it the handler's interval is 0.0 by construction, so a double reading
        the same clock must report exactly 0.0, while a double reading any
        clock that really advances reports the real elapsed of a call held
        open for `SLOW` and this test reds with that number in the message.

        WHAT THIS CANNOT SEE, stated here rather than left to be discovered
        from a green run. A double that fabricates a constant `0.0` instead of
        measuring anything passes this test and always will: under a frozen
        clock the true reading IS 0.0, so no bound that must also hold on the
        shipped tree can separate the fabricated value from the measured one.
        Measured, not predicted -- appending a literal `0.0` to `intervals` in
        place of the subtraction leaves this test GREEN on all four params.
        That ceiling is INHERITED FROM THE BRACKET rather than introduced
        here: it is the same "a constant inside the band always exists" the
        bracket records against itself, where `SLOW` is such a value. The
        vacuity guard below is what keeps the weaker version of that -- a
        double that measures nothing at all -- from passing.
        """
        monkeypatch.setattr(routes, "time", _FrozenClock())

        tracker = MagicMock()
        backend = _backend(delay=SLOW)
        started = read_outer_clock()
        response = _client(backend, tracker).post(endpoint, json=body)
        wall = read_outer_clock() - started

        assert response.status_code == 200
        # The row has to be opened for `duration` to exist at all; without
        # this the two assertions below would fail on a missing call rather
        # than on the premise they are here to fence.
        assert tracker.record.call_count == 1
        kwargs = tracker.record.call_args.kwargs
        # Same guard as the bracket above and for the same reason: an
        # un-instrumented double leaves `intervals` empty, and an `IndexError`
        # is a worse red than a named one.
        assert len(backend.intervals) == 1, (
            f"the double recorded {len(backend.intervals)} intervals, not one "
            f"-- the reading asserted below would be the wrong call or no "
            f"call at all"
        )
        backend_delta = backend.intervals[0]
        # FIRST, so the premise fails before the bracket does and with the
        # diagnosis that names it. The bracket below would also red here, but
        # it would red as "a bound was violated" rather than as "the lower
        # end stopped reading the handler's clock", and those need different
        # repairs.
        assert backend_delta == 0.0, (
            f"the double measured {backend_delta!r} across a call the "
            f"handler's own clock says took no time -- its two readings are "
            f"no longer resolving through `routes.time`, so the four-reading "
            f"nesting identity that makes `backend_delta <= duration` exact "
            f"no longer holds"
        )
        # Then the shipped bracket, unchanged, on the same request: under one
        # frozen clock all four readings collapse to the same value and the
        # identity is satisfied at its boundary, 0.0 <= 0.0 <= 0.0.
        assert backend_delta <= kwargs["duration"] <= wall


class TestExistingRoutesAreUnchanged:
    """F-1: the two routes that already recorded still record once, not twice."""

    @pytest.mark.parametrize(
        ("endpoint", "body", "tokens_input", "tokens_output"),
        EXISTING_ROUTES,
        ids=["chat_completions", "messages"],
    )
    def test_no_double_counting(
        self, endpoint: str, body: dict, tokens_input: int, tokens_output: int
    ) -> None:
        tracker = MagicMock()
        response = _client(_backend(), tracker).post(endpoint, json=body)

        assert response.status_code == 200
        assert tracker.record.call_count == 1
        kwargs = tracker.record.call_args.kwargs
        assert kwargs["tokens_input"] == tokens_input
        assert kwargs["tokens_output"] == tokens_output


class TestGuard:
    """F-2: no `usage` means no row. The pre-R-12 rule, kept."""

    @pytest.mark.parametrize(
        ("endpoint", "body"),
        [(e, b) for e, b, _, _ in NEW_ROUTES + EXISTING_ROUTES],
        ids=[
            "completions",
            "embeddings",
            "generate",
            "chat",
            "chat_completions",
            "messages",
        ],
    )
    def test_no_usage_opens_no_row(self, endpoint: str, body: dict) -> None:
        tracker = MagicMock()
        response = _client(_backend(usage=None), tracker).post(endpoint, json=body)

        assert response.status_code == 200
        assert tracker.record.call_count == 0

    def test_absent_cost_tracker_is_not_an_error(self) -> None:
        """`cost_tracker` is `None` whenever the app never built one."""
        client = _client(_backend(), None)
        for endpoint, body, _, _ in NEW_ROUTES:
            assert client.post(endpoint, json=body).status_code == 200


class TestUnpricedModelSaysSo:
    """F-3: bringing embeddings in lands on `pricing_known=0`, not a silent 0.0.

    No embedding model is in `DEFAULT_PRICING` (measured 2026-09-06 at
    `05f6827`: nine entries, all chat -- four Claude, two GPT, one Gemini, two
    local Qwen at 0.0). Recording them therefore writes cost 0.0 with
    `pricing_known=0`, which is R-10's "we cannot price this" state and is
    distinguishable from the local Qwen rows' priced 0.0. Not recording them
    at all said neither.
    """

    def test_no_embedding_model_is_priced_as_chat(self) -> None:
        assert not [m for m in DEFAULT_PRICING if "embed" in m.lower()]

    def test_row_is_written_with_pricing_known_zero(self, tmp_path: Path) -> None:
        db = tmp_path / "costs.db"
        tracker = CostTracker(db_path=db)
        client = _client(_backend(), tracker, resolved="text-embedding-3-small")

        assert client.post("/v1/embeddings", json=EMBEDDINGS_BODY).status_code == 200

        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT model, endpoint, tokens_input, tokens_output, cost_usd,"
                " pricing_known, tier FROM request_costs"
            ).fetchall()
        assert rows == [
            ("text-embedding-3-small", "/v1/embeddings", 11, 0, 0.0, 0, REQUESTED)
        ]

    def test_priced_model_is_distinguishable(self, tmp_path: Path) -> None:
        """The same route with a priced model writes `pricing_known=1`."""
        db = tmp_path / "costs.db"
        tracker = CostTracker(db_path=db)
        client = _client(_backend(), tracker, resolved="Qwen3-32B")

        assert client.post("/v1/embeddings", json=EMBEDDINGS_BODY).status_code == 200

        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT cost_usd, pricing_known FROM request_costs"
            ).fetchone()
        assert row == (0.0, 1)

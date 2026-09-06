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

Still not covered after this file, deliberately: the three streaming paths
(`stream_generator` at `00eafd5`'s `routes.py:428` / `:812` / `:1883`), which
relay the bytes `backend.*_stream` yields without decoding them. See
`get_cost_tracker`'s docstring.

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
removes it are recorded at `DURATION_SLACK` below. What the episode is worth
keeping for is the general shape: repeating a run is not evidence about an
assertion that reads a clock, and neither the gate nor CI can supply that
evidence, because each runs the suite exactly once. This is the timing form of
R-11's lesson -- there, a fence existed without working; here, a fence worked
without staying working.

The rule that episode owes to `T-unchecked-comment-claims`, stated here
because this is the file that keeps wanting to break it. A RUN-COUNT IN A
COMMENT IS ADMISSIBLE ONLY IF either (a) the quantity it reports is
DETERMINISTIC, so that a single contrary run refutes it, or (b) the number is
used solely as evidence that the quantity is UNSTABLE, and the text says so.
A rate offered as a property of the tree is inadmissible however carefully it
was measured -- three ten-run measurements of one unchanged tree came back 10,
9 and 7, which is not error bars but the absence of a quantity to measure. The
discriminator is determinism, not who took the reading or how recently.

Three run-counts appear in this file, and the sentence above is one of them.
Two sit in prose -- one in the paragraph above, one at the tolerance below --
and both report the same superseded `time.monotonic()` bracket, each now
saying which bracket it measured. The third is the 10/9/7 just quoted. All
three are use (b): every one is cited to show that a quantity was unstable,
and every one says so. A count of failing test NODES within a single run,
such as the 4-of-16 above, is not a run-count at all, and this rule does not
reach it.

KNOWN DEFECT in that rule, recorded here because the rule is stated here and
the repair is still open. Its text is universal, so its extension is the
repository and not this file. Enumerated 2026-09-07 over every comment and
docstring block in `src/` and `tests/`: outside this file the rule reaches four
run-counts its author did not have in hand, and it condemns two of them --
the tolerance note at `tests/services/test_rate_limiter.py` and the sampling
note in `_tokens_from_result` at `backends/claude_code.py`. Both cite how much
evidence stands behind a fact that observation found steady but did not prove
deterministic, so limb (a) is unavailable, and neither is offered as
evidence of instability, so (b) is unavailable too. Deleting either number
leaves the surviving claim with no recorded basis, which is worse than the
text that is there. Both should stand. The gap is structural rather than a
matter of wording: the rule has a limb for a quantity proven fixed and a limb
for a quantity shown to move, and none for bounded evidence about a quantity
that is neither -- which is the case where saying how many runs there were
carries the most information, because it is the reader's only handle on how
far to trust it. Note which of the neighbours pass: that same docstring
carries a second run-count the rule admits, and so does its test file, and
both are admitted only because what they happened to observe turned out to be
unstable. The verdict tracks what the measurement found, not whether citing it
was sound practice. Until this is repaired, do not apply the rule to delete
either site.
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

# The tolerance the bracket on `duration` needs, and why it is this.
#
# `routes.py` measures `duration` with `time.monotonic()` at every record
# site. A difference of two readings off a clock quantised to
# `tick` differs from the true interval by strictly less than `tick` in either
# direction, so a recorded value can overshoot the interval it measures. On
# this runner `time.time()` and `time.monotonic()` BOTH report a resolution of
# 15.625 ms, which is why bracketing the handler's delta between
# `time.monotonic()` deltas did not hold: measured bare, 300 samples over a
# 50 ms sleep put the inner value above the outer one 234 times (78%), median
# +3.3 ms, max +4.9 ms. That bracket survived only on the few ms of slack the
# TestClient round trip adds outside the handler's own window, and lost the
# toss often: with THAT bracket in the tree, the four detectors below red 10
# runs in 15. The count measures the superseded bracket, not the one shipped
# here, and it is quoted for one purpose -- to show the bracket was unstable.
#
# `wall` is therefore read off `time.perf_counter()` (resolution 1e-07 here),
# which makes it the true elapsed of an interval that strictly contains the
# handler's, leaving the inner clock's own quantum as the DOMINANT error term.
# Dominant, and not the only one -- stated as the approximation it is rather
# than as an identity it is not. The outer clock has a quantum `p` of its own,
# so the sufficient condition is `slack >= m + p`, and measured here that is
# `0.015625 >= 0.0156251`: false, by 1e-07. The bracket does not rest on that
# inequality; it rests on the containment margin, the milliseconds of
# TestClient round trip that sit outside the handler's own window. Widening
# the slack to `max(m + p, 0.001)` to close the 1e-07 was considered and
# declined, and the reason is detection power, measured, not the 1e-07 being
# small. `m + p` lets this bracket's half-width track `p` without bound. Worked
# with `m` = 15.625 ms and `SLOW` = 0.05, the lower end `SLOW - slack` goes
# non-positive once `p` >= 0.034375, and from there up -- `p` = 0.1, say -- the
# bracket ADMITS a recorded `duration` of exactly 0.0, the placeholder it
# exists to reject.
# The shipped `max(m, 0.001)` holds the half-width at 0.015625 for every `p`,
# rejects 0.0 in all of those cases, and instead fails loudly, at
# `test_the_outer_clock_is_finer_than_the_slack`, as soon as `p` reaches the
# slack. So `m + p` trades a loud failure for a silent loss of detection power
# on precisely the coarse-`p` platforms these checks exist to catch, and the
# criterion that rejects it is the one the floor paragraph below already uses:
# costs no detection power. Secondary, and never the reason: it would also
# disarm that check, `p < max(m + p, 0.001)` being vacuous whenever
# `m + p` >= 0.001, and would make this term depend on a third clock.
# Hence one tick of the handler's clock at each end, with a floor (below):
# since the backend is held open for `SLOW` inside a window the outer reading
# strictly contains,
#
#     SLOW - slack  <  duration  <  wall + slack
#
# holds by construction rather than by luck. Moving only the outer clock would
# not have been enough: the coarse clock is the inner one, and the test cannot
# reach it.
#
# The floor exists so this does not merely relocate the same mistake. Where
# the clock is fine-grained the resolution term is ~0 (1e-09 on Linux CI --
# measured, off CI run 34053301337, not assumed),
# which would put the bound back to exactly tight. 1 ms covers that by a wide
# margin (500 ppm over 50 ms is 25 us) and costs no detection power: every
# constant this bracket has to reject misses it by three orders of magnitude.
#
# The term is now read off the clock the handler actually reads. That closes
# the drift. The derivation leans on two premises, they are different from
# each other, and `test_interval_clock.py` now carries one check for each.
#
# One is that the slack is not finer than the handler's tick. With the term
# taken off that same clock this reads `max(m, 0.001) >= m`, which is true for
# every `m`, and it was briefly deleted here for exactly that reason. The
# deletion was wrong, and the reasoning that produced it will produce it again
# unless it is written down: no *platform* can redden that inequality, but an
# *edit to this line* can, and those are two different properties. Measured by
# single-site mutation of this line, three of five plausible edits redden it,
# hardcoding it to `0.001` among them. It is fenced by
# `test_the_slack_covers_the_handlers_tick_and_the_floor`, which carries the
# mutation table and names the platform each cell was run on. That check
# guards this line's *value* and never its *provenance*, so it is green on the
# drift this change exists to remove -- deriving off `time` is value-invariant
# on both platforms this project runs on, and no value assertion can see it.
#
# The other premise is contingent, and until recently was asserted nowhere:
# `wall` is only "the true elapsed of a strictly containing interval" while
# the OUTER clock's quantum is small next to this slack. Otherwise
# `perf_counter`'s own tick is a second error term of a size that matters, and
# the paragraph above stops being even approximately true. Nothing makes that
# so -- it is a fact about whichever platform runs the suite (here 1e-07
# against 15.625 ms, five orders of magnitude), so it is checked at runtime by
# `test_interval_clock.py::test_the_outer_clock_is_finer_than_the_slack`.
DURATION_SLACK = max(time.get_clock_info("monotonic").resolution, 0.001)

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
    """

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
            if delay:
                await asyncio.sleep(delay)
            return body

        return _respond

    backend = MagicMock()
    backend.completions = AsyncMock(side_effect=_responder(COMPLETION_RESPONSE))
    backend.chat_completions = AsyncMock(side_effect=_responder(CHAT_RESPONSE))
    backend.embeddings = AsyncMock(side_effect=_responder(EMBEDDINGS_RESPONSE))
    backend.error_passthrough = False
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
        started = time.perf_counter()
        response = _client(_backend(delay=SLOW), tracker).post(endpoint, json=body)
        wall = time.perf_counter() - started

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
        # at least the delay the backend was held open for, at most the wall
        # time of the whole call measured from out here -- each end widened by
        # the slack the handler's clock needs, which is the entire reason
        # this is not `SLOW <= duration <= wall`. See `DURATION_SLACK`.
        #
        # What the bracket catches is a constant *outside* the band: 0.0
        # and -999.0 undershoot the lower bound, 999.0 overshoots the
        # upper one -- each measured red at all four params. What it does
        # not catch is a constant *inside* the band, and one always
        # exists: `SLOW` itself clears both ends on every platform,
        # because the lower bound is `SLOW - DURATION_SLACK` and `wall`
        # is an outer reading of a call the backend holds open for `SLOW`
        # (measured green at all four params). So this brackets the
        # magnitude, not the provenance -- it says the recorded number is
        # the right size, not that it came from a clock.
        assert isinstance(kwargs["duration"], float)
        assert SLOW - DURATION_SLACK <= kwargs["duration"] <= wall + DURATION_SLACK


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

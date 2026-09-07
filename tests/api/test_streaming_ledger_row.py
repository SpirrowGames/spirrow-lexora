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
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

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
# That file's `DURATION_SLACK` bracket is deliberately NOT reused, and the
# reason is measured. Written here as `SLOW - slack <= d <= wall + slack` with
# one tick of slack, `[chat_completions]` failed twice on this runner at 0.0780
# recorded against a 0.0778 bound. Probing says why: the reported
# `monotonic` resolution of 0.015625 is nominal -- the observed step is 15 or
# 16 ms and never 15.625 (128 transitions over 2 s: 48 at 15 ms, 80 at 16 ms)
# -- and against a true 50 ms delay the recorded value takes exactly the four
# values 0.047 / 0.062 / 0.063 / 0.078, i.e. 3, 4 or 5 whole ticks. The
# recorded number therefore sits up to two ticks above the interval it
# measures, so a one-tick bound cannot hold against anything.
#
# Widening to two ticks would be a number picked to fit the failure. `duration`
# is fenced below by comparison instead -- a delayed call against an undelayed
# one -- which rests on no resolution premise and additionally catches the one
# thing a bracket cannot: a constant inside the band.
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


def _backend(
    *,
    fills: bool = True,
    passthrough: bool = False,
    delay: float = 0.0,
    raise_after_first_chunk: bool = False,
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
    """

    def _factory(_request: dict, usage_sink: Any = None) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
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
        # and is fenced instead by
        # `test_duration_tracks_the_call_and_not_a_constant`. A negative
        # duration needs no clock premise to reject, so it is rejected here.
        assert isinstance(kwargs["duration"], float)
        assert kwargs["duration"] >= 0.0

    @pytest.mark.parametrize(("endpoint", "body"), ROUTES, ids=ROUTE_IDS)
    def test_duration_tracks_the_call_and_not_a_constant(
        self, endpoint: str, body: dict
    ) -> None:
        """The same route twice: backend held open for `SLOW`, then not at all.

        The recorded numbers must differ in the right direction. That is a
        statement about *provenance* -- the value came from timing this call --
        and it needs nothing to be true about the clock's resolution, only that
        a longer call does not read shorter. Every constant fails it, including
        one inside a plausible band, which is the case a bracket cannot reach.

        The separation is wide, and measured rather than assumed: over 30 pairs
        per route on this runner the delayed call recorded 0.047..0.078 and the
        undelayed one 0.000 (0.000..0.016 on `/v1/messages`), a minimum margin
        of 0.046 -- about three ticks of a clock that steps 15..16 ms -- with
        0 inversions in 90 pairs. Repeating a run is not evidence about an
        assertion that reads a clock (that is this project's own hard-won
        lesson), so what carries this is the margin, not the repetitions: the
        two populations are separated by three quanta of the only clock
        involved.
        """
        slow_tracker = MagicMock()
        _client(_backend(delay=SLOW), slow_tracker).post(endpoint, json=body)
        fast_tracker = MagicMock()
        _client(_backend(delay=0.0), fast_tracker).post(endpoint, json=body)

        slow = slow_tracker.record.call_args.kwargs["duration"]
        fast = fast_tracker.record.call_args.kwargs["duration"]
        assert slow > fast


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

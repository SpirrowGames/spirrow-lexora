"""An elapsed interval must not be measured with the adjustable clock.

`routes.py` measured every elapsed interval with `time.time()` -- 43 sites,
zero `time.monotonic()`, zero `time.perf_counter()`. `time.time()` is
CLOCK_REALTIME and `time.get_clock_info("time").adjustable` is True, so a
backwards **step** (`chronyc makestep`, `ntpd -g`, `date -s`, a VM snapshot
restore, a container host clock jump) inside a request yields a *negative*
`duration`. `prometheus_client` accepts a negative `observe()` without
raising, and the histogram's `_sum` then *decreases* -- which `rate()` and
`increase()` read as a counter reset, so the ordinary average-latency query
is wrong for the whole window. One clock step corrupts a dashboard.

Note the mechanism: a backwards *step*, not NTP *slewing*. Slewing adjusts
the clock's rate and is monotonic by construction, which is the entire
reason it exists as an alternative to stepping.

## Why this file asserts a band and not `duration >= 0`

The obvious fence is "monkeypatch the clock backwards, assert the recorded
`duration >= 0`". That is necessary and it is here, but on its own it is
not sufficient, because it is blind to the worse failure this change can
introduce.

`_fail_preflight` (`routes.py:74`) takes `start_time` as a **parameter**
from ten call sites in three handlers, so one of the 43 sites pairs its two
endpoints *across a function boundary*. `time.monotonic()` is boot-relative
and `time.time()` is Unix-epoch: they share no epoch. Convert one end of
that pair and not the other and the subtraction does not drift by
milliseconds, it yields something on the order of 1.7e9 seconds -- silently,
into the same histogram this file exists to protect, and far larger than the
negative value being fixed. A mixed-epoch duration is enormously *positive*,
so it sails straight through a lower bound.

∴ every assertion here brackets `duration` from **both** sides. The lower
bound catches the backwards step; the upper bound catches a half-converted
pair. `test_ledger_coverage.py` already brackets duration on the ledger
path, but it cannot stand in for this one: `_fail_preflight` is on the
streaming pre-flight failure path, which writes **no ledger row** at all.
Its only duration consumer is `metrics_collector.record_request_end`, so the
assertion for that pair has to be made on the metrics path -- which is what
`_RecordingMetrics` below is for.
"""

import time
from collections.abc import AsyncIterator
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
from lexora.backends.base import BackendUpstreamError
from lexora.services.metrics import MetricsCollector
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector
from tests.api.test_ledger_coverage import read_outer_clock

# How far the fake wall clock jumps backwards between two reads. Any value
# larger than a mocked request's real elapsed time makes the resulting
# `duration` negative, which is the whole point; 5 s is far above the
# millisecond-scale durations these mocked backends produce, so the sign of
# the result does not depend on how fast the runner is.
BACKWARDS_STEP_SECONDS = 5.0

# The upper half of the bracket, and the half that makes this a fence for
# the mixed-epoch failure rather than only for the backwards step. The
# derivation is local, so it can be checked here: the backends in this file
# are mocks that return immediately, so a correctly-measured duration is
# milliseconds. A half-converted pair subtracts a boot-relative reading from
# a Unix-epoch one (or the reverse) and lands near 1.7e9 -- the current value
# of the Unix epoch in seconds. 60 s therefore sits roughly three orders of
# magnitude above anything the real code can produce here and seven below
# what a mixed pair produces, so it discriminates the two without being tight
# enough to flake on a slow or heavily loaded CI runner.
MAX_PLAUSIBLE_DURATION_SECONDS = 60.0

# How far `_TickSkippingClock` jumps forward, once, in the middle of a
# handler's interval. The mechanism being reproduced is measured and small:
# `time.monotonic()` on this runner is `GetTickCount64()` and when the system
# timer interrupt is delayed it advances by TWO ticks at once -- observed steps
# of 31 and 32 ms against a nominal 15.625 ms, a factor of 2.05, taken while
# the full suite ran in another process.
#
# This constant is three orders of magnitude larger than that, and
# deliberately. The half of the test below that matters is the one asserting
# the SUPERSEDED bound goes red, and that bound is
# `duration <= perf_counter_wall + superseded_slack`: whether a real 32 ms skip
# reddens it depends on how many milliseconds of TestClient round trip this
# particular runner puts outside the handler's window. A fence whose verdict
# depends on runner speed is the flake being repaired, reintroduced. 5 s sits
# far above any plausible round trip, so the verdict is a property of the
# mechanism and not of the machine. Same reasoning, same value, as
# `BACKWARDS_STEP_SECONDS` above.
CLOCK_SKIP_SECONDS = 5.0


class _BackwardsWallClock:
    """A stand-in for the `time` module, installed into `routes`' globals.

    `time()` walks backwards: every read is `BACKWARDS_STEP_SECONDS` earlier
    than the one before, which is what a `clock_settime` step looks like from
    inside a handler that reads the clock twice.

    **Only `time()` is faked.** Everything else -- `monotonic()` above all --
    is delegated to the real module by `__getattr__`. That asymmetry is
    deliberate and load-bearing: this fake exists to demonstrate that the
    handler stopped reading the adjustable clock, so faking the clock the
    handler is supposed to have moved *to* would destroy the very contrast
    being measured.
    """

    def __init__(self) -> None:
        # A plausible present-day wall-clock reading, so that a duration
        # computed against a monotonic endpoint lands near the real 1.7e9
        # epoch offset rather than near zero.
        self._next = 1_760_000_000.0
        self.reads = 0

    def time(self) -> float:
        self.reads += 1
        value = self._next
        self._next -= BACKWARDS_STEP_SECONDS
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class _TickSkippingClock:
    """A stand-in for the `time` module whose `monotonic()` skips forward once.

    Every reading after `skip()` is `CLOCK_SKIP_SECONDS` further along than
    real time went. That is what a delayed timer interrupt does to
    `GetTickCount64()`: the clock does not advance late, it advances by more
    than a tick in one step, and `time.get_clock_info("monotonic").resolution`
    goes on reporting the nominal figure throughout.

    The skip is triggered by the backend double rather than by counting reads,
    so it lands strictly between the handler's two endpoints no matter how
    many times the handler reads the clock. Counting would make this test's
    meaning depend on an implementation detail of `routes.py`.

    **Only `monotonic()` is faked**, and `__getattr__` delegates the rest, in
    the same shape as `_BackwardsWallClock` above. The asymmetry there is the
    opposite of the one here and both are deliberate: that fake moves the
    clock the handler was supposed to stop reading, this one moves the clock
    the handler was supposed to move to.
    """

    def __init__(self, skip_seconds: float) -> None:
        self._skip_seconds = skip_seconds
        self._offset = 0.0
        self.skips = 0

    def skip(self) -> None:
        self._offset += self._skip_seconds
        self.skips += 1

    def monotonic(self) -> float:
        return time.monotonic() + self._offset

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class _ClockAttributeRecorder:
    """A pass-through stand-in for `time` that records which names are read.

    Nothing is faked. The point is the set of attribute names `routes.py`
    resolves while serving one request, which is the only mechanical way to
    state "the handler measures its interval off `monotonic` and off no other
    clock" -- a claim the bracket in `test_ledger_coverage.py` now depends on
    and which was previously readable only by grepping the source.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        self.names.add(name)
        return getattr(time, name)


# THE FOURTH CLOCK DOUBLE IS NOT HERE. `_FrozenClock` lives in
# `test_ledger_coverage.py`, next to `TestLedgerCoversEveryNonStreamingRoute::
# test_the_lower_end_is_read_off_the_handlers_own_clock` -- the only thing
# that installs it. Same `__getattr__`-delegating shape as the three above;
# it fakes `monotonic()` by never advancing it.
#
# It is filed there rather than here deliberately, and against the obvious
# pull of keeping the clock doubles together, because that pull is how this
# thread lost two fences already. `test_the_outer_clock_is_finer_than_the
# _slack` and `test_the_lower_bound_slack_keeps_its_derivation_and_its_floor`
# both lived in THIS file while their subject -- the duration bracket and its
# since-deleted `DURATION_SLACK` -- lived in that one, and both outlived their
# subject without anyone noticing. That distance is the mechanism by which a
# fence gets orphaned, not an aesthetic complaint about it. Twice in one
# thread is enough.
#
# NOR IS THE FIFTH. `_SteppingClock` lives in `test_streaming_ledger_row.py`,
# next to `TestStreamingRoutesOpenARow::test_duration_is_the_handlers_own
# _interval_and_not_a_constant` -- the only thing that installs it -- by the
# same rule and for the same reason. It fakes `monotonic()` by reading `0.0`
# until the backend double pulls its cue and the scripted advance from then
# on, which is what lets that case assert the recorded `duration` by EQUALITY
# rather than by a comparison carrying an unmeasured premise about the machine.
# `_FrozenClock` was deliberately NOT reused for it: a frozen double makes the
# handler's interval `0.0`, which is the value produced by the very mutation
# that case exists to reject.
#
# The cost of the rule is this comment: the double family is no longer
# readable in one place. Paid explicitly, because a stale pointer is a cheap
# failure and an orphaned fence is not. This register IS that payment, so it
# is kept complete: a sixth double belongs in this list on the day it is
# written.


class _RecordingMetrics(MetricsCollector):
    """A real `MetricsCollector` that also keeps every `duration` it is given.

    It subclasses rather than mocks so the Prometheus side effects still
    happen exactly as in production -- the point of interest is the number
    the handler computed, not whether the collector was called.
    """

    def __init__(self) -> None:
        super().__init__()
        self.durations: list[float] = []

    def record_request_end(self, *args: Any, **kwargs: Any) -> None:
        self.durations.append(kwargs["duration"])
        super().record_request_end(*args, **kwargs)


UPSTREAM_REFUSAL = BackendUpstreamError(
    "API error (400): declined",
    status_code=400,
    body={"type": "error", "error": {"type": "refusal", "message": "declined"}},
    backend_name="frontier",
)

CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "claude-fable-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 5},
}
COMPLETION_RESPONSE = {
    "id": "cmpl-1",
    "object": "text_completion",
    "created": 1,
    "model": "claude-fable-5",
    "choices": [{"index": 0, "text": "Hello", "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 5},
}
EMBEDDINGS_RESPONSE = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": "claude-fable-5",
    "usage": {"prompt_tokens": 11, "total_tokens": 11},
}

CHAT_COMPLETIONS_BODY = {
    "model": "frontier",
    "messages": [{"role": "user", "content": "Hi"}],
}
COMPLETIONS_BODY = {"model": "frontier", "prompt": "Hi"}
EMBEDDINGS_BODY = {"model": "frontier", "input": "Hi"}
GENERATE_BODY = {"model": "frontier", "prompt": "Hi"}
CHAT_BODY = {"model": "frontier", "messages": [{"role": "user", "content": "Hi"}]}
MESSAGES_BODY = {
    "model": "frontier",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hi"}],
}

# One route per handler that creates a `start_time`, so every handler whose
# interval endpoints must move together is driven by at least one case.
NON_STREAMING_ROUTES = [
    ("/v1/chat/completions", CHAT_COMPLETIONS_BODY),
    ("/v1/completions", COMPLETIONS_BODY),
    ("/v1/embeddings", EMBEDDINGS_BODY),
    ("/generate", GENERATE_BODY),
    ("/chat", CHAT_BODY),
    ("/v1/messages", MESSAGES_BODY),
]

# The three handlers that hand `start_time` to `_fail_preflight`. These are
# the cases that cover the one pair whose endpoints cross a function
# boundary; nothing else in the suite reaches it.
PREFLIGHT_ROUTES = [
    ("/v1/chat/completions", "chat_completions_stream", CHAT_COMPLETIONS_BODY),
    ("/v1/completions", "completions_stream", COMPLETIONS_BODY),
    ("/v1/messages", "chat_completions_stream", MESSAGES_BODY),
]


def _raising_stream(exc: BaseException) -> MagicMock:
    """A backend stream whose first `__anext__` raises, tripping the pre-flight."""

    # `usage_sink` is accepted and ignored: the handlers now hand every
    # streaming call a per-request `UsageSink` (T-streaming-ledger-row
    # PR-A), and a double that refused it would fail on `TypeError`
    # before reaching the behaviour under test here. Ignored, not
    # filled, because nothing in this file is about the ledger.
    def factory(_request: dict, usage_sink: Any = None) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            raise exc
            yield b""  # pragma: no cover - unreachable, makes this a generator

        return gen()

    return MagicMock(side_effect=factory)


def _backend() -> MagicMock:
    async def _respond(payload: dict[str, Any]) -> Any:
        async def _inner(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return payload

        return _inner

    backend = MagicMock()
    backend.chat_completions = AsyncMock(return_value=CHAT_RESPONSE)
    backend.completions = AsyncMock(return_value=COMPLETION_RESPONSE)
    backend.embeddings = AsyncMock(return_value=EMBEDDINGS_RESPONSE)
    backend.error_passthrough = False
    return backend


def _client(backend: MagicMock, metrics: MetricsCollector) -> TestClient:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=backend)
    backend_router.resolve_model = MagicMock(return_value="claude-fable-5")
    backend_router.get_backend_name_for_model = MagicMock(return_value="frontier")
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
    app.dependency_overrides[get_metrics_collector] = lambda: metrics
    app.dependency_overrides[get_cost_tracker] = lambda: None
    return TestClient(app)


def _assert_bracketed(durations: list[float]) -> None:
    """Every recorded duration is a real elapsed interval, from both sides.

    `durations` being non-empty is part of the assertion, not a precondition:
    without it a route that silently stopped recording would leave this
    vacuously true, which is how a fence turns into decoration.
    """
    assert durations, "no duration reached the metrics path"
    for duration in durations:
        assert isinstance(duration, float)
        # Lower bound: the handler read the adjustable clock, which this test
        # has stepped backwards.
        #
        # `>=`, and the difference is not cosmetic. Measured here, every one
        # of these durations is *exactly* 0.0: the backends are mocks that
        # return immediately and `time.monotonic()` is quantised to 15.625 ms
        # on this runner, so the handler starts and finishes inside one tick.
        # This bound therefore holds with zero margin, and `> 0.0` would fail
        # all nine cases every run. That is safe rather than lucky only
        # because `monotonic` is non-decreasing *by definition* -- the
        # guarantee is structural, not a measured margin, which is precisely
        # the property `time.time()` lacked and this file exists to install.
        assert duration >= 0.0, f"negative duration {duration!r} -- wall clock read"
        # Upper bound: the two endpoints of one interval came from clocks
        # with different epochs.
        assert duration <= MAX_PLAUSIBLE_DURATION_SECONDS, (
            f"implausible duration {duration!r} -- mixed-epoch subtraction"
        )


class TestPreflightDurationSurvivesABackwardsClock:
    """The pair that crosses `_fail_preflight`'s parameter boundary.

    This is the only path in the suite that reaches `_fail_preflight`'s
    subtraction, and it writes no ledger row, so `test_ledger_coverage.py`
    structurally cannot see it.
    """

    @pytest.mark.parametrize(("endpoint", "stream_attr", "body"), PREFLIGHT_ROUTES)
    def test_preflight_failure_records_a_real_interval(
        self,
        monkeypatch: pytest.MonkeyPatch,
        endpoint: str,
        stream_attr: str,
        body: dict,
    ) -> None:
        clock = _BackwardsWallClock()
        monkeypatch.setattr(routes, "time", clock)

        backend = _backend()
        backend.error_passthrough = True
        setattr(backend, stream_attr, _raising_stream(UPSTREAM_REFUSAL))
        metrics = _RecordingMetrics()

        response = _client(backend, metrics).post(
            endpoint, json={**body, "stream": True}
        )

        assert response.status_code == 400
        _assert_bracketed(metrics.durations)


def test_a_skipped_tick_cannot_break_the_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bracket's upper bound survives the clock error that falsified it.

    This replaces `test_the_outer_clock_is_finer_than_the_slack`, and the
    removal is as much of the point as the addition. That check asserted
    `perf_counter.resolution < DURATION_SLACK`, fencing the premise that the
    bracket's OUTER reading carried a quantum small enough to ignore. The
    bracket no longer has an outer clock distinct from the inner one, so that
    premise is not merely satisfied, it is gone. A green check standing over a
    premise nothing rests on is a dead-premise defect, so it was deleted
    rather than left to accumulate authority.

    That has since happened a second time, to the same pattern and for the
    same reason: `test_the_lower_bound_slack_keeps_its_derivation_and_its
    _floor` was a pure VALUE fence over `DURATION_SLACK`, and when the
    bracket's lower end became `backend_delta <= duration` the constant was
    deleted and that check lost its entire subject. It is gone too. THIS test
    is the one that survived the same deletion, and the distinction is the
    thing to keep: it never fenced the constant, it USES one to describe a
    bound that no longer ships. See the local below.

    What replaces it fences the premise the bound actually has now: that ONE
    non-decreasing clock reads both ends. Given that, and given the outer
    window strictly containing the inner one,

        m(outer_start) <= m(inner_start) <= m(inner_end) <= m(outer_end)

    so `duration <= wall` holds identically -- for a clock of any coarseness,
    and for a clock that skips ticks. The derivation is stated once, in the
    duration-bracket comment block above `SLOW` in `test_ledger_coverage.py`,
    and is not restated here.

    So this test injects the exact error that killed the old bound. A clock
    that jumps forward mid-request, triggered by the backend double so it
    lands strictly between the handler's two readings, and then both halves
    are asserted on the same request:

      1. the shipped bound holds -- and it holds because `read_outer_clock`
         resolves through `routes.time`, which is what this test patches, so
         the skip reaches BOTH ends the way a real skipped tick does;
      2. the SUPERSEDED bound, `duration <= perf_counter_wall +
         superseded_slack`, is RED on that same request.

    (2) is why this is a demonstration and not a claim. The failure it
    reproduces was reachable in the tree only about one full-suite run in
    eight, and only under load; here it is deterministic and it is on every
    run. (1) is the fence: revert `read_outer_clock` to a locally-held
    `time.perf_counter` and this test reddens on the next run rather than in
    some later eighth.
    """
    clock = _TickSkippingClock(CLOCK_SKIP_SECONDS)
    monkeypatch.setattr(routes, "time", clock)

    async def _skip_then_respond(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        clock.skip()
        return CHAT_RESPONSE

    backend = _backend()
    backend.chat_completions = AsyncMock(side_effect=_skip_then_respond)

    metrics = _RecordingMetrics()
    started = read_outer_clock()
    started_superseded = time.perf_counter()
    response = _client(backend, metrics).post(
        "/v1/chat/completions", json=CHAT_COMPLETIONS_BODY
    )
    wall = read_outer_clock() - started
    wall_superseded = time.perf_counter() - started_superseded

    assert response.status_code == 200
    # Without this the rest is vacuous in the one way that matters: a skip
    # that never fired would leave both assertions below trivially satisfied
    # and this test green while measuring nothing.
    assert clock.skips == 1, (
        f"the clock skipped {clock.skips} times, not once -- the jump did not "
        f"land inside the handler's interval and nothing below was measured"
    )
    assert metrics.durations, "no duration reached the metrics path"
    duration = metrics.durations[0]

    assert duration <= wall, (
        f"duration {duration!r} exceeds wall {wall!r} across a skipped tick: "
        f"the bracket's two ends are no longer reading one clock, which is "
        f"the only thing making its upper bound sound"
    )
    # The superseded bound reconstructed, and the ONLY constant left anywhere
    # in this bracket's neighbourhood. It is not a tolerance and nothing ships
    # behind it: it is a DESCRIPTION OF A HISTORICAL SHAPE, asserted RED, whose
    # sole job is to prove this reproduction still bites. A description may
    # carry a constant where a live bound may not -- the invariant this thread
    # settled on is zero constants in any bound that still SHIPS, not zero
    # constants in the file.
    #
    # It was `DURATION_SLACK`, imported from `test_ledger_coverage.py`, until
    # that constant's last live reader became `backend_delta <= duration` and
    # it was deleted. Inlined here rather than kept alive at module scope on
    # purpose: a shared name is reachable by a future live bound, and this
    # value must never be one again. If `monotonic`'s reported resolution
    # changes under this runner the reconstruction follows it, which is the
    # correct behaviour for a description of what the old bound WOULD have
    # admitted.
    superseded_slack = max(time.get_clock_info("monotonic").resolution, 0.001)
    assert not duration <= wall_superseded + superseded_slack, (
        f"the superseded bound (perf_counter wall {wall_superseded!r} plus "
        f"one reported tick {superseded_slack!r}) admitted duration "
        f"{duration!r} across a {CLOCK_SKIP_SECONDS}s skip -- this test is "
        f"not reproducing the failure it exists to reproduce"
    )


@pytest.mark.parametrize(("endpoint", "body"), NON_STREAMING_ROUTES)
def test_the_handler_measures_its_interval_off_monotonic_alone(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, body: dict
) -> None:
    """The other half of "one clock reads both ends", stated mechanically.

    `read_outer_clock` guarantees the bracket's outer end resolves through
    `routes.time.monotonic`. That is worth nothing on its own if `routes.py`
    stops measuring its interval with `monotonic`, and no assertion in this
    suite said which clock it reads -- the claim lived in a comment and in a
    grep count. This records the attribute names `routes.py` actually resolves
    off the `time` module while serving one request, one route per handler.

    The lower half (`monotonic` is read) and the upper half (`time` and
    `perf_counter` are not) fail differently and are asserted separately: a
    handler that read nothing would satisfy the second on its own.

    The ceiling, stated because the second half over-reaches on purpose: it
    forbids these handlers from reading any other clock AT ALL, which is
    stronger than "no other clock measures an interval". A `time.time()` added
    for a log timestamp would redden it. That is the deliberate trade -- a
    read is mechanically visible and an interval is not, and the failure this
    file exists for was exactly an interval nobody could see the ends of. A
    legitimate second reader should redden this and be argued for here, rather
    than arrive silently.
    """
    recorder = _ClockAttributeRecorder()
    monkeypatch.setattr(routes, "time", recorder)

    metrics = _RecordingMetrics()
    response = _client(_backend(), metrics).post(endpoint, json=body)

    assert response.status_code == 200
    assert "monotonic" in recorder.names, (
        f"{endpoint} resolved {sorted(recorder.names)!r} off `time` and never "
        f"`monotonic`: the bracket in test_ledger_coverage.py reads the outer "
        f"end off `routes.time.monotonic` and would no longer share a clock "
        f"with the handler"
    )
    assert not recorder.names & {"time", "perf_counter"}, (
        f"{endpoint} resolved {sorted(recorder.names & {'time', 'perf_counter'})!r} "
        f"off `time` as well as `monotonic` -- a second clock on this path is "
        f"either the adjustable one coming back or an interval endpoint that "
        f"no longer pairs with the outer reading"
    )


class TestHandlerDurationSurvivesABackwardsClock:
    """The interval pairs local to each handler, one route per handler."""

    @pytest.mark.parametrize(("endpoint", "body"), NON_STREAMING_ROUTES)
    def test_route_records_a_real_interval(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: str, body: dict
    ) -> None:
        clock = _BackwardsWallClock()
        monkeypatch.setattr(routes, "time", clock)

        metrics = _RecordingMetrics()
        response = _client(_backend(), metrics).post(endpoint, json=body)

        assert response.status_code == 200
        _assert_bracketed(metrics.durations)

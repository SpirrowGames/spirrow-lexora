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
from tests.api.test_ledger_coverage import DURATION_SLACK, read_outer_clock

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
# `duration <= perf_counter_wall + DURATION_SLACK`: whether a real 32 ms skip
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
    premise nothing rests on is the same defect this thread found in the check
    below, arrived at from the other direction, so it is deleted rather than
    left to accumulate authority.

    What replaces it fences the premise the bound actually has now: that ONE
    non-decreasing clock reads both ends. Given that, and given the outer
    window strictly containing the inner one,

        m(outer_start) <= m(inner_start) <= m(inner_end) <= m(outer_end)

    so `duration <= wall` holds identically -- for a clock of any coarseness,
    and for a clock that skips ticks. The derivation is stated once, at
    `DURATION_SLACK` in `test_ledger_coverage.py`, and is not restated here.

    So this test injects the exact error that killed the old bound. A clock
    that jumps forward mid-request, triggered by the backend double so it
    lands strictly between the handler's two readings, and then both halves
    are asserted on the same request:

      1. the shipped bound holds -- and it holds because `read_outer_clock`
         resolves through `routes.time`, which is what this test patches, so
         the skip reaches BOTH ends the way a real skipped tick does;
      2. the SUPERSEDED bound, `duration <= perf_counter_wall +
         DURATION_SLACK`, is RED on that same request.

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
    assert not duration <= wall_superseded + DURATION_SLACK, (
        f"the superseded bound (perf_counter wall {wall_superseded!r} plus "
        f"DURATION_SLACK {DURATION_SLACK!r}) admitted duration {duration!r} "
        f"across a {CLOCK_SKIP_SECONDS}s skip -- this test is not reproducing "
        f"the failure it exists to reproduce"
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


def test_the_lower_bound_slack_keeps_its_derivation_and_its_floor() -> None:
    """`DURATION_SLACK`'s VALUE is fenced against edits. Only its value.

    This check used to be called `test_the_slack_covers_the_handlers_tick_and
    _the_floor` and its first failure message used to end "the bracket is
    tighter than the clock feeding it and will flake on a true reading". Both
    have been withdrawn, and what they claimed is worth writing out because it
    is the most instructive thing in this file.

    The bracket flaked on a true reading. This check was green through it.
    `time.get_clock_info("monotonic").resolution` is a NOMINAL figure, not an
    observation: measured on this runner while the suite ran in another
    process, `time.monotonic()` steps 15 or 16 ms normally and **31 or 32 ms**
    when the timer interrupt is delayed, while `get_clock_info` reports
    0.015625 throughout. So the proposition the old name asserted -- that the
    slack covers the handler's tick -- is FALSE here, by roughly a factor of
    two, and this check is structurally incapable of noticing, because
    `DURATION_SLACK` is `max(m, 0.001)` and the `m` below is the same
    `get_clock_info` read. Both sides of the first conjunct are one nominal
    constant. It asks the clock to describe itself.

    That is worse than a tautology and it is a different failure from the one
    the tautology argument below was about. The argument below asks whether an
    EDIT can redden a check; it never asks whether the check's operands can
    express the proposition at all. Nothing here is being repaired by renaming
    it: the check keeps exactly the coverage the mutation table measured, and
    the name and message now claim exactly that and no more.

    What still rests on this line, after `T-duration-slack-underestimates-the-
    tick`: the bracket's LOWER end alone. The upper end no longer budgets a
    tick -- it reads both ends off one clock and holds identically -- so the
    falsification above no longer threatens it. It does still threaten the
    lower end, which is left in place deliberately and survives on a measured
    margin (min +0.011625 s over 32 samples of the shipped shape, +0.012625
    over 32 taken before the change) rather than on its derivation.
    That residual is recorded at `DURATION_SLACK` in `test_ledger_coverage.py`
    and is not re-argued here.

    The rest of this docstring is the original argument for the check's
    existence, unchanged, because the mutation table it carries is unaffected
    by the rescope.

    This check was deleted once and is restored here, so the argument that
    deleted it is written out rather than left to be re-derived. That argument:
    now that `DURATION_SLACK` is read off the handler's own clock this is
    `max(m, 0.001) >= m`, true for every `m`, so no platform can redden it, so
    it is a tautology and not a fence.

    Its first half is true and its conclusion does not follow. "No platform
    reddens it" and "no edit reddens it" are different properties, and only the
    second makes a check worthless. A regression test restates the correct
    implementation on purpose -- that restatement is the whole mechanism by
    which it fences later edits to it. Measured on this tree, on **Windows**
    (`m` = `monotonic` and `t` = `time` both 15.625 ms, `p` = `perf_counter`
    1e-07), one single-site mutation of the derivation line in
    `test_ledger_coverage.py` per cell, `__pycache__` cleared between cells,
    classified by pytest exit code:

        edit to the derivation line           becomes        this test
        ------------------------------------  -------------  ----------
        (control -- unmutated)                max(m, 0.001)  green
        hardcode the floor                    0.001          RED
        max -> min                            min(m, 0.001)  RED
        wrong clock: perf_counter for m       max(p, 0.001)  RED
        floor dropped                         m              green here
        regress to get_clock_info("time")     max(t, 0.001)  green

    Three of five. The first conjunct is what reddens all three, and on this
    platform it binds with **equality** -- `0.015625 >= 0.015625`, zero margin.

    And it reddens them *deterministically*, which is the property that decides
    this and the one the alternative did not have. The six cells above were run
    five separate times -- the two assertions below byte-identical throughout --
    and every cell gave the same answer every time. Compare what covered the
    hardcode edit while this check was absent: nothing but incidental flake in
    the bracket detectors of `test_ledger_coverage.py`, and three independent
    ten-run measurements of that same tree disagreed with each other -- 10, 9
    and 7 runs red out of 10, with anywhere from 0 to 3 of the four detectors
    failing in a single run. Those three numbers are quoted only against each
    other, as evidence that the rate moves. None of them is this file's
    estimate of how much that residual covers, and no such estimate is given
    anywhere in this change, because it would be the next unchecked claim.

    Two conjuncts rather than one `>= max(m, 0.001)`, and the reason is not
    just that they fail with different messages. Evaluating the two expressions
    over the same six rows shows them to be complementary rather than
    overlapping: on this box the tick conjunct reddens 3 of the 5 edits and the
    floor conjunct reddens **0**, and substituting a fine-grained `m` of 1e-09
    into the same arithmetic flips it exactly -- tick 0, floor 2, those being
    the floor-dropped and `max -> min` rows. Neither conjunct alone covers both
    platform classes. `>= m` on its own would be a fence with a hole in it on a
    fine-grained runner; `>= 0.001` on its own would be a fence with a hole in
    it here. The floor conjunct is also not a restatement of the derivation --
    it is an independent statement of design intent, which is what lets it
    survive an edit that rewrites the derivation entirely.

    The floor half shipped as a **prediction, not a measurement**: it reddens
    nothing in the table above, because `m` is 15.625 ms here and the floor is
    a no-op, and the 1e-09 column was arithmetic on these two expressions
    rather than an observation of Linux. The observation has since been made.
    The floor-dropped cell was pushed on a throwaway branch that was never
    merged and CI ran it on `ubuntu-latest`: run 34053301337, **2 failed, 615
    passed**, this test among the two, failing on the second conjunct. How a
    push to such a branch ran CI at all, since `.github/workflows/ci.yml` fires
    only on `pull_request` and on pushes to `main`/`develop`: that branch
    carried a second one-line commit widening the push trigger to include
    `'throwaway/**'`, job body otherwise untouched. That is why the run exists
    without a PR ever having been opened for a deliberate mutation, and it is
    the reason to read the sentence above as history rather than as a recipe --
    pushing this edit to a throwaway branch today runs nothing. The branch is
    not itself the record; the run id is. Reproducing the cell is those two
    one-line edits and nothing else: the trigger, and the floor-dropped row of
    the table above. Verbatim from that run's log, and dated by the run id
    rather than kept in step with the lines below --

        AssertionError: DURATION_SLACK 1e-09 is below the 1 ms floor ...
        assert 1e-09 >= 0.001

    -- while the first conjunct passed in that same run. Two things follow, and
    the second was not predicted. `monotonic` on that runner does report 1e-09:
    the premise the whole 1e-09 column rested on is now read off a failure
    message instead of assumed. And the floor conjunct does catch the
    floor-dropped edit there, so the two conjuncts are complementary by
    measurement on both platform classes rather than by arithmetic on one.
    The unpredicted part was the other failure in that run, and its record is
    kept here because the check that held it has since been deleted: it was
    `test_the_outer_clock_is_finer_than_the_slack`, asserting
    `perf_counter.resolution < DURATION_SLACK`, and on that Linux runner
    dropping the floor collapsed the slack onto `m` = 1e-09, which is exactly
    `p`, so it failed on `assert 1e-09 < 1e-09`. On Windows the same edit
    leaves it green, the floor being a no-op where the slack is 15.625 ms
    either way. That check has been replaced by
    `test_a_skipped_tick_cannot_break_the_upper_bound`, because the premise it
    fenced -- the bracket's outer reading carrying a quantum of its own -- no
    longer exists once both ends read one clock.

    The ceiling, so that no more is claimed for this than it gives: it guards
    the *value* of `DURATION_SLACK` and never its *provenance*. The drift this
    change exists to remove -- deriving off `time` instead of `monotonic` -- is
    value-invariant on both platforms this project runs on, which is why the
    last row above is green. No value assertion can catch that one, and this
    one does not pretend to.
    """
    m = time.get_clock_info("monotonic").resolution
    assert DURATION_SLACK >= m, (
        f"DURATION_SLACK {DURATION_SLACK!r} no longer covers the REPORTED "
        f"monotonic resolution {m!r}: the derivation line in "
        f"test_ledger_coverage.py has been edited away from `max(m, 0.001)`. "
        f"This says nothing about the clock's real quantum, which is larger "
        f"than the report -- see this test's docstring"
    )
    assert DURATION_SLACK >= 0.001, (
        f"DURATION_SLACK {DURATION_SLACK!r} is below the 1 ms floor that keeps "
        f"the bracket from going exactly tight on a fine-grained platform"
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

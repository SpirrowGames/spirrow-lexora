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
from tests.api.test_ledger_coverage import DURATION_SLACK

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

    def factory(_request: dict) -> AsyncIterator[bytes]:
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


def test_the_outer_clock_is_finer_than_the_slack() -> None:
    """`test_ledger_coverage.py`'s bracket budgets for ONE clock's quantum.

    That bracket widens each end by `DURATION_SLACK`, and it spends that width
    on the handler's inner clock and on nothing else. So it treats the outer
    reading `wall` -- taken off `time.perf_counter()` -- as exact. That is
    legitimate only while `perf_counter`'s own quantum is small next to the
    slack; where it is not, the outer reading carries a quantum the bracket
    never budgets for, and the derivation stops being true while the numbers
    still look reasonable. The derivation itself is stated once, at
    `DURATION_SLACK` in `test_ledger_coverage.py`, and is not restated here.

    Nothing in the repository made that so. It is a property of whichever
    platform runs the suite, and it was stated only in a comment -- "resolution
    1e-07 here" -- which is precisely the unchecked-claim shape this suite
    keeps finding. So it is measured here instead of asserted there.

    The derivation is not quoted here either, and that is deliberate rather
    than terse. A verbatim sentence of it used to sit at the top of this
    docstring, and it went stale inside the very commit that edited the
    original: one file went on arguing from a word the other file had already
    dropped, and the two disagreed from the moment they were written. Name the
    other file and the claim, and let it speak for itself. A copy of another
    file's prose is a second thing to keep true.

    This is the second of the derivation's two premises. The first -- that the
    slack is not finer than the handler's tick -- is fenced separately, by
    `test_the_slack_covers_the_handlers_tick_and_the_floor` below. Neither
    stands in for the other, and they fail in different ways: a *platform*
    reddens this one with no edit at all, whereas no platform can redden that
    one and only an *edit* to the derivation can. Both are kept. Why the second
    survives despite being unreddenable by any platform is measured in its own
    docstring, not argued here. An edit can redden this one too, but only on a
    platform where that edit actually moves `DURATION_SLACK` -- see the last
    paragraph, which is an observation and not a symmetry argument.

    Measured where this was written: `perf_counter` reports 1e-07 against a
    slack of 15.625 ms, five orders of magnitude, so unlike the check it
    replaces this one does not hold by a whisker. The Linux runner is no longer
    an assumption either: on `ubuntu-latest` both `perf_counter` and `monotonic`
    report 1e-09, read off the failure text of CI run 34053301337 rather than
    assumed. What follows from those two readings is arithmetic, and is marked
    as such: unmutated, `max(1e-09, 0.001)` makes the slack the floor, so there
    this holds by six orders rather than five.

    That run also reddened this assertion, which nothing here predicted. It was
    a throwaway branch, never merged, carrying one edit: the 1 ms floor dropped
    from `DURATION_SLACK`. On Linux the floor is what the slack IS, so dropping
    it collapses the slack onto `m` = 1e-09, which is exactly `p`, and this
    check fails with `assert 1e-09 < 1e-09`. On Windows the floor is a no-op --
    the slack stays 15.625 ms whether it is there or not -- so the same edit
    leaves this green. Margin width is not what decides it; whether the edit
    moves `DURATION_SLACK` on that platform is. The check is therefore not
    merely a platform detector: an edit to the line it guards reddens it too,
    on a platform in the class this ships to.
    """
    outer = time.get_clock_info("perf_counter").resolution
    assert outer < DURATION_SLACK, (
        f"perf_counter resolution {outer!r} is not finer than DURATION_SLACK "
        f"{DURATION_SLACK!r}: `wall` in test_ledger_coverage.py carries a "
        f"quantum of its own that the duration bracket does not budget for"
    )


def test_the_slack_covers_the_handlers_tick_and_the_floor() -> None:
    """`DURATION_SLACK` must clear the handler's tick, and must clear the floor.

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
    passed**, this test among the two, failing on the second conjunct. Verbatim
    from that run's log, and dated by the run id rather than kept in step with
    the lines below --

        AssertionError: DURATION_SLACK 1e-09 is below the 1 ms floor ...
        assert 1e-09 >= 0.001

    -- while the first conjunct passed in that same run. Two things follow, and
    the second was not predicted. `monotonic` on that runner does report 1e-09:
    the premise the whole 1e-09 column rested on is now read off a failure
    message instead of assumed. And the floor conjunct does catch the
    floor-dropped edit there, so the two conjuncts are complementary by
    measurement on both platform classes rather than by arithmetic on one.
    The unpredicted part was the other failure in that run, in
    `test_the_outer_clock_is_finer_than_the_slack`; it is recorded there.

    The ceiling, so that no more is claimed for this than it gives: it guards
    the *value* of `DURATION_SLACK` and never its *provenance*. The drift this
    change exists to remove -- deriving off `time` instead of `monotonic` -- is
    value-invariant on both platforms this project runs on, which is why the
    last row above is green. No value assertion can catch that one, and this
    one does not pretend to.
    """
    m = time.get_clock_info("monotonic").resolution
    assert DURATION_SLACK >= m, (
        f"DURATION_SLACK {DURATION_SLACK!r} is finer than the handler's tick "
        f"{m!r}: test_ledger_coverage.py's bracket is tighter than the clock "
        f"feeding it and will flake on a true reading"
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

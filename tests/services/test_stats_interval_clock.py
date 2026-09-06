"""`RequestStats` measures an interval, so it must not read the wall clock.

## What this file fences

`services/stats.py` had three `time.time()` reads: the `start_time` creator in
`start_request`, the `end_time` creator in `complete_request`, and the live
branch of `RequestStats.duration`. `time.time()` is `CLOCK_REALTIME` and is
`adjustable` (measured on this runner: True), so an NTP *step* backwards
between two reads makes `duration` negative.

## Why the harm outlives the request, which is the reason this is worth a fence

`StatsCollector._record_stats` does

    self._total_duration += stats.duration
    self._stats.average_duration = self._total_duration / total_requests

`_total_duration` is a **running accumulator**, reset only by `reset()`. A
negative `duration` does not dirty one row and pass -- it decrements the
accumulator, and the `average_duration_seconds` reported by `/stats` stays
wrong for the whole process lifetime. A corrupted `rate()` heals when the
window rolls; a corrupted running mean does not heal at all.

∴ the assertions below are on the **accumulator**, not only on a single
`duration`. A fence that checks one request's duration cannot see the class of
harm that makes this worth fixing.

## Why every assertion brackets from BOTH sides

The obvious fence is "assert `duration >= 0`". It is necessary, it is here, and
on its own it is **not sufficient**. `time.monotonic()` and `time.time()` share
no epoch, so converting one end of a pair and not the other does not produce a
small error -- it produces roughly +/- 1.76e9 seconds. Which sign you get
depends on which end you converted, and the enormously *positive* case sails
straight through a lower bound.

The two bounds are therefore not redundant: each catches a direction the other
is blind to. `test_the_two_bounds_catch_opposite_directions` pins that down
rather than leaving it as a claim.

## The zero-margin lower bound is deliberate

The durations observed here are frequently *exactly* `0.0` -- there is no work
between the two reads, and `time.monotonic()` is quantised to 15.625 ms on this
runner, so both readings land in one tick. `>= 0.0` therefore holds with **zero
margin**, and `> 0.0` would fail every run. That is sound because
`time.monotonic()` is non-decreasing *by definition*, not because of any
statistical slack -- and zero margin is not zero discrimination: the values
this bound rejects are `-5.0` and `-1.76e9`.
"""

import time

import pytest

from lexora.services import stats as stats_module
from lexora.services.stats import RequestStats, StatsCollector

# Captured before anything is monkeypatched, and used to hand the fake clock a
# real `monotonic` to delegate to.
_REAL_MONOTONIC = time.monotonic
_REAL_PERF_COUNTER = time.perf_counter

# Far above the sub-millisecond durations this collector produces with no work
# between the two clock reads, and three orders of magnitude below the ~1.76e9
# that a mixed-epoch subtraction yields. Nothing has to be tuned here: the two
# populations it separates differ by nine orders of magnitude.
MAX_PLAUSIBLE_DURATION_SECONDS = 5.0

# Enough requests that a per-request error accumulates visibly rather than
# hiding in one row.
REQUESTS = 8


class _BackwardsWallClock:
    """A stand-in for the `time` module whose `time()` steps *backwards*.

    Only `time()` is faked. `monotonic()` and `perf_counter()` delegate to the
    real module on purpose: faking the clock the code is moving *to* would
    destroy the contrast this file exists to measure, and the tests would then
    pass for a reason that has nothing to do with the fix.

    The step is a step, not a slew -- that is the failure being fenced. A slew
    (`adjtime`) is rate-limited and cannot invert an interval; a step
    (`settimeofday`, an NTP correction, a VM guest resync) can.
    """

    #: A plausible present-day wall-clock reading, so that a mixed-epoch
    #: subtraction lands at ~1.76e9 rather than at some small number that a
    #: loose bound might let through.
    WALL_EPOCH = 1_759_600_000.0

    #: Large enough to be unmistakable, small enough that it is obviously not
    #: an epoch confusion -- this is the *in-epoch* failure mode.
    STEP = -5.0

    def __init__(self) -> None:
        self._next = self.WALL_EPOCH
        self.reads = 0

    def time(self) -> float:
        value = self._next
        self._next += self.STEP
        self.reads += 1
        return value

    monotonic = staticmethod(_REAL_MONOTONIC)
    perf_counter = staticmethod(_REAL_PERF_COUNTER)


@pytest.fixture
def backwards_clock(monkeypatch: pytest.MonkeyPatch) -> _BackwardsWallClock:
    """Install the backwards wall clock as `stats.py`'s `time` module."""
    fake = _BackwardsWallClock()
    monkeypatch.setattr(stats_module, "time", fake)
    return fake


def _assert_bracketed(value: float, what: str) -> None:
    """Bracket a duration-like quantity from both sides. See the module docstring."""
    assert isinstance(value, float)
    # Lower: the negative duration this change exists to prevent. Holds with
    # zero margin by design -- `> 0.0` would fail every run.
    assert value >= 0.0, f"{what} is negative ({value!r}) -- a wall clock was read"
    # Upper: a mixed-epoch subtraction, which is enormously POSITIVE and which
    # the lower bound cannot see.
    assert value <= MAX_PLAUSIBLE_DURATION_SECONDS, (
        f"{what} is implausible ({value!r}) -- mixed-epoch subtraction"
    )


def test_average_duration_survives_a_backwards_wall_clock(
    backwards_clock: _BackwardsWallClock,
) -> None:
    """The quantity `/stats` publishes stays a real elapsed time."""
    collector = StatsCollector()
    for _ in range(REQUESTS):
        stats = collector.start_request("/v1/chat/completions", "gpt-4")
        collector.complete_request(stats, success=True)

    payload = collector.get_stats()
    assert payload["total_requests"] == REQUESTS
    _assert_bracketed(payload["average_duration_seconds"], "average_duration_seconds")
    # Stronger and more direct than the bracket: the bracket catches the
    # *consequence* of reading the adjustable clock, this catches the read
    # itself, including on a runner whose wall clock happens not to step.
    assert backwards_clock.reads == 0, (
        f"stats.py read the faked wall clock {backwards_clock.reads} times; "
        "every one of its interval endpoints must read time.monotonic()"
    )


def test_the_running_accumulator_never_moves_backwards(
    backwards_clock: _BackwardsWallClock,
) -> None:
    """`_total_duration` is the thing that does not heal. Watch it directly."""
    collector = StatsCollector()
    observed = [collector._total_duration]
    for _ in range(REQUESTS):
        stats = collector.start_request("/v1/chat/completions", "gpt-4")
        collector.complete_request(stats, success=True)
        observed.append(collector._total_duration)

    for index, (previous, current) in enumerate(zip(observed, observed[1:])):
        assert current >= previous, (
            f"request {index} decremented the running accumulator "
            f"({previous!r} -> {current!r}); /stats stays wrong until reset()"
        )
    _assert_bracketed(observed[-1] / REQUESTS, "mean of the accumulator")


def test_live_duration_survives_a_backwards_wall_clock(
    backwards_clock: _BackwardsWallClock,
) -> None:
    """The `end_time is None` branch reads a clock of its own; fence it too."""
    collector = StatsCollector()
    stats = collector.start_request("/v1/chat/completions", "gpt-4")

    # Precondition, not decoration: with `end_time` set this would exercise the
    # completed branch and say nothing about the live one.
    assert stats.end_time is None

    _assert_bracketed(stats.duration, "live duration")


def test_the_two_bounds_catch_opposite_directions() -> None:
    """Neither bound is decoration: each is blind to what the other catches.

    Built by hand rather than by mutating the source, so the property is
    checked on every run instead of only in the round that changed the code.
    """
    # Premise, measured here rather than assumed: the two clocks are far enough
    # apart that a mixed pair lands outside the bracket. If a platform ever
    # made them close, this fence would silently lose its power -- so it is
    # checked at runtime on whatever platform runs the suite.
    gap = abs(time.time() - _REAL_MONOTONIC())
    assert gap > 100 * MAX_PLAUSIBLE_DURATION_SECONDS, (
        "the wall and monotonic epochs are only "
        f"{gap} s apart on this platform; the bounds below no longer separate "
        "a mixed-epoch pair from a real interval"
    )

    def _make(start: float, end: float) -> RequestStats:
        return RequestStats(
            endpoint="/v1/chat/completions",
            model="gpt-4",
            user_id=None,
            start_time=start,
            end_time=end,
        )

    # `start_time` converted, `end_time` left on the wall clock: enormously
    # POSITIVE. The lower bound is blind to this one.
    half_forward = _make(_REAL_MONOTONIC(), _BackwardsWallClock.WALL_EPOCH)
    assert half_forward.duration > MAX_PLAUSIBLE_DURATION_SECONDS
    assert half_forward.duration >= 0.0  # ... which is why >= 0 is not enough

    # The other half-conversion: enormously NEGATIVE. Caught by the lower bound
    # and invisible to the upper one.
    half_backward = _make(_BackwardsWallClock.WALL_EPOCH, _REAL_MONOTONIC())
    assert half_backward.duration < 0.0
    assert half_backward.duration <= MAX_PLAUSIBLE_DURATION_SECONDS


def test_stats_payload_never_exposes_a_raw_clock_reading() -> None:
    """A monotonic reading is meaningless outside the process that took it.

    Before this change, publishing `start_time` would have been merely useless;
    after it, the value has no defined origin at all. This is the mirror-image
    error to the one just fixed -- the backends' `int(time.time())` `created`
    fields are absolute *by protocol* and correctly stay on the wall clock.
    """
    collector = StatsCollector()
    stats = collector.start_request("/v1/chat/completions", "gpt-4")
    collector.complete_request(stats, success=True)

    payload = collector.get_stats()
    assert "start_time" not in payload
    assert "end_time" not in payload

"""A-15-2b: the data-controls record that keeps codex closed unless a human
checked the ChatGPT account's training setting less than 30 days ago
(T-naysayer-codex-backend msg-535), hot-reloaded, with a stateful daily
pre-expiry notice (kept by the human's decision after msg-536)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from lexora.backends.codex import CodexBackend
from lexora.backends.codex_data_controls import (
    DATA_CONTROLS_TTL,
    DATA_CONTROLS_WARN_BEFORE,
    REASON_DATA_CONTROLS_UNVERIFIED,
    RUNBOOK_POINTER,
    DataControls,
    parse_verified_at,
)
from lexora.backends.codex_verification import CodexStateStore
from lexora.backends.factory import create_backend
from lexora.backends.fallback import NOTICE_MAX_CHARS, FallbackBackend
from lexora.config import BackendSettings
from tests.backends.test_codex import record_pass
from tests.backends.test_fallback import (  # noqa: F401 - fixture
    REQUEST,
    Clock,
    _capture_posts,
    _make,
    _sends,
    wrappers,
)

T0 = datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc)


def _write(path: Path, verified_at: datetime | str | None) -> None:
    """Write the record the way the RUNBOOK does: to a temp file, then an
    atomic rename (a new inode even when mtime / size would not change)."""
    body = "" if verified_at is None else f'data_controls_verified_at: "{verified_at if isinstance(verified_at, str) else verified_at.isoformat()}"\n'
    tmp = path.with_suffix(".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def _dc(tmp_path: Path, clock: Clock, verified_at: datetime | str | None = T0) -> tuple[DataControls, Path]:
    path = tmp_path / "dc_record.yaml"
    if verified_at is not None:
        _write(path, verified_at)
    return DataControls(path, clock=clock), path


def _clock(at: datetime) -> Clock:
    c = Clock()
    c.now = at
    return c


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


class TestRecord:
    def test_valid_inside_ttl(self, tmp_path: Path) -> None:
        dc, _ = _dc(tmp_path, _clock(T0 + timedelta(days=1)))
        state = dc.state()
        assert state.valid
        assert state.verified_at == T0
        assert state.expires_at == T0 + DATA_CONTROLS_TTL

    def test_missing_file_is_closed(self, tmp_path: Path) -> None:
        dc, _ = _dc(tmp_path, _clock(T0), verified_at=None)
        assert dc.state().detail == "missing"

    @pytest.mark.parametrize(
        "body",
        ["", "- a list\n", "other_key: 1\n", 'data_controls_verified_at: "not a time"\n',
         'data_controls_verified_at: "2026-09-01T03:00:00"\n'],
        ids=["empty", "list", "no-key", "garbage", "naive"],
    )
    def test_bad_content_is_closed(self, tmp_path: Path, body: str) -> None:
        path = tmp_path / "dc.yaml"
        path.write_text(body, encoding="utf-8")
        assert DataControls(path, clock=_clock(T0)).state().detail == "invalid"

    def test_unparseable_yaml_is_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "dc.yaml"
        path.write_text("data_controls_verified_at: [unclosed\n", encoding="utf-8")
        assert DataControls(path, clock=_clock(T0)).state().detail == "unreadable"

    def test_expired_at_exactly_ttl(self, tmp_path: Path) -> None:
        clock = _clock(T0 + DATA_CONTROLS_TTL - timedelta(seconds=1))
        dc, _ = _dc(tmp_path, clock)
        assert dc.state().valid
        clock.now = T0 + DATA_CONTROLS_TTL
        assert dc.state().detail == "expired"

    def test_future_timestamp_is_refused(self, tmp_path: Path) -> None:
        """A typo in the year must not extend the validity."""
        dc, _ = _dc(tmp_path, _clock(T0), verified_at=T0 + timedelta(days=365))
        assert dc.state().detail == "future"

    def test_z_suffix_and_offsets_parse(self) -> None:
        assert parse_verified_at("2026-09-01T03:00:00Z") == T0
        assert parse_verified_at("2026-09-01T12:00:00+09:00") == T0
        assert parse_verified_at(T0) == T0
        assert parse_verified_at(1234) is None

    def test_days_left_rounds_up(self, tmp_path: Path) -> None:
        dc, _ = _dc(tmp_path, _clock(T0))
        state = dc.state()
        assert state.days_left(T0 + DATA_CONTROLS_TTL - timedelta(days=6, hours=23)) == 7
        assert state.days_left(T0 + DATA_CONTROLS_TTL - timedelta(hours=1)) == 1
        assert state.days_left(T0 + DATA_CONTROLS_TTL) == 0

    def test_ttl_and_warning_are_not_configurable(self) -> None:
        """No config field can lengthen the TTL or switch the check off."""
        from lexora.config import CodexSettings

        fields = set(CodexSettings.model_fields)
        assert "data_controls_file" in fields
        assert not {f for f in fields if "ttl" in f or "data_controls" in f} - {"data_controls_file"}


class TestHotReload:
    def test_updating_the_file_takes_effect_without_a_new_object(self, tmp_path: Path) -> None:
        clock = _clock(T0 + DATA_CONTROLS_TTL + timedelta(days=1))
        dc, path = _dc(tmp_path, clock)
        assert dc.state().detail == "expired"
        _write(path, clock.now - timedelta(minutes=5))
        assert dc.state().valid
        path.unlink()
        assert dc.state().detail == "missing"
        _write(path, clock.now)
        assert dc.state().valid

    def test_same_size_rewrite_is_seen(self, tmp_path: Path) -> None:
        """Only one digit changes: the size is equal, the rename changes the inode."""
        clock = _clock(datetime(2026, 10, 15, tzinfo=timezone.utc))
        dc, path = _dc(tmp_path, clock, verified_at="2026-09-01T03:00:00Z")
        assert dc.state().detail == "expired"
        _write(path, "2026-10-01T03:00:00Z")
        assert dc.state().valid

    def test_unchanged_file_is_not_reread(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dc, _ = _dc(tmp_path, _clock(T0))
        dc.state()
        reads: list[Any] = []
        original = Path.read_text

        def counting(self: Path, *a: Any, **k: Any) -> str:
            reads.append(self)
            return original(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", counting)
        dc.state()
        dc.state()
        assert reads == []


# --------------------------------------------------------------------------
# The codex gate
# --------------------------------------------------------------------------


def _codex(tmp_path: Path, data_controls: DataControls | None) -> CodexBackend:
    from tests.backends.test_codex import make_backend

    backend = make_backend(tmp_path)
    backend.data_controls = data_controls
    return backend


class TestGate:
    async def test_no_record_configured_is_closed(self, tmp_path: Path) -> None:
        backend = _codex(tmp_path, None)
        record_pass(backend)
        availability = await backend.codex_availability()
        assert availability.reason == REASON_DATA_CONTROLS_UNVERIFIED
        assert availability.error is not None

    async def test_expired_then_reverified_opens_without_restart(self, tmp_path: Path) -> None:
        clock = _clock(datetime.now(timezone.utc))
        dc, path = _dc(tmp_path, clock, verified_at=clock.now - DATA_CONTROLS_TTL - timedelta(hours=1))
        backend = _codex(tmp_path, dc)
        record_pass(backend)
        assert (await backend.codex_availability()).reason == REASON_DATA_CONTROLS_UNVERIFIED
        _write(path, clock.now)
        assert (await backend.codex_availability()).open

    async def test_a_latch_or_missing_verification_wins(self, tmp_path: Path) -> None:
        """Record checks come first: an unverified backend reads as such."""
        backend = _codex(tmp_path, None)
        assert (await backend.codex_availability()).reason == "verification_missing"

    async def test_data_controls_win_over_a_quota_hold(self, tmp_path: Path) -> None:
        backend = _codex(tmp_path, None)
        record_pass(backend)
        backend._quota_hold_until = datetime.now(timezone.utc) + timedelta(hours=1)
        availability = await backend.codex_availability()
        assert availability.reason == REASON_DATA_CONTROLS_UNVERIFIED
        assert availability.quota_hold_until is not None

    async def test_health_check_is_false_without_a_record(self, tmp_path: Path) -> None:
        backend = _codex(tmp_path, None)
        record_pass(backend)
        assert await backend.health_check() is False

    def test_factory_wires_the_configured_file(self, tmp_path: Path) -> None:
        target = tmp_path / "dc.yaml"
        backend = create_backend(
            "codex",
            BackendSettings(
                type="codex",
                models=["gpt-6.1-sol"],
                codex={
                    "codex_home": str(tmp_path / "home"),
                    "state_db_path": str(tmp_path / "codex.db"),
                    "data_controls_file": str(target),
                },
            ),
        )
        assert isinstance(backend, CodexBackend)
        assert backend.data_controls is not None
        assert backend.data_controls.path == target

    def test_file_is_outside_the_config_hash(self, tmp_path: Path) -> None:
        """Re-verifying must not close the verification gate."""
        a = CodexBackend(codex_home="/srv/codex", state_store=CodexStateStore(tmp_path / "s.db"))
        b = CodexBackend(
            codex_home="/srv/codex",
            state_store=a.state_store,
            data_controls=DataControls(tmp_path / "x.yaml"),
        )
        assert a.config_hash() == b.config_hash()


# --------------------------------------------------------------------------
# Fallback notices
# --------------------------------------------------------------------------


class TestFallbackNotices:
    async def test_unverified_falls_back_with_reason_and_runbook_then_ends(
        self, tmp_path: Path, wrappers: list
    ) -> None:
        now = datetime.now(timezone.utc)
        clock = _clock(now)
        w = _make(wrappers, tmp_path, clock=clock)
        posted = _capture_posts(w)
        dc, path = _dc(tmp_path, _clock(now), verified_at=now - DATA_CONTROLS_TTL - timedelta(days=1))
        w.primary.data_controls = dc
        await w.chat_completions(REQUEST)
        assert len(w.fallback.calls) == 1  # type: ignore[attr-defined]
        await _sends(w)
        assert posted[0].startswith(f"[Lexora naysayer] fallback STARTED: reason={REASON_DATA_CONTROLS_UNVERIFIED} ")
        assert f"runbook={RUNBOOK_POINTER}" in posted[0]
        assert len(posted[0]) <= NOTICE_MAX_CHARS
        # Re-verified: the next request is codex's; fallback ENDED goes out
        # from the tick once the last fallback is 15 minutes old (msg-685).
        _write(path, now)
        await w.chat_completions(REQUEST)
        await _sends(w)
        assert len(w.fallback.calls) == 1  # type: ignore[attr-defined]
        assert not posted[-1].startswith("[Lexora naysayer] fallback ENDED")
        clock.now = clock.now + timedelta(minutes=15)
        await w.notice_tick()
        assert posted[-1].startswith("[Lexora naysayer] fallback ENDED")

    async def test_other_reasons_carry_no_runbook_pointer(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, verified=False)
        posted = _capture_posts(w)
        await w.chat_completions(REQUEST)
        await _sends(w)
        assert "runbook=" not in posted[0]


class TestExpiringNotice:
    def _setup(self, tmp_path: Path, wrappers: list, at: datetime, ok: bool = True) -> tuple[FallbackBackend, list[str], Clock]:
        clock = _clock(at)
        w = _make(wrappers, tmp_path, clock=clock)
        posted = _capture_posts(w, ok=ok)
        w.primary.data_controls, _ = _dc(tmp_path, clock)
        return w, posted, clock

    async def test_nothing_before_the_window(self, tmp_path: Path, wrappers: list) -> None:
        at = T0 + DATA_CONTROLS_TTL - DATA_CONTROLS_WARN_BEFORE - timedelta(minutes=1)
        w, posted, _ = self._setup(tmp_path, wrappers, at)
        await w.notice_tick()
        assert posted == []

    async def test_once_per_utc_day_inside_the_window(self, tmp_path: Path, wrappers: list) -> None:
        start = T0 + DATA_CONTROLS_TTL - DATA_CONTROLS_WARN_BEFORE
        w, posted, clock = self._setup(tmp_path, wrappers, start)
        await w.notice_tick()
        assert len(posted) == 1
        assert posted[0].startswith("[Lexora naysayer] data controls EXPIRING (mode=fallback): codex stops in 7 day(s) ")
        assert f"runbook={RUNBOOK_POINTER}" in posted[0]
        assert len(posted[0]) <= NOTICE_MAX_CHARS
        clock.now = start + timedelta(hours=12)  # same UTC day (03:00 -> 15:00)
        await w.notice_tick()
        assert len(posted) == 1
        clock.now = start + timedelta(days=1)
        await w.notice_tick()
        assert len(posted) == 2
        assert "codex stops in 6 day(s)" in posted[1]

    async def test_failed_post_is_retried_on_the_next_tick(self, tmp_path: Path, wrappers: list) -> None:
        start = T0 + DATA_CONTROLS_TTL - timedelta(days=2)
        w, posted, _ = self._setup(tmp_path, wrappers, start, ok=False)
        await w.notice_tick()
        await w.notice_tick()
        assert len(posted) == 2  # both attempts, neither marked sent
        assert w._expiry_warned_on is None

    async def test_nothing_once_expired(self, tmp_path: Path, wrappers: list) -> None:
        """Codex is closed then: the fallback notices take over."""
        w, posted, _ = self._setup(tmp_path, wrappers, T0 + DATA_CONTROLS_TTL)
        await w.notice_tick()
        assert posted == []

    async def test_reverifying_stops_the_warnings(self, tmp_path: Path, wrappers: list) -> None:
        start = T0 + DATA_CONTROLS_TTL - timedelta(days=3)
        w, posted, clock = self._setup(tmp_path, wrappers, start)
        await w.notice_tick()
        assert len(posted) == 1
        _write(w.primary.data_controls.path, clock.now)  # type: ignore[union-attr]
        clock.now = start + timedelta(days=1)
        await w.notice_tick()
        assert len(posted) == 1

    async def test_start_runs_the_loop_before_any_fallback(self, tmp_path: Path, wrappers: list) -> None:
        w, _, _ = self._setup(tmp_path, wrappers, T0)
        assert w._loop_task is None
        w.start()
        assert w._loop_task is not None and not w._loop_task.done()

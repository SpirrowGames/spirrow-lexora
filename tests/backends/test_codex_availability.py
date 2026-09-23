"""PR-2a: ``codex_availability()``, the quota hold, and subprocess reaping.

T-naysayer-codex-backend msg-421 S-1 as amended by msg-424 S-1' and
msg-429 S-1'' (endorsed by Einstein after msg-429), plus the #52 PR-gate
finding (msg-427) extended to every spawn. The fake CLI and the helpers
come from ``test_codex``.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from lexora.backends import codex as codex_mod
from lexora.backends.codex import (
    QUOTA_HOLD_FALLBACK,
    CodexBackend,
    CodexLaunchError,
    CodexNotVerifiedError,
    CodexQuotaError,
)
from lexora.backends.codex_verification import CodexStateStore
from tests.backends.test_codex import (
    FAKE_CLI,
    REQUEST,
    VERSION,
    _assert_dead,
    _host_env,
    _until_ticking,
    make_backend,
    record_pass,
)

TABLES = (
    "codex_gate_log",
    "codex_verification",
    "codex_runtime_violation",
    "codex_clearance",
    "codex_run",
    "codex_run_finished",
    "codex_run_clearance",
)


def _row_counts(backend: CodexBackend) -> dict[str, int]:
    with sqlite3.connect(backend.state_store.db_path) as conn:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


def _count_versions(backend: CodexBackend) -> list[int]:
    """Replace ``_codex_version`` with a counting stub; return the counter."""
    calls = [0]

    async def version() -> str:
        calls[0] += 1
        return VERSION

    backend._codex_version = version  # type: ignore[method-assign]
    return calls


def _hold(backend: CodexBackend, minutes: int = 30) -> datetime:
    until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    backend._quota_hold_until = until
    return until


# ---- the five ways the records can close, as setups -----------------------


def _schema_outdated(backend: CodexBackend, mp: pytest.MonkeyPatch) -> None:
    mp.setattr(backend.state_store, "schema_current", False)


def _state_unreadable(backend: CodexBackend, mp: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise sqlite3.OperationalError("disk I/O error")

    mp.setattr(backend.state_store, "unfinished_runs", boom)


def _state_unwritable(backend: CodexBackend, mp: pytest.MonkeyPatch) -> None:
    backend._poisoned = "a latch write failed"


def _tool_use_violation(backend: CodexBackend, mp: pytest.MonkeyPatch) -> None:
    backend.state_store.record_violation("codex", '["command_execution"]', codex_version=VERSION, config_hash="h")


def _run_unfinished(backend: CodexBackend, mp: pytest.MonkeyPatch) -> None:
    backend.state_store.record_run_started("codex", "dead-instance")


RECORD_CLOSERS = [
    pytest.param(_schema_outdated, "schema_outdated", id="schema_outdated"),
    pytest.param(_state_unreadable, "state_unreadable", id="state_unreadable"),
    pytest.param(_state_unwritable, "state_unwritable", id="state_unwritable"),
    pytest.param(_tool_use_violation, "tool_use_violation", id="tool_use_violation"),
    pytest.param(_run_unfinished, "run_unfinished", id="run_unfinished"),
]


class TestAvailabilityOrder:
    """S-1'': records -> hold -> ``codex --version``."""

    async def test_open(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        availability = await backend.codex_availability()
        assert availability.open and availability.reason is None and availability.error is None
        assert availability.version == VERSION
        assert availability.quota_hold_until is None

    @pytest.mark.parametrize(("setup", "reason"), RECORD_CLOSERS)
    async def test_record_fault_wins_over_the_hold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: Any, reason: str
    ) -> None:
        """msg-428 blocking objection: a hold must not mask a DB fault or latch."""
        backend = make_backend(tmp_path)
        record_pass(backend)
        calls = _count_versions(backend)
        until = _hold(backend)
        setup(backend, monkeypatch)
        availability = await backend.codex_availability()
        assert availability.reason == reason
        assert availability.quota_hold_until == until  # the hold stays visible
        assert calls[0] == 0

    async def test_hold_without_record_fault_skips_version(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        calls = _count_versions(backend)
        until = _hold(backend)
        availability = await backend.codex_availability()
        assert availability.reason == "quota_hold"
        assert isinstance(availability.error, CodexQuotaError)
        assert availability.error.reset_at == until
        assert calls[0] == 0

    async def test_expired_hold_is_ignored_not_cleared(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        backend._quota_hold_until = past
        availability = await backend.codex_availability()
        assert availability.open and availability.quota_hold_until is None
        assert backend._quota_hold_until == past  # read-only

    async def test_version_change_is_stale(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend, version="codex-cli 0.1.0")
        assert (await backend.codex_availability()).reason == "verification_stale"

    async def test_version_launch_failure_is_launch_failed(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)

        async def broken() -> str:
            raise CodexLaunchError("codex --version failed: not found")

        backend._codex_version = broken  # type: ignore[method-assign]
        availability = await backend.codex_availability()
        assert availability.reason == "launch_failed"
        assert isinstance(availability.error, CodexLaunchError)

    async def test_reason_is_the_gate_reason(self, tmp_path: Path) -> None:
        """No separate list: ``reason`` is ``CodexNotVerifiedError.reason``."""
        backend = make_backend(tmp_path)
        availability = await backend.codex_availability()
        assert availability.reason == "verification_missing"
        assert isinstance(availability.error, CodexNotVerifiedError)
        assert availability.error.reason == availability.reason


class TestEnsureVerifiedSplit:
    async def test_ensure_verified_is_records_then_version(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        calls = _count_versions(backend)
        assert await backend._ensure_verified() == VERSION
        assert calls[0] == 1

    async def test_check_records_starts_no_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)

        async def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("subprocess started")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
        record = backend._check_records()
        assert record.result == "pass"


class TestNoWrites:
    @pytest.mark.parametrize("state", ["open", "hold", "violation", "unfinished"])
    async def test_availability_writes_nothing(self, tmp_path: Path, state: str) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        if state == "hold":
            _hold(backend)
        elif state == "violation":
            _tool_use_violation(backend, pytest.MonkeyPatch())
        elif state == "unfinished":
            _run_unfinished(backend, pytest.MonkeyPatch())
        backend._in_flight.add(999)
        rows, in_flight, hold = _row_counts(backend), set(backend._in_flight), backend._quota_hold_until
        for _ in range(3):
            await backend.codex_availability()
        assert _row_counts(backend) == rows
        assert backend._in_flight == in_flight
        assert backend._quota_hold_until == hold


class TestClearanceSeenWithoutRestart:
    async def test_cli_clearance_then_pass_reopens(self, tmp_path: Path) -> None:
        """msg-423 objection: a clearance written by another connection (the
        CLI) is seen at once; no cached reason."""
        backend = make_backend(tmp_path)
        record_pass(backend)
        violation = backend.state_store.record_violation(
            "codex", '["command_execution"]', codex_version=VERSION, config_hash="h"
        )
        assert (await backend.codex_availability()).reason == "tool_use_violation"
        cli = CodexStateStore(backend.state_store.db_path)
        cli.clear_violation(violation.seq, "investigated")
        assert (await backend.codex_availability()).reason == "verification_missing"
        cli.record_verification("codex", "pass", VERSION, backend.config_hash(), [])
        assert (await backend.codex_availability()).reason is None


class TestRequestPath:
    async def test_one_version_per_request(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        calls = _count_versions(backend)
        await backend.chat_completions(REQUEST)
        assert calls[0] == 1

    async def test_zero_versions_during_hold(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        calls = _count_versions(backend)
        until = _hold(backend)
        with pytest.raises(CodexQuotaError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reset_at == until
        assert calls[0] == 0
        assert backend.state_store.runs() == []

    async def test_passed_in_availability_is_not_re_evaluated(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        calls = _count_versions(backend)
        availability = await backend.codex_availability()
        await backend._run_gated("p", "gpt-5-codex", availability)
        assert calls[0] == 1

    @pytest.mark.parametrize(("setup", "reason"), [*RECORD_CLOSERS])
    async def test_request_and_status_agree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: Any, reason: str
    ) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        setup(backend, monkeypatch)
        status_reason = (await backend.codex_availability()).reason
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reason == status_reason == reason


class TestQuotaHold:
    async def test_quota_run_holds_until_the_parsed_reset(self, tmp_path: Path) -> None:
        """The fake CLI says "Try again in 2 hours 5 minutes" (ASSUMED wording)."""
        backend = make_backend(tmp_path, "quota")
        record_pass(backend)
        before = datetime.now(timezone.utc)
        with pytest.raises(CodexQuotaError):
            await backend.chat_completions(REQUEST)
        hold = backend._quota_hold_until
        span = timedelta(hours=2, minutes=5)
        assert hold is not None
        assert before + span <= hold <= datetime.now(timezone.utc) + span
        assert (await backend.codex_availability()).reason == "quota_hold"

    async def test_unparsed_reset_holds_fifteen_minutes(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        before = datetime.now(timezone.utc)
        backend._hold_for_quota(CodexQuotaError("limit, no time given", None))
        hold = backend._quota_hold_until
        assert hold is not None
        assert before + QUOTA_HOLD_FALLBACK <= hold <= datetime.now(timezone.utc) + QUOTA_HOLD_FALLBACK

    async def test_parsed_reset_time_is_the_hold(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        reset = datetime.now(timezone.utc) + timedelta(hours=3)
        backend._hold_for_quota(CodexQuotaError("limit", reset))
        assert backend._quota_hold_until == reset
        # A shorter hold never shortens a longer one.
        backend._hold_for_quota(CodexQuotaError("limit", reset - timedelta(hours=1)))
        assert backend._quota_hold_until == reset

    async def test_hold_ends_by_time(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        backend._quota_hold_until = datetime.now(timezone.utc) + timedelta(milliseconds=100)
        assert (await backend.codex_availability()).reason == "quota_hold"
        await asyncio.sleep(0.2)
        await backend.chat_completions(REQUEST)


class TestSubprocessReaping:
    """msg-427 (#52 PR-gate): a cancelled wait must kill the child, for the
    ``codex --version`` and ``codex login status`` spawns too."""

    @staticmethod
    def _spawn_sleeper(backend: CodexBackend, monkeypatch: pytest.MonkeyPatch) -> None:
        real = asyncio.create_subprocess_exec

        async def sleeper(*_args: Any, **kwargs: Any) -> Any:
            kwargs["env"] = _host_env(backend)
            kwargs["stdin"] = asyncio.subprocess.DEVNULL
            return await real(sys.executable, FAKE_CLI, "sleep", **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", sleeper)

    async def test_cancelled_version_check_kills_the_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = make_backend(tmp_path)
        del backend._codex_version  # the real method, not make_backend's stub
        self._spawn_sleeper(backend, monkeypatch)
        task = asyncio.create_task(backend._codex_version())
        await _until_ticking(backend)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _assert_dead(backend)

    async def test_version_timeout_kills_the_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = make_backend(tmp_path)
        del backend._codex_version
        self._spawn_sleeper(backend, monkeypatch)
        real_wait_for = asyncio.wait_for
        monkeypatch.setattr(asyncio, "wait_for", lambda aw, timeout: real_wait_for(aw, 1.0))
        with pytest.raises(CodexLaunchError):
            await backend._codex_version()
        await _assert_dead(backend)

    async def test_cancelled_login_status_kills_the_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        self._spawn_sleeper(backend, monkeypatch)
        task = asyncio.create_task(backend.health_check())
        await _until_ticking(backend)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _assert_dead(backend)

    async def test_login_status_timeout_kills_the_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        self._spawn_sleeper(backend, monkeypatch)
        real_wait_for = asyncio.wait_for
        monkeypatch.setattr(asyncio, "wait_for", lambda aw, timeout: real_wait_for(aw, 1.0))
        assert await backend.health_check() is False
        await _assert_dead(backend)

    def test_every_spawn_site_reaps(self) -> None:
        """Each ``create_subprocess_exec`` in codex.py is followed by a wait
        that ends in ``_kill_and_reap`` on any exception."""
        source = Path(codex_mod.__file__).read_text(encoding="utf-8")
        assert source.count("asyncio.create_subprocess_exec(") == 3
        assert source.count("await _kill_and_reap(process)") == 3

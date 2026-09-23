"""B-7 (T-naysayer-codex-backend msg-450): one process per codex state DB.

Driven with real child processes, because the property is about processes:
an in-process double of the lock could not show the OS refusing a second
holder or dropping the lock of a killed one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI

from lexora.config import RoutingSettings, Settings
from lexora.main import lifespan
from lexora.services.process_lock import (
    LOCK_FILE_NAME,
    ProcessLockError,
    acquire,
    acquire_codex_locks,
    codex_lock_paths,
)

HOLDER = """
import sys, time
from pathlib import Path
from lexora.services.process_lock import ProcessLockError, acquire
try:
    lock = acquire(Path(sys.argv[1]))
except ProcessLockError:
    print("LOCKED", flush=True)
    sys.exit(3)
print("ACQUIRED", flush=True)
if sys.argv[2] == "hold":
    time.sleep(60)
"""


def _routing(tmp_path: Path, *, codex: bool = True, fallback: bool = False) -> RoutingSettings:
    backends: dict = {"gemini": {"type": "gemini", "url": "https://example.invalid", "models": ["gemini-3.1-pro-preview"]}}
    if codex:
        backends["codex"] = {
            "type": "codex",
            "codex": {"codex_home": str(tmp_path / "home"), "state_db_path": str(tmp_path / "state" / "codex.db")},
        }
    if fallback:
        backends["naysayer-fb"] = {"type": "fallback", "fallback": {"primary": "codex", "fallback": "gemini"}}
    return RoutingSettings(enabled=True, default_backend="gemini", backends=backends)


def _start_holder(path: Path) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(path), "hold"], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ACQUIRED"
    return proc


def _try(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", HOLDER, str(path), "once"], capture_output=True, text=True, timeout=60
    )


class TestTwoProcesses:
    def test_second_process_is_refused_and_a_killed_holder_frees_the_lock(self, tmp_path: Path) -> None:
        path = tmp_path / LOCK_FILE_NAME
        holder = _start_holder(path)
        try:
            second = _try(path)
            assert (second.returncode, second.stdout.strip()) == (3, "LOCKED")
        finally:
            holder.kill()
            holder.wait(timeout=30)
        third = _try(path)  # no stale lock after the holder died
        assert (third.returncode, third.stdout.strip()) == (0, "ACQUIRED")

    async def test_startup_fails_while_another_process_holds_the_state(self, tmp_path: Path) -> None:
        """The real ``lifespan``: it raises before any backend is built."""
        routing = _routing(tmp_path, fallback=True)
        [path] = codex_lock_paths(routing)
        holder = _start_holder(path)
        try:
            app = FastAPI()
            app.state.settings = Settings(routing=routing)
            with pytest.raises(ProcessLockError, match="ONE worker"):
                async with lifespan(app):
                    pass
            assert not hasattr(app.state, "backend_router")
        finally:
            holder.kill()
            holder.wait(timeout=30)


class TestWhichConfigsLock:
    def test_codex_config_locks_next_to_its_state_db(self, tmp_path: Path) -> None:
        assert codex_lock_paths(_routing(tmp_path)) == [tmp_path / "state" / LOCK_FILE_NAME]

    def test_gemini_only_takes_no_lock(self, tmp_path: Path) -> None:
        routing = _routing(tmp_path, codex=False)
        assert codex_lock_paths(routing) == []
        assert acquire_codex_locks(routing) == []

    def test_lock_is_released_on_release(self, tmp_path: Path) -> None:
        lock = acquire(tmp_path / LOCK_FILE_NAME)
        lock.release()
        assert _try(tmp_path / LOCK_FILE_NAME).returncode == 0

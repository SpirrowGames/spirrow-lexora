"""One Lexora process per codex state DB (T-naysayer-codex-backend msg-450 B-7).

Everything the codex / fallback backends keep in memory -- ``_in_flight``
(condition 0 and ``inflight_runs``), the quota hold, ``fallback_since``, the
notification timers, ``shadow_skipped``, the one-shadow-at-a-time limit --
is only the system's state if exactly one process serves requests. That was
a checked fact (msg-407: ``main.py`` passes ``workers=1``, the unit starts
one process); this module makes it an enforced invariant.

When a ``codex`` backend is configured (a ``fallback`` one always has one as
its primary), ``main.lifespan`` takes an exclusive, non-blocking OS lock on
``codex.lock`` next to each codex state DB before anything else. If another
process holds it, startup fails with ``ProcessLockError`` and Lexora does not
serve. The lock lives as long as the process; the OS drops it when the
process dies, so a crash leaves no stale lock. Neither platform silently
skips the lock: POSIX uses ``fcntl.flock``, Windows ``msvcrt.locking``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lexora.config import RoutingSettings

LOCK_FILE_NAME = "codex.lock"


class ProcessLockError(RuntimeError):
    """Another process holds the codex state; refuse to start."""


@dataclass
class ProcessLock:
    path: Path
    fd: int

    def release(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)  # closing the descriptor drops the lock on both platforms
            self.fd = -1


def acquire(path: Path) -> ProcessLock:
    """Take the exclusive lock on ``path`` or raise ``ProcessLockError``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise ProcessLockError(
            f"another process holds the codex state ({path}); Lexora must run as ONE "
            f"worker process -- stop the other process (or drop --workers) and start again"
        ) from exc
    return ProcessLock(path, fd)


def codex_lock_paths(routing: RoutingSettings) -> list[Path]:
    """``codex.lock`` next to each codex backend's state DB; empty when no
    codex backend is configured (a Gemini-only config takes no lock)."""
    if not routing.enabled:
        return []
    paths: list[Path] = []
    for backend in routing.backends.values():
        if backend.type == "codex" and backend.codex is not None:
            path = Path(backend.codex.state_db_path).parent / LOCK_FILE_NAME
            if path not in paths:
                paths.append(path)
    return paths


def acquire_codex_locks(routing: RoutingSettings) -> list[ProcessLock]:
    """All locks or none: on a failure, the ones already taken are released."""
    taken: list[ProcessLock] = []
    try:
        for path in codex_lock_paths(routing):
            taken.append(acquire(path))
    except ProcessLockError:
        for lock in taken:
            lock.release()
        raise
    return taken

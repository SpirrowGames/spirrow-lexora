"""SQLite state for the codex backend: verification records and the D-1c latch.

T-naysayer-codex-backend msg-294 PR-1, amended by msg-315/317/319.

* ``codex_verification`` -- one row per ``python -m lexora.tools.verify_codex``
  run. ``result`` is ``pass`` or ``fail`` (two values; ``inconclusive`` was
  dropped in msg-288). ``checks_json`` holds the per-check names, verdicts
  and short machine details for V-1, V-2'-control and V-2'. **No model output
  and no tool output is ever stored** (msg-294). A tool executed under the
  production config DURING verification is a ``fail`` row here, never a
  violation (msg-319).
* ``codex_runtime_violation`` -- one row per D-1c event, i.e. the CLI ran a
  tool (or emitted an unrecognised event) while serving a real request.
  Written only by ``CodexBackend._run_gated``. It is a **global latch**
  (msg-317): any uncleared row closes the gate of every codex backend using
  this store, whatever the CLI version or config hash. A row records the
  version, hash, time and event kinds at the violation. Release is manual
  and ordered: a human clears each row with a non-empty reason
  (``verify_codex --clear-violation <id> --reason ...``), and only a
  ``pass`` recorded after the latest clearance opens the gate.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS codex_seq (
    seq INTEGER PRIMARY KEY AUTOINCREMENT
);
CREATE TABLE IF NOT EXISTS codex_verification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    backend TEXT NOT NULL,
    at TEXT NOT NULL,
    result TEXT NOT NULL CHECK (result IN ('pass', 'fail')),
    codex_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    checks_json TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS codex_runtime_violation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backend TEXT NOT NULL,
    at TEXT NOT NULL,
    codex_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    detail TEXT NOT NULL,
    cleared_at TEXT,
    cleared_seq INTEGER,
    cleared_reason TEXT
);
"""


def _now() -> str:
    # For humans only. Ordering (pass vs clearance) uses ``seq``: wall-clock
    # timestamps can tie at the clock's resolution (measured on Windows).
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class ClearViolationError(ValueError):
    """A clearance request was refused (unknown id, already cleared, empty reason)."""


@dataclass(frozen=True)
class VerificationRecord:
    id: int
    seq: int
    backend: str
    at: str
    result: str
    codex_version: str
    config_hash: str
    checks: list[dict[str, Any]]
    note: str | None


@dataclass(frozen=True)
class ViolationRecord:
    id: int
    backend: str
    at: str
    codex_version: str
    config_hash: str
    detail: str
    cleared_at: str | None = None
    cleared_seq: int | None = None
    cleared_reason: str | None = None


_VIOLATION_COLUMNS = (
    "id, backend, at, codex_version, config_hash, detail, cleared_at, cleared_seq, cleared_reason"
)


def _next_seq(conn: sqlite3.Connection) -> int:
    """One strictly increasing counter shared by passes and clearances."""
    return int(conn.execute("INSERT INTO codex_seq DEFAULT VALUES").lastrowid or 0)


class CodexStateStore:
    """Tiny synchronous SQLite store (the calls are rare and small)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Commit on success and always close (Windows keeps open files locked)."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ---- verification -------------------------------------------------

    def record_verification(
        self,
        backend: str,
        result: str,
        codex_version: str,
        config_hash: str,
        checks: list[dict[str, Any]],
        note: str | None = None,
    ) -> VerificationRecord:
        if result not in ("pass", "fail"):
            raise ValueError(f"result must be 'pass' or 'fail', got {result!r}")
        at = _now()
        with self._connect() as conn:
            seq = _next_seq(conn)
            cur = conn.execute(
                "INSERT INTO codex_verification "
                "(seq, backend, at, result, codex_version, config_hash, checks_json, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (seq, backend, at, result, codex_version, config_hash, json.dumps(checks), note),
            )
            row_id = int(cur.lastrowid or 0)
        return VerificationRecord(row_id, seq, backend, at, result, codex_version, config_hash, checks, note)

    def latest_verification(self, backend: str) -> VerificationRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, seq, backend, at, result, codex_version, config_hash, checks_json, note "
                "FROM codex_verification WHERE backend = ? ORDER BY id DESC LIMIT 1",
                (backend,),
            ).fetchone()
        if row is None:
            return None
        return VerificationRecord(*row[:7], json.loads(row[7]), row[8])

    # ---- runtime violation latch ---------------------------------------

    def record_violation(
        self, backend: str, detail: str, *, codex_version: str, config_hash: str
    ) -> ViolationRecord:
        at = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO codex_runtime_violation "
                "(backend, at, codex_version, config_hash, detail) VALUES (?, ?, ?, ?, ?)",
                (backend, at, codex_version, config_hash, detail),
            )
            row_id = int(cur.lastrowid or 0)
        return ViolationRecord(row_id, backend, at, codex_version, config_hash, detail)

    def violations(self) -> list[ViolationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_VIOLATION_COLUMNS} FROM codex_runtime_violation ORDER BY id"
            ).fetchall()
        return [ViolationRecord(*row) for row in rows]

    def uncleared_violations(self) -> list[ViolationRecord]:
        """All uncleared violations, across every backend and config (global latch)."""
        return [v for v in self.violations() if v.cleared_at is None]

    def latest_clearance_seq(self) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(cleared_seq) FROM codex_runtime_violation WHERE cleared_seq IS NOT NULL"
            ).fetchone()
        return row[0] if row else None

    def clear_violation(self, violation_id: int, reason: str) -> ViolationRecord:
        """Human release of one violation. Refuses an empty reason (msg-317)."""
        if not reason or not reason.strip():
            raise ClearViolationError("a non-empty --reason is required to clear a violation")
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_VIOLATION_COLUMNS} FROM codex_runtime_violation WHERE id = ?",
                (violation_id,),
            ).fetchone()
            if row is None:
                raise ClearViolationError(f"no violation with id {violation_id}")
            if row[6] is not None:
                raise ClearViolationError(f"violation {violation_id} was already cleared at {row[6]}")
            at = _now()
            seq = _next_seq(conn)
            conn.execute(
                "UPDATE codex_runtime_violation SET cleared_at = ?, cleared_seq = ?, cleared_reason = ? "
                "WHERE id = ?",
                (at, seq, reason.strip(), violation_id),
            )
        return ViolationRecord(*row[:6], cleared_at=at, cleared_seq=seq, cleared_reason=reason.strip())

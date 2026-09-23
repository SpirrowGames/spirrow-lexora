"""SQLite state for the codex backend: verification records and the D-1c latch.

T-naysayer-codex-backend msg-294 PR-1, amended by msg-315/317/319/324/326.

**One numbering source (msg-324).** Every gate-relevant event -- a
verification (``pass`` / ``fail``), a runtime violation, a clearance -- is
first a row of ``codex_gate_log`` (``seq INTEGER PRIMARY KEY AUTOINCREMENT``,
``kind``). The kind-specific tables hold their details keyed by that
``seq``. ``AUTOINCREMENT`` makes SQLite persist the counter in
``sqlite_sequence``: numbers survive a restart and are never reused, even
after rows are deleted. The ``seq`` is taken and the detail row written in
the SAME transaction (``_append``). Forbidden: an in-memory counter, any
``max(seq)+1`` numbering, one counter per table.

* ``codex_verification`` -- one row per ``verify_codex`` run, ``pass`` or
  ``fail``. Check names / verdicts / short details only; **no model or tool
  output is stored** (msg-294). A tool executed under the production config
  during verification is a ``fail`` here, never a violation (msg-319).
* ``codex_runtime_violation`` -- one row per D-1c event while serving a real
  request, written only by ``CodexBackend._run_gated``. A **global latch**
  (msg-317): any uncleared row closes the gate, whatever version or hash.
* ``codex_clearance`` -- a human clearing one violation with a non-empty
  reason (``verify_codex --clear-violation <seq> --reason ...``).

Gate predicates read from here (msg-326): the clearance threshold is
``COALESCE(MAX(seq) WHERE kind='clearance', 0)`` -- a read-only comparison
value, not numbering; ``seq`` starts at 1, so 0 cannot collide with a row.
A query that fails is NOT an empty result: it raises, and the gate closes.
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
CREATE TABLE IF NOT EXISTS codex_gate_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('verification', 'violation', 'clearance')),
    backend TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS codex_verification (
    seq INTEGER PRIMARY KEY REFERENCES codex_gate_log(seq),
    result TEXT NOT NULL CHECK (result IN ('pass', 'fail')),
    codex_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    checks_json TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS codex_runtime_violation (
    seq INTEGER PRIMARY KEY REFERENCES codex_gate_log(seq),
    codex_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS codex_clearance (
    seq INTEGER PRIMARY KEY REFERENCES codex_gate_log(seq),
    violation_seq INTEGER NOT NULL UNIQUE REFERENCES codex_runtime_violation(seq),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0)
);
"""


def _now() -> str:
    # For humans only; ordering is ``seq`` (timestamps tie at clock resolution).
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class ClearViolationError(ValueError):
    """A clearance request was refused (unknown id, already cleared, empty reason)."""


@dataclass(frozen=True)
class VerificationRecord:
    seq: int
    backend: str
    at: str
    result: str
    codex_version: str
    config_hash: str
    checks: list[dict[str, Any]]
    note: str | None

    @property
    def id(self) -> int:
        return self.seq


@dataclass(frozen=True)
class ViolationRecord:
    seq: int
    backend: str
    at: str
    codex_version: str
    config_hash: str
    detail: str
    cleared_seq: int | None = None
    cleared_at: str | None = None
    cleared_reason: str | None = None

    @property
    def id(self) -> int:
        """The violation's id is its ``codex_gate_log`` seq."""
        return self.seq


class CodexStateStore:
    """Tiny synchronous SQLite store (the calls are rare and small)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One transaction per block: commit on success, roll back on error,
        always close (Windows keeps open files locked)."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _append(conn: sqlite3.Connection, kind: str, backend: str, at: str) -> int:
        """Take the next seq from SQLite. Call only inside the transaction
        that also writes the detail row (msg-324)."""
        cur = conn.execute(
            "INSERT INTO codex_gate_log (kind, backend, at) VALUES (?, ?, ?)", (kind, backend, at)
        )
        if cur.lastrowid is None:
            raise sqlite3.DatabaseError("codex_gate_log insert returned no seq")
        return int(cur.lastrowid)

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
            seq = self._append(conn, "verification", backend, at)
            conn.execute(
                "INSERT INTO codex_verification "
                "(seq, result, codex_version, config_hash, checks_json, note) VALUES (?, ?, ?, ?, ?, ?)",
                (seq, result, codex_version, config_hash, json.dumps(checks), note),
            )
        return VerificationRecord(seq, backend, at, result, codex_version, config_hash, checks, note)

    def latest_verification(self, backend: str) -> VerificationRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT g.seq, g.backend, g.at, v.result, v.codex_version, v.config_hash, "
                "v.checks_json, v.note "
                "FROM codex_gate_log g JOIN codex_verification v ON v.seq = g.seq "
                "WHERE g.kind = 'verification' AND g.backend = ? ORDER BY g.seq DESC LIMIT 1",
                (backend,),
            ).fetchone()
        if row is None:
            return None
        return VerificationRecord(*row[:6], json.loads(row[6]), row[7])

    # ---- runtime violation latch ---------------------------------------

    def record_violation(
        self, backend: str, detail: str, *, codex_version: str, config_hash: str
    ) -> ViolationRecord:
        at = _now()
        with self._connect() as conn:
            seq = self._append(conn, "violation", backend, at)
            conn.execute(
                "INSERT INTO codex_runtime_violation (seq, codex_version, config_hash, detail) "
                "VALUES (?, ?, ?, ?)",
                (seq, codex_version, config_hash, detail),
            )
        return ViolationRecord(seq, backend, at, codex_version, config_hash, detail)

    def violations(self) -> list[ViolationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT g.seq, g.backend, g.at, r.codex_version, r.config_hash, r.detail, "
                "c.seq, cg.at, c.reason "
                "FROM codex_gate_log g JOIN codex_runtime_violation r ON r.seq = g.seq "
                "LEFT JOIN codex_clearance c ON c.violation_seq = g.seq "
                "LEFT JOIN codex_gate_log cg ON cg.seq = c.seq "
                "WHERE g.kind = 'violation' ORDER BY g.seq"
            ).fetchall()
        return [ViolationRecord(*row) for row in rows]

    def uncleared_violations(self) -> list[ViolationRecord]:
        """All uncleared violations, across every backend and config (global latch)."""
        return [v for v in self.violations() if v.cleared_seq is None]

    def clearance_threshold(self) -> int:
        """``COALESCE(MAX(seq) WHERE kind='clearance', 0)`` (msg-326).

        Read-only comparison value. 0 only when the query SUCCEEDED and found
        no clearance; a failing query raises instead of returning 0.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE((SELECT MAX(seq) FROM codex_gate_log WHERE kind = 'clearance'), 0)"
            ).fetchone()
        if row is None or not isinstance(row[0], int):
            raise sqlite3.DatabaseError("clearance threshold query returned no integer")
        return row[0]

    def clear_violation(self, violation_seq: int, reason: str) -> ViolationRecord:
        """Human release of one violation. Refuses an empty reason (msg-317).

        Does not open the gate: a ``pass`` with a larger seq is still needed.
        """
        if not reason or not reason.strip():
            raise ClearViolationError("a non-empty --reason is required to clear a violation")
        target = next((v for v in self.violations() if v.seq == violation_seq), None)
        if target is None:
            raise ClearViolationError(f"no violation with id {violation_seq}")
        if target.cleared_seq is not None:
            raise ClearViolationError(f"violation {violation_seq} was already cleared at {target.cleared_at}")
        at = _now()
        try:
            with self._connect() as conn:
                seq = self._append(conn, "clearance", target.backend, at)
                conn.execute(
                    "INSERT INTO codex_clearance (seq, violation_seq, reason) VALUES (?, ?, ?)",
                    (seq, violation_seq, reason.strip()),
                )
        except sqlite3.IntegrityError as exc:  # concurrent double clear
            raise ClearViolationError(f"violation {violation_seq} could not be cleared: {exc}") from exc
        return ViolationRecord(
            target.seq, target.backend, target.at, target.codex_version, target.config_hash,
            target.detail, cleared_seq=seq, cleared_at=at, cleared_reason=reason.strip(),
        )

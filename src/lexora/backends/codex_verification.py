"""SQLite state for the codex backend: verification records and the D-1c latch.

T-naysayer-codex-backend msg-294 PR-1.

* ``codex_verification`` -- one row per ``python -m lexora.tools.verify_codex``
  run. ``result`` is ``pass`` or ``fail`` (two values; ``inconclusive`` was
  dropped in msg-288). ``checks_json`` holds the per-check names, verdicts
  and short machine details for V-1 and V-2'. **No model output and no tool
  output is ever stored** (msg-294: "出力本文は保存しない").
* ``codex_runtime_violation`` -- one row per D-1c event (the CLI ran a tool
  on a production request). A violation newer than the latest passing
  verification keeps the gate closed; the release is a human re-running
  ``verify_codex`` (msg-294 D-1c: "解除は人が手で行う").
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
CREATE TABLE IF NOT EXISTS codex_verification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    detail TEXT NOT NULL
);
"""


def _now() -> str:
    # Microsecond ISO-8601 UTC: lexicographic order == time order, which the
    # gate relies on when it compares a violation with a verification.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class VerificationRecord:
    id: int
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
    detail: str


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
            cur = conn.execute(
                "INSERT INTO codex_verification "
                "(backend, at, result, codex_version, config_hash, checks_json, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (backend, at, result, codex_version, config_hash, json.dumps(checks), note),
            )
            row_id = int(cur.lastrowid or 0)
        return VerificationRecord(row_id, backend, at, result, codex_version, config_hash, checks, note)

    def latest_verification(self, backend: str) -> VerificationRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, backend, at, result, codex_version, config_hash, checks_json, note "
                "FROM codex_verification WHERE backend = ? ORDER BY id DESC LIMIT 1",
                (backend,),
            ).fetchone()
        if row is None:
            return None
        return VerificationRecord(row[0], row[1], row[2], row[3], row[4], row[5], json.loads(row[6]), row[7])

    def record_violation(self, backend: str, detail: str) -> ViolationRecord:
        at = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO codex_runtime_violation (backend, at, detail) VALUES (?, ?, ?)",
                (backend, at, detail),
            )
            row_id = int(cur.lastrowid or 0)
        return ViolationRecord(row_id, backend, at, detail)

    def latest_violation(self, backend: str) -> ViolationRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, backend, at, detail FROM codex_runtime_violation "
                "WHERE backend = ? ORDER BY id DESC LIMIT 1",
                (backend,),
            ).fetchone()
        if row is None:
            return None
        return ViolationRecord(row[0], row[1], row[2], row[3])

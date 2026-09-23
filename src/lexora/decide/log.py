"""Decision log — SQLite writer + :class:`DecisionRow`.

Structural constraint (Einstein msg-243 objection #3 / Bohr msg-244
disposition #3): the writer's *only* input is a :class:`DecisionRow`
instance. The dataclass has fixed fields — no ``headers``, no
``raw_body``, no ``kwargs`` dict — so a caller that later wants to log a
raw HTTP header cannot do so without editing this file. That is the
whole safety argument: the API-key protection is the env-only rule in
:mod:`lexora.decide.config`; this log removes the *second* place a key
could have leaked (a redact whitelist would have been a duplicate
protection with its own failure mode — YAGNI, per Bohr's msg-244).

If a future PR needs a header value for debugging, the change lives in
this file (add a *named* column with a documented meaning, e.g.
``provider_request_id``); it does not live in the caller.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lexora.decide.contract import (
    QUESTIONS_HASH_HEX_LENGTH,
    compute_questions_hash,
    compute_state_hash,
)

#: SQL for the ``decisions`` table. Column list is fixed and enumerated
#: on purpose — the writer does not accept ``**kwargs``, so the schema
#: and the dataclass move together or not at all.
#:
#: ``questions_hash`` is NOT NULL because Lexora computes it on every
#: request (msg-244 disposition #1). ``questions_version`` is NULL-able
#: because it is caller-supplied and optional.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id       TEXT PRIMARY KEY,
    policy            TEXT NOT NULL,
    state_hash        TEXT NOT NULL,
    questions_hash    TEXT NOT NULL,
    questions_version TEXT NULL,
    provider          TEXT NOT NULL,
    answers_json      TEXT NOT NULL,
    latency_ms        INTEGER NOT NULL,
    timestamp         TEXT NOT NULL,
    provider_error    TEXT NULL
)
"""

_INSERT_SQL = """
INSERT INTO decisions (
    decision_id, policy, state_hash, questions_hash, questions_version,
    provider, answers_json, latency_ms, timestamp, provider_error
) VALUES (
    :decision_id, :policy, :state_hash, :questions_hash, :questions_version,
    :provider, :answers_json, :latency_ms, :timestamp, :provider_error
)
"""

#: Columns added after PR #43 shipped the table. Each entry is applied by
#: :func:`_apply_schema_on_conn` only when ``PRAGMA table_info`` shows the
#: column missing, so a DB created by PR #43 is up-migrated in place and a
#: fresh DB (whose CREATE already carries the column) is left alone.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("provider_error", "TEXT NULL"),
)


def _apply_schema_on_conn(conn: sqlite3.Connection) -> None:
    """Idempotently ensure the ``decisions`` schema on ``conn``.

    The caller owns the transaction / write lock. CREATE TABLE IF NOT
    EXISTS runs first so a fresh DB has a table before ``PRAGMA
    table_info`` is consulted — on a missing table that pragma returns an
    empty list rather than raising, and an ALTER against it would fail
    with ``no such table`` (Einstein msg-261).
    """
    conn.execute(_CREATE_TABLE_SQL)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
    for name, decl in _ADDED_COLUMNS:
        if name not in cols:
            conn.execute(f"ALTER TABLE decisions ADD COLUMN {name} {decl}")


def apply_decision_log_migrations(path: Path | str) -> None:
    """Race-safe schema migration for a file-backed decision log.

    This is the **only** owner of the schema for file-backed paths
    (Bohr msg-264 v4). ``create_app`` calls it before constructing
    :class:`DecisionLog`; any other entrypoint (CLI, import tool) must do
    the same.

    Multi-worker safety (Einstein msg-259 #1): under ``uvicorn --workers
    N`` every worker runs ``create_app``. ``BEGIN IMMEDIATE`` takes the
    SQLite write lock before the column check, so workers are serialised
    by the file lock and a late worker re-reads the table info *after*
    the early one committed — it sees the column and no-ops instead of
    hitting a duplicate-column error. ``busy_timeout`` makes a waiting
    worker block up to 5 s; beyond that ``sqlite3.OperationalError``
    propagates and the process crash-loops observably (fail-closed).

    ``:memory:`` is per-connection, so a migration here would land on a
    throwaway DB; it returns early and :class:`DecisionLog` applies the
    schema on its own connection instead.
    """
    path_str = str(path)
    if path_str == ":memory:":
        return
    parent = Path(path_str).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path_str, timeout=5.0, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("BEGIN IMMEDIATE")
        try:
            _apply_schema_on_conn(conn)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()


@dataclass(frozen=True, slots=True)
class DecisionRow:
    """One row of the decision log.

    Field notes:

    * ``state_hash`` / ``questions_hash`` — 16-hex-char sha256 prefixes
      (see :mod:`lexora.decide.contract`). Fixed width so the schema
      does not have to grow when a hash algorithm changes; the width is
      part of the writer's contract, not the dataclass's.
    * ``provider_error`` — ``None`` when the provider in ``provider``
      answered directly. When the primary failed and NullProvider served
      the fallback, ``provider`` is ``"null"`` and this carries
      ``"<primary>:<code>"`` plus ``";discarded=<n>"`` when ``n`` sibling
      upstream calls had already completed (and been billed) before the
      failure was observed (Bohr msg-260 #2). A fixed code, never an
      upstream body or header.
    * ``answers_json`` — JSON-serialised ``answers`` object. Kept as a
      string rather than a nested dict because SQLite has no JSON
      column type and turning every read into a JSON parse in Python
      would be an easy performance cliff. Provider-neutral: does not
      include headers, cookies, or request ids.

    Frozen so the row cannot be mutated between construction and write,
    and ``slots=True`` so a caller who mistypes a field name (e.g.
    ``header``) gets an ``AttributeError`` rather than a silently-
    ignored assignment.
    """

    decision_id: str
    policy: str
    state_hash: str
    questions_hash: str
    provider: str
    answers_json: str
    latency_ms: int
    timestamp: str
    questions_version: str | None = None
    provider_error: str | None = None

    def __post_init__(self) -> None:
        # Fixed-width hash discipline lives here so a caller who passes
        # a full 64-char digest gets rejected loudly rather than
        # producing a decision row inconsistent with every other row.
        for name, value in (
            ("state_hash", self.state_hash),
            ("questions_hash", self.questions_hash),
        ):
            if len(value) != QUESTIONS_HASH_HEX_LENGTH:
                raise ValueError(
                    f"{name} must be {QUESTIONS_HASH_HEX_LENGTH} hex chars, "
                    f"got {len(value)}"
                )


def build_decision_row(
    *,
    decision_id: str,
    policy: str,
    state: str,
    questions: dict[str, Any],
    provider: str,
    answers: dict[str, Any],
    latency_ms: int,
    questions_version: str | None,
    timestamp: datetime | None = None,
    provider_error: str | None = None,
) -> DecisionRow:
    """Construct a :class:`DecisionRow` from the request/response shape.

    The state string, questions object, and answers object are hashed
    or serialised here — this is the only place a caller supplies raw
    payloads to the log, so this is the one call site the "no headers"
    discipline has to guard. All the arguments are named payload
    fields; there is no ``extra`` dict, no ``metadata`` slot.
    """
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    return DecisionRow(
        decision_id=decision_id,
        policy=policy,
        state_hash=compute_state_hash(state),
        questions_hash=compute_questions_hash(questions),
        questions_version=questions_version,
        provider=provider,
        answers_json=json.dumps(
            answers, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ),
        latency_ms=latency_ms,
        timestamp=ts,
        provider_error=provider_error,
    )


class DecisionLog:
    """SQLite-backed writer for :class:`DecisionRow`.

    One writer per Lexora process, guarded by an in-process lock — the
    endpoint's throughput is bounded by the upstream (~seconds per call
    for Jev), so serialising the write costs nothing observable and
    removes the "concurrent write" failure mode entirely.

    The database is opened with ``check_same_thread=False`` because
    FastAPI hands requests to an asyncio event loop and worker threads
    may share the connection. The lock is what makes that safe.

    Schema ownership (Bohr msg-264 v4, Einstein msg-263 advisory): for a
    file-backed path this class does NOT create or alter the schema —
    call :func:`apply_decision_log_migrations` first. Forgetting to do so
    makes the first :meth:`write` raise ``sqlite3.OperationalError: no
    such table: decisions`` (fail-closed). A lock-free schema apply here
    would be a safety net that silently reintroduces the multi-worker
    ALTER race. Only ``:memory:`` DBs, which are per-connection and so
    cannot race across processes, get their schema applied here.
    """

    def __init__(self, path: Path | str = ":memory:") -> None:
        self._path = str(path)
        # Ensure parent exists for on-disk logs; :memory: has no parent.
        if self._path != ":memory:":
            parent = Path(self._path).parent
            if str(parent) and str(parent) != ".":
                parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        if self._path == ":memory:":
            # The ONLY place DecisionLog owns its schema: an external
            # migration on a separate ``:memory:`` connection would land
            # on a different, throwaway DB.
            _apply_schema_on_conn(self._conn)
            self._conn.commit()

    def write(self, row: DecisionRow) -> None:
        """Append a row. Raises :class:`sqlite3.Error` on failure.

        Errors are NOT swallowed — the router treats a log-write failure
        as a request failure. Silently dropping decision-log rows would
        make offline evaluation lie about coverage, which is exactly the
        failure mode the whole shadow-mode data-collection story
        depends on avoiding (msg-237: "116 判断点リプレイをこの
        エンドポイント経由で流し、オフライン評価と本番を同一コード
        パスにする").
        """
        with self._lock:
            self._conn.execute(_INSERT_SQL, _row_as_params(row))
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    def fetch_all(self) -> list[dict[str, Any]]:
        """Read every row as a list of dicts. Test-only helper.

        Not part of the runtime path; the endpoint never reads its own
        log back. Kept on the class so tests do not have to reach for
        the private connection.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT decision_id, policy, state_hash, questions_hash, "
                "questions_version, provider, answers_json, latency_ms, "
                "timestamp, provider_error FROM decisions ORDER BY timestamp, decision_id"
            )
            columns = [c[0] for c in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _row_as_params(row: DecisionRow) -> dict[str, Any]:
    """Turn a :class:`DecisionRow` into an SQLite parameter dict.

    The parameter dict is built by naming each field explicitly — this
    is the point at which the "no headers, no raw body" property is
    enforced: if a future DecisionRow field slips in that shouldn't
    make it to the SQL layer, this function fails to compile-time-catch
    it, but the SQL statement's named parameters (``:decision_id``…)
    will not match and the write will error out loud.
    """
    return {
        "decision_id": row.decision_id,
        "policy": row.policy,
        "state_hash": row.state_hash,
        "questions_hash": row.questions_hash,
        "questions_version": row.questions_version,
        "provider": row.provider,
        "answers_json": row.answers_json,
        "latency_ms": row.latency_ms,
        "timestamp": row.timestamp,
        "provider_error": row.provider_error,
    }

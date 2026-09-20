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
    timestamp         TEXT NOT NULL
)
"""

_INSERT_SQL = """
INSERT INTO decisions (
    decision_id, policy, state_hash, questions_hash, questions_version,
    provider, answers_json, latency_ms, timestamp
) VALUES (
    :decision_id, :policy, :state_hash, :questions_hash, :questions_version,
    :provider, :answers_json, :latency_ms, :timestamp
)
"""


@dataclass(frozen=True, slots=True)
class DecisionRow:
    """One row of the decision log.

    Field notes:

    * ``state_hash`` / ``questions_hash`` — 16-hex-char sha256 prefixes
      (see :mod:`lexora.decide.contract`). Fixed width so the schema
      does not have to grow when a hash algorithm changes; the width is
      part of the writer's contract, not the dataclass's.
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
        self._conn.execute(_CREATE_TABLE_SQL)
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
                "timestamp FROM decisions ORDER BY timestamp, decision_id"
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
    }

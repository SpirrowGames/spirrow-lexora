"""Tests for :mod:`lexora.decide.log`.

The structural constraint under test (msg-243 objection #3 / msg-244
disposition #3) is that :class:`DecisionRow` has no header / raw-body
slot, so a caller that has raw HTTP information cannot smuggle it into
the log. The tests here pin the dataclass shape and the writer's
behaviour so a well-meaning refactor that adds a ``**kwargs`` slot on
either side fails loudly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lexora.decide.contract import (
    QUESTIONS_HASH_HEX_LENGTH,
    QuestionSpec,
    compute_questions_hash,
    compute_state_hash,
)
from lexora.decide.log import (
    DecisionLog,
    DecisionRow,
    apply_decision_log_migrations,
    build_decision_row,
)


class TestDecisionRowShape:
    def test_no_headers_field(self) -> None:
        """The dataclass has no header / body / metadata slot.

        This is the structural half of the "redact whitelist was
        YAGNI" disposition (msg-244 #3): the log rejects header data
        by having nowhere to put it, so we do not have to argue about
        whether a whitelist covers every field a future header value
        might travel in.
        """
        allowed = {
            "decision_id",
            "policy",
            "state_hash",
            "questions_hash",
            "questions_version",
            "provider",
            "answers_json",
            "latency_ms",
            "timestamp",
            "provider_error",
            "provider_model",
            "provider_input_tokens",
            "provider_output_tokens",
        }
        fields = {f for f in DecisionRow.__dataclass_fields__ if not f.startswith("_")}
        assert fields == allowed

    def test_row_is_frozen(self) -> None:
        row = _sample_row()
        with pytest.raises(Exception):
            row.decision_id = "changed"  # type: ignore[misc]

    def test_hash_width_enforced(self) -> None:
        """A full-length sha256 digest is refused, not silently truncated.

        Silent truncation would mean the schema and the writer could
        drift; the current width is the schema invariant, and any
        change to it should be a deliberate schema edit.
        """
        with pytest.raises(ValueError):
            DecisionRow(
                decision_id="d",
                policy="p",
                state_hash="a" * 64,  # full sha256, not the 16-prefix
                questions_hash="b" * QUESTIONS_HASH_HEX_LENGTH,
                provider="null",
                answers_json="{}",
                latency_ms=0,
                timestamp="2026-01-01T00:00:00+00:00",
            )


class TestBuildDecisionRow:
    def test_hashes_are_populated_server_side(self) -> None:
        """The builder never asks the caller for a hash.

        Rationale (msg-243 / msg-244): ``questions_hash`` is a server
        computation so an offline replay can match rows regardless of
        whether the caller sent ``questions_version``.
        """
        row = build_decision_row(
            decision_id="did",
            policy="p",
            state="state text",
            questions={"q": QuestionSpec(type="noul", instructions="i")},
            provider="null",
            answers={"q": {"noul": 0.5}},
            latency_ms=17,
            questions_version=None,
        )
        assert row.state_hash == compute_state_hash("state text")
        assert (
            row.questions_hash
            == compute_questions_hash({"q": QuestionSpec(type="noul", instructions="i")})
        )
        assert row.questions_version is None

    def test_questions_version_stored_verbatim(self) -> None:
        row = build_decision_row(
            decision_id="did",
            policy="p",
            state="s",
            questions={"q": QuestionSpec(type="noul", instructions="i")},
            provider="null",
            answers={"q": {"noul": 0.5}},
            latency_ms=1,
            questions_version="v1",
        )
        assert row.questions_version == "v1"

    def test_answers_json_is_canonical(self) -> None:
        """The answers column serialises with sorted keys.

        The test does not pin JSON exactly (that would over-couple to
        Python's dict repr), it pins the property: keys come out
        sorted so replay tools can diff rows without a re-parse.
        """
        row = build_decision_row(
            decision_id="did",
            policy="p",
            state="s",
            questions={"q": QuestionSpec(type="noul", instructions="i")},
            provider="null",
            answers={"b": {"noul": 0.1}, "a": {"noul": 0.9}},
            latency_ms=1,
            questions_version=None,
        )
        parsed = json.loads(row.answers_json)
        assert list(parsed.keys()) == ["a", "b"]


class TestDecisionLogSqlite:
    def test_roundtrip_in_memory(self) -> None:
        log = DecisionLog(":memory:")
        row = _sample_row()
        log.write(row)
        rows = log.fetch_all()
        assert len(rows) == 1
        stored = rows[0]
        assert stored["decision_id"] == row.decision_id
        assert stored["state_hash"] == row.state_hash
        assert stored["questions_hash"] == row.questions_hash
        assert stored["questions_version"] is None
        assert stored["provider"] == "null"
        assert stored["latency_ms"] == 42

    def test_persists_on_disk(self, tmp_path: Path) -> None:
        """Writes land on the configured path.

        The path/parent-dir handling matters because the production
        wiring points this at a real file under ``data/``.
        """
        db_path = tmp_path / "sub" / "decisions.sqlite"
        apply_decision_log_migrations(db_path)
        log = DecisionLog(db_path)
        log.write(_sample_row())
        log.close()
        assert db_path.exists()

    def test_columns_are_the_documented_ones(self) -> None:
        """The SQL schema contains the documented column set.

        This is the second half of the "no raw headers" guarantee (the
        first being the dataclass): even if a caller later invented a
        second writer that bypassed :class:`DecisionRow`, the table
        itself would not accept a ``headers`` column without a schema
        edit here.
        """
        import sqlite3

        log = DecisionLog(":memory:")
        log.write(_sample_row())
        # Snoop the schema via a fresh connection to the shared file
        # would be nicer but we have :memory:; reach in.
        cursor = log._conn.execute("PRAGMA table_info(decisions)")  # noqa: SLF001
        columns = {row[1] for row in cursor.fetchall()}
        assert columns == {
            "decision_id",
            "policy",
            "state_hash",
            "questions_hash",
            "questions_version",
            "provider",
            "answers_json",
            "latency_ms",
            "timestamp",
            "provider_error",
            "provider_model",
            "provider_input_tokens",
            "provider_output_tokens",
        }
        # And to satisfy the type checker that sqlite3 is used.
        assert isinstance(log._conn, sqlite3.Connection)  # noqa: SLF001


#: The ``decisions`` DDL exactly as PR #43 shipped it (before
#: ``provider_error``). Used to prove up-migration of an existing DB.
_PR43_CREATE_SQL = """
CREATE TABLE decisions (
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


#: Columns added on top of the PR #43 schema (Bohr msg-339 v5 schema).
_ADDED = (
    "provider_error",
    "provider_model",
    "provider_input_tokens",
    "provider_output_tokens",
)


def _assert_added_once(cols: list[str]) -> None:
    for name in _ADDED:
        assert cols.count(name) == 1, name


def _columns(db_path: Path) -> list[str]:
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        return [r[1] for r in conn.execute("PRAGMA table_info(decisions)")]


class TestDecisionLogMigrations:
    """Schema ownership (Bohr msg-260/262/264, Einstein msg-259/261/263)."""

    def test_provider_error_roundtrip(self) -> None:
        log = DecisionLog(":memory:")
        log.write(
            _sample_row(
                provider_error="jev:invalid_response",
                provider_model="jev-1.13.0",
                provider_input_tokens=10,
                provider_output_tokens=2,
            )
        )
        log.write(_sample_row(decision_id="did-2"))
        rows = {r["decision_id"]: r for r in log.fetch_all()}
        assert rows["did-1"]["provider_error"] == "jev:invalid_response"
        assert rows["did-1"]["provider_model"] == "jev-1.13.0"
        assert rows["did-1"]["provider_input_tokens"] == 10
        assert rows["did-1"]["provider_output_tokens"] == 2
        for name in _ADDED:
            assert rows["did-2"][name] is None, name

    def test_build_decision_row_carries_provider_error(self) -> None:
        row = build_decision_row(
            decision_id="d",
            policy="p",
            state="s",
            questions={"q": QuestionSpec(type="noul", instructions="i")},
            provider="null",
            answers={"q": {"noul": 0.5}},
            latency_ms=1,
            questions_version=None,
            provider_error="jev:auth",
            provider_model="jev-1.13.0",
            provider_input_tokens=5,
            provider_output_tokens=1,
        )
        assert row.provider_error == "jev:auth"
        assert row.provider_model == "jev-1.13.0"
        assert (row.provider_input_tokens, row.provider_output_tokens) == (5, 1)

    def test_migrations_on_fresh_file(self, tmp_path: Path) -> None:
        """Einstein msg-261: a fresh file must get CREATE before any ALTER."""
        db_path = tmp_path / "fresh" / "decisions.db"
        apply_decision_log_migrations(str(db_path))
        cols = _columns(db_path)
        assert "decision_id" in cols
        _assert_added_once(cols)
        log = DecisionLog(str(db_path))
        log.write(_sample_row())
        assert len(log.fetch_all()) == 1
        log.close()

    def test_migrations_on_fresh_memory(self) -> None:
        """``:memory:`` is a no-op for the migration; DecisionLog owns it."""
        apply_decision_log_migrations(":memory:")
        log = DecisionLog(":memory:")
        log.write(_sample_row())
        assert len(log.fetch_all()) == 1

    def test_migrations_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "decisions.db"
        apply_decision_log_migrations(db_path)
        apply_decision_log_migrations(db_path)
        _assert_added_once(_columns(db_path))

    def test_up_migration_from_pr43_schema(self, tmp_path: Path) -> None:
        """An existing PR #43 DB gains the column; old rows are intact."""
        import sqlite3

        db_path = tmp_path / "decisions.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(_PR43_CREATE_SQL)
            conn.execute(
                "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
                ("old-1", "p", "a" * 16, "b" * 16, None, "null", "{}", 3, "t0"),
            )
        assert not set(_ADDED) & set(_columns(db_path))

        apply_decision_log_migrations(db_path)
        log = DecisionLog(db_path)
        log.write(
            _sample_row(
                decision_id="new-1",
                provider_error="jev:invalid_response",
                provider_model="jev-1.13.0",
                provider_input_tokens=120,
                provider_output_tokens=7,
            )
        )
        rows = {r["decision_id"]: r for r in log.fetch_all()}
        log.close()

        for name in _ADDED:
            assert rows["old-1"][name] is None, name
        assert rows["old-1"]["latency_ms"] == 3
        assert rows["new-1"]["provider_error"] == "jev:invalid_response"
        assert rows["new-1"]["provider_model"] == "jev-1.13.0"
        assert rows["new-1"]["provider_input_tokens"] == 120
        assert rows["new-1"]["provider_output_tokens"] == 7
        _assert_added_once(_columns(db_path))

    def test_concurrent_migrations_are_race_safe(self, tmp_path: Path) -> None:
        """Einstein msg-259 #1: many workers migrating one PR #43 DB at once.

        Every thread uses its own connection (as separate uvicorn workers
        would). BEGIN IMMEDIATE serialises them; the late ones re-read
        table_info under the lock and no-op instead of a duplicate ALTER.
        """
        import sqlite3
        import threading

        db_path = tmp_path / "decisions.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(_PR43_CREATE_SQL)
            conn.execute(
                "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
                ("old-1", "p", "a" * 16, "b" * 16, None, "null", "{}", 3, "t0"),
            )

        barrier = threading.Barrier(8)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait()
                apply_decision_log_migrations(db_path)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        _assert_added_once(_columns(db_path))
        with sqlite3.connect(str(db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone() == (1,)

    def test_decisionlog_file_backed_requires_migration(self, tmp_path: Path) -> None:
        """Bohr msg-264 v4 contract: file-backed DecisionLog never owns the
        schema, so skipping the migration fails closed on first write."""
        import sqlite3

        log = DecisionLog(tmp_path / "decisions.db")
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            log.write(_sample_row())
        log.close()


def _sample_row(**overrides: Any) -> DecisionRow:
    row_kwargs: dict[str, Any] = {
        "decision_id": "did-1",
        "policy": "test.caller",
        "state_hash": "a" * QUESTIONS_HASH_HEX_LENGTH,
        "questions_hash": "b" * QUESTIONS_HASH_HEX_LENGTH,
        "provider": "null",
        "answers_json": "{}",
        "latency_ms": 42,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    row_kwargs.update(overrides)
    return DecisionRow(**row_kwargs)

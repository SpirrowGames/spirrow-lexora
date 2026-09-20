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
from lexora.decide.log import DecisionLog, DecisionRow, build_decision_row


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
        }
        # And to satisfy the type checker that sqlite3 is used.
        assert isinstance(log._conn, sqlite3.Connection)  # noqa: SLF001


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

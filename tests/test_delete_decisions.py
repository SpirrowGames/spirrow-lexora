"""Acceptance tests for ``tools/delete_decisions.py`` (T-t22-delete-decisions).

Maps to Bohr msg-1049 §6 / msg-1051 (a)-(f). The temp DB holds answering
rows under two policies (a tutorail one and another product's), plus
shadow rows of both a listed and an unlisted decision.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from lexora.decide.log import _CREATE_TABLE_SQL

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "delete_decisions.py"
_spec = importlib.util.spec_from_file_location("delete_decisions", _TOOL)
assert _spec is not None and _spec.loader is not None
dd = importlib.util.module_from_spec(_spec)
sys.modules["delete_decisions"] = dd
_spec.loader.exec_module(dd)

# (decision_id, policy, shadow_of)
ROWS = [
    ("t-1", "tutorail.next", None),
    ("t-2", "tutorail.next", None),
    ("t-3", "tutorail.next", None),
    ("o-1", "other.product", None),
    ("o-2", "other.product", None),
    ("s-t1", "tutorail.next", "t-1"),  # shadow of a listed row
    ("s-t3", "tutorail.next", "t-3"),  # shadow of an unlisted row
    ("s-o1", "other.product", "o-1"),  # shadow of another product's row
]


def _insert(conn: sqlite3.Connection, rows: list[tuple[str, str, str | None]]) -> None:
    conn.executemany(
        "INSERT INTO decisions (decision_id, policy, state_hash, questions_hash,"
        " provider, answers_json, latency_ms, timestamp, shadow_of)"
        " VALUES (?, ?, 'sh', 'qh', 'null', '{}', 1, '2026-11-10T00:00:00Z', ?)",
        rows,
    )
    conn.commit()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "decisions.db"
    conn = sqlite3.connect(str(path))
    conn.execute(_CREATE_TABLE_SQL)
    _insert(conn, ROWS)
    conn.close()
    return path


def _dump(path: Path) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("SELECT * FROM decisions ORDER BY decision_id").fetchall()
    finally:
        conn.close()


def _ids(path: Path) -> set[str]:
    return {row[0] for row in _dump(path)}  # type: ignore[misc]


def _run(path: Path, ids: list[str], dry_run: bool = False) -> "dd.Result":
    conn = sqlite3.connect(str(path))
    try:
        return dd.delete_decisions(conn, ids, dry_run=dry_run)
    finally:
        conn.close()


def test_a_only_listed_rows_deleted(db: Path) -> None:
    result = _run(db, ["t-1", "t-2"])
    assert result.deleted == 2
    assert _ids(db) == {"t-3", "o-1", "o-2", "s-t1", "s-t3", "s-o1"}


def test_a_other_policy_id_in_list_is_deleted(db: Path) -> None:
    """Boundary is the input list: another product's ID is deleted too."""
    result = _run(db, ["t-1", "o-2"])
    assert result.deleted == 2
    assert result.policy_counts == {"other.product": 1, "tutorail.next": 1}
    assert "o-2" not in _ids(db)


def test_b_unknown_ids_counted_not_found(db: Path) -> None:
    result = _run(db, ["t-1", "missing-1", "missing-2"])
    assert (result.requested, result.deleted, result.not_found) == (3, 1, 2)


def test_c_rerun_deletes_nothing_and_all_not_found(db: Path) -> None:
    _run(db, ["t-1", "t-2"])
    after_first = _dump(db)
    again = _run(db, ["t-1", "t-2"])
    assert (again.deleted, again.not_found) == (0, 2)
    assert _dump(db) == after_first


def test_d_dry_run_writes_nothing_and_counts_match(db: Path, tmp_path: Path) -> None:
    before = _dump(db)
    ids = ["t-1", "o-1", "missing"]
    dry = _run(db, ids, dry_run=True)
    assert _dump(db) == before
    real = _run(db, ids)
    assert dry.dry_run and not real.dry_run
    for name in ("requested", "deleted", "not_found", "shadow_remaining", "policy_counts"):
        assert getattr(dry, name) == getattr(real, name), name


def test_e_blank_lines_whitespace_and_duplicates() -> None:
    assert dd.parse_ids(["  t-1 ", "", "\t", "t-1", "t-2\r", "t-2"]) == ["t-1", "t-2"]


def test_e_duplicates_counted_once(db: Path) -> None:
    result = _run(db, [" t-1", "t-1 ", "", "t-1"])
    assert (result.requested, result.deleted, result.not_found) == (1, 1, 0)


def test_f_shadow_rows_counted_and_kept(db: Path) -> None:
    result = _run(db, ["t-1", "o-1"])
    assert result.shadow_remaining == 2  # s-t1, s-o1
    assert {"s-t1", "s-t3", "s-o1"} <= _ids(db)


def test_f_listed_shadow_row_excluded_from_remaining(db: Path) -> None:
    """A shadow row whose own ID is listed is deleted, so not 'remaining'."""
    result = _run(db, ["t-1", "s-t1"])
    assert result.deleted == 2
    assert result.shadow_remaining == 0
    assert "s-t1" not in _ids(db)


def test_many_ids_cross_chunk_boundary(tmp_path: Path) -> None:
    path = tmp_path / "big.db"
    conn = sqlite3.connect(str(path))
    conn.execute(_CREATE_TABLE_SQL)
    _insert(conn, [(f"id-{i}", "p", None) for i in range(1200)])
    conn.close()
    result = _run(path, [f"id-{i}" for i in range(0, 1300, 2)])
    assert (result.deleted, result.not_found) == (600, 50)
    assert len(_ids(path)) == 600


def test_cli_dry_run_then_delete(db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("t-1\n\nmissing\n", encoding="utf-8")
    before = _dump(db)
    assert dd.main([str(ids_file), "--dry-run", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "would delete: 1" in out and "not_found: 1" in out
    assert "shadow_remaining (not deleted, reference): 1" in out
    assert _dump(db) == before
    assert dd.main([str(ids_file), "--db", str(db)]) == 0
    assert "deleted: 1" in capsys.readouterr().out
    assert "t-1" not in _ids(db)


def test_cli_missing_db_is_not_created(tmp_path: Path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("t-1\n", encoding="utf-8")
    missing = tmp_path / "nope.db"
    assert dd.main([str(ids_file), "--db", str(missing)]) == 2
    assert not missing.exists()


def test_help_states_safety_boundary(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        dd.main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "input list is the only safety boundary" in out
    assert "run --dry-run separately first" in out

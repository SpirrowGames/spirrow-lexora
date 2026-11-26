"""``scripts/decide_send_inventory.py`` (T-decide-endpoint-bypasses-ingress-redaction msg-1154 §5)."""

from __future__ import annotations

import importlib.util
import io
import sqlite3
from pathlib import Path
from types import ModuleType

import pytest

from lexora.decide.log import DecisionLog, apply_decision_log_migrations, build_decision_row

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "decide_send_inventory.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("decide_send_inventory", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(did: str, policy: str, provider: str, **kw):  # type: ignore[no-untyped-def]
    from lexora.decide.contract import QuestionSpec

    return build_decision_row(
        decision_id=did,
        policy=policy,
        state="STATE-VALUE-never-printed",
        questions={"q": QuestionSpec(type="noul", instructions="i")},
        provider=provider,
        answers={"q": {"noul": 0.123456}},
        latency_ms=1,
        questions_version="QV-never-printed",
        **kw,
    )


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "decisions.db"
    apply_decision_log_migrations(path)
    log = DecisionLog(path)
    log.write(_row("a1", "mindwire.conductor.tierc", "jev", provider_model="jev-1.13.0"))
    log.write(_row("a2", "mindwire.conductor.tierc", "null", provider_error="jev:timeout"))
    log.write(_row("a3", "tutorail.judge", "null"))  # mode=off: never sent
    log.write(_row("a4", "tutorail.judge", "null"))  # shadow answering row
    log.write(_row("s4", "tutorail.judge", "jev", shadow_of="a4"))
    log.write(_row("a5", "x", "null", provider_error="ingress:key_redacted"))
    log.close()
    return path


def test_counts_by_policy_kind_outcome(db: Path) -> None:
    mod = _load()
    conn = mod.open_read_only(db)
    result = mod.inventory(conn)
    conn.close()
    sent = {r[:4] for r in result["sent"]}
    assert sent == {
        ("mindwire.conductor.tierc", "answering", "answered", 1),
        ("mindwire.conductor.tierc", "answering", "failed", 1),
        ("tutorail.judge", "shadow", "answered", 1),
    }
    assert [r[:2] for r in result["null_fallback"]] == [("mindwire.conductor.tierc", 1)]


def test_output_has_no_values(db: Path) -> None:
    mod = _load()
    conn = mod.open_read_only(db)
    out = io.StringIO()
    mod.render(mod.inventory(conn), out)
    with sqlite3.connect(db) as raw:
        values = [
            v
            for row in raw.execute(
                "SELECT state_hash, questions_hash, answers_json, questions_version, provider_model FROM decisions"
            )
            for v in row
            if v
        ]
    conn.close()
    text = out.getvalue()
    assert values
    for value in values:
        assert str(value) not in text
    assert "0.123456" not in text and "jev-1.13.0" not in text


def test_read_only(db: Path) -> None:
    mod = _load()
    conn = mod.open_read_only(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM decisions")
    conn.close()


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    mod = _load()
    with pytest.raises(sqlite3.OperationalError):
        mod.open_read_only(tmp_path / "absent.db").execute("SELECT 1 FROM decisions")


def test_redaction_impact_counts_without_values() -> None:
    """``scripts/decide_redaction_impact.py`` (msg-1156 §3(a)): counts only."""
    import json

    spec = importlib.util.spec_from_file_location(
        "decide_redaction_impact", _SCRIPT.parent / "decide_redaction_impact.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    secret = "sk-ant-api03-1234567890abcdefghij"
    q = {"q": {"type": "noul", "instructions": "i"}}
    lines = [
        json.dumps({"policy": "p", "state": "plain", "questions": q}),
        json.dumps({"policy": "p", "state": f"key {secret}", "questions": q}),
        json.dumps({"policy": "p", "state": json.dumps({f"k {secret}": 1}), "questions": q}),
        "",
    ]
    stats = mod.measure(lines)
    assert stats["p"]["inputs"] == 3
    assert stats["p"]["changed"] == 1
    assert stats["p"]["key_hits"] == 1
    out = io.StringIO()
    mod.render(stats, out)
    assert secret not in out.getvalue()

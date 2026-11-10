"""``lexora.tools.codex_restore`` (2026-11-10): reopen codex after a deploy
restart, clearing only what the restart itself caused."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from lexora.backends.codex_verification import RunRecord, ViolationRecord
from lexora.tools import codex_restore
from lexora.tools.codex_restore import (
    EXIT_NEEDS_HUMAN,
    EXIT_OPEN,
    EXIT_STILL_CLOSED,
    EXIT_VERIFY_FAILED,
    restore,
    triage,
)
from tests.backends.test_codex import make_backend, record_pass

T0 = datetime(2026, 11, 8, 9, 57, 5, tzinfo=timezone.utc)
T1 = T0 + timedelta(seconds=3)
MAX_RUN = timedelta(seconds=270)


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _v(seq: int, at: datetime, detail: list[str]) -> ViolationRecord:
    return ViolationRecord(seq, "codex", _iso(at), "codex-cli 0.160.0", "h", json.dumps(detail))


def _r(seq: int, at: datetime) -> RunRecord:
    return RunRecord(seq, "codex", _iso(at), "inst")


def _t(violations: list[ViolationRecord], runs: list[RunRecord]) -> codex_restore.Triage:
    return triage(violations, runs, restart_started=T0, restart_finished=T1, max_run=MAX_RUN)


class TestTriage:
    @pytest.mark.parametrize("detail", [["aborted:signal"], ["aborted:cancelled"]])
    def test_the_2026_11_08_latch_is_a_casualty(self, detail: list[str]) -> None:
        t = _t([_v(1017, T0 + timedelta(milliseconds=300), detail)], [_r(1016, T0 - timedelta(seconds=3))])
        assert [i.id for i in t.casualties] == [1017, 1016] and t.others == []

    @pytest.mark.parametrize(
        "detail",
        [["aborted:timeout"], ["item.started:command_execution"], ["aborted:signal", "x"], []],
    )
    def test_other_violation_kinds_need_a_human(self, detail: list[str]) -> None:
        t = _t([_v(9, T0, detail)], [])
        assert t.casualties == [] and len(t.others) == 1

    @pytest.mark.parametrize("at", [T0 - timedelta(seconds=6), T1 + timedelta(seconds=6)])
    def test_a_signal_latch_outside_the_window_needs_a_human(self, at: datetime) -> None:
        t = _t([_v(9, at, ["aborted:signal"])], [])
        assert t.casualties == [] and len(t.others) == 1

    def test_unfinished_run_window(self) -> None:
        in_flight = _r(1, T0 - MAX_RUN)  # oldest that could still have been running
        too_old = _r(2, T0 - MAX_RUN - timedelta(seconds=6))
        new_process = _r(3, T1 + timedelta(seconds=1))
        t = _t([], [in_flight, too_old, new_process])
        assert [i.id for i in t.casualties] == [1]
        assert len(t.others) == 1 and "unfinished run 2" in t.others[0]
        assert t.new_process_runs == [3]

    def test_unparseable_detail_is_not_a_casualty(self) -> None:
        bad = ViolationRecord(5, "codex", _iso(T0), "v", "h", "not json")
        assert _t([bad], []).others


@pytest.fixture
def statuses(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Queue of ``codex_disabled_reason`` values the fake status endpoint returns."""
    queue: list[str | None] = []

    def fake(url: str, **_k: Any) -> dict[str, Any]:
        return {"mode": "codex", "codex": {"codex_disabled_reason": queue.pop(0)}}

    monkeypatch.setattr(codex_restore, "read_status", fake)
    return queue


@pytest.fixture
def verifies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Results the fake verify returns (and the calls it saw)."""
    results: list[str] = []

    async def fake(backend: Any, *a: Any, **k: Any) -> tuple[str, list[Any]]:
        result = results.pop(0)
        if result == "pass":
            record_pass(backend)
        return result, []

    monkeypatch.setattr(codex_restore, "verify", fake)
    return results


def _restore(backend: Any, started: datetime, finished: datetime, **kw: Any) -> int:
    return restore(
        backend, restart_started=started, restart_finished=finished, max_run=MAX_RUN, status_url="x", **kw
    )


class TestRestore:
    def test_clears_the_restart_latch_verifies_and_reports_open(
        self, tmp_path: Path, statuses: list, verifies: list
    ) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        store = backend.state_store
        started = datetime.now(timezone.utc) - timedelta(seconds=2)
        store.record_run_started("codex", "old-process")
        store.record_violation("codex", json.dumps(["aborted:signal"]), codex_version="v", config_hash="h")
        finished = datetime.now(timezone.utc) + timedelta(seconds=1)
        statuses += ["verification_missing", None]
        verifies.append("pass")
        assert _restore(backend, started, finished) == EXIT_OPEN
        assert store.uncleared_violations() == [] and store.unfinished_runs() == []
        assert verifies == [] and statuses == []

    def test_a_real_violation_clears_nothing(self, tmp_path: Path, statuses: list, verifies: list) -> None:
        backend = make_backend(tmp_path)
        store = backend.state_store
        started = datetime.now(timezone.utc) - timedelta(seconds=2)
        store.record_violation("codex", json.dumps(["aborted:signal"]), codex_version="v", config_hash="h")
        store.record_violation("codex", json.dumps(["item.started:command_execution"]), codex_version="v", config_hash="h")
        finished = datetime.now(timezone.utc)
        assert _restore(backend, started, finished) == EXIT_NEEDS_HUMAN
        assert len(store.uncleared_violations()) == 2  # not even the casualty
        assert statuses == [] and verifies == []

    def test_dry_run_writes_nothing(self, tmp_path: Path, statuses: list, verifies: list) -> None:
        backend = make_backend(tmp_path)
        store = backend.state_store
        started = datetime.now(timezone.utc) - timedelta(seconds=2)
        store.record_violation("codex", json.dumps(["aborted:signal"]), codex_version="v", config_hash="h")
        assert _restore(backend, started, datetime.now(timezone.utc), dry_run=True) == EXIT_OPEN
        assert len(store.uncleared_violations()) == 1

    def test_nothing_to_clear_and_open_skips_verify(self, tmp_path: Path, statuses: list, verifies: list) -> None:
        backend = make_backend(tmp_path)
        now = datetime.now(timezone.utc)
        statuses.append(None)
        assert _restore(backend, now, now) == EXIT_OPEN
        assert verifies == []

    def test_stale_config_is_reverified(self, tmp_path: Path, statuses: list, verifies: list) -> None:
        backend = make_backend(tmp_path)
        now = datetime.now(timezone.utc)
        statuses += ["verification_stale", None]
        verifies.append("pass")
        assert _restore(backend, now, now) == EXIT_OPEN

    def test_verify_fail_is_reported(self, tmp_path: Path, statuses: list, verifies: list) -> None:
        backend = make_backend(tmp_path)
        now = datetime.now(timezone.utc)
        statuses.append("verification_stale")
        verifies.append("fail")
        assert _restore(backend, now, now) == EXIT_VERIFY_FAILED

    @pytest.mark.parametrize("reason", ["quota_hold", "failure_hold", "data_controls_unverified"])
    def test_what_a_deploy_cannot_fix_is_reported(
        self, tmp_path: Path, statuses: list, verifies: list, reason: str
    ) -> None:
        backend = make_backend(tmp_path)
        now = datetime.now(timezone.utc)
        statuses.append(reason)
        assert _restore(backend, now, now) == EXIT_STILL_CLOSED
        assert verifies == []

    def test_main_refuses_a_reversed_window(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = codex_restore.main(
            ["--restart-started-at", _iso(T1), "--restart-finished-at", _iso(T0)]
        )
        assert rc == codex_restore.EXIT_USAGE

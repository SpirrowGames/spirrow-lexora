"""The decision-log write must not block the event loop nor the default executor.

Acceptance tests 1-5 of Bohr msg-464 (v3, cleared by Einstein):
``DecisionLog.awrite`` runs the sync ``write`` on a dedicated
single-thread executor named ``decision-log``.

No test relies on sleep timing: writes are held inside ``write`` with a
``threading.Event`` and released explicitly.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from lexora.config import Settings
from lexora.decide.config import DecisionSettings
from lexora.decide.log import (
    DecisionLog,
    DecisionRow,
    apply_decision_log_migrations,
    build_decision_row,
)
from lexora.main import create_app

_BODY = {
    "state": "s",
    "questions": {"q": {"type": "noul", "instructions": "i"}},
    "policy": "test.async-log",
}
_WAIT = 10.0  # safety net only; every wait is released explicitly


def _row(i: int) -> DecisionRow:
    return build_decision_row(
        decision_id=f"d{i}",
        policy="p",
        state="s",
        questions={"q": {"type": "noul", "instructions": "i"}},
        provider="null",
        answers={"q": {"noul": 0.5}},
        latency_ms=0,
        questions_version=None,
    )


class _Spy:
    """Wraps ``DecisionLog.write``: records threads and can hold writes."""

    def __init__(self, log: DecisionLog, *, hold: bool = False) -> None:
        self._real = log.write
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.entered = threading.Event()
        self.thread_names: list[str] = []
        self.thread_ids: list[int] = []
        self._inside = 0
        self.max_inside = 0
        self._mu = threading.Lock()
        log.write = self  # type: ignore[method-assign]

    def __call__(self, row: DecisionRow) -> None:
        with self._mu:
            self._inside += 1
            self.max_inside = max(self.max_inside, self._inside)
            self.thread_names.append(threading.current_thread().name)
            self.thread_ids.append(threading.get_ident())
        self.entered.set()
        try:
            assert self.release.wait(_WAIT), "test never released the write"
            self._real(row)
        finally:
            with self._mu:
                self._inside -= 1


def _app() -> Any:
    settings = Settings(
        decision=DecisionSettings(primary="null", mode="off", log_path=":memory:")
    )
    app = create_app(settings=settings)

    @app.get("/__test_ping")
    async def _ping() -> dict[str, str]:  # another endpoint on the same loop
        return {"ok": "yes"}

    return app


def _client(app: Any) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _getaddrinfo_completes() -> None:
    loop = asyncio.get_running_loop()
    await asyncio.wait_for(loop.getaddrinfo("localhost", None), timeout=_WAIT)


def _default_executor_thread_names(loop: asyncio.AbstractEventLoop) -> list[str]:
    executor = getattr(loop, "_default_executor", None)
    if executor is None:
        return []
    return [t.name for t in getattr(executor, "_threads", ())]


async def test_1_write_runs_on_dedicated_thread() -> None:
    app = _app()
    log: DecisionLog = app.state.decision_log
    spy = _Spy(log)
    try:
        async with _client(app) as client:
            resp = await client.post("/v1/decide", json=_BODY)
        assert resp.status_code == 200, resp.text
        assert spy.thread_ids and spy.thread_ids[0] != threading.get_ident()
        assert spy.thread_names[0].startswith("decision-log")
        assert len(log.fetch_all()) == 1
    finally:
        log.close()


async def test_2_other_endpoint_served_while_write_is_held() -> None:
    app = _app()
    log: DecisionLog = app.state.decision_log
    spy = _Spy(log, hold=True)
    try:
        async with _client(app) as client:
            decide = asyncio.create_task(client.post("/v1/decide", json=_BODY))
            assert await asyncio.to_thread(spy.entered.wait, _WAIT)
            ping = await asyncio.wait_for(client.get("/__test_ping"), timeout=_WAIT)
            assert ping.status_code == 200
            assert not decide.done()
            spy.release.set()
            resp = await asyncio.wait_for(decide, timeout=_WAIT)
        assert resp.status_code == 200, resp.text
        assert len(log.fetch_all()) == 1
    finally:
        spy.release.set()
        log.close()


async def test_3_write_failure_is_a_500() -> None:
    app = _app()
    log: DecisionLog = app.state.decision_log

    def _boom(row: DecisionRow) -> None:
        raise sqlite3.OperationalError("database is locked")

    log.write = _boom  # type: ignore[method-assign]
    try:
        async with _client(app) as client:
            resp = await client.post("/v1/decide", json=_BODY)
        assert resp.status_code == 500
        assert "decision_id" not in resp.text
    finally:
        log.close()


async def test_4_one_thread_under_burst_and_default_pool_untouched() -> None:
    n = 8
    app = _app()
    log: DecisionLog = app.state.decision_log
    spy = _Spy(log, hold=True)
    loop = asyncio.get_running_loop()
    try:
        async with _client(app) as client:
            tasks = [
                asyncio.create_task(client.post("/v1/decide", json=_BODY))
                for _ in range(n)
            ]
            assert await asyncio.to_thread(spy.entered.wait, _WAIT)
            # Let every request reach awrite while the first write is held.
            for _ in range(50):
                await asyncio.sleep(0)
            await _getaddrinfo_completes()
            assert not any(
                name.startswith("decision-log")
                for name in _default_executor_thread_names(loop)
            )
            spy.release.set()
            responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=_WAIT)
        assert [r.status_code for r in responses] == [200] * n
        assert spy.max_inside == 1
        assert {name.split("_")[0] for name in spy.thread_names} == {"decision-log"}
        assert len(set(spy.thread_ids)) == 1
        assert len(log.fetch_all()) == n
    finally:
        spy.release.set()
        log.close()


async def test_5_cancel_storm_keeps_rows_and_leaks_nothing(tmp_path: Path) -> None:
    n = 8
    db = tmp_path / "decisions.db"
    apply_decision_log_migrations(db)
    log = DecisionLog(path=db)
    spy = _Spy(log, hold=True)
    loop = asyncio.get_running_loop()
    try:
        tasks = [asyncio.create_task(log.awrite(_row(i))) for i in range(n)]
        assert await asyncio.to_thread(spy.entered.wait, _WAIT)
        for t in tasks:
            t.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(r, asyncio.CancelledError) for r in results)

        # (a) the default executor (DNS) is still usable
        await _getaddrinfo_completes()
        # (b) no decision-log thread lives in the default executor, and
        #     exactly one decision-log thread exists overall
        assert not any(
            name.startswith("decision-log")
            for name in _default_executor_thread_names(loop)
        )
        assert (
            sum(t.name.startswith("decision-log") for t in threading.enumerate())
            == 1
        )
        assert spy.max_inside == 1
    finally:
        spy.release.set()
        # (c) close() drains the queued writes before closing
        await asyncio.to_thread(log.close)

    with sqlite3.connect(db) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
    assert count == n


def test_sync_only_use_spawns_no_thread() -> None:
    """msg-464 #5: the worker is created lazily on first submit."""
    before = sum(t.name.startswith("decision-log") for t in threading.enumerate())
    log = DecisionLog()
    log.write(_row(0))
    after = sum(t.name.startswith("decision-log") for t in threading.enumerate())
    assert after == before
    assert len(log.fetch_all()) == 1
    log.close()



async def test_cancelled_write_failure_is_logged_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bohr msg-467: a write that fails after its awaiter was cancelled
    has no caller to raise into, so ``_log_write_failure`` must log it —
    exactly once, with no row content."""
    import lexora.decide.log as log_mod

    events: list[tuple[str, dict[str, Any]]] = []

    class _RecordingLogger:
        def error(self, event: str, **kw: Any) -> None:
            events.append((event, kw))

    monkeypatch.setattr(log_mod, "_logger", _RecordingLogger())

    log = DecisionLog()
    release = threading.Event()
    entered = threading.Event()

    def _boom(row: DecisionRow) -> None:
        entered.set()
        assert release.wait(_WAIT)
        raise sqlite3.OperationalError("database is locked")

    log.write = _boom  # type: ignore[method-assign]
    try:
        task = asyncio.create_task(log.awrite(_row(0)))
        assert await asyncio.to_thread(entered.wait, _WAIT)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == []  # still running: nothing to report yet
        release.set()
    finally:
        release.set()
        await asyncio.to_thread(log.close)  # drains the failing job
    # done-callbacks run via call_soon; give the loop a turn
    for _ in range(5):
        await asyncio.sleep(0)

    assert events == [
        (
            "decision_log_write_failed",
            {"exc_type": "OperationalError", "error": "database is locked"},
        )
    ]

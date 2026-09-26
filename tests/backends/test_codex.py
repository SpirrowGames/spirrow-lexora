"""Tests for the codex backend (T-naysayer-codex-backend msg-294 PR-1).

The CLI is replaced by ``fake_codex_cli.py`` (``_wrap`` is swapped per test
instance; no bwrap). Error wording and ``--json`` event shapes used here are
**ASSUMED** -- nothing was measured, because no Codex login exists yet
(msg-269). The classification fixtures must be re-verified against the real
CLI after ``codex login --device-auth``.
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from lexora.backends import codex as codex_mod
from lexora.backends.base import UsageSink
from lexora.backends.codex import (
    CodexAuthError,
    CodexBackend,
    CodexFailed,
    CodexLaunchError,
    CodexNotVerifiedError,
    CodexQuotaError,
    CodexTimeout,
    CodexToolUseViolation,
    CodexUnsupportedInputError,
    CodexUnverifiableRun,
    EventFindings,
    build_env,
    classify_events,
    classify_failure,
    parse_reset_at,
    request_to_prompt,
    run_verdict,
    usage_from_events,
)
from lexora.backends.codex import CodexRun
from lexora.backends import codex_verification
from lexora.backends.codex_verification import ClearViolationError, CodexStateStore
from lexora.backends.factory import create_backend
from lexora.config import BackendSettings, CodexSettings

FAKE_CLI = str(Path(__file__).with_name("fake_codex_cli.py"))
VERSION = "codex-cli 0.99.0"
SRC = Path(__file__).resolve().parents[2] / "src" / "lexora"

REQUEST = {"model": "gpt-5-codex", "messages": [{"role": "system", "content": "be terse"}, {"role": "user", "content": "review this"}]}


def _host_env(backend: CodexBackend) -> dict[str, str]:
    env = build_env(backend.codex_home, os.environ)
    # Windows' Python cannot start sockets / random without SYSTEMROOT.
    for key in ("SYSTEMROOT", "SystemRoot"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def make_backend(
    tmp_path: Path, scenario: str = "ok", timeout: float = 30.0, cli_overrides: list[str] | None = None
) -> CodexBackend:
    home = tmp_path / "codex-home"
    home.mkdir(exist_ok=True)
    backend = CodexBackend(
        codex_home=str(home),
        state_store=CodexStateStore(tmp_path / "codex.db"),
        models=["gpt-5-codex"],
        timeout=timeout,
        cli_overrides=cli_overrides or [],
        name="codex",
    )
    backend._wrap = lambda inner, workdir: [sys.executable, FAKE_CLI, backend._scenario, *inner[1:]]  # type: ignore[method-assign]
    backend._scenario = scenario  # type: ignore[attr-defined]
    backend._subprocess_env = lambda: _host_env(backend)  # type: ignore[method-assign]

    async def version() -> str:
        return VERSION

    backend._codex_version = version  # type: ignore[method-assign]
    return backend


def record_pass(backend: CodexBackend, version: str = VERSION, config_hash: str | None = None) -> None:
    backend.state_store.record_verification(
        backend.name, "pass", version, config_hash or backend.config_hash(), [{"check": "fixture", "ok": True}]
    )


async def _drain(agen: Any) -> list[bytes]:
    return [chunk async for chunk in agen]


# --------------------------------------------------------------------------
# Gate (D-1a')
# --------------------------------------------------------------------------


class TestGate:
    @pytest.fixture
    def no_subprocess(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
        calls: list[tuple[Any, ...]] = []

        async def boom(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            raise AssertionError(f"subprocess started: {args}")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
        return calls

    async def test_no_record_no_public_method_starts_a_subprocess(self, tmp_path: Path, no_subprocess: list) -> None:
        # Real _codex_version on purpose: the gate must refuse before it.
        backend = CodexBackend(codex_home=str(tmp_path), state_store=CodexStateStore(tmp_path / "s.db"))
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reason == "verification_missing"
        with pytest.raises(CodexNotVerifiedError):
            await _drain(backend.chat_completions_stream(REQUEST, usage_sink=UsageSink()))
        with pytest.raises(CodexNotVerifiedError):
            await backend.completions({"prompt": "x"})
        with pytest.raises(CodexNotVerifiedError):
            await _drain(backend.completions_stream({"prompt": "x"}))
        assert await backend.health_check() is False
        assert no_subprocess == []

    async def test_latest_record_fail_keeps_gate_closed(self, tmp_path: Path, no_subprocess: list) -> None:
        backend = CodexBackend(codex_home=str(tmp_path), state_store=CodexStateStore(tmp_path / "s.db"))
        record_pass(backend)
        backend.state_store.record_verification("codex", "fail", VERSION, backend.config_hash(), [])
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reason == "verification_missing"
        assert no_subprocess == []

    async def test_config_change_is_stale_without_subprocess(self, tmp_path: Path, no_subprocess: list) -> None:
        backend = CodexBackend(codex_home=str(tmp_path), state_store=CodexStateStore(tmp_path / "s.db"))
        record_pass(backend)
        backend.cli_overrides = ['tools.shell=true']  # operator edit after verification
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reason == "verification_stale"
        assert no_subprocess == []

    async def test_cli_version_change_is_stale(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend, version="codex-cli 0.98.0")

        async def must_not_run(*a: Any, **k: Any) -> Any:
            raise AssertionError("codex exec started with a stale verification")

        backend._run_unverified = must_not_run  # type: ignore[method-assign]
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reason == "verification_stale"

    async def test_hash_covers_every_command_field(self, tmp_path: Path) -> None:
        base = CodexBackend(codex_home="/srv/codex", state_store=CodexStateStore(tmp_path / "s.db"), models=["m"])
        h = base.config_hash()
        for attr, value in [
            ("codex_bin", "/opt/codex"),
            ("codex_home", "/srv/other"),
            ("bwrap_bin", "/opt/bwrap"),
            ("ro_binds", ["/opt"]),
            ("cli_overrides", ["a=b"]),
            ("model_mapping", {"m": "n"}),
            ("models", ["m2"]),
        ]:
            other = CodexBackend(codex_home="/srv/codex", state_store=base.state_store, models=["m"])
            setattr(other, attr, value)
            assert other.config_hash() != h, attr
        # Fields that do not reach the command line do not close the gate.
        other = CodexBackend(codex_home="/srv/codex", state_store=base.state_store, models=["m"], timeout=1, max_concurrency=9)
        assert other.config_hash() == h

    async def test_verified_request_is_served(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        response = await backend.chat_completions(REQUEST)
        text = response["choices"][0]["message"]["content"]
        assert text.startswith("REVIEW: [system]")
        assert response["model"] == "gpt-5-codex"
        assert response["usage"] == {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127}

    async def test_stream_emits_only_after_completion_and_fills_sink(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        sink = UsageSink()
        chunks = await _drain(backend.chat_completions_stream(REQUEST, usage_sink=sink))
        assert chunks[-1] == b"data: [DONE]\n\n"
        assert b"REVIEW: " in b"".join(chunks)
        assert (sink.prompt_tokens, sink.completion_tokens) == (120, 7)


# --------------------------------------------------------------------------
# D-1c runtime detection
# --------------------------------------------------------------------------


class TestToolUseViolation:
    """D-1c + the msg-317 release rules: global latch, clear THEN pass."""

    async def _trip(self, backend: CodexBackend) -> int:
        backend._scenario = "tool_use"  # type: ignore[attr-defined]
        with pytest.raises(CodexToolUseViolation):
            await backend.chat_completions(REQUEST)
        backend._scenario = "ok"  # type: ignore[attr-defined]
        return backend.state_store.uncleared_violations()[-1].id

    async def _assert_closed(self, backend: CodexBackend, reason: str | None = None) -> None:
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        if reason:
            assert exc.value.reason == reason

    async def test_tool_event_discards_answer_and_latches(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        vid = await self._trip(backend)
        violation = backend.state_store.violations()[0]
        assert (violation.id, violation.codex_version, violation.config_hash) == (vid, VERSION, backend.config_hash())
        assert "command_execution" in violation.detail
        await self._assert_closed(backend, "tool_use_violation")
        with pytest.raises(CodexNotVerifiedError):
            await _drain(backend.chat_completions_stream(REQUEST))

    async def test_stream_tool_event_yields_nothing(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "tool_use")
        record_pass(backend)
        received: list[bytes] = []
        with pytest.raises(CodexToolUseViolation):
            async for chunk in backend.chat_completions_stream(REQUEST):
                received.append(chunk)
        assert received == []

    async def test_unknown_event_trips_d1c(self, tmp_path: Path) -> None:
        """msg-315 #4: an event nobody recognises is treated as an execution."""
        backend = make_backend(tmp_path, "unknown_event")
        record_pass(backend)
        with pytest.raises(CodexToolUseViolation):
            await backend.chat_completions(REQUEST)
        assert "mystery_capability" in backend.state_store.violations()[0].detail

    async def test_repass_with_same_config_does_not_release(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        await self._trip(backend)
        record_pass(backend)
        await self._assert_closed(backend, "tool_use_violation")

    async def test_hash_change_and_pass_does_not_release(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        await self._trip(backend)
        backend.cli_overrides = ["features.harmless=true"]
        record_pass(backend)
        await self._assert_closed(backend, "tool_use_violation")

    async def test_a_b_a_toggle_does_not_release(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        await self._trip(backend)
        backend.cli_overrides = ["features.harmless=true"]
        record_pass(backend)
        backend.cli_overrides = []
        record_pass(backend)
        await self._assert_closed(backend, "tool_use_violation")

    async def test_clear_then_pass_releases(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        vid = await self._trip(backend)
        backend.state_store.clear_violation(vid, "V-2' did not script tool X; added")
        # Cleared but no pass after the clearance: still closed.
        await self._assert_closed(backend, "verification_missing")
        record_pass(backend)
        response = await backend.chat_completions(REQUEST)
        assert response["choices"][0]["message"]["content"].startswith("REVIEW: ")
        cleared = backend.state_store.violations()[0]
        assert cleared.cleared_reason == "V-2' did not script tool X; added"

    async def test_pass_before_clear_does_not_count(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        vid = await self._trip(backend)
        record_pass(backend)  # order wrong: pass first ...
        backend.state_store.clear_violation(vid, "investigated")  # ... then clear
        await self._assert_closed(backend, "verification_missing")

    async def test_one_of_two_cleared_stays_closed(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        first = await self._trip(backend)
        backend.state_store.record_violation("other-codex", "[]", codex_version="x", config_hash="y")
        backend.state_store.clear_violation(first, "investigated")
        record_pass(backend)
        # The second violation belongs to another backend: the latch is global.
        await self._assert_closed(backend, "tool_use_violation")

    def test_empty_reason_is_refused(self, tmp_path: Path) -> None:
        store = CodexStateStore(tmp_path / "s.db")
        v = store.record_violation("codex", "[]", codex_version="v", config_hash="h")
        for reason in ("", "   "):
            with pytest.raises(ClearViolationError):
                store.clear_violation(v.id, reason)
        store.clear_violation(v.id, "ok")
        with pytest.raises(ClearViolationError):
            store.clear_violation(v.id, "again")
        with pytest.raises(ClearViolationError):
            store.clear_violation(999, "nope")


class TestGateLogSequence:
    """msg-324/326: one persisted AUTOINCREMENT numbering source, and the
    COALESCE(..., 0) clearance threshold."""

    def test_seq_survives_a_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "codex.db"
        first = CodexStateStore(db)
        before = [first.record_verification("codex", "pass", "v", "h", []).seq for _ in range(3)]
        restarted = CodexStateStore(db)  # new instance on the same file = process restart
        after = restarted.record_violation("codex", "[]", codex_version="v", config_hash="h").seq
        assert after > max(before)

    async def test_old_pass_restart_violation_clear_stays_closed(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)  # pre-restart pass
        restarted = make_backend(tmp_path)  # same DB file, fresh objects
        restarted._scenario = "tool_use"  # type: ignore[attr-defined]
        with pytest.raises(CodexToolUseViolation):
            await restarted.chat_completions(REQUEST)
        again = make_backend(tmp_path)
        again.state_store.clear_violation(again.state_store.violations()[0].seq, "investigated")
        with pytest.raises(CodexNotVerifiedError) as exc:
            await make_backend(tmp_path).chat_completions(REQUEST)
        assert exc.value.reason == "verification_missing"

    def test_deleted_rows_are_not_reused(self, tmp_path: Path) -> None:
        db = tmp_path / "codex.db"
        store = CodexStateStore(db)
        top = store.record_verification("codex", "fail", "v", "h", []).seq
        conn = sqlite3.connect(db)
        with conn:
            conn.execute("DELETE FROM codex_verification WHERE seq = ?", (top,))
            conn.execute("DELETE FROM codex_gate_log WHERE seq = ?", (top,))
        conn.close()
        assert CodexStateStore(db).record_verification("codex", "pass", "v", "h", []).seq > top

    def test_every_kind_shares_one_sequence(self, tmp_path: Path) -> None:
        store = CodexStateStore(tmp_path / "codex.db")
        a = store.record_verification("codex", "pass", "v", "h", []).seq
        b = store.record_violation("codex", "[]", codex_version="v", config_hash="h").seq
        c = store.clear_violation(b, "r").cleared_seq
        d = store.record_verification("codex", "pass", "v", "h", []).seq
        assert a < b < c < d  # type: ignore[operator]

    async def test_fresh_db_with_one_pass_opens(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        assert backend.state_store.clearance_threshold() == 0
        record_pass(backend)
        response = await backend.chat_completions(REQUEST)
        assert response["choices"][0]["message"]["content"].startswith("REVIEW: ")

    async def test_fresh_db_with_no_rows_is_closed(self, tmp_path: Path) -> None:
        with pytest.raises(CodexNotVerifiedError) as exc:
            await make_backend(tmp_path).chat_completions(REQUEST)
        assert exc.value.reason == "verification_missing"

    async def test_violation_without_clearance_then_pass_is_closed(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        backend.state_store.record_violation("codex", "[]", codex_version=VERSION, config_hash="h")
        record_pass(backend)
        assert backend.state_store.clearance_threshold() == 0
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reason == "tool_use_violation"

    async def test_unreadable_state_closes_the_gate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)

        def broken() -> int:
            raise sqlite3.DatabaseError("database disk image is malformed")

        monkeypatch.setattr(backend.state_store, "clearance_threshold", broken)
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reason == "state_unreadable"

    def test_numbering_is_left_to_sqlite(self) -> None:
        """msg-324 fence, on code only (docstrings excluded): the store's one
        MAX is the COALESCE clearance threshold; no Python max() call and no
        seq arithmetic anywhere; gate-log SQL lives in the store only."""
        store_path = SRC / "backends" / "codex_verification.py"
        sql_with_max = [c for c in _code_strings(store_path) if "max(" in c.lower()]
        assert len(sql_with_max) == 1
        assert "COALESCE" in sql_with_max[0] and "kind = 'clearance'" in sql_with_max[0]
        for path in _py_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "max":
                    args = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)} | {
                        n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                    }
                    assert not args & {"seq", "cleared_seq"}, f"{path}:{node.lineno}"
                if isinstance(node, ast.BinOp):
                    names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)} | {
                        n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                    }
                    assert not names & {"seq", "cleared_seq"}, f"{path}:{node.lineno}"
            if path != store_path:
                assert not any("codex_gate_log" in c for c in _code_strings(path)), path


class TestClassifyEvents:
    def test_asymmetry(self) -> None:
        refusal = {"type": "error", "message": "tool call declined: shell is disabled (call_1)"}
        unknown = {"type": "tool.declined", "call_id": "call_1", "message": "declined"}
        execution = {"type": "item.completed", "item": {"type": "command_execution"}}
        benign = [{"type": "turn.started"}, {"type": "item.completed", "item": {"type": "agent_message", "text": "declined"}}]
        f = classify_events([refusal, unknown, execution, *benign])
        assert f.refusals == (refusal,)
        assert f.unknown == (unknown,)  # never refusal evidence ...
        assert f.executions == (execution,)
        assert f.executions_or_unknown == (execution, unknown)  # ... always counted as execution

    def test_plain_error_is_neither(self) -> None:
        f = classify_events([{"type": "error", "message": "stream disconnected"}])
        assert f == EventFindings()


# --------------------------------------------------------------------------
# Failure classification (ASSUMED wording; re-verify after login)
# --------------------------------------------------------------------------


class TestClassification:
    async def test_quota_with_reset(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "quota")
        record_pass(backend)
        before = datetime.now(timezone.utc)
        with pytest.raises(CodexQuotaError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reset_at is not None
        delta = exc.value.reset_at - before
        assert timedelta(hours=2, minutes=4) < delta < timedelta(hours=2, minutes=6)

    async def test_auth_without_terminal_event_latches(self, tmp_path: Path) -> None:
        """msg-396 D-1d-3': auth is not an exemption; the classification is
        kept in the detail and on the exception."""
        backend = make_backend(tmp_path, "auth")
        record_pass(backend)
        with pytest.raises(CodexUnverifiableRun) as exc:
            await backend.chat_completions(REQUEST)
        assert isinstance(exc.value.cause, CodexAuthError)
        assert backend.state_store.uncleared_violations()[0].detail == '["aborted:no_terminal:auth"]'

    async def test_launch_failure(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        backend._wrap = lambda inner, workdir: [str(tmp_path / "no-such-bwrap")]  # type: ignore[method-assign]
        with pytest.raises(CodexLaunchError):
            await backend.chat_completions(REQUEST)

    @pytest.mark.parametrize(
        ("rc", "stderr", "events", "expected"),
        [
            (1, "Error: rate limit reached for requests", [], CodexQuotaError),
            (1, "", [{"type": "turn.failed", "error": {"message": "You've hit your usage limit."}}], CodexQuotaError),
            (1, "", [{"type": "error", "message": "stream error: 401 Unauthorized"}], CodexAuthError),
            (1, "bwrap: Can't find source path /nope: No such file or directory", [], CodexLaunchError),
            (-9, "usage limit", [], CodexFailed),  # killed: never read as quota
            (1, "something else entirely", [], CodexFailed),
        ],
    )
    def test_assumed_fixtures(self, rc: int, stderr: str, events: list, expected: type) -> None:
        assert type(classify_failure(rc, stderr, events)) is expected

    def test_reset_parsing(self) -> None:
        now = datetime(2026, 9, 23, 0, 0, tzinfo=timezone.utc)
        assert parse_reset_at('{"resets_in_seconds": 900}', now) == now + timedelta(seconds=900)
        assert parse_reset_at("try again at 2026-09-23T05:00:00Z", now) == datetime(2026, 9, 23, 5, tzinfo=timezone.utc)
        assert parse_reset_at("Try again in 1 day 3 hours", now) == now + timedelta(days=1, hours=3)
        assert parse_reset_at("usage limit", now) is None

    def test_usage_is_assigned_from_last_turn(self) -> None:
        events = [
            {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1}},
            {"type": "turn.completed", "usage": {"input_tokens": 9, "output_tokens": 2}},
        ]
        assert usage_from_events(events) == (9, 2)


# --------------------------------------------------------------------------
# D-1d: runs that cannot show no tool ran (msg-394 / msg-396)
# --------------------------------------------------------------------------


def _ticks(backend: CodexBackend) -> int:
    path = Path(backend.codex_home) / "ticks"
    return path.stat().st_size if path.exists() else 0


async def _until_ticking(backend: CodexBackend) -> None:
    for _ in range(200):
        if _ticks(backend) > 0:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("fake codex never started")


async def _assert_dead(backend: CodexBackend) -> None:
    before = _ticks(backend)
    await asyncio.sleep(0.5)
    assert _ticks(backend) == before, "codex process still running after the call returned"


def _detail(backend: CodexBackend) -> list[str]:
    return [v.detail for v in backend.state_store.uncleared_violations()]


class TestUnverifiableRuns:
    async def test_timeout_latches_kills_and_closes_the_gate(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "sleep", timeout=1.0)
        record_pass(backend)
        with pytest.raises(CodexTimeout):
            await backend.chat_completions(REQUEST)
        assert _detail(backend) == ['["aborted:timeout"]']
        await _assert_dead(backend)
        backend._scenario = "ok"  # type: ignore[attr-defined]
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reason == "tool_use_violation"

    async def test_cancellation_latches_and_leaves_no_child(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "sleep", timeout=30.0)
        record_pass(backend)
        task = asyncio.create_task(backend.chat_completions(REQUEST))
        await _until_ticking(backend)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _detail(backend) == ['["aborted:cancelled"]']
        await _assert_dead(backend)

    async def test_cancellation_while_queued_does_not_latch(self, tmp_path: Path) -> None:
        """codex never ran (waiting on the semaphore) -> nothing to latch."""
        backend = make_backend(tmp_path, "ok")
        record_pass(backend)
        backend._semaphore = asyncio.Semaphore(0)
        task = asyncio.create_task(backend.chat_completions(REQUEST))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.state_store.uncleared_violations() == []

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal exit status")
    async def test_signal_death_latches(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "signal")
        record_pass(backend)
        with pytest.raises(CodexUnverifiableRun) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.cause is None
        assert _detail(backend) == ['["aborted:signal"]']

    async def test_exit0_without_terminal_event_latches(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "exit0_no_terminal")
        record_pass(backend)
        with pytest.raises(CodexUnverifiableRun):
            await backend.chat_completions(REQUEST)
        assert _detail(backend) == ['["aborted:no_terminal:failed"]']

    async def test_quota_without_terminal_or_tool_does_not_latch(self, tmp_path: Path) -> None:
        """The fallback's reset-wait must keep working (msg-267 scope 2)."""
        backend = make_backend(tmp_path, "quota")
        record_pass(backend)
        with pytest.raises(CodexQuotaError):
            await backend.chat_completions(REQUEST)
        assert backend.state_store.uncleared_violations() == []

    async def test_quota_with_tool_event_latches(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "quota_with_tool")
        record_pass(backend)
        with pytest.raises(CodexToolUseViolation):
            await backend.chat_completions(REQUEST)
        assert _detail(backend) == ['["command_execution"]']

    async def test_launch_failure_without_events_does_not_latch(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "launch_no_events")
        record_pass(backend)
        with pytest.raises(CodexLaunchError):
            await backend.chat_completions(REQUEST)
        assert backend.state_store.uncleared_violations() == []

    async def test_launch_failure_with_events_latches(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "launch_with_events")
        record_pass(backend)
        with pytest.raises(CodexUnverifiableRun) as exc:
            await backend.chat_completions(REQUEST)
        assert isinstance(exc.value.cause, CodexLaunchError)
        assert _detail(backend) == ['["aborted:no_terminal:launch"]']

    async def test_spawn_failure_does_not_latch(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        backend._wrap = lambda inner, workdir: [str(tmp_path / "no-such-bwrap")]  # type: ignore[method-assign]
        with pytest.raises(CodexLaunchError):
            await backend.chat_completions(REQUEST)
        assert backend.state_store.uncleared_violations() == []


def _run(rc: int | None, stderr: str = "", events: list[dict[str, Any]] | None = None) -> CodexRun:
    return CodexRun(returncode=rc, stdout="", stderr=stderr, events=events or [])


TERMINAL = {"type": "turn.completed", "usage": {}}
STARTED = {"type": "turn.started"}
TOOL = {"type": "item.started", "item": {"type": "command_execution"}}
QUOTA = "ERROR: You've hit your usage limit."


class TestRunVerdict:
    """Pure table of msg-396 D-1d-3' (first match wins)."""

    @pytest.mark.parametrize(
        ("run", "latch", "error"),
        [
            # 1. tool event -> latch, however it ended
            (_run(0, "", [TOOL, TERMINAL]), ["command_execution"], None),
            (_run(1, QUOTA, [TOOL]), ["command_execution"], None),
            (_run(-9, "", [TOOL]), ["command_execution"], None),
            # 2. signal
            (_run(-9, QUOTA, [TERMINAL]), ["aborted:signal"], None),
            (_run(-15), ["aborted:signal"], None),
            # 3. no terminal event
            (_run(0, "", [STARTED]), ["aborted:no_terminal:failed"], None),
            (_run(0, QUOTA, [STARTED]), ["aborted:no_terminal:quota"], None),  # exit 0 never exempt
            (_run(0, "bwrap: x"), ["aborted:no_terminal:launch"], None),
            (_run(1, QUOTA, [STARTED]), None, CodexQuotaError),
            (_run(1, QUOTA), None, CodexQuotaError),
            (_run(1, "bwrap: Can't find source path"), None, CodexLaunchError),
            (_run(1, "bwrap: Can't find source path", [STARTED]), ["aborted:no_terminal:launch"], None),
            (_run(1, "Not logged in", [STARTED]), ["aborted:no_terminal:auth"], None),
            (_run(1, "boom"), ["aborted:no_terminal:failed"], None),
            # 4. terminal present
            (_run(0, "", [TERMINAL]), None, None),
            (_run(1, "", [{"type": "turn.failed", "error": {"message": QUOTA}}]), None, CodexQuotaError),
            (_run(1, "Not logged in", [TERMINAL]), None, CodexAuthError),
        ],
    )
    def test_table(self, run: CodexRun, latch: list[str] | None, error: type | None) -> None:
        got_latch, got_error = run_verdict(run, classify_events(run.events))
        assert got_latch == latch
        assert (type(got_error) if got_error is not None else None) is error


# --------------------------------------------------------------------------
# Input gate
# --------------------------------------------------------------------------


class TestInputGate:
    @pytest.mark.parametrize(
        "request_body",
        [
            {**REQUEST, "tools": [{"type": "function", "function": {"name": "x"}}]},
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
            {"messages": [{"role": "tool", "tool_call_id": "a", "content": "x"}]},
            {"messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]}]},
            {"messages": []},
        ],
    )
    async def test_refused_before_any_subprocess(self, tmp_path: Path, request_body: dict, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("subprocess started")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
        backend = CodexBackend(codex_home=str(tmp_path), state_store=CodexStateStore(tmp_path / "s.db"))
        with pytest.raises(CodexUnsupportedInputError):
            await backend.chat_completions(request_body)

    def test_text_blocks_are_flattened(self) -> None:
        prompt = request_to_prompt({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]})
        assert prompt == "[user]\na\nb"


# --------------------------------------------------------------------------
# Command shape and blast radius (D-1e)
# --------------------------------------------------------------------------


class TestCommand:
    def test_exec_argv(self, tmp_path: Path) -> None:
        backend = CodexBackend(codex_home="/srv/codex", state_store=CodexStateStore(tmp_path / "s.db"), cli_overrides=["k=v"])
        assert backend._build_exec_argv("gpt-5-codex", "/tmp/w/last.txt") == [
            "codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--json",
            "-c", "k=v", "--output-last-message", "/tmp/w/last.txt", "--model", "gpt-5-codex", "-",
        ]

    def test_bwrap_layout_and_env_allowlist(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "secret-g")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-a")
        monkeypatch.setenv("LANG", "C.UTF-8")
        backend = CodexBackend(codex_home="/srv/codex", state_store=CodexStateStore(tmp_path / "s.db"))
        argv = backend._wrap(["codex", "exec"], "/tmp/lexora-codex-x")
        joined = " ".join(argv)
        assert argv[0] == "bwrap"
        assert "--tmpfs /home" in joined
        assert "--ro-bind /usr /usr" in joined
        assert "--bind /srv/codex /srv/codex" in joined
        assert "--clearenv" in argv
        setenv = {argv[i + 1]: argv[i + 2] for i, a in enumerate(argv) if a == "--setenv"}
        assert setenv == {"PATH": codex_mod.SANDBOX_PATH, "HOME": "/srv/codex", "CODEX_HOME": "/srv/codex", "LANG": "C.UTF-8"}
        assert "secret-g" not in joined and "secret-a" not in joined
        assert argv[argv.index("--") + 1:] == ["codex", "exec"]
        # The host-side bwrap process gets the allow-listed env too.
        assert "GEMINI_API_KEY" not in backend._subprocess_env()


# --------------------------------------------------------------------------
# Config schema and factory
# --------------------------------------------------------------------------

_BYPASS_RE = re.compile(r"verif|gate|skip|unsafe|bypass|trust|insecure|disable|allow_unverified", re.IGNORECASE)


class TestConfig:
    def test_no_field_can_turn_the_gate_off(self) -> None:
        # ``governance_gate_enabled`` predates this backend and is the gemini
        # data-governance gate; the factory forwards it to GeminiBackend only
        # (asserted below), so it cannot reach the codex gate.
        preexisting = {"governance_gate_enabled"}
        names = list(CodexSettings.model_fields) + [n for n in BackendSettings.model_fields if n not in preexisting]
        assert [n for n in names if _BYPASS_RE.search(n)] == []
        factory_src = (SRC / "backends" / "factory.py").read_text(encoding="utf-8")
        codex_branch = factory_src.split('settings.type == "codex"', 1)[1]
        assert "governance_gate_enabled" not in codex_branch

    def test_unknown_key_is_refused(self) -> None:
        with pytest.raises(ValueError):
            CodexSettings(codex_home="/srv/codex", skip_verification=True)  # type: ignore[call-arg]

    def test_codex_section_required_and_exclusive(self) -> None:
        with pytest.raises(ValueError):
            BackendSettings(type="codex")
        with pytest.raises(ValueError):
            BackendSettings(type="gemini", codex={"codex_home": "/srv/codex"})

    def test_timeout_defaults_to_600_unless_set(self) -> None:
        assert BackendSettings(type="codex", codex={"codex_home": "/x"}).timeout == 600.0
        assert BackendSettings(type="codex", codex={"codex_home": "/x"}, timeout=42).timeout == 42

    @pytest.mark.parametrize(
        "ro_bind",
        ["opt/codex", "/opt/../home/x", "/home", "/home/sgadmin/.local/bin", "/root/x", "/", "/srv",
         "/srv/codex/sub"],
    )
    def test_ro_binds_that_widen_the_sandbox_are_refused(self, ro_bind: str) -> None:
        # codex_home /srv/codex/home -> its parent /srv/codex is sensitive;
        # "/" and "/srv" contain it, "/srv/codex/sub" lies under it.
        with pytest.raises(ValueError):
            CodexSettings(codex_home="/srv/codex/home", ro_binds=[ro_bind])

    @pytest.mark.parametrize(
        "codex_home",
        ["../../sandbox", "sandbox/home", "./home", "/var/lib/codex/../sandbox", "/srv/./codex/home", "/srv//codex/home"],
    )
    def test_codex_home_must_be_absolute_and_normalised(self, codex_home: str) -> None:
        """#53 PR-gate (msg-433): the parent of codex_home is taken syntactically,
        so a relative or unnormalised codex_home would dodge the overlap check."""
        with pytest.raises(ValueError, match="codex_home"):
            CodexSettings(codex_home=codex_home)

    def test_msg_433_bypass_is_refused(self) -> None:
        """The exact pair from msg-433: /var/lib/codex/../sandbox resolves under
        /var, so binding /var/sandbox was the parent it failed to protect."""
        with pytest.raises(ValueError):
            CodexSettings(codex_home="/var/lib/codex/../sandbox", ro_binds=["/var/sandbox"])
        with pytest.raises(ValueError, match="overlaps"):
            CodexSettings(codex_home="/var/sandbox/home", ro_binds=["/var/sandbox"])

    def test_normalised_codex_home_with_trailing_slash_is_accepted(self) -> None:
        settings = CodexSettings(codex_home="/srv/codex/home/", ro_binds=["/opt/codex"])
        assert settings.codex_home == "/srv/codex/home/"
        with pytest.raises(ValueError, match="overlaps"):
            CodexSettings(codex_home="/srv/codex/home/", ro_binds=["/srv/codex"])

    def test_ro_bind_outside_sensitive_roots_is_accepted(self) -> None:
        assert CodexSettings(codex_home="/srv/codex/home", ro_binds=["/opt/codex"]).ro_binds == ["/opt/codex"]

    @pytest.mark.parametrize(
        "override",
        [
            'model_provider="x"',
            "model_providers.x.base_url=http://evil",
            'sandbox_mode="danger-full-access"',
            "sandbox_workspace_write.network_access=true",
            'approval_policy="never"',
            "shell_environment_policy.inherit=all",
            'profile="loose"',
            "no_equals_sign",
            "=value",
        ],
    )
    def test_forbidden_cli_overrides_are_refused(self, override: str) -> None:
        with pytest.raises(ValueError):
            CodexSettings(codex_home="/srv/codex/home", cli_overrides=[override])

    def test_other_cli_overrides_are_accepted(self) -> None:
        assert CodexSettings(codex_home="/x/h", cli_overrides=["features.foo=false"]).cli_overrides == ["features.foo=false"]

    def test_codex_settings_ignore_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CODEX_BIN", "/tmp/evil")
        assert CodexSettings(codex_home="/x").codex_bin == "codex"

    def test_factory_builds_without_cli_or_login(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("subprocess started")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
        settings = BackendSettings(
            type="codex",
            models=["gpt-5-codex"],
            codex={"codex_home": str(tmp_path / "home"), "state_db_path": str(tmp_path / "codex.db"), "max_concurrency": 2},
        )
        backend = create_backend("codex_naysayer", settings)
        assert isinstance(backend, CodexBackend)
        assert backend.timeout == 600.0
        assert backend.name == "codex_naysayer"
        assert (tmp_path / "codex.db").exists()


# --------------------------------------------------------------------------
# Source fences
# --------------------------------------------------------------------------


def _py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _code_strings(path: Path) -> list[str]:
    """String literals of a module, docstrings excluded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


class TestSourceFences:
    @staticmethod
    def _callers(attr: str) -> dict[str, list[str]]:
        callers: dict[str, list[str]] = {}
        for path in _py_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for func in ast.walk(tree):
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(func):
                    if isinstance(node, ast.Attribute) and node.attr == attr and isinstance(node.ctx, ast.Load):
                        callers.setdefault(path.relative_to(SRC).as_posix(), []).append(func.name)
        return callers

    def test_run_unverified_callers(self) -> None:
        """Only verify_codex starts an ungated run (msg-294)."""
        assert self._callers("_run_unverified") == {"tools/verify_codex.py": ["_drive"]}

    def test_execute_callers(self) -> None:
        assert self._callers("_execute") == {"backends/codex.py": ["_run_unverified", "_run_gated"]}

    def test_only_run_gated_writes_the_latch(self) -> None:
        """msg-319: the violation writer is called from _run_gated only."""
        assert self._callers("record_violation") == {"backends/codex.py": ["_run_gated"]}

    def test_run_gated_checks_the_gate_first(self) -> None:
        tree = ast.parse((SRC / "backends" / "codex.py").read_text(encoding="utf-8"))
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_gated")
        body = func.body[1:] if isinstance(func.body[0], ast.Expr) and isinstance(func.body[0].value, ast.Constant) else func.body
        # S-1'' (msg-429): the gate is codex_availability() -- evaluated here
        # unless the caller passes this request's evaluation in -- and a
        # closed result raises before anything else runs.
        assert "self.codex_availability()" in ast.unparse(body[0])
        assert ast.unparse(body[1]) == "if availability.error is not None:\n    raise availability.error"

    def test_nothing_opens_the_login_file(self) -> None:
        """No string literal outside docstrings names the CLI's credential file."""
        hits: list[str] = []
        for path in _py_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = getattr(node, "body", [])
                    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                        docstrings.add(id(body[0].value))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    if "auth.json" in node.value:
                        hits.append(f"{path.relative_to(SRC)}:{node.lineno}")
        assert hits == []



# --------------------------------------------------------------------------
# D-1e': write-ahead run log (msg-403, amended by msg-405)
# --------------------------------------------------------------------------


def _db_error(*_a: Any, **_k: Any) -> Any:
    raise sqlite3.OperationalError("database is locked")


def _failing(real: Any, times: int) -> Any:
    """Wrap ``real`` so its first ``times`` calls raise, then delegate."""
    calls = {"n": 0}

    def wrapper(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        if calls["n"] <= times:
            raise sqlite3.OperationalError("database is locked")
        return real(*a, **k)

    return wrapper


def _restart(tmp_path: Path, backend: CodexBackend, scenario: str = "ok") -> CodexBackend:
    """A new process: same DB, empty in-memory state."""
    fresh = make_backend(tmp_path, scenario)
    assert fresh.state_store.db_path == backend.state_store.db_path
    return fresh


async def _reason(backend: CodexBackend) -> str | None:
    try:
        await backend._ensure_verified()
    except CodexNotVerifiedError as exc:
        return exc.reason
    return None


class TestWriteAheadRunLog:
    async def test_clean_run_is_paired(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        await backend.chat_completions(REQUEST)
        runs = backend.state_store.runs()
        assert len(runs) == 1 and runs[0].finished_seq is not None
        assert runs[0].instance_id == codex_mod.INSTANCE_ID
        assert backend.state_store.unfinished_runs() == []
        assert await _reason(backend) is None

    async def test_run_started_unwritable_starts_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)

        async def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("codex started without run_started")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
        monkeypatch.setattr(backend.state_store, "record_run_started", _db_error)
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reason == "state_unwritable"

    async def test_latched_run_leaves_run_started_unpaired(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "tool_use")
        record_pass(backend)
        with pytest.raises(CodexToolUseViolation):
            await backend.chat_completions(REQUEST)
        assert len(backend.state_store.unfinished_runs()) == 1
        # Both conditions hold; the violation is the reported reason.
        assert await _reason(backend) == "tool_use_violation"
        vid = backend.state_store.uncleared_violations()[0].id
        backend.state_store.clear_violation(vid, "investigated")
        # The clearance is newer than the run_started, so it covers it too.
        assert backend.state_store.unfinished_runs() == []
        assert await _reason(backend) == "verification_missing"
        record_pass(backend)
        assert await _reason(backend) is None

    async def test_violation_and_finish_unwritable_closes_across_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = make_backend(tmp_path, "tool_use")
        record_pass(backend)
        monkeypatch.setattr(backend.state_store, "record_violation", _db_error)
        monkeypatch.setattr(backend.state_store, "record_run_finished", _db_error)
        with pytest.raises(CodexToolUseViolation) as exc:
            await backend.chat_completions(REQUEST)
        assert "NOT written" in str(exc.value)
        assert await _reason(backend) == "state_unwritable"  # in-process poison
        monkeypatch.undo()
        assert await _reason(backend) == "state_unwritable"  # poison survives a healthy DB
        fresh = _restart(tmp_path, backend)
        assert await _reason(fresh) == "run_unfinished"  # condition 0, after restart

    async def test_finish_unwritable_latches_without_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """msg-408 #1: no retry queue. A clean run whose run_finished cannot be
        written stays unpaired -> run_unfinished, now and after a restart."""
        backend = make_backend(tmp_path)
        record_pass(backend)
        monkeypatch.setattr(backend.state_store, "record_run_finished", _db_error)
        response = await backend.chat_completions(REQUEST)  # the clean answer is still returned
        assert response["choices"][0]["message"]["content"].startswith("REVIEW: ")
        monkeypatch.undo()
        assert await _reason(backend) == "run_unfinished"  # no self-recovery
        assert await _reason(_restart(tmp_path, backend)) == "run_unfinished"

    async def test_lock_beyond_busy_timeout_latches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A real SQLite lock held past the busy timeout -> run_unfinished."""
        monkeypatch.setattr(codex_verification, "BUSY_TIMEOUT_S", 0.2)
        backend = make_backend(tmp_path)
        record_pass(backend)
        store = backend.state_store
        real = store.record_run_finished
        holder = sqlite3.connect(store.db_path, check_same_thread=False, isolation_level=None)

        def locked_finish(*a: Any, **k: Any) -> None:
            holder.execute("BEGIN IMMEDIATE")
            try:
                real(*a, **k)
            finally:
                holder.execute("COMMIT")

        monkeypatch.setattr(store, "record_run_finished", locked_finish)
        try:
            await backend.chat_completions(REQUEST)
        finally:
            holder.close()
        monkeypatch.undo()
        assert len(store.unfinished_runs()) == 1
        assert await _reason(backend) == "run_unfinished"

    async def test_crash_mid_run_needs_clearance_and_a_new_pass(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        # What a process that died inside codex exec leaves behind.
        run_seq = backend.state_store.record_run_started("codex", "dead-instance")
        fresh = _restart(tmp_path, backend)
        assert await _reason(fresh) == "run_unfinished"
        with pytest.raises(ClearViolationError):
            fresh.state_store.clear_unfinished_run(run_seq, " ")
        fresh.state_store.clear_unfinished_run(run_seq, "deploy restarted lexora mid-review")
        assert await _reason(fresh) == "verification_missing"  # the old pass no longer counts
        record_pass(fresh)
        assert await _reason(fresh) is None
        with pytest.raises(ClearViolationError):
            fresh.state_store.clear_unfinished_run(run_seq, "again")

    async def test_own_in_flight_run_does_not_close_the_gate(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "sleep", timeout=30.0)
        record_pass(backend)
        task = asyncio.create_task(backend.chat_completions(REQUEST))
        await _until_ticking(backend)
        assert len(backend.state_store.unfinished_runs()) == 1
        assert await _reason(backend) is None  # same process: excused
        other = _restart(tmp_path, backend)
        assert await _reason(other) == "run_unfinished"  # another process: not excused
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_cancel_while_queued_pairs_the_run(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        backend._semaphore = asyncio.Semaphore(0)
        task = asyncio.create_task(backend.chat_completions(REQUEST))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.state_store.unfinished_runs() == []
        assert backend.state_store.uncleared_violations() == []

    async def test_quota_and_spawn_failure_pair_the_run(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "quota")
        record_pass(backend)
        with pytest.raises(CodexQuotaError):
            await backend.chat_completions(REQUEST)
        # PR-2: the quota failure set a hold that would refuse the next call
        # before any spawn; this test is about pairing, so lift it.
        backend._quota_hold_until = None
        backend._wrap = lambda inner, workdir: [str(tmp_path / "no-such-bwrap")]  # type: ignore[method-assign]
        with pytest.raises(CodexLaunchError):
            await backend.chat_completions(REQUEST)
        assert backend.state_store.unfinished_runs() == []
        assert await _reason(backend) is None

    async def test_lock_within_busy_timeout_does_not_close(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        holder = sqlite3.connect(backend.state_store.db_path, check_same_thread=False, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        release = threading.Timer(0.5, holder.execute, args=("COMMIT",))
        release.start()
        try:
            await backend.chat_completions(REQUEST)
        finally:
            release.join()
            holder.close()
        assert backend.state_store.unfinished_runs() == []
        assert await _reason(backend) is None

    async def test_outdated_schema_closes_the_gate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """msg-408 #5: a DB created by the PR-1 schema is reported, not migrated."""
        db = tmp_path / "codex.db"
        old = sqlite3.connect(db)
        old.execute(
            "CREATE TABLE codex_gate_log (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
            "kind TEXT NOT NULL CHECK (kind IN ('verification', 'violation', 'clearance')), "
            "backend TEXT NOT NULL, at TEXT NOT NULL)"
        )
        old.commit()
        old.close()
        backend = make_backend(tmp_path)
        assert backend.state_store.schema_current is False
        record_pass(backend)

        async def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("subprocess started on an outdated schema")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
        with pytest.raises(CodexNotVerifiedError) as exc:
            await backend.chat_completions(REQUEST)
        assert exc.value.reason == "schema_outdated"
        assert CodexStateStore(tmp_path / "fresh" / "codex.db").schema_current is True


def _row_run_started_fails(backend: CodexBackend, mp: pytest.MonkeyPatch) -> None:
    mp.setattr(backend.state_store, "record_run_started", _db_error)


def _row_violation_written(backend: CodexBackend, mp: pytest.MonkeyPatch) -> None:
    backend._scenario = "tool_use"  # type: ignore[attr-defined]


def _row_nothing_writable(backend: CodexBackend, mp: pytest.MonkeyPatch) -> None:
    backend._scenario = "tool_use"  # type: ignore[attr-defined]
    mp.setattr(backend.state_store, "record_violation", _db_error)
    mp.setattr(backend.state_store, "record_run_finished", _db_error)


def _row_finish_unwritable(backend: CodexBackend, mp: pytest.MonkeyPatch) -> None:
    mp.setattr(backend.state_store, "record_run_finished", _db_error)


class TestStateTable:
    """msg-403's table, as amended by msg-408 (no retry queue). Columns: what happened ->
    (reason in this process, reason after a restart)."""

    @pytest.mark.parametrize(
        ("setup", "here", "after_restart"),
        [
            pytest.param(_row_run_started_fails, "state_unwritable", None, id="run_started-unwritable"),
            pytest.param(_row_violation_written, "tool_use_violation", "tool_use_violation", id="violation-written"),
            pytest.param(_row_nothing_writable, "state_unwritable", "run_unfinished", id="violation-and-finish-unwritable"),
            pytest.param(_row_finish_unwritable, "run_unfinished", "run_unfinished", id="finish-unwritable"),
        ],
    )
    async def test_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: Any, here: str, after_restart: str | None
    ) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        setup(backend, monkeypatch)
        try:
            await backend.chat_completions(REQUEST)
        except (CodexToolUseViolation, CodexNotVerifiedError):
            pass
        if here == "state_unwritable" and after_restart is None:
            # The request itself was refused before any run; the gate is fine.
            monkeypatch.undo()
            assert backend.state_store.runs() == []
            assert await _reason(backend) is None
        else:
            monkeypatch.undo()
            assert await _reason(backend) == here
            assert await _reason(_restart(tmp_path, backend)) == after_restart

    async def test_row_crash_mid_run(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        record_pass(backend)
        backend.state_store.record_run_started("codex", "dead-instance")
        assert await _reason(_restart(tmp_path, backend)) == "run_unfinished"

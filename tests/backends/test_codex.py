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
import sys
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
    EventFindings,
    build_env,
    classify_events,
    classify_failure,
    parse_reset_at,
    request_to_prompt,
    usage_from_events,
)
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

    async def test_auth(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "auth")
        record_pass(backend)
        with pytest.raises(CodexAuthError):
            await backend.chat_completions(REQUEST)

    async def test_timeout_kills_the_process(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "sleep", timeout=1.0)
        record_pass(backend)
        with pytest.raises(CodexTimeout):
            await backend.chat_completions(REQUEST)

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
        first = func.body[1] if isinstance(func.body[0], ast.Expr) and isinstance(func.body[0].value, ast.Constant) else func.body[0]
        assert "_ensure_verified" in ast.unparse(first)

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

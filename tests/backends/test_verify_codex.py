"""Tests for ``python -m lexora.tools.verify_codex`` (msg-294 PR-1 + msg-315/317/319).

V-2'-control and V-2' run the real mock model server against
``fake_codex_cli.py``. The five fixtures (msg-292 + msg-315 #3):

1. pass (``contained``)
2. a CLI that executes tools under the production config (``executes``)
3. a CLI that never contacts the mock (``no_connect``)
4. a CLI that dies after receiving the tool calls (``crash``)
5. a control run in which no tool executes -- tool names do not match
   (``wrong_names``)

The real tool-disabling setting is unmeasured, so the tests register the
fake CLI's stand-in (``lexora_test.tools_disabled``) in
``TOOL_DISABLE_OVERRIDE_KEYS``. Tool names, next-turn shapes and event
shapes the fake CLI uses are ASSUMED; the real CLI's are measured after login.

V-1 needs bwrap and Linux, so its evaluation is tested on probe output here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexora.backends import codex as codex_mod
from lexora.backends.codex import CodexBackend, CodexNotVerifiedError, EventFindings
from lexora.backends.codex_verification import ClearViolationError
from lexora.tools import verify_codex as vc
from tests.backends.test_codex import REQUEST, VERSION, make_backend, record_pass

FLAG = "lexora_test.tools_disabled"


def _names(checks: list[vc.Check], ok: bool) -> set[str]:
    return {c.name for c in checks if c.ok is ok}


@pytest.fixture
def v1_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    async def ok(backend: object, extra_hidden: object = ()) -> list[vc.Check]:
        return [vc.Check("V-1/stub", True)]

    monkeypatch.setattr(vc, "run_v1", ok)


@pytest.fixture
def flag_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex_mod, "TOOL_DISABLE_OVERRIDE_KEYS", frozenset({FLAG}))


def v2_backend(tmp_path: Path, scenario: str) -> CodexBackend:
    return make_backend(tmp_path, scenario, cli_overrides=[f"{FLAG}=true"])


def _note(backend: CodexBackend) -> dict:
    record = backend.state_store.latest_verification("codex")
    assert record is not None
    return json.loads(record.note or "{}")


# --------------------------------------------------------------------------
# The five fixtures, end to end through verify()
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("v1_ok", "flag_registered")
class TestFixtures:
    async def test_1_contained_cli_passes_and_opens_the_gate(self, tmp_path: Path) -> None:
        backend = v2_backend(tmp_path, "contained")
        with pytest.raises(CodexNotVerifiedError):
            await backend.chat_completions(REQUEST)
        result, checks = await vc.verify(backend)
        assert result == "pass", [c for c in checks if not c.ok]
        assert {
            "V-2c/tool_executed_under_control",
            "V-2c/tool_execution_events_present",
            "V-2/0c_refusal_evidence",
            "V-2/2_no_tool_events",
        } <= _names(checks, True)
        backend._scenario = "ok"  # type: ignore[attr-defined]
        response = await backend.chat_completions(REQUEST)
        assert response["choices"][0]["message"]["content"].startswith("REVIEW: ")

    async def test_1b_contained_cli_passes_on_chat_wire(self, tmp_path: Path) -> None:
        result, checks = await vc.verify(v2_backend(tmp_path, "contained"), wire_api="chat")
        assert result == "pass", [c for c in checks if not c.ok]

    async def test_2_cli_that_executes_under_prod_config_fails(self, tmp_path: Path) -> None:
        backend = v2_backend(tmp_path, "executes")
        result, checks = await vc.verify(backend)
        assert result == "fail"
        failed = _names(checks, False)
        assert {"V-2/1_no_leak_to_model", "V-2/2_no_tool_events", "V-2/0c_refusal_evidence"} <= failed
        assert "tool_executed_under_prod_config" in _note(backend)["fail_reasons"]
        # msg-319: a verification failure is a record, never a runtime violation.
        assert backend.state_store.violations() == []

    async def test_3_cli_that_never_connects_fails(self, tmp_path: Path) -> None:
        backend = v2_backend(tmp_path, "no_connect")
        result, checks = await vc.verify(backend)
        assert result == "fail"
        assert {"V-2c/0a_nonce_request_received", "V-2c/tool_executed_under_control", "V-2/not_run_control_failed"} <= _names(checks, False)

    async def test_4_cli_that_crashes_after_receiving_fails(self, tmp_path: Path) -> None:
        backend = v2_backend(tmp_path, "crash")
        result, checks = await vc.verify(backend)
        assert result == "fail"
        failed = _names(checks, False)
        assert {"V-2/0_cli_terminated_normally", "V-2/0c_refusal_evidence"} <= failed
        assert {"V-2c/tool_executed_under_control", "V-2/0b_tool_calls_sent", "V-2/1_no_leak_to_model"} <= _names(checks, True)

    async def test_5_control_that_executes_nothing_fails(self, tmp_path: Path) -> None:
        """msg-315 #3: a CLI answering 'unknown tool' would pass 0(c) on its
        own; the control catches it, and the production result is not used."""
        backend = v2_backend(tmp_path, "wrong_names")
        result, checks = await vc.verify(backend)
        assert result == "fail"
        assert {"V-2c/tool_executed_under_control", "V-2c/tool_execution_events_present"} <= _names(checks, False)
        assert "V-2/not_run_control_failed" in _names(checks, False)
        assert not any(c.name.startswith("V-2/0c") for c in checks)
        assert "control_tool_not_executed" in _note(backend)["fail_reasons"]

    async def test_control_executing_tools_writes_no_violation(self, tmp_path: Path) -> None:
        """msg-318/319: the control runs tools on purpose; nothing latches."""
        backend = v2_backend(tmp_path, "contained")
        await vc.verify(backend)
        assert backend.state_store.violations() == []
        assert backend.state_store.uncleared_violations() == []

    async def test_canary_files_are_removed(self, tmp_path: Path) -> None:
        backend = v2_backend(tmp_path, "contained")
        await vc.verify(backend)
        assert list(Path(backend.codex_home).glob(".lexora-verify-canary-*")) == []

    async def test_record_stores_no_output(self, tmp_path: Path) -> None:
        backend = v2_backend(tmp_path, "executes")
        await vc.verify(backend)
        record = backend.state_store.latest_verification("codex")
        assert record is not None
        blob = repr(record.checks) + (record.note or "")
        assert "LEXORA-CANARY-" not in blob and "CODEX_HOME=" not in blob


@pytest.mark.usefixtures("v1_ok")
async def test_cannot_pass_until_the_tool_disable_key_is_registered(tmp_path: Path) -> None:
    """With TOOL_DISABLE_OVERRIDE_KEYS empty (shipped state) the control
    equals production, so a contained CLI refuses in the control too."""
    assert codex_mod.TOOL_DISABLE_OVERRIDE_KEYS == frozenset()
    result, checks = await vc.verify(v2_backend(tmp_path, "contained"))
    assert result == "fail"
    assert "V-2c/tool_executed_under_control" in _names(checks, False)


def test_control_clone_drops_only_the_disable_keys(tmp_path: Path, flag_registered: None) -> None:
    backend = make_backend(tmp_path, cli_overrides=[f"{FLAG}=true", "features.x=1"])
    clone = backend.control_clone("/tmp/dummy-home")
    assert clone.cli_overrides == ["features.x=1"]
    assert clone.codex_home == "/tmp/dummy-home"
    assert backend.cli_overrides == [f"{FLAG}=true", "features.x=1"]
    assert backend.codex_home != clone.codex_home


def test_provider_override_is_not_hashed(tmp_path: Path) -> None:
    backend = make_backend(tmp_path)
    before = backend.config_hash()
    backend._build_exec_argv("m", "/w/l", vc.provider_overrides("http://127.0.0.1:1/v1", "responses"))
    assert backend.config_hash() == before


# --------------------------------------------------------------------------
# Tool discovery and the unknown-event asymmetry
# --------------------------------------------------------------------------


class TestDiscovery:
    def test_responses_shape(self) -> None:
        raw = json.dumps({"tools": [
            {"type": "function", "name": "exec_command", "parameters": {"properties": {"cmd": {"type": "string"}}}},
            {"type": "function", "name": "view_file", "parameters": {"properties": {"file_path": {"type": "string"}}}},
            {"type": "local_shell"},
            {"type": "function", "name": "update_plan"},
        ]})
        tools = vc.discover_tools(raw)
        assert [(t.name, t.role) for t in tools] == [("exec_command", "shell"), ("view_file", "read"), ("local_shell", "shell")]
        calls = vc.calls_for(tools, "/h/canary")
        assert calls[0].arguments == {"cmd": "cat /h/canary"}
        assert calls[2].arguments == {"file_path": "/h/canary"}
        assert calls[3].kind == "local_shell_call"

    def test_chat_shape(self) -> None:
        raw = json.dumps({"tools": [{"type": "function", "function": {"name": "shell", "parameters": {"properties": {"command": {"type": "array"}}}}}]})
        tools = vc.discover_tools(raw)
        assert vc.calls_for(tools, "/c")[0].arguments == {"command": ["cat", "/c"]}

    def test_no_tools_declared(self) -> None:
        assert vc.discover_tools(json.dumps({"input": []})) == []


def test_unknown_event_is_not_refusal_evidence() -> None:
    """msg-315 #4: an unrecognised event naming the call and saying
    'declined' proves nothing for 0(c) and counts against 条件 2."""
    call = vc.ScriptedCall("call_abc", "function_call", "shell", {"command": ["env"]})
    state = vc.MockState(nonce="n", planner=lambda raw: ([], [call]))
    state.calls, state.nonce_seen, state.tool_calls_sent = [call], True, True
    state.requests = ["{\"n\": 1}"]
    unknown = {"type": "tool.declined", "call_id": "call_abc", "message": "declined"}
    events = [unknown, {"type": "turn.completed"}]
    checks = vc.evaluate_v2(state, codex_mod.classify_events(events), events, 0, "", "CANARY")
    assert {"V-2/0c_refusal_evidence", "V-2/2_no_tool_events"} <= _names(checks, False)
    # The same call refused by a recognised refusal event is evidence.
    refusal = {"type": "error", "message": "tool call_abc declined: tools disabled"}
    events = [refusal, {"type": "turn.completed"}]
    checks = vc.evaluate_v2(state, codex_mod.classify_events(events), events, 0, "", "CANARY")
    assert _names(checks, False) == set()


def test_evaluate_v2_without_calls_cannot_pass() -> None:
    state = vc.MockState(nonce="n", planner=lambda raw: ([], []))
    state.nonce_seen = state.tool_calls_sent = True
    checks = vc.evaluate_v2(state, EventFindings(), [{"type": "turn.completed"}], 0, "", "C")
    assert "V-2/0c_refusal_evidence" in _names(checks, False)


# --------------------------------------------------------------------------
# --clear-violation
# --------------------------------------------------------------------------


class TestClearViolationCli:
    def _config(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "lexora.yaml"
        cfg.write_text(
            "routing:\n"
            "  default_backend: codex\n"
            "  backends:\n"
            "    codex:\n"
            "      type: codex\n"
            "      models: [gpt-5-codex]\n"
            "      codex:\n"
            f"        codex_home: {(tmp_path / 'home').as_posix()}\n"
            f"        state_db_path: {(tmp_path / 'codex.db').as_posix()}\n",
            encoding="utf-8",
        )
        return cfg

    def test_clear_requires_reason_and_does_not_open_the_gate(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = self._config(tmp_path)
        backend = vc._load_backend(str(cfg), None)
        record_pass(backend, VERSION)
        v = backend.state_store.record_violation("codex", "[]", codex_version=VERSION, config_hash="h")
        assert vc.main(["--config", str(cfg), "--clear-violation", str(v.id)]) == 2
        assert vc.main(["--config", str(cfg), "--clear-violation", str(v.id), "--reason", " "]) == 2
        assert backend.state_store.uncleared_violations() != []
        assert vc.main(["--config", str(cfg), "--clear-violation", str(v.id), "--reason", "V-2' missed X"]) == 0
        assert backend.state_store.uncleared_violations() == []
        # The pass above predates the clearance -> still closed.
        assert backend.state_store.clearance_threshold() > backend.state_store.latest_verification("codex").seq  # type: ignore[union-attr]
        with pytest.raises(ClearViolationError):
            vc.clear_violation(backend.state_store, v.id, "again")


# --------------------------------------------------------------------------
# V-1 evaluation
# --------------------------------------------------------------------------

GOOD_PROBE = "\n".join(
    [
        "HIDDEN /home/sgadmin/.ssh",
        "HIDDEN /srv/lexora",
        "ENTRY /home/sgadmin .codex-lexora",
        "LISTED /home/sgadmin",
        "CODEX_HOME_READABLE",
        "USR_READONLY",
        "ENV PATH",
        "ENV HOME",
        "ENV CODEX_HOME",
        vc.PROBE_END,
    ]
)


class TestV1:
    def _plan(self) -> tuple[list[str], dict[str, set[str]]]:
        return vc.plan_v1(
            ["/home/sgadmin", "/home/sgadmin/.ssh", "/srv/lexora"], ["/home/sgadmin/.codex-lexora"]
        )

    def test_plan_splits_ancestors(self) -> None:
        plain, ancestors = self._plan()
        assert plain == ["/home/sgadmin/.ssh", "/srv/lexora"]
        assert ancestors == {"/home/sgadmin": {".codex-lexora"}}

    def test_hidden_path_inside_bind_is_probed_and_fails(self) -> None:
        plain, _ = vc.plan_v1(["/srv/codex"], ["/srv/codex"])
        assert plain == ["/srv/codex"]
        checks = vc.evaluate_v1(f"VISIBLE /srv/codex\n{vc.PROBE_END}", 0, plain, {}, "CANARY")
        assert "V-1/hidden:/srv/codex" in _names(checks, False)

    def test_good_probe_passes(self) -> None:
        plain, ancestors = self._plan()
        assert _names(vc.evaluate_v1(GOOD_PROBE, 0, plain, ancestors, "LEXORA_VERIFY_CANARY_X"), False) == set()

    @pytest.mark.parametrize(
        ("mutation", "failing"),
        [
            (("HIDDEN /srv/lexora", "VISIBLE /srv/lexora"), "V-1/hidden:/srv/lexora"),
            (("LISTED /home/sgadmin", "ENTRY /home/sgadmin .ssh\nLISTED /home/sgadmin"), "V-1/ancestor_only:/home/sgadmin"),
            (("ENV HOME", "ENV HOME\nENV GEMINI_API_KEY"), "V-1/api_keys_absent"),
            (("ENV HOME", "ENV HOME\nENV LEXORA_VERIFY_CANARY_X"), "V-1/canary_env_absent"),
            (("USR_READONLY", "USR_WRITABLE"), "V-1/usr_readonly"),
            (("CODEX_HOME_READABLE", "CODEX_HOME_UNREADABLE"), "V-1/codex_home_readable"),
            ((vc.PROBE_END, ""), "V-1/probe_completed"),
        ],
    )
    def test_each_leak_fails(self, mutation: tuple[str, str], failing: str) -> None:
        plain, ancestors = self._plan()
        output = GOOD_PROBE.replace(*mutation)
        assert failing in _names(vc.evaluate_v1(output, 0, plain, ancestors, "LEXORA_VERIFY_CANARY_X"), False)

    def test_probe_that_did_not_finish_cannot_pass_env_checks(self) -> None:
        plain, ancestors = self._plan()
        checks = vc.evaluate_v1("", 1, plain, ancestors, "C")
        assert {"V-1/probe_completed", "V-1/canary_env_absent", "V-1/api_keys_absent"} <= _names(checks, False)

"""Tests for ``python -m lexora.tools.verify_codex`` (msg-294 PR-1, V-1 / V-2').

V-2' runs the real mock model server against ``fake_codex_cli.py`` in the
four scenarios msg-292 names: pass / a CLI that executes tools / a CLI that
never contacts the mock / a CLI that dies after receiving the tool calls.
The tool names, next-turn shapes and event shapes the fake CLI uses are
ASSUMED; the real CLI's are measured after login.

V-1 needs bwrap and Linux, so its evaluation is tested on probe output here;
the probe itself is exercised for real by ``verify_codex`` on the host.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lexora.backends.codex import CodexNotVerifiedError
from lexora.tools import verify_codex as vc
from tests.backends.test_codex import REQUEST, make_backend


def _names(checks: list[vc.Check], ok: bool) -> set[str]:
    return {c.name for c in checks if c.ok is ok}


# --------------------------------------------------------------------------
# V-2' -- the four msg-292 fixtures
# --------------------------------------------------------------------------


class TestV2:
    async def test_contained_cli_passes(self, tmp_path: Path) -> None:
        checks = await vc.run_v2(make_backend(tmp_path, "contained"))
        assert _names(checks, False) == set()
        assert {
            "V-2/0a_nonce_request_received",
            "V-2/0b_tool_calls_sent",
            "V-2/0c_refusal_evidence",
            "V-2/1_no_leak_to_model",
            "V-2/2_no_tool_events",
        } <= _names(checks, True)

    async def test_contained_cli_passes_on_chat_wire(self, tmp_path: Path) -> None:
        checks = await vc.run_v2(make_backend(tmp_path, "contained"), wire_api="chat")
        assert _names(checks, False) == set()

    async def test_cli_that_executes_tools_fails(self, tmp_path: Path) -> None:
        checks = await vc.run_v2(make_backend(tmp_path, "executes"))
        failed = _names(checks, False)
        assert "V-2/1_no_leak_to_model" in failed
        assert "V-2/2_no_tool_events" in failed
        assert "V-2/0c_refusal_evidence" in failed

    async def test_cli_that_never_connects_fails(self, tmp_path: Path) -> None:
        """Zero requests look exactly like perfect containment to 条件 1/2;
        条件 0 is what turns it into a fail (msg-290)."""
        checks = await vc.run_v2(make_backend(tmp_path, "no_connect"))
        failed = _names(checks, False)
        assert {"V-2/0a_nonce_request_received", "V-2/0b_tool_calls_sent", "V-2/0c_refusal_evidence"} <= failed
        assert {"V-2/1_no_leak_to_model", "V-2/2_no_tool_events"} <= _names(checks, True)

    async def test_cli_that_crashes_after_receiving_fails(self, tmp_path: Path) -> None:
        """Pass-on-crash (msg-291): 条件 1/2 hold, the tool calls were sent, but
        there is no evidence of refusal and the CLI panicked."""
        checks = await vc.run_v2(make_backend(tmp_path, "crash"))
        failed = _names(checks, False)
        assert "V-2/0_cli_terminated_normally" in failed
        assert "V-2/0c_refusal_evidence" in failed
        assert {"V-2/0b_tool_calls_sent", "V-2/1_no_leak_to_model", "V-2/2_no_tool_events"} <= _names(checks, True)

    async def test_canary_file_is_removed(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path, "contained")
        await vc.run_v2(backend)
        assert list(Path(backend.codex_home).glob(".lexora-verify-canary-*")) == []

    def test_provider_override_is_not_hashed(self, tmp_path: Path) -> None:
        backend = make_backend(tmp_path)
        before = backend.config_hash()
        backend._build_exec_argv("m", "/w/l", vc.provider_overrides("http://127.0.0.1:1/v1", "responses"))
        assert backend.config_hash() == before


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


# --------------------------------------------------------------------------
# Orchestration: the record is what opens the gate
# --------------------------------------------------------------------------


def _v1_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    async def ok(backend: object, extra_hidden: object = ()) -> list[vc.Check]:
        return [vc.Check("V-1/stub", True)]

    monkeypatch.setattr(vc, "run_v1", ok)


class TestVerify:
    async def test_pass_opens_gate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _v1_ok(monkeypatch)
        backend = make_backend(tmp_path, "contained")
        with pytest.raises(CodexNotVerifiedError):
            await backend.chat_completions(REQUEST)
        result, _ = await vc.verify(backend)
        assert result == "pass"
        record = backend.state_store.latest_verification("codex")
        assert record is not None and record.result == "pass"
        assert record.config_hash == backend.config_hash()
        backend._scenario = "ok"  # type: ignore[attr-defined]
        response = await backend.chat_completions(REQUEST)
        assert response["choices"][0]["message"]["content"].startswith("REVIEW: ")

    @pytest.mark.parametrize("scenario", ["executes", "no_connect", "crash"])
    async def test_fail_keeps_gate_closed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str) -> None:
        _v1_ok(monkeypatch)
        backend = make_backend(tmp_path, scenario)
        result, _ = await vc.verify(backend)
        assert result == "fail"
        with pytest.raises(CodexNotVerifiedError):
            await backend.chat_completions(REQUEST)

    async def test_record_stores_no_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _v1_ok(monkeypatch)
        backend = make_backend(tmp_path, "executes")
        await vc.verify(backend)
        record = backend.state_store.latest_verification("codex")
        assert record is not None
        blob = repr(record.checks)
        assert "LEXORA-CANARY-" not in blob and "CODEX_HOME=" not in blob

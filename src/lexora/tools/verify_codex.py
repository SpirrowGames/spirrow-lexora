"""``python -m lexora.tools.verify_codex`` -- open the codex backend's gate.

T-naysayer-codex-backend msg-294 PR-1 (design converged over msg-277..292,
approved msg-293). Runs two model-free, deterministic checks against the
exact command the backend uses in production and records one
``codex_verification`` row whose ``result`` is ``pass`` or ``fail``:

**V-1 -- the sandbox itself.** The backend's own bwrap wrapper and env
allow-list run a fixed ``/bin/sh`` probe instead of ``codex``. Pass requires,
from the OS's point of view inside the sandbox:

* every hidden path (the operator's home, ``~/.ssh``, the Lexora working
  tree, plus ``--hidden`` extras) is ENOENT -- or, when it is an ancestor of
  a bind target, lists nothing but the path components leading to it;
* a canary env var and every ``*_API_KEY`` set in this process are absent;
* ``CODEX_HOME`` is readable (the boundary the design accepts);
* ``/usr`` is read-only;
* the probe reached its end marker (positive evidence it ran at all).

**V-2' -- the CLI refuses tool calls a model asks for.** A local mock model
server is started and ``codex exec`` is pointed at it with ``-c`` provider
overrides -- the only difference from production, and excluded from the
config hash. Whatever it is asked, the mock answers with scripted tool calls
(shell ``cat $CODEX_HOME/<canary file>``, shell ``env``, a file read).
Verdict:

* 条件 0 (prerequisite): (a) the mock received the request carrying this
  run's nonce; (b) the mock finished sending the tool calls; (c) for every
  scripted call there is positive evidence the CLI received it and did not
  run it -- a refusal event on ``--json``, or a next-turn request carrying a
  tool result for that call id with no canary / tool output in it. A CLI
  killed by a signal, a panic, or an event stream with no terminal event is
  a ``fail``: a crash must never look like a refusal (msg-291/292).
* 条件 1: no request the mock received contains the canary value or tool
  output.
* 条件 2: no tool-execution event on ``--json``.

``pass`` iff V-1 passes and 条件 0, 1, 2 all hold. No model output and no
tool output is stored; ``checks_json`` holds check names, verdicts and short
machine details only.

Wire-format details (tool names, event shapes, refusal wording) are ASSUMED
until measured after login; the go/no-go conditions in msg-294 decide what
happens if they turn out not to exist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

from lexora.backends.codex import (
    CodexBackend,
    CodexError,
    is_terminal_event,
    is_tool_execution_event,
    tool_execution_events,
)

#: Timeout for the V-2' ``codex exec`` run (the mock answers instantly).
V2_TIMEOUT_S = 120.0

PROBE_END = "LEXORA_PROBE_END"

# One ``/bin/sh`` script, arguments: <n_hidden> <hidden...> <n_ancestors> <ancestor...>
# Prints env var NAMES only, never values.
_PROBE_SCRIPT = r"""
n=$1; shift
i=0
while [ "$i" -lt "$n" ]; do
  p=$1; shift; i=$((i+1))
  if [ -e "$p" ] || [ -L "$p" ]; then echo "VISIBLE $p"; else echo "HIDDEN $p"; fi
done
n=$1; shift
i=0
while [ "$i" -lt "$n" ]; do
  p=$1; shift; i=$((i+1))
  if [ -d "$p" ]; then
    for e in $(ls -A "$p" 2>/dev/null); do echo "ENTRY $p $e"; done
    echo "LISTED $p"
  else
    echo "HIDDEN $p"
  fi
done
if [ -r "$CODEX_HOME" ] && ls "$CODEX_HOME" >/dev/null 2>&1; then echo CODEX_HOME_READABLE; else echo CODEX_HOME_UNREADABLE; fi
if touch /usr/.lexora_verify_probe 2>/dev/null; then echo USR_WRITABLE; else echo USR_READONLY; fi
env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/ENV \1/p'
echo LEXORA_PROBE_END
"""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.name, "ok": self.ok, "detail": self.detail[:200]}


# --------------------------------------------------------------------------
# V-1
# --------------------------------------------------------------------------


def _is_ancestor(parent: str, child: str) -> bool:
    p, c = PurePosixPath(parent), PurePosixPath(child)
    return p != c and p in c.parents


def plan_v1(hidden: Sequence[str], bind_targets: Sequence[str]) -> tuple[list[str], dict[str, set[str]]]:
    """Split hidden paths into plain ENOENT checks and ancestor listings.

    A hidden path that is an ancestor of a bind target necessarily exists in
    the sandbox (bwrap creates it); for it, the check is that it lists only
    the components leading to the bind targets.
    """
    plain: list[str] = []
    ancestors: dict[str, set[str]] = {}
    for path in hidden:
        allowed = {
            PurePosixPath(target).relative_to(PurePosixPath(path)).parts[0]
            for target in bind_targets
            if _is_ancestor(path, target)
        }
        if allowed:
            ancestors[path] = allowed
        else:
            # Includes a hidden path lying inside (or equal to) a bind
            # target: it is visible, so its ENOENT check fails -- as it must.
            plain.append(path)
    return plain, ancestors


def evaluate_v1(
    output: str,
    returncode: int | None,
    plain: Sequence[str],
    ancestors: dict[str, set[str]],
    canary_env: str,
) -> list[Check]:
    """Turn the probe's stdout into V-1 checks. Pure; unit-tested."""
    lines = output.splitlines()
    checks: list[Check] = []
    ended = PROBE_END in lines and returncode == 0
    checks.append(Check("V-1/probe_completed", ended, f"exit={returncode}"))
    for path in plain:
        checks.append(Check(f"V-1/hidden:{path}", f"HIDDEN {path}" in lines, "must be ENOENT"))
    for path, allowed in ancestors.items():
        entries = {ln.split(" ", 2)[2] for ln in lines if ln.startswith(f"ENTRY {path} ") and len(ln.split(" ", 2)) == 3}
        listed = f"LISTED {path}" in lines or f"HIDDEN {path}" in lines
        extra = sorted(entries - allowed)
        checks.append(
            Check(
                f"V-1/ancestor_only:{path}",
                listed and not extra,
                f"unexpected entries: {extra}" if extra else "only bind path components",
            )
        )
    checks.append(Check("V-1/codex_home_readable", "CODEX_HOME_READABLE" in lines))
    checks.append(Check("V-1/usr_readonly", "USR_READONLY" in lines))
    env_names = {ln[4:] for ln in lines if ln.startswith("ENV ")}
    checks.append(Check("V-1/canary_env_absent", ended and canary_env not in env_names))
    leaked_keys = sorted(n for n in env_names if n.upper().endswith("_API_KEY"))
    checks.append(Check("V-1/api_keys_absent", ended and not leaked_keys, ",".join(leaked_keys)))
    return checks


async def run_v1(backend: CodexBackend, extra_hidden: Sequence[str] = ()) -> list[Check]:
    """Run the probe through the backend's own wrapper (not ``codex``)."""
    canary_env = f"LEXORA_VERIFY_CANARY_{secrets.token_hex(4).upper()}"
    injected = {canary_env: secrets.token_hex(8)}
    for key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if key not in os.environ:
            injected[key] = "lexora-verify-dummy"
    home = str(PurePosixPath(Path.home().as_posix()))
    hidden = [home, f"{home}/.ssh", PurePosixPath(Path.cwd().as_posix()).as_posix(), *extra_hidden]
    bind_targets = [backend.codex_home, *backend.ro_binds]
    plain, ancestors = plan_v1(hidden, bind_targets)
    anc_paths = list(ancestors)
    workdir = tempfile.mkdtemp(prefix="lexora-codex-verify-")
    saved = {k: os.environ.get(k) for k in injected}
    os.environ.update(injected)
    try:
        argv = backend._wrap(
            ["/bin/sh", "-c", _PROBE_SCRIPT, "probe", str(len(plain)), *plain, str(len(anc_paths)), *anc_paths],
            workdir,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=backend._subprocess_env(),
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=60.0)
        except (OSError, asyncio.TimeoutError) as exc:
            return [Check("V-1/probe_completed", False, f"probe could not run: {exc}")]
        return evaluate_v1(stdout.decode("utf-8", "replace"), process.returncode, plain, ancestors, canary_env)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------
# V-2' mock model
# --------------------------------------------------------------------------


@dataclass
class ScriptedCall:
    call_id: str
    kind: str  # "function_call" | "local_shell_call"
    name: str
    arguments: dict[str, Any]


def scripted_calls(canary_path: str) -> list[ScriptedCall]:
    """The tool calls the mock returns (ASSUMED tool names; see module doc).

    Several names for the shell tool are sent because the CLI's current
    tool name is unmeasured; every one of them must be refused.
    """

    def cid() -> str:
        return f"call_{secrets.token_hex(6)}"

    return [
        ScriptedCall(cid(), "function_call", "shell", {"command": ["cat", canary_path]}),
        ScriptedCall(cid(), "function_call", "shell", {"command": ["env"]}),
        ScriptedCall(cid(), "function_call", "shell_command", {"command": f"cat {canary_path}"}),
        ScriptedCall(cid(), "local_shell_call", "local_shell", {"command": ["cat", canary_path]}),
        ScriptedCall(cid(), "function_call", "read_file", {"path": canary_path}),
    ]


@dataclass
class MockState:
    nonce: str
    calls: list[ScriptedCall]
    requests: list[str] = field(default_factory=list)
    nonce_seen: bool = False
    tool_calls_sent: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def _sse(event: str | None, data: Any) -> bytes:
    head = f"event: {event}\n" if event else ""
    body = data if isinstance(data, str) else json.dumps(data)
    return f"{head}data: {body}\n\n".encode()


def responses_tool_stream(calls: Sequence[ScriptedCall]) -> bytes:
    out = [_sse("response.created", {"type": "response.created", "response": {"id": "resp_verify_1"}})]
    for index, call in enumerate(calls):
        if call.kind == "local_shell_call":
            item: dict[str, Any] = {
                "type": "local_shell_call",
                "id": f"lsh_{index}",
                "call_id": call.call_id,
                "status": "completed",
                "action": {"type": "exec", **call.arguments},
            }
        else:
            item = {
                "type": "function_call",
                "id": f"fc_{index}",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": json.dumps(call.arguments),
            }
        out.append(
            _sse("response.output_item.done", {"type": "response.output_item.done", "output_index": index, "item": item})
        )
    out.append(
        _sse(
            "response.completed",
            {"type": "response.completed", "response": {"id": "resp_verify_1", "usage": _usage()}},
        )
    )
    return b"".join(out)


def responses_text_stream(text: str) -> bytes:
    item = {
        "type": "message",
        "id": "msg_verify_2",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    return b"".join(
        [
            _sse("response.created", {"type": "response.created", "response": {"id": "resp_verify_2"}}),
            _sse("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": item}),
            _sse("response.completed", {"type": "response.completed", "response": {"id": "resp_verify_2", "usage": _usage()}}),
        ]
    )


def chat_tool_stream(calls: Sequence[ScriptedCall]) -> bytes:
    tool_calls = [
        {
            "index": i,
            "id": c.call_id,
            "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
        }
        for i, c in enumerate(calls)
    ]
    first = {"id": "chatcmpl-verify-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": tool_calls}, "finish_reason": None}]}
    last = {"id": "chatcmpl-verify-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    return _sse(None, first) + _sse(None, last) + _sse(None, "[DONE]")


def chat_text_stream(text: str) -> bytes:
    first = {"id": "chatcmpl-verify-2", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
    last = {"id": "chatcmpl-verify-2", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    return _sse(None, first) + _sse(None, last) + _sse(None, "[DONE]")


def _usage() -> dict[str, int]:
    return {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


def _make_handler(state: MockState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            return

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            body = b'{"object":"list","data":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8", "replace") if length else ""
            chat = self.path.rstrip("/").endswith("/chat/completions")
            with state.lock:
                state.requests.append(raw)
                first_with_nonce = not state.nonce_seen and state.nonce in raw
                if first_with_nonce:
                    state.nonce_seen = True
            if first_with_nonce:
                payload = chat_tool_stream(state.calls) if chat else responses_tool_stream(state.calls)
            else:
                payload = chat_text_stream("ok") if chat else responses_text_stream("ok")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            if first_with_nonce:
                with state.lock:
                    state.tool_calls_sent = True

    return Handler


class MockModelServer:
    """Loopback mock model; lives only for one verification run."""

    def __init__(self, state: MockState) -> None:
        self.state = state
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> MockModelServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def provider_overrides(base_url: str, wire_api: str) -> list[str]:
    """``-c`` overrides pointing the CLI at the mock (excluded from the hash)."""
    return [
        'model_provider="lexora_verify"',
        'model_providers.lexora_verify.name="lexora-verify"',
        f'model_providers.lexora_verify.base_url="{base_url}"',
        f'model_providers.lexora_verify.wire_api="{wire_api}"',
    ]


# --------------------------------------------------------------------------
# V-2' evaluation
# --------------------------------------------------------------------------

#: ASSUMED wording of a CLI-side refusal event on ``--json``.
_REFUSAL_RE = re.compile(
    r"declin|reject|denied|not allowed|disabled|unsupported|unknown tool|not available|no such tool",
    re.IGNORECASE,
)


def _tool_results(raw_request: str) -> dict[str, str]:
    """call_id -> tool result text, from a next-turn request (both wires)."""
    try:
        body = json.loads(raw_request)
    except json.JSONDecodeError:
        return {}
    results: dict[str, str] = {}
    if not isinstance(body, dict):
        return results
    for item in body.get("input") or []:
        if isinstance(item, dict) and str(item.get("type", "")).endswith("_output") and item.get("call_id"):
            output = item.get("output")
            results[str(item["call_id"])] = output if isinstance(output, str) else json.dumps(output)
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("tool_call_id"):
            content = msg.get("content")
            results[str(msg["tool_call_id"])] = content if isinstance(content, str) else json.dumps(content)
    return results


def _leak_markers(canary_value: str) -> list[str]:
    # The canary itself (cat / read_file), and what ``env`` prints inside
    # the sandbox (build_env always sets these two).
    return [canary_value, "CODEX_HOME=", "PATH=/usr"]


def evaluate_v2(
    state: MockState,
    events: Sequence[dict[str, Any]],
    returncode: int | None,
    stderr: str,
    canary_value: str,
) -> list[Check]:
    """Pure V-2' verdict from what the mock saw and what the CLI printed."""
    markers = _leak_markers(canary_value)
    checks: list[Check] = []
    checks.append(Check("V-2/0a_nonce_request_received", state.nonce_seen))
    checks.append(Check("V-2/0b_tool_calls_sent", state.tool_calls_sent))

    crashed = (returncode is not None and returncode < 0) or "panicked at" in stderr
    terminal = any(is_terminal_event(e) for e in events)
    checks.append(
        Check(
            "V-2/0_cli_terminated_normally",
            not crashed and terminal,
            f"exit={returncode} terminal_event={terminal} panic={'panicked at' in stderr}",
        )
    )

    next_turn: dict[str, str] = {}
    for raw in state.requests:
        next_turn.update(_tool_results(raw))
    refusal_events = [
        json.dumps(e) for e in events if not is_tool_execution_event(e) and _REFUSAL_RE.search(json.dumps(e))
    ]
    missing: list[str] = []
    for call in state.calls:
        result = next_turn.get(call.call_id)
        by_result = result is not None and not any(m in result for m in markers)
        by_event = any(call.call_id in text for text in refusal_events)
        if not (by_result or by_event):
            missing.append(f"{call.name}:{call.call_id}")
    checks.append(
        Check(
            "V-2/0c_refusal_evidence",
            state.tool_calls_sent and not missing and not crashed and terminal,
            f"no evidence for {missing}" if missing else "every scripted call refused",
        )
    )

    leaked = [m for m in markers if any(m in raw for raw in state.requests)]
    checks.append(
        Check(
            "V-2/1_no_leak_to_model",
            not leaked,
            "leak markers present in a request to the model" if leaked else "",
        )
    )
    executions = tool_execution_events(events)
    checks.append(Check("V-2/2_no_tool_events", not executions, f"{len(executions)} tool events"))
    return checks


async def run_v2(backend: CodexBackend, wire_api: str = "responses") -> list[Check]:
    """Run ``codex exec`` against the mock model; no real model is called."""
    nonce = secrets.token_hex(12)
    canary_value = f"LEXORA-CANARY-{secrets.token_hex(16)}"
    canary_path = Path(backend.codex_home) / f".lexora-verify-canary-{nonce}"
    calls = scripted_calls(PurePosixPath(Path(backend.codex_home).as_posix(), canary_path.name).as_posix())
    state = MockState(nonce=nonce, calls=calls)
    canary_path.write_text(canary_value, encoding="utf-8")
    try:
        with MockModelServer(state) as server:
            prompt = f"lexora verify_codex nonce {nonce}. Reply with the single word ok."
            try:
                run = await backend._run_unverified(
                    prompt,
                    backend.resolve_model(None),
                    extra_overrides=provider_overrides(server.base_url, wire_api),
                    timeout=V2_TIMEOUT_S,
                )
            except CodexError as exc:
                checks = evaluate_v2(state, [], None, "", canary_value)
                checks.append(Check("V-2/cli_run", False, f"{type(exc).__name__}: {exc}"))
                return checks
        return evaluate_v2(state, run.events, run.returncode, run.stderr, canary_value)
    finally:
        canary_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Orchestration + CLI
# --------------------------------------------------------------------------


async def verify(
    backend: CodexBackend,
    wire_api: str = "responses",
    extra_hidden: Sequence[str] = (),
) -> tuple[str, list[Check]]:
    """Run V-1 and V-2', record the verdict, return (result, checks)."""
    checks: list[Check] = []
    try:
        version_before = await backend._codex_version()
    except CodexError as exc:
        checks.append(Check("cli_version", False, str(exc)))
        backend.state_store.record_verification(backend.name, "fail", "unknown", backend.config_hash(), [c.as_dict() for c in checks])
        return "fail", checks
    checks.extend(await run_v1(backend, extra_hidden))
    checks.extend(await run_v2(backend, wire_api))
    try:
        version_after = await backend._codex_version()
    except CodexError as exc:
        version_after = f"error: {exc}"
    checks.append(Check("cli_version_stable", version_before == version_after, f"{version_before!r} -> {version_after!r}"))
    result = "pass" if checks and all(c.ok for c in checks) else "fail"
    backend.state_store.record_verification(
        backend.name,
        result,
        version_before,
        backend.config_hash(),
        [c.as_dict() for c in checks],
        note=f"wire_api={wire_api}",
    )
    return result, checks


def _load_backend(config: str | None, backend_name: str | None) -> CodexBackend:
    from lexora.backends.factory import create_backend
    from lexora.config import create_settings

    settings = create_settings(Path(config) if config else None)
    codex_backends = {n: b for n, b in settings.routing.backends.items() if b.type == "codex"}
    if not codex_backends:
        raise SystemExit("no backend of type 'codex' in the configuration")
    if backend_name is None:
        if len(codex_backends) != 1:
            raise SystemExit(f"several codex backends ({sorted(codex_backends)}); pass --backend")
        backend_name = next(iter(codex_backends))
    if backend_name not in codex_backends:
        raise SystemExit(f"'{backend_name}' is not a codex backend ({sorted(codex_backends)})")
    backend = create_backend(backend_name, codex_backends[backend_name])
    assert isinstance(backend, CodexBackend)
    return backend


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lexora.tools.verify_codex", description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="Lexora YAML config (default: Lexora's usual lookup)")
    parser.add_argument("--backend", help="name of the codex backend (required if several)")
    parser.add_argument("--wire-api", choices=("responses", "chat"), default="responses")
    parser.add_argument("--hidden", action="append", default=[], help="extra path that must be invisible (repeatable)")
    args = parser.parse_args(argv)
    backend = _load_backend(args.config, args.backend)
    result, checks = asyncio.run(verify(backend, args.wire_api, args.hidden))
    for check in checks:
        print(f"{'ok  ' if check.ok else 'FAIL'} {check.name} {check.detail}")
    print(f"result: {result}")
    return 0 if result == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

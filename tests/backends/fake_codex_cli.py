"""Stand-in for ``codex exec`` in the codex backend / verify_codex tests.

Invoked as ``python fake_codex_cli.py <scenario> <codex argv without argv[0]>``
(the tests replace ``CodexBackend._wrap`` so no bwrap is involved). It
parses the same flags the real command carries (``-c`` overrides,
``--output-last-message``) and emits ``--json`` JSONL events in the ASSUMED
shape the backend reads.

Scenarios for the backend:

* ``ok``            -- final message + ``turn.completed`` with usage.
* ``tool_use``      -- reports a ``command_execution`` item (D-1c).
* ``unknown_event`` -- an item type nobody recognises, then a normal answer
                       (D-1c must treat it as an execution, msg-315 #4).
* ``quota``         -- ASSUMED usage-limit wording on stderr, exit 1.
* ``auth``          -- ASSUMED not-logged-in wording on stderr, exit 1.
* ``sleep``         -- never finishes (timeout / cancellation path); appends a
                       byte to ``$CODEX_HOME/ticks`` every 50 ms, so a test can
                       tell whether the process is still alive.
* ``exit0_no_terminal`` -- answers, exit 0, but never emits a terminal event.
* ``quota_with_tool``   -- a ``command_execution`` item, then usage-limit
                           wording, exit 1, no terminal event.
* ``launch_no_events``  -- ``bwrap:`` on stderr, exit 1, no event at all.
* ``launch_with_events``-- ``bwrap:`` on stderr, exit 1, after one event.
* ``signal``            -- kills itself with SIGKILL (POSIX only).

Scenarios for V-2'-control and V-2' (talk to the mock model named by the
provider override). The CLI declares ``shell`` and ``read_file`` in its first
request. ``DISABLE_FLAG`` among the ``-c`` overrides is the stand-in for the
real (unmeasured) tool-disabling setting; the control run drops it.

* ``contained``   -- runs its tools only when the flag is absent: control
                     executes, production refuses -> pass.
* ``executes``    -- always runs its tools -> production run fails.
* ``no_connect``  -- never contacts the mock -> control fails.
* ``crash``       -- runs tools in the control; with the flag it panics
                     right after receiving the calls -> production fails.
* ``wrong_names`` -- declares no tools and knows only a tool the mock cannot
                     guess, so every call is "unknown tool" -> control fails
                     (fixture 5, msg-315).
* ``stderr_leak``  -- like ``contained``, but copies a tool's stderr onto its
                      own stderr -> ``V-2'/tool_stderr_isolated`` fails.
* ``no_item_started`` -- like ``contained``, but reports tool runs only as
                      ``item.completed`` -> ``V-2'/tool_stderr_isolated`` fails.

A tool run emits ``item.started`` then ``item.completed`` (ASSUMED order).
The stderr probe (``sh -c <script> lexora-probe OUT NONCE ERR NONCE``) is
simulated, not run: its stdout is OUT+NONCE and its stderr ERR+NONCE; both go
into the tool result and ``aggregated_output``, never onto this process's
stderr (except under ``stderr_leak``).
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
import time
import urllib.request
from pathlib import Path

DISABLE_FLAG = "lexora_test.tools_disabled"
KNOWN_TOOLS = {"shell", "read_file", "local_shell"}
TOOL_DECLARATIONS = [
    {"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}}}},
    {"type": "function", "name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
]


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def parse_args(argv: list[str]) -> tuple[dict[str, str], str | None]:
    overrides: dict[str, str] = {}
    last_message: str | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "-c":
            key, _, value = argv[i + 1].partition("=")
            overrides[key] = value.strip('"')
            i += 2
        elif argv[i] == "--output-last-message":
            last_message = argv[i + 1]
            i += 2
        else:
            i += 1
    return overrides, last_message


def post(url: str, body: dict) -> list[dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode()
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            out.append(json.loads(line[6:]))
    return out


def tool_calls_from(events: list[dict], chat: bool) -> list[dict]:
    calls = []
    for ev in events:
        if chat:
            for choice in ev.get("choices", []):
                for tc in choice.get("delta", {}).get("tool_calls", []) or []:
                    calls.append({"call_id": tc["id"], "name": tc["function"]["name"], "args": json.loads(tc["function"]["arguments"])})
        elif ev.get("type") == "response.output_item.done":
            item = ev["item"]
            if item["type"] == "function_call":
                calls.append({"call_id": item["call_id"], "name": item["name"], "args": json.loads(item["arguments"])})
            elif item["type"] == "local_shell_call":
                calls.append({"call_id": item["call_id"], "name": "local_shell", "args": item["action"]})
    return calls


def really_run(call: dict) -> tuple[str, str]:
    """(stdout, stderr) of the tool."""
    args = call["args"]
    command = args.get("command") or args.get("cmd")
    if call["name"] == "read_file":
        return Path(args["path"]).read_text(), ""
    if isinstance(command, str):
        command = shlex.split(command)
    if command and command[0] == "cat":
        return Path(command[1]).read_text(), ""
    if command and command[0] == "env":
        return "\n".join(f"{k}={v}" for k, v in os.environ.items()), ""
    if command and command[0] == "sh" and len(command) == 8:
        return command[4] + command[5] + "\n", command[6] + command[7] + "\n"
    return "", ""


def v2(scenario: str, overrides: dict[str, str], prompt: str, last_message: str | None) -> int:
    emit({"type": "thread.started", "thread_id": "t1"})
    emit({"type": "turn.started"})
    if scenario == "no_connect":
        emit({"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": "ok"}})
        emit({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
        return 0
    disabled = DISABLE_FLAG in overrides
    base = overrides["model_providers.lexora_verify.base_url"]
    chat = overrides.get("model_providers.lexora_verify.wire_api") == "chat"
    url = base + ("/chat/completions" if chat else "/responses")
    known = {"exec_real"} if scenario == "wrong_names" else KNOWN_TOOLS
    decls = [] if scenario == "wrong_names" else TOOL_DECLARATIONS
    if chat:
        first_body = {"messages": [{"role": "user", "content": prompt}], "tools": [{"type": "function", "function": {k: v for k, v in d.items() if k != "type"}} for d in decls]}
    else:
        first_body = {"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}], "tools": decls}
    calls = tool_calls_from(post(url, first_body), chat)
    if scenario == "crash" and disabled:
        sys.stderr.write("thread 'main' panicked at codex-rs/core/src/tool.rs:1:1\n")
        sys.stderr.flush()
        os._exit(101)
    contained_like = scenario in ("contained", "stderr_leak", "no_item_started")
    run_tools = scenario in ("executes", "crash") or (contained_like and not disabled)
    results = []
    for call in calls:
        if call["name"] not in known:
            output = f"unknown tool: {call['name']}"
        elif run_tools:
            item = {"id": call["call_id"], "type": "command_execution", "command": call["name"]}
            if scenario != "no_item_started":
                emit({"type": "item.started", "item": {**item, "status": "in_progress"}})
            out, err = really_run(call)
            if scenario == "stderr_leak" and err:
                sys.stderr.write(err)
                sys.stderr.flush()
            output = out + err
            emit({"type": "item.completed", "item": {**item, "aggregated_output": output, "exit_code": 0, "status": "completed"}})
        else:
            output = "tool call rejected: tools are disabled for this session"
        results.append((call["call_id"], output))
    if chat:
        body = {"messages": [{"role": "user", "content": prompt}] + [{"role": "tool", "tool_call_id": cid, "content": out} for cid, out in results]}
    else:
        body = {"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}] + [{"type": "function_call_output", "call_id": cid, "output": out} for cid, out in results]}
    post(url, body)
    if last_message:
        Path(last_message).write_text("ok")
    emit({"type": "item.completed", "item": {"id": "i9", "type": "agent_message", "text": "ok"}})
    emit({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
    return 0


def main() -> int:
    scenario = sys.argv[1]
    overrides, last_message = parse_args(sys.argv[2:])
    prompt = sys.stdin.read()
    if scenario in ("contained", "executes", "no_connect", "crash", "wrong_names", "stderr_leak", "no_item_started"):
        return v2(scenario, overrides, prompt, last_message)
    if scenario == "sleep":
        ticks = Path(os.environ["CODEX_HOME"]) / "ticks"
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            with ticks.open("a") as fh:
                fh.write(".")
            time.sleep(0.05)
        return 0
    if scenario == "signal":
        os.kill(os.getpid(), signal.SIGKILL)
        return 0
    if scenario == "launch_no_events":
        sys.stderr.write("bwrap: Can't find source path /nope: No such file or directory\n")
        return 1
    emit({"type": "thread.started", "thread_id": "t1"})
    emit({"type": "turn.started"})
    if scenario == "ok":
        text = "REVIEW: " + prompt[:40]
        emit({"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": text}})
        emit({"type": "turn.completed", "usage": {"input_tokens": 120, "cached_input_tokens": 20, "output_tokens": 7}})
        if last_message:
            Path(last_message).write_text(text)
        return 0
    if scenario == "tool_use":
        emit({"type": "item.started", "item": {"id": "i1", "type": "command_execution", "command": "cat x", "status": "in_progress"}})
        emit({"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "leaked"}})
        emit({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
        if last_message:
            Path(last_message).write_text("leaked")
        return 0
    if scenario == "unknown_event":
        emit({"type": "item.completed", "item": {"id": "i1", "type": "mystery_capability", "status": "completed"}})
        emit({"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "fine"}})
        emit({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
        if last_message:
            Path(last_message).write_text("fine")
        return 0
    if scenario == "exit0_no_terminal":
        text = "REVIEW: truncated"
        emit({"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": text}})
        if last_message:
            Path(last_message).write_text(text)
        return 0
    if scenario == "quota_with_tool":
        emit({"type": "item.started", "item": {"id": "i1", "type": "command_execution", "command": "sleep 1", "status": "in_progress"}})
        sys.stderr.write("ERROR: You've hit your usage limit. Try again in 2 hours.\n")
        return 1
    if scenario == "launch_with_events":
        sys.stderr.write("bwrap: Can't find source path /nope: No such file or directory\n")
        return 1
    if scenario == "quota":
        sys.stderr.write("ERROR: You've hit your usage limit. Try again in 2 hours 5 minutes.\n")
        return 1
    if scenario == "auth":
        sys.stderr.write("ERROR: Not logged in. Run `codex login` to authenticate.\n")
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

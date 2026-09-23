"""Stand-in for ``codex exec`` in the codex backend / verify_codex tests.

Invoked as ``python fake_codex_cli.py <scenario> <codex argv without argv[0]>``
(the tests replace ``CodexBackend._wrap`` so no bwrap is involved). It
parses the same flags the real command carries (``-c`` overrides,
``--output-last-message``) and emits ``--json`` JSONL events in the ASSUMED
shape the backend reads.

Scenarios for the backend:

* ``ok``           -- final message + ``turn.completed`` with usage.
* ``tool_use``     -- reports a ``command_execution`` item (D-1c).
* ``quota``        -- ASSUMED usage-limit wording on stderr, exit 1.
* ``auth``         -- ASSUMED not-logged-in wording on stderr, exit 1.
* ``sleep``        -- never finishes (timeout path).

Scenarios for V-2' (talk to the mock model given by the provider override):

* ``contained``    -- receives the tool calls, runs none, reports a refusal
                      as the tool result on the next turn, finishes.
* ``executes``     -- actually runs them (reads the canary file, dumps env)
                      and sends the output back: must FAIL.
* ``no_connect``   -- ignores the override and never contacts the mock.
* ``crash``        -- receives the tool calls, then panics: must FAIL.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path


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


def really_run(call: dict) -> str:
    args = call["args"]
    command = args.get("command") or args.get("cmd")
    if call["name"] == "read_file":
        return Path(args["path"]).read_text()
    if isinstance(command, str):
        command = command.split()
    if command and command[0] == "cat":
        return Path(command[1]).read_text()
    if command and command[0] == "env":
        return "\n".join(f"{k}={v}" for k, v in os.environ.items())
    return ""


def v2(scenario: str, overrides: dict[str, str], prompt: str, last_message: str | None) -> int:
    emit({"type": "thread.started", "thread_id": "t1"})
    emit({"type": "turn.started"})
    if scenario == "no_connect":
        emit({"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": "ok"}})
        emit({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
        return 0
    base = overrides["model_providers.lexora_verify.base_url"]
    chat = overrides.get("model_providers.lexora_verify.wire_api") == "chat"
    url = base + ("/chat/completions" if chat else "/responses")
    first = post(url, {"messages": [{"role": "user", "content": prompt}]} if chat else {"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}]})
    calls = tool_calls_from(first, chat)
    if scenario == "crash":
        sys.stderr.write("thread 'main' panicked at codex-rs/core/src/tool.rs:1:1\n")
        sys.stderr.flush()
        os._exit(101)
    results = []
    for call in calls:
        if scenario == "executes":
            output = really_run(call)
            emit({"type": "item.completed", "item": {"id": call["call_id"], "type": "command_execution", "command": call["name"], "aggregated_output": "", "exit_code": 0, "status": "completed"}})
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
    if scenario in ("contained", "executes", "no_connect", "crash"):
        return v2(scenario, overrides, prompt, last_message)
    if scenario == "sleep":
        time.sleep(60)
        return 0
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
    if scenario == "quota":
        sys.stderr.write("ERROR: You've hit your usage limit. Try again in 2 hours 5 minutes.\n")
        return 1
    if scenario == "auth":
        sys.stderr.write("ERROR: Not logged in. Run `codex login` to authenticate.\n")
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

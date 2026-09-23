"""Codex CLI backend (``codex exec``) for the naysayer tier.

T-naysayer-codex-backend, msg-294 PR-1 (the design Einstein approved in
msg-293). Runs ``codex exec`` non-interactively, read-only, under bubblewrap,
and returns the CLI's final message as an OpenAI-shaped completion. Both
Lexora entry points (``/v1/chat/completions`` and ``/v1/messages``, which
``api/anthropic_compat.py`` converts to the same chat call) land here.

Safety structure, in the order a request meets it:

1. **Verification gate (D-1a')** -- ``_ensure_verified``. Every public
   method runs it before any subprocess is started. It opens only when the
   latest ``codex_verification`` record for this backend is ``pass`` and was
   taken against the same ``codex --version`` and the same config hash
   (``config_hash``) as now, and no runtime tool-use violation has been
   latched since. Otherwise ``CodexNotVerifiedError``. The gate has no
   config switch. The only ungated entry is ``_run_unverified``, whose only
   caller is ``lexora/tools/verify_codex.py`` (pinned by a test).
2. **Input gate** -- non-text content blocks, ``tools`` / ``functions`` and
   tool-role messages are refused (``CodexUnsupportedInputError``).
3. **Blast radius (D-1e)** -- ``_wrap``: bwrap with ``/home`` covered by a
   tmpfs, only the dedicated ``CODEX_HOME`` bound, ``/usr`` read-only, and an
   environment built from an allow-list (``PATH``, ``HOME``, ``CODEX_HOME``,
   locale).
4. **Runtime detection (D-1c)** -- any tool-execution or unrecognised event
   in the ``--json`` stream (``classify_events``, pure) discards the answer,
   latches a global violation in the state DB and raises
   ``CodexToolUseViolation``. Only ``_run_gated`` writes the latch. Release:
   a human clears each violation with ``verify_codex --clear-violation``,
   THEN ``verify_codex`` must pass again (msg-317).
5. **Unverifiable runs (D-1d, msg-394/396)** -- a run whose ``--json``
   stream was cut short cannot show that no tool ran, so it latches too:
   timeout (``aborted:timeout``), cancellation after the spawn
   (``aborted:cancelled``), death by signal (``aborted:signal``) and a
   stream with no terminal event (``aborted:no_terminal:<class>``). The only
   no-terminal runs that do NOT latch are a usage-window exhaustion or a
   sandbox launch failure with a non-zero exit and no tool/UNKNOWN event (a
   launch failure additionally with no event at all) -- see
   ``run_verdict``. ``_execute`` kills the process on ANY exception,
   cancellation included, so no orphan keeps running tools.

Wire-format facts marked **ASSUMED** below (event names, error wording) were
not measured: no Codex login exists yet (msg-269). They are to be re-verified
after ``codex login --device-auth`` on sg-ai-server-01, together with the four
go/no-go conditions listed in msg-294.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from lexora.backends.base import Backend, BackendError, UsageSink
from lexora.backends.codex_verification import CodexStateStore
from lexora.utils.logging import get_logger

logger = get_logger(__name__)

#: Random per process start; written into every ``run_started`` row so an
#: operator can tell which process left an unfinished run (msg-403).
INSTANCE_ID = uuid.uuid4().hex

#: Backoff ceiling for re-writing a ``run_finished`` (msg-405 D-1e'-2'.3).
FINISH_RETRY_MAX_S = 30.0
#: A ``run_finished`` still pending after this long is logged as WARNING
#: (msg-405 D-1e'-2'.7).
FINISH_PENDING_WARN_S = 60.0

#: Bumped whenever ``_build_exec_argv`` / ``_wrap`` / ``build_env`` change
#: shape, so a code change to the command closes the gate like a config
#: change does.
COMMAND_TEMPLATE_VERSION = "codex-exec-v1"

#: Environment variables copied from Lexora's environment into the sandbox,
#: beyond the three the backend sets itself (``PATH``, ``HOME``,
#: ``CODEX_HOME``). Allow-list, never deny-list (msg-294 D-1e).
LOCALE_ENV_ALLOWLIST: tuple[str, ...] = ("LANG", "LC_ALL", "LC_CTYPE", "LANGUAGE")

SANDBOX_PATH = "/usr/local/bin:/usr/bin:/bin"

#: ``-c`` override keys that disable the CLI's tools. EMPTY until measured
#: after login (msg-294 go/no-go: the tool-disabling CLI setting exists). The
#: V-2'-control run drops exactly these keys; while the set is empty the
#: control equals production, so verification cannot pass -- by design.
TOOL_DISABLE_OVERRIDE_KEYS: frozenset[str] = frozenset()


def override_key(override: str) -> str:
    """Key part of a ``key=value`` ``-c`` override, stripped and lowercased."""
    return override.split("=", 1)[0].strip().lower()


#: Output file name inside the per-request working directory.
LAST_MESSAGE_FILE = "last_message.txt"


# --------------------------------------------------------------------------
# Errors (msg-294 "失敗の分類" + gate / D-1c / input gate)
# --------------------------------------------------------------------------


class CodexError(BackendError):
    """Base class for codex backend failures.

    Deliberately plain ``BackendError`` subclasses, never
    ``BackendTimeoutError`` / ``BackendRateLimitError``: those two are in the
    retry handler's retryable set, and a retried ``codex exec`` spends the
    subscription window again (and a retried 600s timeout holds a caller for
    30 minutes).
    """


class CodexQuotaError(CodexError):
    """The subscription window is exhausted (rate limit / usage limit)."""

    def __init__(self, message: str, reset_at: datetime | None = None) -> None:
        super().__init__(message)
        self.reset_at = reset_at


class CodexAuthError(CodexError):
    """The CLI is not logged in, or its login was rejected."""


class CodexLaunchError(CodexError):
    """The CLI (or bwrap) could not be started."""


class CodexTimeout(CodexError):
    """``codex exec`` did not finish within the backend timeout."""


class CodexFailed(CodexError):
    """Any failure the classifier could not place in a narrower class."""


class CodexNotVerifiedError(CodexError):
    """The verification gate is closed; no subprocess was started.

    ``reason`` is one of ``verification_missing`` (no record, the latest
    record is not ``pass``, or the pass predates the latest clearance),
    ``verification_stale`` (CLI version or config hash changed since the
    pass), ``tool_use_violation`` (an uncleared runtime violation exists),
    ``state_unreadable`` (the state DB could not be read; fail-closed),
    ``state_unwritable`` (``run_started`` could not be written, so codex was
    not started -- or, for the rest of the process, a latch write failed),
    ``run_unfinished`` (a ``run_started`` with no ``run_finished``: human
    clearance needed) or ``run_finish_pending`` (the only unpaired runs are
    ones this process knows ended normally and is still re-writing; opens by
    itself once written). D-1e', msg-403/405.
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class CodexToolUseViolation(CodexError):
    """The CLI executed a tool (D-1c). The answer was discarded."""


class CodexUnverifiableRun(CodexToolUseViolation):
    """D-1d: the stream cannot show that no tool ran (signal death, or no
    terminal event). Latched and handled exactly like
    ``CodexToolUseViolation`` (msg-394 #4); ``cause`` keeps the class the
    failure classifier read, for the operator (``None`` for a signal death).
    """

    def __init__(self, message: str, cause: CodexError | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class CodexUnsupportedInputError(CodexError):
    """The request carries tools or non-text content (input gate)."""


# --------------------------------------------------------------------------
# Pure helpers (unit-tested without a subprocess)
# --------------------------------------------------------------------------


def config_hash(
    *,
    codex_bin: str,
    codex_home: str,
    bwrap_bin: str,
    ro_binds: Sequence[str],
    cli_overrides: Sequence[str],
    model_mapping: Mapping[str, str],
    models: Sequence[str],
) -> str:
    """sha256 over every field that changes the command line (D-1a').

    Excludes fields that do not reach the command (timeout, concurrency,
    state DB path) and the provider override ``verify_codex`` adds for V-2'
    (msg-288: "provider の上書きは...ハッシュの対象から外す").
    """
    payload = {
        "template": COMMAND_TEMPLATE_VERSION,
        "codex_bin": codex_bin,
        "codex_home": codex_home,
        "bwrap_bin": bwrap_bin,
        "ro_binds": list(ro_binds),
        "cli_overrides": list(cli_overrides),
        "model_mapping": dict(sorted(model_mapping.items())),
        "models": sorted(models),
        "env_allowlist": list(LOCALE_ENV_ALLOWLIST),
        "sandbox_path": SANDBOX_PATH,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def parse_events(stdout: str) -> list[dict[str, Any]]:
    """Parse the ``--json`` JSONL stream; non-JSON lines are skipped."""
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


#: ASSUMED ``--json`` item / message types that mean "the CLI ran a tool".
#: Two event generations are covered: ``item.*`` events carrying
#: ``item.type`` (``command_execution`` ...) and the older ``{"msg": {"type":
#: "exec_command_begin"}}`` shape. Re-verify after login.
_TOOL_ITEM_TYPES = frozenset(
    {"command_execution", "file_change", "mcp_tool_call", "web_search", "local_shell_call"}
)
_TOOL_MSG_PREFIXES = ("exec_command", "patch_apply", "mcp_tool_call", "web_search")

#: ASSUMED event types known to be neither a tool run nor a refusal. Anything
#: outside this set, the tool set and the refusal shape is UNKNOWN.
_BENIGN_EVENT_TYPES = frozenset(
    {"thread.started", "turn.started", "turn.completed", "turn.failed", "task_complete"}
)
_ITEM_EVENT_TYPES = frozenset({"item.started", "item.updated", "item.completed"})
_BENIGN_ITEM_TYPES = frozenset({"agent_message", "reasoning", "todo_list"})

#: ASSUMED wording of a CLI-side refusal of a tool call.
REFUSAL_RE = re.compile(
    r"declin|reject|denied|not allowed|disabled|unsupported|unknown tool|not available|no such tool",
    re.IGNORECASE,
)


def _event_type_and_item_type(event: Mapping[str, Any]) -> tuple[str, str]:
    etype = str(event.get("type", ""))
    item = event.get("item")
    itype = str(item.get("type", "")) if isinstance(item, dict) else ""
    msg = event.get("msg")
    if isinstance(msg, dict) and not etype:
        etype = str(msg.get("type", ""))
    return etype, itype


@dataclass(frozen=True)
class EventFindings:
    """Side-effect-free classification of a ``--json`` stream (msg-319).

    ``executions`` -- events recording a tool being run.
    ``refusals``   -- events recording the CLI refusing a tool call.
    ``unknown``    -- everything this code does not recognise.

    The asymmetry of msg-315 #4 lives here and in the two readers: an unknown
    event counts as an execution for D-1c and for V-2' 条件 2
    (``executions_or_unknown``), and never as refusal evidence for 条件 0(c)
    (only ``refusals`` is). Either way an unrecognised event makes the
    outcome fail.
    """

    executions: tuple[Mapping[str, Any], ...] = ()
    refusals: tuple[Mapping[str, Any], ...] = ()
    unknown: tuple[Mapping[str, Any], ...] = ()

    @property
    def executions_or_unknown(self) -> tuple[Mapping[str, Any], ...]:
        return self.executions + self.unknown

    def kinds(self) -> list[str]:
        """Event kinds of executions + unknown, for records (no content)."""
        out = set()
        for event in self.executions_or_unknown:
            etype, itype = _event_type_and_item_type(event)
            out.add(itype or etype or "?")
        return sorted(out)


def _is_refusal(etype: str, itype: str, event: Mapping[str, Any]) -> bool:
    if etype == "error" or itype == "error":
        return bool(REFUSAL_RE.search(json.dumps(event)))
    return False


def classify_events(events: Sequence[Mapping[str, Any]]) -> EventFindings:
    """Pure: no DB, no latch, no I/O. See ``EventFindings``."""
    executions: list[Mapping[str, Any]] = []
    refusals: list[Mapping[str, Any]] = []
    unknown: list[Mapping[str, Any]] = []
    for event in events:
        etype, itype = _event_type_and_item_type(event)
        if itype in _TOOL_ITEM_TYPES or any(etype.startswith(p) for p in _TOOL_MSG_PREFIXES):
            # A declined call reported as an item of a tool type is still
            # counted as an execution (fail-closed).
            executions.append(event)
        elif _is_refusal(etype, itype, event):
            refusals.append(event)
        elif etype in _BENIGN_EVENT_TYPES or etype == "error":
            continue
        elif etype in _ITEM_EVENT_TYPES and itype in _BENIGN_ITEM_TYPES | {"error"}:
            continue
        else:
            unknown.append(event)
    return EventFindings(tuple(executions), tuple(refusals), tuple(unknown))


#: ASSUMED terminal event types (a finished, non-truncated stream).
TERMINAL_EVENT_TYPES = frozenset({"turn.completed", "turn.failed", "task_complete"})


def is_terminal_event(event: Mapping[str, Any]) -> bool:
    etype, _ = _event_type_and_item_type(event)
    return etype in TERMINAL_EVENT_TYPES


def usage_from_events(events: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) from the last ``turn.completed``.

    ASSUMED shape: ``{"type": "turn.completed", "usage": {"input_tokens",
    "cached_input_tokens", "output_tokens"}}``, with ``input_tokens``
    already including the cached part (OpenAI convention). Assigned from the
    last such event, never summed.
    """
    prompt = completion = 0
    for event in events:
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
            prompt = int(usage.get("input_tokens", 0) or 0)
            completion = int(usage.get("output_tokens", 0) or 0)
    return prompt, completion


def last_agent_message(events: Sequence[Mapping[str, Any]]) -> str | None:
    """Text of the last ``agent_message`` item, the fallback when the
    ``--output-last-message`` file is missing (ASSUMED shape)."""
    text: str | None = None
    for event in events:
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            value = item.get("text")
            if isinstance(value, str):
                text = value
    return text


# ASSUMED wording. Order matters: quota before auth ("log in again after the
# limit resets" must stay a quota error).
_QUOTA_RE = re.compile(
    r"usage limit|rate limit|rate_limit|quota|too many requests|\b429\b", re.IGNORECASE
)
_AUTH_RE = re.compile(
    r"not logged in|codex login|unauthori[sz]ed|\b401\b|authentication|"
    r"token (?:has )?expired|invalid[_ ]token|refresh token",
    re.IGNORECASE,
)
_LAUNCH_RE = re.compile(r"^bwrap:", re.MULTILINE)
_RESET_IN_RE = re.compile(
    r"(?:try again|resets?)\s+in\s+((?:\d+\s*(?:days?|hours?|hrs?|minutes?|mins?|seconds?|secs?)[\s,and]*)+)",
    re.IGNORECASE,
)
_RESET_UNIT_RE = re.compile(r"(\d+)\s*(day|hour|hr|minute|min|second|sec)", re.IGNORECASE)
_RESET_SECONDS_RE = re.compile(r'"?resets?_in_seconds"?\s*[:=]\s*(\d+)', re.IGNORECASE)
_RESET_ISO_RE = re.compile(
    r"(?:try again at|resets? at)\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)",
    re.IGNORECASE,
)


def parse_reset_at(text: str, now: datetime | None = None) -> datetime | None:
    """Best-effort window reset time from CLI error text (ASSUMED wording).

    ``None`` when nothing parses; the caller (PR-2) then holds for a fixed
    15 minutes instead.
    """
    now = now or datetime.now(timezone.utc)
    m = _RESET_SECONDS_RE.search(text)
    if m:
        return now + timedelta(seconds=int(m.group(1)))
    m = _RESET_ISO_RE.search(text)
    if m:
        raw = m.group(1).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    m = _RESET_IN_RE.search(text)
    if m:
        seconds = 0
        for amount, unit in _RESET_UNIT_RE.findall(m.group(1)):
            u = unit.lower()
            n = int(amount)
            if u.startswith("day"):
                seconds += n * 86400
            elif u.startswith("h"):
                seconds += n * 3600
            elif u.startswith("min"):
                seconds += n * 60
            else:
                seconds += n
        if seconds > 0:
            return now + timedelta(seconds=seconds)
    return None


def _error_text(stderr: str, events: Sequence[Mapping[str, Any]]) -> str:
    parts = [stderr]
    for event in events:
        etype = event.get("type")
        if etype == "error" and isinstance(event.get("message"), str):
            parts.append(event["message"])
        elif etype == "turn.failed" and isinstance(event.get("error"), dict):
            msg = event["error"].get("message")
            if isinstance(msg, str):
                parts.append(msg)
    return "\n".join(p for p in parts if p)


def classify_failure(
    returncode: int | None,
    stderr: str,
    events: Sequence[Mapping[str, Any]],
    now: datetime | None = None,
) -> CodexError:
    """Map a failed ``codex exec`` run to one of the msg-294 classes.

    Reads stderr and the ``error`` / ``turn.failed`` events of the ``--json``
    stream. Wording is ASSUMED (fixtures in the tests say so) and is to be
    re-verified after login.
    """
    text = _error_text(stderr, events)
    snippet = text.strip()[:300]
    if returncode is not None and returncode < 0:
        return CodexFailed(f"codex exec killed by signal {-returncode}: {snippet}")
    if _LAUNCH_RE.search(stderr):
        return CodexLaunchError(f"sandbox failed to start: {snippet}")
    if _QUOTA_RE.search(text):
        return CodexQuotaError(f"codex usage window exhausted: {snippet}", parse_reset_at(text, now))
    if _AUTH_RE.search(text):
        return CodexAuthError(f"codex authentication failed: {snippet}")
    return CodexFailed(f"codex exec failed (exit {returncode}): {snippet}")


def failure_label(error: CodexError) -> str:
    """``<class>`` of ``aborted:no_terminal:<class>`` (msg-396)."""
    if isinstance(error, CodexQuotaError):
        return "quota"
    if isinstance(error, CodexLaunchError):
        return "launch"
    if isinstance(error, CodexAuthError):
        return "auth"
    return "failed"


def run_verdict(run: CodexRun, findings: EventFindings) -> tuple[list[str] | None, CodexError | None]:
    """Pure D-1c / D-1d-3' decision for a run whose process has exited.

    Returns ``(latch_detail, error)``. A non-``None`` ``latch_detail`` means
    latch a violation with that detail. Otherwise a non-``None`` ``error``
    is raised without latching. Both ``None``: the run succeeded. Rules,
    first match wins (msg-396):

    1. A tool or UNKNOWN event -> latch (``findings.kinds()``), however the
       process ended.
    2. Killed by a signal (``returncode < 0``) -> latch ``aborted:signal``.
    3. No terminal event -> latch ``aborted:no_terminal:<class>``, exit 0
       included, UNLESS all of: the classifier says quota or launch failure;
       ``returncode != 0``; no tool/UNKNOWN event (guaranteed by rule 1);
       and, for a launch failure only, no event at all.
    4. Terminal event present -> a non-zero exit is classified, not latched.

    Timeout and cancellation never reach here (``_execute`` raised them);
    ``_run_gated`` latches those itself.
    """
    if findings.executions_or_unknown:
        return findings.kinds(), None
    if run.returncode is not None and run.returncode < 0:
        return ["aborted:signal"], None
    if not any(is_terminal_event(e) for e in run.events):
        error = classify_failure(run.returncode, run.stderr, run.events)
        exempt = run.returncode != 0 and (
            isinstance(error, CodexQuotaError)
            or (isinstance(error, CodexLaunchError) and not run.events)
        )
        if exempt:
            return None, error
        return [f"aborted:no_terminal:{failure_label(error)}"], None
    if run.returncode != 0:
        return None, classify_failure(run.returncode, run.stderr, run.events)
    return None, None


def _content_to_text(content: Any) -> str:
    """Text of an OpenAI message ``content``; refuses non-text blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                btype = block.get("type") if isinstance(block, dict) else type(block).__name__
                raise CodexUnsupportedInputError(
                    f"codex backend accepts text content only; got a '{btype}' block"
                )
        return "\n".join(parts)
    raise CodexUnsupportedInputError(
        f"codex backend accepts text content only; got {type(content).__name__}"
    )


def request_to_prompt(request: Mapping[str, Any]) -> str:
    """Input gate + flattening of ``messages`` into one stdin prompt.

    Refuses ``tools`` / ``functions`` / ``tool_choice``, tool-role messages
    and assistant ``tool_calls``, and any non-text block (msg-294 入力ゲート).
    """
    for key in ("tools", "functions", "tool_choice", "function_call"):
        if request.get(key):
            raise CodexUnsupportedInputError(f"codex backend refuses '{key}' in the request")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise CodexUnsupportedInputError("codex backend requires a non-empty 'messages' list")
    sections: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise CodexUnsupportedInputError("each message must be an object")
        role = msg.get("role", "user")
        if role not in ("system", "developer", "user", "assistant"):
            raise CodexUnsupportedInputError(f"codex backend refuses '{role}' messages")
        if msg.get("tool_calls") or msg.get("function_call"):
            raise CodexUnsupportedInputError("codex backend refuses assistant tool calls")
        text = _content_to_text(msg.get("content"))
        sections.append(f"[{role}]\n{text}")
    return "\n\n".join(sections)


def build_env(codex_home: str, source_env: Mapping[str, str]) -> dict[str, str]:
    """Allow-listed environment for anything run in the sandbox (D-1e)."""
    env = {"PATH": SANDBOX_PATH, "HOME": codex_home, "CODEX_HOME": codex_home}
    for key in LOCALE_ENV_ALLOWLIST:
        if key in source_env:
            env[key] = source_env[key]
    return env


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


@dataclass
class CodexRun:
    """Outcome of one ``codex exec`` process (never persisted)."""

    returncode: int | None
    stdout: str
    stderr: str
    events: list[dict[str, Any]] = field(default_factory=list)
    last_message: str | None = None


@dataclass
class ExecProgress:
    """How far ``_execute`` got; read by ``_run_gated`` on cancellation."""

    spawned: bool = False


class CodexBackend(Backend):
    """``codex exec`` backend. See the module docstring for the safety model."""

    #: Filled from the ``turn.completed`` usage (ASSUMED shape).
    fills_usage_sink: bool = True

    def __init__(
        self,
        *,
        codex_home: str,
        state_store: CodexStateStore,
        codex_bin: str = "codex",
        bwrap_bin: str = "bwrap",
        ro_binds: Sequence[str] = (),
        cli_overrides: Sequence[str] = (),
        model_mapping: Mapping[str, str] | None = None,
        models: Sequence[str] = (),
        timeout: float = 600.0,
        max_concurrency: int = 2,
        name: str = "codex",
    ) -> None:
        self.codex_home = codex_home
        self.state_store = state_store
        self.codex_bin = codex_bin
        self.bwrap_bin = bwrap_bin
        self.ro_binds = list(ro_binds)
        self.cli_overrides = list(cli_overrides)
        self.model_mapping = dict(model_mapping or {})
        self.models = list(models)
        self.timeout = timeout
        self.name = name
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # D-1e' in-process state (msg-403/405). None of it can OPEN the gate:
        # ``_in_flight`` only excuses this process's own running runs from
        # condition 0, ``_finish_pending`` only chooses the closed reason,
        # ``_poisoned`` only closes.
        self._in_flight: set[int] = set()
        self._finish_pending: dict[int, float] = {}
        self._finish_warned: set[int] = set()
        self._finish_retry_task: asyncio.Task[None] | None = None
        self._poisoned: str | None = None

    # ---- configuration identity -------------------------------------

    def config_hash(self) -> str:
        return config_hash(
            codex_bin=self.codex_bin,
            codex_home=self.codex_home,
            bwrap_bin=self.bwrap_bin,
            ro_binds=self.ro_binds,
            cli_overrides=self.cli_overrides,
            model_mapping=self.model_mapping,
            models=self.models,
        )

    def resolve_model(self, requested: str | None) -> str:
        model = requested or (self.models[0] if self.models else "")
        return self.model_mapping.get(model, model)

    # ---- command construction ---------------------------------------

    def _build_exec_argv(
        self, model: str, last_message_path: str, extra_overrides: Sequence[str] = ()
    ) -> list[str]:
        """The msg-294 command, prompt read from stdin (``-``)."""
        argv = [self.codex_bin, "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--json"]
        for override in [*self.cli_overrides, *extra_overrides]:
            argv.extend(["-c", override])
        argv.extend(["--output-last-message", last_message_path])
        if model:
            argv.extend(["--model", model])
        argv.append("-")
        return argv

    def _wrap(self, inner_argv: Sequence[str], workdir: str) -> list[str]:
        """bwrap around ``inner_argv`` (D-1e).

        ``/home`` is a tmpfs, only ``codex_home`` and ``workdir`` are bound
        writable, ``/usr`` and ``ro_binds`` read-only. ``--clearenv`` so the
        subprocess sees only what ``build_env`` passed through ``--setenv``.
        Network is shared (the CLI must reach its provider).
        """
        argv = [
            self.bwrap_bin,
            "--die-with-parent",
            "--unshare-all",
            "--share-net",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind-try", "/bin", "/bin",
            "--ro-bind-try", "/sbin", "/sbin",
            "--ro-bind-try", "/lib", "/lib",
            "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/etc", "/etc",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--tmpfs", "/home",
        ]
        for path in self.ro_binds:
            argv.extend(["--ro-bind", path, path])
        argv.extend(["--bind", self.codex_home, self.codex_home])
        argv.extend(["--bind", workdir, workdir, "--chdir", workdir, "--clearenv"])
        for key, value in build_env(self.codex_home, os.environ).items():
            argv.extend(["--setenv", key, value])
        argv.append("--")
        argv.extend(inner_argv)
        return argv

    def _subprocess_env(self) -> dict[str, str]:
        """Environment handed to the host-side ``bwrap`` process itself."""
        return build_env(self.codex_home, os.environ)

    # ---- gate (D-1a') -----------------------------------------------

    async def _codex_version(self) -> str:
        """``codex --version``, run directly (not the model process)."""
        try:
            process = await asyncio.create_subprocess_exec(
                self.codex_bin,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._subprocess_env(),
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=15.0)
        except (OSError, asyncio.TimeoutError) as exc:
            raise CodexLaunchError(f"codex --version failed: {exc}") from exc
        if process.returncode != 0:
            raise CodexLaunchError(f"codex --version exited {process.returncode}")
        return stdout.decode("utf-8", errors="replace").strip()

    async def _ensure_verified(self) -> str:
        """Open the gate or raise ``CodexNotVerifiedError``; return the CLI version.

        The msg-324/326 conditions, in order. Record checks come first and
        need no subprocess, so a never-verified backend starts nothing.

        1. No uncleared runtime violation, in the whole store (global latch,
           msg-317 -- no release by a version or hash change).
        2. The latest ``verification`` row for this backend is ``pass`` and
           its ``seq`` is greater than ``threshold = COALESCE(MAX(seq) of
           clearances, 0)`` (msg-326). Ordering is SQLite's persisted
           ``AUTOINCREMENT`` seq, never a clock or an in-memory counter.
        3. Its config hash and ``codex --version`` equal the current ones.

        Before them (D-1e', msg-403/405): a latch write that failed in this
        process closes it for good (``state_unwritable``); pending
        ``run_finished`` writes are retried once; then condition 0 -- any
        ``run_started`` newer than the latest clearance, without a
        ``run_finished``, and not running in this process, closes the gate
        (``run_finish_pending`` if every such run is one this process is
        still re-writing, else ``run_unfinished``).

        A state DB that cannot be read closes the gate (``state_unreadable``);
        it is never treated as "no violations".
        """
        if self._poisoned is not None:
            raise CodexNotVerifiedError(self._poisoned, reason="state_unwritable")
        if self._finish_pending:
            self._flush_finish_pending()
        try:
            unfinished = self.state_store.unfinished_runs()
            uncleared = self.state_store.uncleared_violations()
            record = self.state_store.latest_verification(self.name)
            threshold = self.state_store.clearance_threshold()
        except sqlite3.Error as exc:
            raise CodexNotVerifiedError(
                f"codex state DB unreadable ({exc}); gate closed", reason="state_unreadable"
            ) from exc
        # Condition 0 and condition 1 both close; when both hold, the violation
        # is reported (it is the specific diagnosis, and its clearance also
        # covers the latched run's unpaired run_started).
        if uncleared:
            ids = ", ".join(str(v.id) for v in uncleared)
            raise CodexNotVerifiedError(
                f"codex is disabled by uncleared runtime tool-use violation(s) [{ids}]; "
                f"a human must investigate and run `verify_codex --clear-violation <id> "
                f"--reason ...`, then verify_codex must pass again",
                reason="tool_use_violation",
            )
        unpaired = [r for r in unfinished if r.seq not in self._in_flight]
        if unpaired:
            ids = ", ".join(str(r.id) for r in unpaired)
            if all(r.seq in self._finish_pending for r in unpaired):
                raise CodexNotVerifiedError(
                    f"codex run(s) [{ids}] ended normally but run_finished is not written yet; "
                    f"retrying, the gate reopens by itself once written",
                    reason="run_finish_pending",
                )
            raise CodexNotVerifiedError(
                f"codex run(s) [{ids}] started but never finished normally (crash, restart, or an "
                f"unwritable latch); a human must investigate and run `verify_codex "
                f"--clear-violation <id> --reason ...` (for a latched run, clearing its "
                f"violation is enough), then verify_codex must pass again",
                reason="run_unfinished",
            )
        if record is None or record.result != "pass":
            raise CodexNotVerifiedError(
                f"codex backend '{self.name}' has no passing verification; "
                f"run `python -m lexora.tools.verify_codex --backend {self.name}`",
                reason="verification_missing",
            )
        if not record.seq > threshold:
            raise CodexNotVerifiedError(
                f"codex backend '{self.name}': the latest pass predates the most recent "
                f"violation clearance; run verify_codex again",
                reason="verification_missing",
            )
        if record.config_hash != self.config_hash():
            raise CodexNotVerifiedError(
                f"codex backend '{self.name}' config changed since verification",
                reason="verification_stale",
            )
        version = await self._codex_version()
        if record.codex_version != version:
            raise CodexNotVerifiedError(
                f"codex CLI version changed since verification "
                f"({record.codex_version!r} -> {version!r})",
                reason="verification_stale",
            )
        return version

    # ---- execution ---------------------------------------------------

    async def _execute(
        self,
        prompt: str,
        model: str,
        extra_overrides: Sequence[str] = (),
        timeout: float | None = None,
        progress: ExecProgress | None = None,
    ) -> tuple[CodexRun, EventFindings]:
        """Start ``codex exec`` and collect its output. NO side effects.

        Launch, timeout and collection only; writes neither the state DB nor
        the latch (msg-319). Called by ``_run_gated`` and ``_run_unverified``.

        D-1d-1 (msg-394): ANY exception while the process runs --
        ``TimeoutError`` and ``CancelledError`` included -- kills and reaps
        the process before it propagates, so no orphan keeps running tools.
        A timeout becomes ``CodexTimeout``; anything else is re-raised as is.
        ``progress.spawned`` is set just before the spawn is attempted, so
        the caller can tell "cancelled while queued" (codex never ran) from
        "cancelled while running".
        """
        workdir = tempfile.mkdtemp(prefix="lexora-codex-")
        last_message_path = str(Path(workdir) / LAST_MESSAGE_FILE)
        argv = self._wrap(self._build_exec_argv(model, last_message_path, extra_overrides), workdir)
        limit = timeout if timeout is not None else self.timeout
        try:
            async with self._semaphore:
                if progress is not None:
                    progress.spawned = True
                try:
                    process = await asyncio.create_subprocess_exec(
                        *argv,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=workdir,
                        env=self._subprocess_env(),
                        limit=4 * 1024 * 1024,
                    )
                except OSError as exc:
                    raise CodexLaunchError(f"could not start codex exec: {exc}") from exc
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(
                        process.communicate(input=prompt.encode("utf-8")), timeout=limit
                    )
                except BaseException as exc:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.shield(process.wait())
                    except asyncio.CancelledError:
                        # Cancelled again while reaping: the kill is already
                        # sent; the original exception still propagates.
                        pass
                    if isinstance(exc, asyncio.TimeoutError):
                        raise CodexTimeout(f"codex exec timed out after {limit}s") from exc
                    raise
            stdout = stdout_b.decode("utf-8", errors="replace")
            last_message: str | None = None
            path = Path(last_message_path)
            if path.is_file():
                last_message = path.read_text(encoding="utf-8", errors="replace")
            events = parse_events(stdout)
            run = CodexRun(
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr_b.decode("utf-8", errors="replace"),
                events=events,
                last_message=last_message,
            )
            return run, classify_events(events)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _run_unverified(
        self,
        prompt: str,
        model: str,
        extra_overrides: Sequence[str] = (),
        timeout: float | None = None,
    ) -> tuple[CodexRun, EventFindings]:
        """``_execute`` WITHOUT the gate and WITHOUT the latch.

        The only caller is ``lexora/tools/verify_codex.py``, which cannot
        pass a gate it is there to open, and whose control run executes tools
        on purpose (msg-318/319) -- so this path must never latch. A test
        greps for the caller.
        """
        return await self._execute(prompt, model, extra_overrides, timeout)

    async def _run_gated(self, prompt: str, model: str) -> tuple[str, int, int]:
        """The production path: gate -> ``_execute`` -> D-1c/D-1d latch -> classify.

        The one place that writes a runtime violation (a test greps for it).
        Executions and UNKNOWN events both trip D-1c (msg-315 #4). A run cut
        short (timeout; cancellation after the spawn) or whose stream cannot
        be trusted (``run_verdict``) latches too (D-1d, msg-394/396); the
        timeout / cancellation is re-raised after the latch is written. A
        launch failure from the spawn (``CodexLaunchError``) and a
        cancellation before the spawn never latch: codex did not run.

        Write-ahead (D-1e', msg-403/405): ``run_started`` is written before
        ``_execute``; if that write fails, codex is not started
        (``state_unwritable``). ``run_finished`` is written only for a
        non-latching outcome (clean verdict, quota / launch / classified
        failure, spawn failure, cancellation before the spawn). A latching
        outcome writes only the violation; any other exception after the
        spawn writes nothing -- in both cases the unpaired ``run_started``
        keeps the gate closed (condition 0). If the violation write fails,
        this process's gate is also poisoned (``state_unwritable``).
        """
        version = await self._ensure_verified()
        try:
            run_seq = self.state_store.record_run_started(self.name, INSTANCE_ID)
        except sqlite3.Error as exc:
            raise CodexNotVerifiedError(
                f"codex run_started could not be written ({exc}); codex not started",
                reason="state_unwritable",
            ) from exc
        self._in_flight.add(run_seq)
        try:
            progress = ExecProgress()
            run: CodexRun | None = None
            latch: list[str] | None
            pending: BaseException | None
            try:
                run, findings = await self._execute(prompt, model, progress=progress)
            except CodexTimeout as exc:
                latch, pending = ["aborted:timeout"], exc
            except asyncio.CancelledError as exc:
                if not progress.spawned:
                    self._finish_run(run_seq)
                    raise
                latch, pending = ["aborted:cancelled"], exc
            except CodexLaunchError:
                self._finish_run(run_seq)  # the spawn itself failed: codex never ran
                raise
            except BaseException:
                if not progress.spawned:
                    self._finish_run(run_seq)
                raise  # after the spawn: run_started stays unpaired (condition 0)
            else:
                latch, pending = run_verdict(run, findings)
            if latch is not None:
                detail = json.dumps(latch)
                violation_note: str
                try:
                    violation = self.state_store.record_violation(
                        self.name, detail, codex_version=version, config_hash=self.config_hash()
                    )
                except sqlite3.Error as exc:
                    self._poisoned = (
                        f"a codex runtime violation ({detail}, run {run_seq}) could not be written "
                        f"({exc}); codex disabled in this process, and run {run_seq} stays unfinished"
                    )
                    logger.error(
                        "codex_latch_write_failed", backend=self.name, events=detail, run_seq=run_seq, error=str(exc)
                    )
                    violation_note = f"violation NOT written: {exc}; run {run_seq} left unfinished"
                else:
                    logger.error(
                        "codex_tool_use_violation", backend=self.name, events=detail, violation_id=violation.id
                    )
                    violation_note = f"violation {violation.id}"
                if pending is None:
                    message = (
                        f"codex exec ran a tool, emitted an unrecognised event, or left a stream that "
                        f"cannot show it did not ({detail}); answer discarded, codex disabled "
                        f"({violation_note})"
                    )
                    if run is not None and latch[0].startswith("aborted:"):
                        cause = None if latch == ["aborted:signal"] else classify_failure(
                            run.returncode, run.stderr, run.events
                        )
                        pending = CodexUnverifiableRun(message, cause)
                    else:
                        pending = CodexToolUseViolation(message)
                raise pending
            self._finish_run(run_seq)
            if pending is not None:
                raise pending
            assert run is not None
            text = run.last_message if run.last_message is not None else last_agent_message(run.events)
            if text is None:
                raise CodexFailed("codex exec exited 0 without a final message")
            prompt_tokens, completion_tokens = usage_from_events(run.events)
            return text, prompt_tokens, completion_tokens
        finally:
            self._in_flight.discard(run_seq)

    # ---- run_finished write + self-recovery (msg-405 D-1e'-2') ----------

    def _finish_run(self, run_seq: int) -> None:
        """Write ``run_finished`` for a NON-latching outcome. On failure the
        run is queued for re-writing: the gate stays closed
        (``run_finish_pending``) until it is written, then opens by itself.
        Latching outcomes never come here (msg-405 D-1e'-2'.5)."""
        try:
            self.state_store.record_run_finished(self.name, run_seq)
        except sqlite3.Error as exc:
            self._finish_pending.setdefault(run_seq, time.monotonic())
            logger.warning("codex_run_finish_pending", backend=self.name, run_seq=run_seq, error=str(exc))
            self._ensure_finish_retry()

    def _flush_finish_pending(self) -> None:
        """One write attempt per pending ``run_finished``."""
        now = time.monotonic()
        for run_seq, since in list(self._finish_pending.items()):
            try:
                self.state_store.record_run_finished(self.name, run_seq)
            except sqlite3.Error as exc:
                if now - since > FINISH_PENDING_WARN_S and run_seq not in self._finish_warned:
                    self._finish_warned.add(run_seq)
                    logger.warning(
                        "codex_run_finish_pending_long",
                        backend=self.name,
                        run_seq=run_seq,
                        pending_s=round(now - since),
                        error=str(exc),
                    )
            else:
                del self._finish_pending[run_seq]
                self._finish_warned.discard(run_seq)
                logger.info("codex_run_finish_recovered", backend=self.name, run_seq=run_seq)

    def _ensure_finish_retry(self) -> None:
        if self._finish_retry_task is not None and not self._finish_retry_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop: the next _ensure_verified() retries instead
        self._finish_retry_task = loop.create_task(self._finish_retry_loop())

    async def _finish_retry_loop(self) -> None:
        delay = 1.0
        while self._finish_pending:
            await asyncio.sleep(delay)
            self._flush_finish_pending()
            delay = min(delay * 2, FINISH_RETRY_MAX_S)

    def finish_pending_count(self) -> int:
        """Runs whose ``run_finished`` is still being re-written (msg-405
        D-1e'-2'.7 -- for a status surface)."""
        return len(self._finish_pending)

    def control_clone(self, codex_home: str) -> CodexBackend:
        """A copy for V-2'-control: dummy ``CODEX_HOME``, tool-disabling overrides removed.

        Everything else (bwrap layout, env allow-list, the rest of the
        overrides) is kept, so the control differs from production only in
        the setting whose effect is being measured (msg-315 #3).
        """
        clone = copy.copy(self)
        clone.codex_home = codex_home
        clone.cli_overrides = [
            o for o in self.cli_overrides if override_key(o) not in TOOL_DISABLE_OVERRIDE_KEYS
        ]
        return clone

    # ---- Backend interface -------------------------------------------

    def _response(self, content: str, model: str, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        prompt = request_to_prompt(request)
        requested = request.get("model")
        text, p, c = await self._run_gated(prompt, self.resolve_model(requested))
        return self._response(text, requested or self.resolve_model(None), p, c)

    async def chat_completions_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None = None
    ) -> AsyncIterator[bytes]:
        """Run to completion, then emit the answer as one SSE burst.

        Nothing is yielded before the run has finished and passed the D-1c
        check: streaming partial text would hand the caller output from a
        run that is then discarded for having executed a tool.
        """
        prompt = request_to_prompt(request)
        requested = request.get("model")
        text, p, c = await self._run_gated(prompt, self.resolve_model(requested))
        if usage_sink is not None:
            usage_sink.prompt_tokens, usage_sink.completion_tokens = p, c
        model = requested or self.resolve_model(None)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def chunk(delta: dict[str, Any], finish: str | None) -> bytes:
            body = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(body)}\n\n".encode()

        yield chunk({"role": "assistant", "content": ""}, None)
        if text:
            yield chunk({"content": text}, None)
        yield chunk({}, "stop")
        yield b"data: [DONE]\n\n"

    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_verified()
        raise CodexError("text completions are not supported by the codex backend")

    async def completions_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None = None
    ) -> AsyncIterator[bytes]:
        await self._ensure_verified()
        raise CodexError("text completions are not supported by the codex backend")
        yield b""  # pragma: no cover - makes this an async generator

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        raise CodexError("embeddings are not supported by the codex backend")

    async def list_models(self) -> dict[str, Any]:
        """Empty: a CLI has no catalogue; config-declared names are advertised
        by the router, as for ``claude_code``."""
        return {"object": "list", "data": []}

    async def health_check(self) -> bool:
        """Gate, then ``codex login status``. Never calls the model.

        A closed gate is reported as unhealthy (False) rather than raised, so
        ``/health`` keeps working on a host with no login or no CLI.
        """
        try:
            await self._ensure_verified()
        except CodexError:
            return False
        try:
            process = await asyncio.create_subprocess_exec(
                self.codex_bin,
                "login",
                "status",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._subprocess_env(),
            )
            await asyncio.wait_for(process.wait(), timeout=15.0)
        except (OSError, asyncio.TimeoutError):
            return False
        return process.returncode == 0

    async def close(self) -> None:
        """Stops the run_finished retry task. Anything still pending stays
        unpaired in the DB, so the gate stays closed after a restart
        (msg-405 D-1e'-2'.6)."""
        task = self._finish_retry_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

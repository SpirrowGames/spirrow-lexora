"""``python -m lexora.tools.codex_restore`` -- reopen codex after a deploy restart.

2026-11-10 (Takahito): the Lexora release ends with codex answering. Since the
naysayer tier is ``codex_only`` (no Gemini behind it), a codex left closed
after a deploy stops every naysayer review until someone notices. That is
what happened on 2026-11-08: the restart of the lexora#108 deploy killed a
codex run that had started 3 seconds earlier, the run was latched
(``aborted:signal``), and codex stayed closed for two days.

The magickit deploy runner calls this after it has restarted the service
(``scripts/post_restart.sh`` passes the restart window and the service's own
environment). It does three things, in order:

1. **Clear what the restart itself caused, and nothing else.** A *restart
   casualty* is

   * an uncleared runtime violation whose detail is exactly
     ``["aborted:signal"]`` or ``["aborted:cancelled"]`` (the two ways the
     old process's codex child dies when systemd stops the service) and that
     was written inside ``[restart_started - SLACK, restart_finished + SLACK]``;
   * an unfinished run (``run_started`` with no ``run_finished``, not yet
     cleared) that started inside ``[restart_started - max_run - SLACK,
     restart_finished]`` -- it was in flight when the old process stopped.
     ``max_run`` is the wrapper's ``codex_timeout_s``: an older run cannot
     still have been running.

   Runs started after ``restart_finished`` belong to the new process and are
   not looked at. If **any** uncleared violation or unfinished run is not a
   casualty -- a real tool-use event, a timeout latch, a run left over from
   an earlier crash -- nothing at all is cleared and the exit code is 3: that
   is the human investigation of RUNBOOK §3, and this tool does not do it.
   (Deciding everything before writing anything matters: one clearance hides
   every older unfinished run, ``CodexStateStore.unfinished_runs``.)

2. **Verify.** After a clearance, or when the service reports
   ``verification_missing`` / ``verification_stale`` (a deploy that changed
   the codex config or catalog), run the same checks as ``verify_codex`` and
   record the result. A ``fail`` is exit 4.

3. **Read the running service's status.** ``codex_disabled_reason`` must be
   ``null``. Any other reason (quota, failure hold, data controls, ...) is
   not something a deploy can fix: exit 5, with the reason printed.

``--dry-run`` prints what would be cleared and stops before writing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lexora.backends.codex import CodexBackend
from lexora.backends.codex_verification import RunRecord, ViolationRecord
from lexora.tools.verify_codex import _load_backend, verify

#: Violation details that a stopped service produces on its own codex child.
RESTART_LATCH_DETAILS: frozenset[tuple[str, ...]] = frozenset({("aborted:signal",), ("aborted:cancelled",)})
#: Clock and write slack around the restart window.
SLACK = timedelta(seconds=5)
#: Reasons a fresh ``verify_codex`` pass fixes.
VERIFY_FIXES = frozenset({"verification_missing", "verification_stale"})
DEFAULT_STATUS_URL = "http://127.0.0.1:8110/v1/naysayer/status"

EXIT_OPEN = 0
EXIT_USAGE = 2
EXIT_NEEDS_HUMAN = 3
EXIT_VERIFY_FAILED = 4
EXIT_STILL_CLOSED = 5


def _parse_time(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError(f"{text!r} has no timezone")
    return value.astimezone(timezone.utc)


@dataclass
class Triage:
    """What the state DB holds, split into restart casualties and the rest."""

    casualties: list[ViolationRecord | RunRecord] = field(default_factory=list)
    others: list[str] = field(default_factory=list)
    #: Unfinished runs started after the restart (the new process's own, in flight).
    new_process_runs: list[int] = field(default_factory=list)


def triage(
    violations: Sequence[ViolationRecord],
    unfinished: Sequence[RunRecord],
    *,
    restart_started: datetime,
    restart_finished: datetime,
    max_run: timedelta,
) -> Triage:
    """Pure: classify every uncleared violation and unfinished run."""
    out = Triage()
    for v in violations:
        try:
            detail = tuple(json.loads(v.detail))
        except (ValueError, TypeError):
            detail = ()
        at = _parse_time(v.at)
        in_window = restart_started - SLACK <= at <= restart_finished + SLACK
        if detail in RESTART_LATCH_DETAILS and in_window:
            out.casualties.append(v)
        else:
            out.others.append(f"violation {v.id} at {v.at}: {v.detail}")
    for r in unfinished:
        at = _parse_time(r.at)
        if at > restart_finished:
            out.new_process_runs.append(r.id)
        elif at >= restart_started - max_run - SLACK:
            out.casualties.append(r)
        else:
            out.others.append(f"unfinished run {r.id} started {r.at} (before the restart window)")
    return out


def _max_run(config: str | None, backend_name: str) -> timedelta:
    """The longest a run of ``backend_name`` may take: the largest
    ``codex_timeout_s`` of a fallback wrapper over it (the wrapper is what
    cuts codex off), else the caller budget."""
    from lexora.config import LEXORA_BACKEND_TIMEOUT_S, create_settings

    settings = create_settings(Path(config) if config else None)
    limits = [
        b.fallback.codex_timeout_s
        for b in settings.routing.backends.values()
        if b.type == "fallback" and b.fallback is not None and b.fallback.primary == backend_name
    ]
    return timedelta(seconds=max(limits) if limits else LEXORA_BACKEND_TIMEOUT_S)


def read_status(url: str, *, attempts: int = 12, wait_s: float = 5.0) -> dict:
    """GET the naysayer status; retried while the new process finishes starting."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                body = json.load(resp)
            if isinstance(body, dict) and isinstance(body.get("codex"), dict):
                return body
            last = ValueError(f"unexpected status body: {str(body)[:200]}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
        if i + 1 < attempts:
            time.sleep(wait_s)
    raise RuntimeError(f"status unreadable at {url}: {last}")


def restore(
    backend: CodexBackend,
    *,
    restart_started: datetime,
    restart_finished: datetime,
    max_run: timedelta,
    status_url: str,
    dry_run: bool = False,
    reason_prefix: str = "",
) -> int:
    store = backend.state_store
    t = triage(
        store.uncleared_violations(),
        store.unfinished_runs(),
        restart_started=restart_started,
        restart_finished=restart_finished,
        max_run=max_run,
    )
    window = f"{restart_started.isoformat()}..{restart_finished.isoformat()}"
    if t.new_process_runs:
        print(f"in flight in the new process (not touched): runs {t.new_process_runs}")
    if t.others:
        for line in t.others:
            print(f"NOT a restart casualty: {line}")
        print(
            f"result: needs-human -- {len(t.others)} latch item(s) outside the restart window {window} "
            f"or of another kind; nothing was cleared (deploy/RUNBOOK.md §3)"
        )
        return EXIT_NEEDS_HUMAN
    for item in t.casualties:
        kind = "violation" if isinstance(item, ViolationRecord) else "unfinished run"
        print(f"restart casualty: {kind} {item.id} at {item.at}")
    if dry_run:
        print(f"result: dry-run -- would clear {len(t.casualties)} item(s)")
        return EXIT_OPEN
    for item in t.casualties:
        why = (
            f"{reason_prefix}codex run killed by the service restart of a deploy "
            f"(restart window {window}); cleared automatically by lexora.tools.codex_restore"
        )
        if isinstance(item, ViolationRecord):
            store.clear_violation(item.id, why)
        else:
            store.clear_unfinished_run(item.id, why)
        print(f"cleared {item.id}")

    status = read_status(status_url)
    reason = status["codex"].get("codex_disabled_reason")
    if reason in VERIFY_FIXES or (t.casualties and reason is not None):
        result, checks = asyncio.run(verify(backend))
        failed = [f"{c.name} {c.detail}" for c in checks if not c.ok]
        print(f"verify_codex: {result}" + (f" (failed: {'; '.join(failed)[:600]})" if failed else ""))
        if result != "pass":
            print("result: verify-failed -- codex stays closed (deploy/RUNBOOK.md §2)")
            return EXIT_VERIFY_FAILED
        status = read_status(status_url)
        reason = status["codex"].get("codex_disabled_reason")
    if reason is not None:
        print(f"result: still-closed -- codex_disabled_reason={reason} (deploy/RUNBOOK.md §2)")
        return EXIT_STILL_CLOSED
    print(f"result: open -- codex_disabled_reason=null mode={status.get('mode')}")
    return EXIT_OPEN


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lexora.tools.codex_restore", description=__doc__.splitlines()[0])
    parser.add_argument("--restart-started-at", required=True, help="ISO-8601 time just before the restart")
    parser.add_argument("--restart-finished-at", required=True, help="ISO-8601 time just after the restart returned")
    parser.add_argument("--config", help="Lexora YAML config (default: Lexora's usual lookup)")
    parser.add_argument("--backend", help="name of the codex backend (required if several)")
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    parser.add_argument("--reason-prefix", default="", help="prepended to every clearance reason (e.g. the deploy id)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        started = _parse_time(args.restart_started_at)
        finished = _parse_time(args.restart_finished_at)
    except ValueError as exc:
        print(f"result: usage -- {exc}", file=sys.stderr)
        return EXIT_USAGE
    if finished < started:
        print("result: usage -- restart finished before it started", file=sys.stderr)
        return EXIT_USAGE
    backend = _load_backend(args.config, args.backend)
    try:
        return restore(
            backend,
            restart_started=started,
            restart_finished=finished,
            max_run=_max_run(args.config, backend.name),
            status_url=args.status_url,
            dry_run=args.dry_run,
            reason_prefix=args.reason_prefix,
        )
    except RuntimeError as exc:
        print(f"result: still-closed -- {exc}")
        return EXIT_STILL_CLOSED


if __name__ == "__main__":
    sys.exit(main())

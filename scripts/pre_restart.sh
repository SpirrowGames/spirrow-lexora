#!/bin/bash
# Wait for in-flight codex runs to finish before the deploy runner restarts
# lexora (deploy/RUNBOOK.md §1 and §11).
#
# A restart kills the codex child of a run that is still going; the run is
# latched (aborted:signal) and codex closes until it is cleared. Since the
# naysayer tier is codex_only (2026-11-10) a closed codex stops every
# naysayer review, so the release first waits for `codex.inflight_runs == 0`.
#
# This only narrows the window: a request can still arrive between the last
# read and the restart (RUNBOOK §1 note). Whatever slips through is cleared
# by scripts/post_restart.sh. So this script never fails the deploy: after
# WAIT_S it reports and exits 0.
#
# Env (optional): STATUS_URL (default http://127.0.0.1:8110/v1/naysayer/status),
# WAIT_S (default 300).
set -uo pipefail

STATUS_URL="${STATUS_URL:-http://127.0.0.1:8110/v1/naysayer/status}"
WAIT_S="${WAIT_S:-300}"

python3 - "$STATUS_URL" "$WAIT_S" <<'PY'
import json, sys, time, urllib.request

url, wait_s = sys.argv[1], float(sys.argv[2])
deadline = time.monotonic() + wait_s
last = "unread"
while True:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            inflight = json.load(resp)["codex"]["inflight_runs"]
        last = str(inflight)
        if inflight == 0:
            print("pre-restart: codex inflight_runs=0")
            sys.exit(0)
    except Exception as exc:  # the old process may be unhealthy; do not block the deploy
        last = f"status unreadable ({type(exc).__name__})"
    if time.monotonic() >= deadline:
        print(f"pre-restart: gave up waiting after {wait_s:.0f}s (inflight_runs={last}); "
              f"post_restart.sh will clear what the restart kills")
        sys.exit(0)
    time.sleep(2)
PY

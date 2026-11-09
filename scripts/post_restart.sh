#!/bin/bash
# Reopen codex after the deploy runner has restarted lexora
# (deploy/RUNBOOK.md §11).
#
# Runs lexora.tools.codex_restore with the restart window the runner passes
# (MAGICKIT_DEPLOY_RESTART_STARTED_AT / _FINISHED_AT, ISO-8601 UTC) and with
# the running service's own environment, read from /proc/<MainPID>/environ:
# verify_codex has to hash the same ro_binds the service does
# (LEXORA_CODEX_RO_BINDS comes from the unit's EnvironmentFile, RUNBOOK §9
# C-2'), and the file itself is root-only while the process is ours.
#
# Exit status is codex_restore's: 0 = codex open; anything else = the deploy
# is not complete (codex closed) and the last line says why.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT="${LEXORA_UNIT:-spirrow-lexora.service}"
: "${MAGICKIT_DEPLOY_RESTART_STARTED_AT:?the deploy runner sets this}"
: "${MAGICKIT_DEPLOY_RESTART_FINISHED_AT:?the deploy runner sets this}"

pid="$(systemctl show -p MainPID --value "$UNIT")"
if [ -z "$pid" ] || [ "$pid" = "0" ]; then
    echo "result: still-closed -- $UNIT has no main PID"
    exit 5
fi

started="$MAGICKIT_DEPLOY_RESTART_STARTED_AT"
finished="$MAGICKIT_DEPLOY_RESTART_FINISHED_AT"
prefix="${MAGICKIT_DEPLOY_REQUEST_ID:+deploy $MAGICKIT_DEPLOY_REQUEST_ID: }"

# Replace this shell's environment with the service's (one NUL-separated
# entry per variable), then run the tool from the service's venv.
mapfile -d '' service_env < "/proc/$pid/environ"
cd "$REPO_DIR"
exec env -i "${service_env[@]}" "$REPO_DIR/venv/bin/python" -m lexora.tools.codex_restore \
    --restart-started-at "$started" \
    --restart-finished-at "$finished" \
    --reason-prefix "$prefix"

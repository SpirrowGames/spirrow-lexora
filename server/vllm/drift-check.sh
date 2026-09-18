#!/bin/bash
# Report where the vLLM launch scripts on sg-ai-server-01 differ from this repo.
#
# The repository is the source of truth and the host copies are deployed by
# hand (README.md). Nothing enforces that by itself, so this is what keeps the
# arrangement checkable rather than a promise.
#
#   drift-check.sh           # print the diffs; exit 1 if any file differs
#   drift-check.sh --apply   # copy repo -> host for the files that differ
#
# Direction is one-way on purpose: a host-side edit is drift to be undone, not
# a change to be pulled back in. Edit here, open a pull request, then --apply.
#
# Unlike the docs-host sibling, --apply is NOT enough to make a change take
# effect: these scripts are the ExecStart of a systemd unit, so the running
# server keeps the arguments it was started with until it is restarted. The
# restart is left to a human because it takes the GPU down and every Lexora
# tier with it -- see README.md.

set -euo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VLLM=/home/sgadmin/services/vllm

# repo file -> deployed path. Every start-*.sh in this directory is canonical;
# the disabled ones are kept because they are the documented rollback targets
# (start-32b.sh is where qwen38.conf rolls back to).
declare -A TARGET=(
    [start-qwen38-27b.sh]="$VLLM/start-qwen38-27b.sh"
    [start-32b.sh]="$VLLM/start-32b.sh"
    [start-35b.sh]="$VLLM/start-35b.sh"
    [start-14b.sh]="$VLLM/start-14b.sh"
    [start-1.7b.sh]="$VLLM/start-1.7b.sh"
    [start-1.5b.sh]="$VLLM/start-1.5b.sh"
)

apply=0
case "${1:-}" in
    --apply) apply=1 ;;
    "") ;;
    *) echo "usage: $0 [--apply]" >&2; exit 2 ;;
esac

drift=0
applied=0
for name in "${!TARGET[@]}"; do
    src="$HERE/$name"
    dst=${TARGET[$name]}
    if [ ! -e "$dst" ]; then
        echo "MISSING  $dst"
        drift=1
    else
        # `|| true` because diff exits 1 on a difference, which set -e would
        # read as this script failing.
        delta=$(diff -u --label "repo/$name" --label "$dst" "$src" "$dst" || true)
        if [ -z "$delta" ]; then
            echo "same     $dst"
            continue
        fi
        echo "DIFFERS  $dst"
        echo "$delta"
        drift=1
    fi
    if [ "$apply" = 1 ]; then
        install -m "$(stat -c %a "$src")" "$src" "$dst"
        echo "applied  $dst"
        applied=1
    fi
done

if [ "$apply" = 1 ]; then
    if [ "$applied" = 1 ]; then
        echo
        echo "deployed. The running server still has its OLD arguments."
        echo "To pick them up (this stops the GPU and every Lexora tier for ~40s):"
        echo "    sudo systemctl restart vllm-32b.service"
        echo "Then confirm the process really carries them:"
        echo "    ps -eo args | grep -m1 api_server"
    else
        echo "nothing to deploy"
    fi
    exit 0
fi

exit $drift

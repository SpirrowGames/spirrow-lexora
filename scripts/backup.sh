#!/bin/bash
# Snapshot of lexora's accumulated state.
#
# What is here and why it is worth keeping: data/costs.db is the running
# record of what every LLM call has cost. It is not reproducible -- there
# is nowhere to re-derive it from -- and nothing else on this host backs
# it up. Everything else lexora needs comes from git or from
# /etc/spirrow-lexora.env.
#
# The deploy runner invokes this before touching anything, so it has to
# be quick, idempotent, and safe to run against a live service.
#
# Usage:
#   ./scripts/backup.sh                       # default backup dir
#   BACKUP_DIR=/path/to/dest ./scripts/backup.sh
#
# Env (optional overrides):
#   DB_PATH         (default: <repo>/data/costs.db)
#   BACKUP_DIR      (default: <repo>/backups)
#   RETENTION_DAYS  (default: 30)
set -euo pipefail

# Overridable so the script can be exercised against a tree other than the
# one it lives in; defaults to its own repo, which is what the deploy
# runner wants.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DB_PATH="${DB_PATH:-$REPO_DIR/data/costs.db}"
BACKUP_DIR="${BACKUP_DIR:-$REPO_DIR/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

if [[ ! -f "$DB_PATH" ]]; then
    # Not an error: a lexora that has never billed anything has no db yet,
    # and a deploy must not be blocked by the absence of a file whose
    # absence means "nothing to lose".
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) backup skipped: no database at $DB_PATH"
    exit 0
fi

mkdir -p "$BACKUP_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$BACKUP_DIR/costs-${TS}.db.gz"

# `sqlite3 .backup` rather than `cp`: the service is running and writing,
# and copying a sqlite file underneath a live writer can capture a torn
# page. The backup API takes a consistent snapshot without blocking
# writers for more than a moment.
TMP=$(mktemp "${TMPDIR:-/tmp}/costs-backup.XXXXXX.db")
trap 'rm -f "$TMP" "$TMP-wal" "$TMP-shm"' EXIT
sqlite3 "$DB_PATH" ".backup '$TMP'"
gzip -c "$TMP" > "$OUT"

# Owner-only: cost records say which models were used how often, which is
# more than we want world-readable.
chmod 600 "$OUT"

find "$BACKUP_DIR" -maxdepth 1 -name 'costs-*.db.gz' -mtime +"$RETENTION_DAYS" -delete

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) backup ok: $OUT ($(stat -c%s "$OUT") bytes)"

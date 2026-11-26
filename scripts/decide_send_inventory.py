"""Count what ``/v1/decide`` has sent to an external provider, by policy.

T-decide-endpoint-bypasses-ingress-redaction (Bohr msg-1154 §5, msg-1156
Obj-2). Read-only: the database is opened with ``mode=ro`` and nothing is
written. The output carries policies, row kinds, counts and timestamps --
never a hash, an answer, or any other value from a row.

Usage::

    python scripts/decide_send_inventory.py data/decisions.db

Two tables are printed (tab-separated):

``sent``
    Rows whose request reached Jev or may have: ``provider='jev'`` (an
    answer came back) or ``provider_error LIKE 'jev:%'`` (the call failed,
    a timeout included, so the request may have been sent in part).
    Shadow rows count too. ``outcome`` is ``answered`` or ``failed``.

``null_fallback``
    Answering rows where Jev failed and NullProvider answered the caller
    (``provider='null' AND provider_error LIKE 'jev:%'``): how often each
    caller already receives the fallback answer in production.

Limit: the log has no state text, so the number of requests that carried
a secret cannot be counted from it.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import TextIO

_SENT_SQL = """
SELECT policy,
       CASE WHEN shadow_of IS NULL THEN 'answering' ELSE 'shadow' END AS kind,
       CASE WHEN provider = 'jev' THEN 'answered' ELSE 'failed' END AS outcome,
       COUNT(*), MIN(timestamp), MAX(timestamp)
FROM decisions
WHERE provider = 'jev' OR provider_error LIKE 'jev:%'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
"""

_NULL_FALLBACK_SQL = """
SELECT policy, COUNT(*), MIN(timestamp), MAX(timestamp)
FROM decisions
WHERE shadow_of IS NULL AND provider = 'null' AND provider_error LIKE 'jev:%'
GROUP BY 1
ORDER BY 1
"""


def open_read_only(path: Path | str) -> sqlite3.Connection:
    """Open ``path`` read-only; a missing file is an error, not a new DB."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def inventory(conn: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    return {
        "sent": list(conn.execute(_SENT_SQL)),
        "null_fallback": list(conn.execute(_NULL_FALLBACK_SQL)),
    }


def render(result: dict[str, list[tuple[object, ...]]], out: TextIO) -> None:
    headers = {
        "sent": ("policy", "kind", "outcome", "rows", "first", "last"),
        "null_fallback": ("policy", "rows", "first", "last"),
    }
    for name, rows in result.items():
        out.write(f"# {name}\n")
        out.write("\t".join(headers[name]) + "\n")
        for row in rows:
            out.write("\t".join(str(v) for v in row) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", help="path to decisions.db")
    args = parser.parse_args(argv)
    conn = open_read_only(args.db)
    try:
        render(inventory(conn), sys.stdout)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

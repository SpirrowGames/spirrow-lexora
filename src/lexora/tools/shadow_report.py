"""``python -m lexora.tools.shadow_report`` -- summarise shadow comparisons.

T-naysayer-codex-backend msg-448 B-5. Reads the ``shadow_comparisons`` table
the ``type: fallback`` backend writes in ``mode: shadow`` (verdicts and
timings only, never text) and prints:

* compared: rows where both sides produced a parsed verdict;
* agreement rate over those rows, and the number of disagreements;
* ``unparsed`` counts per side;
* codex failures by reason (``codex_verdict = error``), and Gemini errors;
* median seconds per side.

These are the figures Bohr takes to Takahito for the shadow -> codex-primary
switch (a Tier-C billing decision, B-5). Read-only: opens the DB, never
writes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_DB = "data/costs.db"
_PARSED = {"APPROVE", "REQUEST_CHANGES", "blocking", "non_blocking"}


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    compared = [r for r in rows if r["gemini_verdict"] in _PARSED and r["codex_verdict"] in _PARSED]
    agree = sum(1 for r in compared if r["gemini_verdict"] == r["codex_verdict"])

    def median(key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return round(statistics.median(values), 3) if values else None

    return {
        "rows": len(rows),
        "compared": len(compared),
        "agreement_rate": round(agree / len(compared), 4) if compared else None,
        "disagreements": len(compared) - agree,
        "unparsed": {
            "gemini": sum(1 for r in rows if r["gemini_verdict"] == "unparsed"),
            "codex": sum(1 for r in rows if r["codex_verdict"] == "unparsed"),
        },
        "codex_failures": dict(Counter(r["codex_reason"] or "unknown" for r in rows if r["codex_verdict"] == "error")),
        "gemini_errors": sum(1 for r in rows if r["gemini_verdict"] == "error"),
        "median_seconds": {"gemini": median("gemini_seconds"), "codex": median("codex_seconds")},
    }


def load_rows(db_path: Path, since: str | None = None) -> list[dict[str, Any]]:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        sql = "SELECT * FROM shadow_comparisons"
        params: tuple[Any, ...] = ()
        if since:
            sql += " WHERE timestamp >= ?"
            params = (since,)
        return [dict(r) for r in conn.execute(sql + " ORDER BY id", params)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lexora.tools.shadow_report", description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=DEFAULT_DB, help=f"cost DB path (default {DEFAULT_DB})")
    parser.add_argument("--since", help="only rows at or after this ISO timestamp (UTC)")
    args = parser.parse_args(argv)
    db = Path(args.db)
    if not db.is_file():
        parser.error(f"no cost DB at {db}")
    try:
        rows = load_rows(db, args.since)
    except sqlite3.OperationalError as exc:
        parser.error(f"cannot read shadow_comparisons from {db}: {exc}")
    print(json.dumps(summarise(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

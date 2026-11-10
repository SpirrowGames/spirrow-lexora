"""Delete decisions by ``decision_id`` from Lexora's decision log.

Usage::

    python tools/delete_decisions.py ids.txt [--dry-run] [--db data/decisions.db]

``ids.txt`` holds one ``decision_id`` per line -- the same shape that the
caller-side ``--export-decision-ids`` writes. Blank lines and surrounding
whitespace are ignored; a repeated ID counts once.

Scope (spirrow-mindwire T-t22-delete-decisions, Bohr msg-1049 / msg-1051):

* Only rows whose ``decision_id`` is in the list are deleted. Nothing is
  filtered by ``policy`` or by product.
* **Every ID in the list is deleted, whatever its ``policy`` or product.**
  The ``decisions`` table has no product column, and ``policy`` is a
  call-site label, not a product boundary. The only thing protecting
  another product's decisions is the correctness of the input list.
* The per-``policy`` counts printed are **reference information only**.
  An ID from another product that uses the same ``policy`` is
  indistinguishable in them, so they do not guarantee that a mix-up is
  detected. To inspect what will be deleted *before* deleting, run with
  ``--dry-run`` first as a separate run; the counts printed by a real run
  are a record made after the fact.
* Shadow rows (``shadow_of`` = a listed ID) are **not** deleted. Their
  number is reported as ``shadow_remaining`` (rows whose ``shadow_of`` is
  a listed ID and whose own ``decision_id`` is not listed). Whether shadow
  rows belong to a withdrawal is an open decision tracked in
  spirrow-tutorail ``T-trial-readiness``; until it is made, deleting with
  this tool alone does not remove every row derived from a decision.
* Re-running with the same list is safe: already-deleted IDs are counted
  as ``not_found`` and nothing else happens.
* ``--dry-run`` counts the same numbers and writes nothing.
* No retention period is added.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DB = "data/decisions.db"

#: Stay well under SQLite's bound-parameter limit (999 on old builds).
_CHUNK = 500


@dataclass(frozen=True)
class Result:
    """Counts produced by :func:`delete_decisions` (identical for dry-run)."""

    requested: int
    deleted: int
    not_found: int
    shadow_remaining: int
    policy_counts: dict[str, int] = field(default_factory=dict)
    dry_run: bool = False


def parse_ids(lines: Iterable[str]) -> list[str]:
    """Strip whitespace, drop blank lines, de-duplicate keeping first order."""
    seen: dict[str, None] = {}
    for line in lines:
        value = line.strip()
        if value:
            seen.setdefault(value, None)
    return list(seen)


def _chunks(ids: Sequence[str]) -> Iterator[Sequence[str]]:
    for start in range(0, len(ids), _CHUNK):
        yield ids[start : start + _CHUNK]


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


def delete_decisions(
    conn: sqlite3.Connection, ids: Iterable[str], dry_run: bool = False
) -> Result:
    """Delete the rows whose ``decision_id`` is in ``ids``.

    All counting and deleting happens in one transaction. Under
    ``dry_run`` the transaction is rolled back, so nothing is written.
    """
    unique = parse_ids(ids)
    id_set = set(unique)
    previous_isolation = conn.isolation_level
    conn.isolation_level = None  # manage the transaction explicitly
    try:
        conn.execute("BEGIN IMMEDIATE" if not dry_run else "BEGIN")
        try:
            found: set[str] = set()
            policy_counts: dict[str, int] = {}
            shadow_rows: set[str] = set()
            for chunk in _chunks(unique):
                ph = _placeholders(len(chunk))
                for decision_id, policy in conn.execute(
                    f"SELECT decision_id, policy FROM decisions "
                    f"WHERE decision_id IN ({ph})",
                    tuple(chunk),
                ):
                    found.add(decision_id)
                    policy_counts[policy] = policy_counts.get(policy, 0) + 1
                for (shadow_id,) in conn.execute(
                    f"SELECT decision_id FROM decisions WHERE shadow_of IN ({ph})",
                    tuple(chunk),
                ):
                    if shadow_id not in id_set:
                        shadow_rows.add(shadow_id)
            deleted = 0
            if not dry_run:
                for chunk in _chunks(unique):
                    cur = conn.execute(
                        f"DELETE FROM decisions WHERE decision_id IN "
                        f"({_placeholders(len(chunk))})",
                        tuple(chunk),
                    )
                    deleted += cur.rowcount
                if deleted != len(found):  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"deleted {deleted} rows but counted {len(found)}"
                    )
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("ROLLBACK" if dry_run else "COMMIT")
    finally:
        conn.isolation_level = previous_isolation
    return Result(
        requested=len(unique),
        deleted=len(found),
        not_found=len(unique) - len(found),
        shadow_remaining=len(shadow_rows),
        policy_counts=dict(sorted(policy_counts.items())),
        dry_run=dry_run,
    )


def format_result(result: Result) -> str:
    verb = "would delete" if result.dry_run else "deleted"
    lines = [
        f"mode: {'dry-run (nothing written)' if result.dry_run else 'delete'}",
        f"requested: {result.requested}",
        f"{verb}: {result.deleted}",
        f"not_found: {result.not_found}",
        f"shadow_remaining (not deleted, reference): {result.shadow_remaining}",
        "by policy (reference only; does not detect other-product IDs):",
    ]
    if result.policy_counts:
        lines += [f"  {p}: {n}" for p, n in result.policy_counts.items()]
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="delete_decisions.py",
        description=(
            "Delete Lexora decisions by decision_id. Every listed ID is "
            "deleted regardless of policy or product: the input list is the "
            "only safety boundary. Per-policy counts are reference only and "
            "do not detect other-product IDs; run --dry-run separately first "
            "to inspect. Shadow rows are not deleted (counted only)."
        ),
    )
    parser.add_argument("ids_file", help="file with one decision_id per line")
    parser.add_argument(
        "--dry-run", action="store_true", help="count only; write nothing"
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB, help=f"decision log path (default {DEFAULT_DB})"
    )
    args = parser.parse_args(argv)

    db = Path(args.db)
    if not db.is_file():
        # Do not let sqlite3.connect create an empty DB at a mistyped path.
        print(f"error: decision log not found: {db}", file=sys.stderr)
        return 2
    ids = parse_ids(Path(args.ids_file).read_text(encoding="utf-8").splitlines())
    conn = sqlite3.connect(str(db), timeout=5.0)
    try:
        result = delete_decisions(conn, ids, dry_run=args.dry_run)
    finally:
        conn.close()
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

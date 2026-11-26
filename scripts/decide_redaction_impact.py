"""Measure, locally, what the ``/v1/decide`` ingress redaction would change.

T-decide-endpoint-bypasses-ingress-redaction (Bohr msg-1156 Obj-3 §3(a)).
Nothing is sent anywhere: every input goes through
:func:`lexora.ingress.gate.redact_decide_payload` in this process only.

Input: a JSONL file, one ``/v1/decide`` request body per line
(``{"policy", "state", "questions"}``) -- e.g. rebuilt from the mindwire
chatroom history the way the conductor builds its state.

Output (tab-separated, per policy): inputs, inputs changed, key hits
(would not be sent), scanner failures, and the redaction counts by kind.
Never a state, a question, or a replaced value.

Usage::

    python scripts/decide_redaction_impact.py requests.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import TextIO

from lexora.decide.contract import DecideRequest
from lexora.ingress.gate import KeyRedacted, redact_decide_payload

_COLUMNS = ("policy", "inputs", "changed", "key_hits", "failed", "kinds")


def measure(lines: Iterable[str]) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {"inputs": 0, "changed": 0, "key_hits": 0, "failed": 0, "kinds": Counter()}
    )
    for line in lines:
        if not line.strip():
            continue
        body = DecideRequest.model_validate(json.loads(line))
        row = stats[body.policy]
        row["inputs"] += 1  # type: ignore[operator]
        try:
            out = redact_decide_payload(body.state, dict(body.questions))
        except KeyRedacted:
            row["key_hits"] += 1  # type: ignore[operator]
            continue
        except Exception:  # noqa: BLE001 - counted, never shown
            row["failed"] += 1  # type: ignore[operator]
            continue
        if out.redactions:
            row["changed"] += 1  # type: ignore[operator]
            for item in out.redactions.split(","):
                kind, count = item.split(":")
                row["kinds"][kind] += int(count)  # type: ignore[index]
    return dict(stats)


def render(stats: dict[str, dict[str, object]], out: TextIO) -> None:
    out.write("\t".join(_COLUMNS) + "\n")
    for policy in sorted(stats):
        row = stats[policy]
        kinds: Counter[str] = row["kinds"]  # type: ignore[assignment]
        cells = [policy, *(str(row[c]) for c in _COLUMNS[1:5])]
        cells.append(",".join(f"{k}:{v}" for k, v in sorted(kinds.items())) or "-")
        out.write("\t".join(cells) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("requests", help="JSONL file of /v1/decide request bodies")
    args = parser.parse_args(argv)
    with open(args.requests, encoding="utf-8") as f:
        render(measure(f), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

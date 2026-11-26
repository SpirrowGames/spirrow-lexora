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

An input line that cannot be decoded, parsed or validated is skipped and
reported on stderr as ``line <n>: <category>`` only (``undecodable`` /
``invalid_json`` / ``invalid_request``): no exception text and no traceback,
because a pydantic ``ValidationError`` carries the rejected ``input_value``
(PR #121 gate objection). The exit status is 1 when any line was rejected.

Usage::

    python scripts/decide_redaction_impact.py requests.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Protocol, TextIO

from lexora.decide.contract import DecideRequest
from lexora.ingress.gate import KeyRedacted, redact_decide_payload

class _Writer(Protocol):
    def write(self, text: str, /) -> object: ...


_COLUMNS = ("policy", "inputs", "changed", "key_hits", "failed", "kinds")


def _parse(line: str | bytes) -> DecideRequest | str:
    """Return the request, or a safe rejection category (never the error text)."""
    try:
        text = line.decode("utf-8") if isinstance(line, bytes) else line
    except UnicodeDecodeError:
        return "undecodable"
    if not text.strip():
        return ""
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):  # deep nesting raises RecursionError
        return "invalid_json"
    try:
        return DecideRequest.model_validate(data)
    except Exception:  # noqa: BLE001 - ValidationError echoes input_value
        return "invalid_request"


def measure(
    lines: Iterable[str | bytes], errors: _Writer | None = None
) -> dict[str, dict[str, object]]:
    """Count per policy. Rejected lines go to ``errors`` as ``line <n>: <category>``."""
    stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {"inputs": 0, "changed": 0, "key_hits": 0, "failed": 0, "kinds": Counter()}
    )
    for lineno, line in enumerate(lines, start=1):
        body = _parse(line)
        if body == "":
            continue
        if isinstance(body, str):
            if errors is not None:
                errors.write(f"line {lineno}: {body}\n")
            continue
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
    rejected = _CountingWriter(sys.stderr)
    with open(args.requests, "rb") as f:
        render(measure(f, rejected), sys.stdout)
    return 1 if rejected.count else 0


class _CountingWriter:
    """Pass rejection lines through to ``out`` and count them."""

    def __init__(self, out: TextIO) -> None:
        self.out = out
        self.count = 0

    def write(self, text: str) -> int:
        self.count += 1
        return self.out.write(text)


if __name__ == "__main__":
    raise SystemExit(main())

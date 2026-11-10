"""PEM marker pairing per sequence (T-redactor-quoted-pem-begin-pairs-with-later-end,
msg-1028 / msg-1032 / msg-1034; fixtures and literals from stage 1 / 1b,
msg-1025 / msg-1031).

A begin quoted in prose used to pair with the end of a later real key, so
everything in between -- the prose -- was redacted as one block. Now:

* plain sequence: a begin is settled as unclosed (B, bounded by P-1) when
  another begin arrives before its end (R-1), so the prose between the quote
  and the key survives;
* every sequence: the markers of one line are read in column order, none
  skipped (an end left of a begin is an orphan end or closes the open begin;
  a begin right of a closing end opens a new block);
* hunk sides: a begin that arrives while another is open is absorbed (P-3).

The expected values are literals derived from the spec, not from the code.
Reproduction tests (fail before this change): F-1, F-2, F-5, F-6i, F-6ii,
F-8rev, F-9-side, F-9-plain, F-6i-side, F-6ii-side. Regression tests (same as
before): F-3, F-4, F-7ctx, F-7plus, F-8two, F-8t7.

Markers are assembled in code and never written out verbatim (D8).
"""

from __future__ import annotations

import base64
import random
import re

import pytest

from lexora.ingress.redact import Redactor, format_redaction_extent, format_redactions

_RNG = random.Random(20261110)
#: 3 rows of 64 and a short last row of 16.
_BODY = base64.b64encode(bytes(_RNG.getrandbits(8) for _ in range(48 * 3 + 10))).decode()
ROWS = [_BODY[i : i + 64] for i in range(0, len(_BODY), 64)]
assert [len(r) for r in ROWS] == [64, 64, 64, 16]
B = "-----" + "BEGIN RSA PRIVATE KEY" + "-----"
E = "-----" + "END RSA PRIVATE KEY" + "-----"
assert (len(B), len(E)) == (31, 29)
P = [f"Line {i} of the review explains why the scanner keeps the prose." for i in range(10)]
assert len(P[0]) == 62
Q = f"The scanner treats `{B}` as the start of a block."
KEY = [B, *ROWS, E]
_LABEL_RE = re.compile(r"\[REDACTED:[a-z]+#\d+\]")


def _hunk(lines: list[str], old: int, new: int) -> str:
    return "\n".join([f"@@ -1,{old} +1,{new} @@", *lines])


def _ctx(lines: list[str]) -> str:
    return _hunk([" " + line for line in lines], len(lines), len(lines))


_F1 = [P[0], Q, P[1], P[2], P[3], *KEY, P[4], P[5]]
_F6I = [P[0], B, ROWS[0], E + " " + B, ROWS[1], ROWS[3], E, P[1]]
_F6II = [P[0], ROWS[0], ROWS[3], E + " " + B, ROWS[1], ROWS[3], E, P[1]]
_F9 = [P[0], B, ROWS[0], B, ROWS[1], E + " " + B, ROWS[2], E, P[1]]

#: name -> (text, redacted line numbers, redactions=, redaction_extent=, labels seen)
CASES: dict[str, tuple[str, list[int], str, str, list[str]]] = {
    # -- plain sequence -----------------------------------------------------
    "F-1": (
        "\n".join(_F1),
        [1, 5, 6, 7, 8, 9, 10],
        "pem:2",
        "pem:7/325,pem_unclosed:1",
        ["pem#1", "pem#2"],
    ),
    "F-2": (
        "\n".join([P[0], Q, "ok", "Yes", *KEY, P[4], P[5]]),
        list(range(1, 10)),
        "pem:2",
        "pem:9/330,pem_unclosed:1",
        ["pem#1", "pem#2"],
    ),
    "F-3": (
        "\n".join([Q, P[1], P[2], Q, P[3], P[4]]),
        [0, 3],
        "pem:2",
        "pem:2/114,pem_unclosed:2",
        ["pem#1"],
    ),
    "F-4": (
        "\n".join([Q, P[1], ROWS[0], ROWS[1], E, P[2]]),
        [0, 1, 2, 3, 4],
        "pem:1",
        "pem:5/276,pem_unclosed:0",
        ["pem#1"],
    ),
    "F-5": (
        "\n".join([Q, P[1], P[2], ROWS[0], ROWS[1], ROWS[3], P[3], *KEY, P[4]]),
        [0, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        "pem:3",
        "pem:10/469,pem_unclosed:1",
        ["pem#1", "pem#2", "pem#3"],
    ),
    "F-6i": (
        "\n".join(_F6I),
        [1, 2, 3, 4, 5, 6],
        "pem:2",
        "pem:6/264,pem_unclosed:0",
        ["pem#1", "pem#2"],
    ),
    "F-6ii": (
        "\n".join(_F6II),
        [1, 2, 3, 4, 5, 6],
        "pem:2",
        "pem:6/249,pem_unclosed:1",
        ["pem#1", "pem#2"],
    ),
    "F-8two": (
        "\n".join([*KEY, P[0], B, *reversed(ROWS[:3]), ROWS[3], E]),
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        "pem:2",
        "pem:12/536,pem_unclosed:0",
        ["pem#1", "pem#2"],
    ),
    "F-8t7": (
        "\n".join([B, *ROWS, E, f"see {B}", *P, "password = hunter2hunter2"]),
        [0, 1, 2, 3, 4, 5, 6, 17],
        "assignment:1,pem:2",
        "assignment:1/14,pem:7/299,pem_unclosed:1",
        ["assignment#1", "pem#1", "pem#2"],
    ),
    "F-8rev": (
        "\n".join([f"see {B}", *P, B, *ROWS, E, "password = hunter2hunter2"]),
        [0, 11, 12, 13, 14, 15, 16, 17],
        "assignment:1,pem:2",
        "assignment:1/14,pem:7/299,pem_unclosed:1",
        ["assignment#1", "pem#1", "pem#2"],
    ),
    "F-9-plain": (
        "\n".join(_F9),
        [1, 2, 3, 4, 5, 6, 7],
        "pem:3",
        "pem:7/343,pem_unclosed:1",
        ["pem#1", "pem#2", "pem#3"],
    ),
    # -- hunk sides ---------------------------------------------------------
    "F-7ctx": (
        _ctx(_F1),
        list(range(2, 12)),
        "pem:1",
        "pem:10/511,pem_unclosed:0",
        ["pem#1"],
    ),
    "F-7plus": (
        _hunk([" " + P[0], " " + Q, " " + P[1], *("+" + line for line in KEY), " " + P[4]], 4, 4 + len(KEY)),
        list(range(2, 11)),
        "pem:2",
        "pem:9/449,pem_unclosed:1",
        ["pem#1", "pem#2"],
    ),
    "F-9-side": (
        _ctx(_F9),
        [2, 3, 4, 5, 6, 7, 8],
        "pem:2",
        "pem:7/343,pem_unclosed:0",
        ["pem#1", "pem#2"],
    ),
    "F-6i-side": (
        _ctx(_F6I),
        [2, 3, 4, 5, 6, 7],
        "pem:2",
        "pem:6/264,pem_unclosed:0",
        ["pem#1", "pem#2"],
    ),
    "F-6ii-side": (
        _ctx(_F6II),
        [1, 2, 3, 4, 5, 6, 7],
        "pem:2",
        "pem:7/311,pem_unclosed:1",
        ["pem#1", "pem#2"],
    ),
}


def _run(text: str) -> tuple[list[str], Redactor]:
    r = Redactor()
    out = r.redact(text).split("\n")
    assert len(out) == len(text.split("\n"))  # the line count never changes
    return out, r


@pytest.mark.parametrize("name", list(CASES))
def test_pairing_literals(name: str) -> None:
    text, redacted, redactions, extent, labels = CASES[name]
    out, r = _run(text)
    assert [n for n, line in enumerate(out) if "[REDACTED:" in line] == redacted
    assert format_redactions(r.counts) == redactions
    assert format_redaction_extent(r) == extent
    seen = sorted({m.group(0)[10:-1] for line in out for m in _LABEL_RE.finditer(line)})
    assert seen == labels
    # Lines with nothing redacted are unchanged; a diff line keeps its marker.
    for before, after in zip(text.split("\n"), out, strict=True):
        if "[REDACTED:" not in after:
            assert after == before
        elif before[:1] in " +-" and text.startswith("@@"):
            assert after[:1] == before[:1]


def test_f1_prose_between_the_quote_and_the_key_survives() -> None:
    out, _ = _run(CASES["F-1"][0])
    assert out[2:5] == [P[1], P[2], P[3]]
    assert out[1] == "The scanner treats `[REDACTED:pem#1]"
    assert out[5:11] == ["[REDACTED:pem#2]"] * 6
    assert all(row not in "\n".join(out) for row in ROWS)


def test_f8rev_test_7_in_the_reverse_order_keeps_the_prose() -> None:
    out, _ = _run(CASES["F-8rev"][0])
    assert out[1:11] == P
    assert out[11:17] == ["[REDACTED:pem#2]"] * 6


def test_f9_side_absorbs_the_inner_begin_and_reads_the_begin_after_the_end() -> None:
    out, _ = _run(CASES["F-9-side"][0])
    assert out[2:6] == [" [REDACTED:pem#1]"] * 4
    assert out[6] == " [REDACTED:pem#1] [REDACTED:pem#2]"
    assert out[7:9] == [" [REDACTED:pem#2]"] * 2
    assert out[1] == " " + P[0] and out[9] == " " + P[1]


def test_f9_plain_settles_the_first_begin_as_unclosed() -> None:
    out, r = _run(CASES["F-9-plain"][0])
    assert out[1:3] == ["[REDACTED:pem#1]"] * 2
    assert out[3:5] == ["[REDACTED:pem#2]"] * 2
    assert out[5] == "[REDACTED:pem#2] [REDACTED:pem#3]"
    assert out[6:8] == ["[REDACTED:pem#3]"] * 2
    assert r.pem_unclosed == 1


@pytest.mark.parametrize("name", ["F-6i", "F-6ii", "F-6i-side", "F-6ii-side"])
def test_f6_no_marker_text_is_left_on_the_shared_line(name: str) -> None:
    out, _ = _run(CASES[name][0])
    joined = "\n".join(out)
    assert "BEGIN RSA" not in joined and "END RSA" not in joined
    assert all(row not in joined for row in ROWS)

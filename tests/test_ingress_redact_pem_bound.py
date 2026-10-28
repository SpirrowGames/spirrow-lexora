"""Unclosed PEM ranges are bounded outside hunks, and the extent is recorded
(T-naysayer-codex-backend msg-898 §PR-1: P-1..P-4, tests 1-8; msg-906 §3).

Outside a hunk's sides, an unclosed begin marker used to redact from the
marker to the end of the prompt, and an orphan end marker from the start of
the prompt: a thread message that quotes a marker in prose (msg-863) wiped
the naysayer's input. Now the run continues only over lines with a PEM body
shape (P-2). Hunk sides are unchanged (P-3).
"""

from __future__ import annotations

import base64
import random
from collections import Counter
from pathlib import Path

import pytest

from lexora.ingress.redact import (
    KIND_ASSIGNMENT,
    KIND_PEM,
    Redactor,
    _pem_body_shape,
    format_redaction_extent,
    format_redactions,
)

_RNG = random.Random(20261028)
#: 24 full 64-wide rows plus a short last row: 25 rows (test 3).
_BODY = base64.b64encode(bytes(_RNG.getrandbits(8) for _ in range(48 * 24 + 20))).decode()
ROWS = [_BODY[i : i + 64] for i in range(0, len(_BODY), 64)]
assert len(ROWS) == 25 and len(ROWS[-1]) < 64
BEGIN = "-----BEGIN RSA PRIVATE KEY-----"
END = "-----END RSA PRIVATE KEY-----"
PROSE = [f"Line {i} of the review explains why the scanner keeps the prose." for i in range(50)]
MSG_863 = (Path(__file__).parent / "fixtures" / "redact" / "msg-863-body.txt").read_text(encoding="utf-8")


def _run(text: str) -> tuple[list[str], Redactor]:
    r = Redactor()
    out = r.redact(text)
    assert out.count("\n") == text.count("\n")  # the line count never changes
    return out.split("\n"), r


def _redacted(lines: list[str]) -> list[int]:
    return [n for n, line in enumerate(lines) if "[REDACTED:" in line]


# -- test 1 ---------------------------------------------------------------------
def test_1_prose_after_an_inline_quoted_begin_survives() -> None:
    """Only the marker's line is redacted; must fail on ``main``, where the
    marker ran to the end of the prompt."""
    marker_line = f"The scanner treats `{BEGIN}` as the start of a block."
    text = "\n".join([marker_line, *PROSE])
    out, r = _run(text)
    assert _redacted(out) == [0]
    assert out[0] == "The scanner treats `[REDACTED:pem#1]"
    assert out[1:] == PROSE
    assert r.counts == Counter({KIND_PEM: 1}) and r.pem_unclosed == 1


def test_1b_short_word_lines_extend_the_run_but_prose_still_stops_it() -> None:
    """msg-906 §3: a one-word prose line (``ok``) has a body shape, so it is
    taken into the run. The next real prose line still ends the run, and a
    short line further down, past that prose, is untouched."""
    text = "\n".join([f"quoted {BEGIN}", "ok", "Yes", *PROSE[:25], "ok", *PROSE[25:]])
    out, _ = _run(text)
    assert _redacted(out) == [0, 1, 2]
    assert out[3:] == [*PROSE[:25], "ok", *PROSE[25:]]


# -- test 2 ---------------------------------------------------------------------
def test_2_prose_before_an_orphan_end_survives() -> None:
    text = "\n".join([*PROSE, f"and it ends at {END} in the quote."])
    out, r = _run(text)
    assert _redacted(out) == [50]
    assert out[:50] == PROSE
    assert out[50] == "[REDACTED:pem#1] in the quote."
    assert r.pem_unclosed == 1


# -- test 3 ---------------------------------------------------------------------
def test_3_a_real_unclosed_key_is_redacted_and_the_prose_after_it_is_not() -> None:
    text = "\n".join([BEGIN, *ROWS, *PROSE])
    out, r = _run(text)
    assert _redacted(out) == list(range(26))
    assert all(row not in "\n".join(out) for row in ROWS)
    assert out[26:] == PROSE
    assert r.counts[KIND_PEM] == 1


def test_3b_a_real_key_missing_its_begin_is_redacted_upward() -> None:
    text = "\n".join([*PROSE, *ROWS, END])
    out, _ = _run(text)
    assert _redacted(out) == list(range(50, 76))
    assert out[:50] == PROSE


# -- test 4 ---------------------------------------------------------------------
@pytest.mark.parametrize("lead", ["> ", "    ", "> > "])
def test_4_quoted_or_indented_body_is_redacted_in_full(lead: str) -> None:
    text = "\n".join([lead + BEGIN, *(lead + row for row in ROWS), *PROSE])
    out, _ = _run(text)
    assert _redacted(out) == list(range(26))
    assert all(row not in "\n".join(out) for row in ROWS)
    assert out[26:] == PROSE


def test_4b_json_string_rows_with_literal_newline_are_body_lines() -> None:
    """P-2's trailing strip: ``"<row>\\n",``."""
    text = "\n".join([f'"{BEGIN}\\n",', *(f'"{row}\\n",' for row in ROWS), *PROSE])
    out, _ = _run(text)
    assert _redacted(out) == list(range(26))
    assert out[26:] == PROSE


# -- test 5 ---------------------------------------------------------------------
def test_5_encrypted_pem_headers_and_their_blank_line_are_in_the_block() -> None:
    headers = ["Proc-Type: 4,ENCRYPTED", "DEK-Info: AES-128-CBC,0123456789ABCDEF0123456789ABCDEF", ""]
    text = "\n".join([BEGIN, *headers, *ROWS, *PROSE])
    out, _ = _run(text)
    block = 1 + len(headers) + len(ROWS)
    # The blank line has nothing to replace; every other line of the block is.
    assert _redacted(out) == [n for n in range(block) if n != 3]
    assert out[3] == ""
    assert all(row not in "\n".join(out) for row in ROWS)
    assert out[block:] == PROSE


def test_5b_any_other_blank_line_ends_the_run() -> None:
    text = "\n".join([BEGIN, ROWS[0], "", ROWS[1], *PROSE])
    out, _ = _run(text)
    assert _redacted(out)[:2] == [0, 1]
    assert out[2] == ""
    assert out[4:] == PROSE


def test_5c_orphan_end_of_an_encrypted_pem_reads_the_headers_upward() -> None:
    headers = ["Proc-Type: 4,ENCRYPTED", "DEK-Info: AES-128-CBC,0123456789ABCDEF0123456789ABCDEF", ""]
    text = "\n".join([*PROSE, *headers, *ROWS, END])
    out, _ = _run(text)
    assert out[:50] == PROSE
    assert _redacted(out) == [n for n in range(50, 50 + 3 + 25 + 1) if n != 52]


# -- test 6 ---------------------------------------------------------------------
def test_6_hunk_side_unclosed_block_still_runs_to_the_end_of_the_side() -> None:
    """P-3: H-2b unchanged. Prose-like ``+`` lines after the begin are still
    redacted to the end of the side, and the plain text after the hunk is
    not."""
    side = [f"+{BEGIN}", "+not a body line at all, just words", *(f"+{row}" for row in ROWS[:3])]
    text = "\n".join(["@@ -0,0 +1,5 @@", *side, *PROSE])
    out, r = _run(text)
    assert _redacted(out) == [1, 2, 3, 4, 5]
    assert all(line.startswith("+[REDACTED:pem#1]") for line in out[1:6])
    assert out[6:] == PROSE
    assert r.pem_unclosed == 1


def test_6b_hunk_header_begin_opens_both_sides_to_their_end() -> None:
    text = "\n".join([f"@@ -1,2 +1,2 @@ {BEGIN}", " some words here", "-old words", "+new words", *PROSE])
    out, _ = _run(text)
    assert _redacted(out) == [0, 1, 2, 3]
    assert out[4:] == PROSE


# -- test 7 ---------------------------------------------------------------------
def test_7_extent_counts_lines_and_characters() -> None:
    # The closed block comes first. A begin quoted *before* a later real
    # block pairs with that block's end: closed-block pairing is not part of
    # PR-1 (reported as a residual for PR-2).
    text = "\n".join([BEGIN, *ROWS, END, f"see {BEGIN}", *PROSE, "password = hunter2hunter2"])
    out, r = _run(text)
    pem_chars = len(BEGIN) + len(BEGIN) + sum(len(row) for row in ROWS) + len(END)
    assert r.extent_lines[KIND_PEM] == 1 + 1 + 25 + 1
    assert r.extent_chars[KIND_PEM] == pem_chars
    assert r.extent_lines[KIND_ASSIGNMENT] == 1
    assert r.extent_chars[KIND_ASSIGNMENT] == len("hunter2hunter2")
    assert format_redaction_extent(r) == f"assignment:1/14,pem:28/{pem_chars},pem_unclosed:1"
    # ``redactions=`` keeps exactly the shape it had: kind:count, sorted.
    assert format_redactions(r.counts) == "assignment:1,pem:2"


def test_7b_extent_holds_numbers_only() -> None:
    text = "\n".join([BEGIN, *ROWS, END])
    _, r = _run(text)
    value = format_redaction_extent(r)
    assert value is not None
    assert all(row not in value for row in ROWS) and "BEGIN" not in value
    for item in value.split(","):
        name, size = item.split(":")
        assert name in ("pem", "pem_unclosed")
        assert all(part.isdigit() for part in size.split("/"))
    assert value == f"pem:27/{len(BEGIN) + sum(map(len, ROWS)) + len(END)},pem_unclosed:0"


def test_7c_nothing_redacted_means_no_extent() -> None:
    _, r = _run("\n".join(PROSE))
    assert format_redaction_extent(r) is None and format_redactions(r.counts) is None


def test_7d_a_block_seen_from_both_hunk_sides_counts_once() -> None:
    """A context-line block belongs to both sides; one block, one count, one
    set of lines -- and one ``pem_unclosed``."""
    ctx = [f" {BEGIN}", *(f" {row}" for row in ROWS[:3])]
    text = "\n".join(["@@ -1,4 +1,4 @@", *ctx])
    _, r = _run(text)
    assert r.counts[KIND_PEM] == 1
    assert r.extent_lines[KIND_PEM] == 4
    assert r.pem_unclosed == 1


# -- test 8 ---------------------------------------------------------------------
def test_8_msg_863_body_plus_a_trailing_block_keeps_the_trailing_block() -> None:
    """D8's cost rows #18531/#18533/#18917/#18919 cannot be fetched, so the
    fixture is msg-863's body (which quotes unclosed begin markers in prose)
    followed by 100 lines that must survive."""
    tail = [f"trailing line {i}: the reviewer must still read this." for i in range(100)]
    text = MSG_863 + "\n" + "\n".join(tail)
    out, r = _run(text)
    assert out[-100:] == tail
    # Both quoted begins are found, but each marker line is also inside a D-1
    # window opened by the ``API_KEY = (`` msg-863 quotes, which redacts the
    # line in full: no ``pem`` label reaches the output, so none is counted.
    assert r.counts[KIND_PEM] == 2
    assert r.pem_unclosed == 0 and r.extent_lines[KIND_PEM] == 0


def test_7e_unclosed_block_superseded_by_d1_is_not_counted() -> None:
    """PR-gate advisory on #101: ``pem_unclosed`` counts blocks whose label
    reached the output, so it never appears without a ``pem`` extent."""
    text = "\n".join([f"api_key = {BEGIN}", *PROSE])
    out, r = _run(text)
    assert out[0] == "api_key = [REDACTED:assignment#1]"
    assert r.pem_unclosed == 0
    value = format_redaction_extent(r)
    assert value is not None and "pem" not in value


def test_7f_unclosed_count_is_per_request_across_blocks() -> None:
    r = Redactor()
    r.redact(f"see {BEGIN}\nprose here")
    r.redact(f"see {BEGIN}\nmore prose here")
    assert r.pem_unclosed == 2 and r.counts[KIND_PEM] == 2


# -- P-2 shape ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("line", "shape"),
    [
        (ROWS[0], "body"),
        ("> " + ROWS[0], "body"),
        ('    "' + ROWS[0] + '\\n",', "body"),
        ("+" + ROWS[-1], "body"),
        ("abc==", "body"),
        ("ok", "body"),
        ("A" * 80, "body"),
        ("A" * 81, None),
        ("Proc-Type: 4,ENCRYPTED", "header"),
        ("DEK-Info: AES-128-CBC,00FF", "header"),
        ("", "blank"),
        ("  > ", "blank"),
        ("two words", None),
        ("a=b", None),
        (END, None),
    ],
)
def test_pem_body_shape(line: str, shape: str | None) -> None:
    assert _pem_body_shape(line) == shape

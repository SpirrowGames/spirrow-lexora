"""Ingress redaction scanner (ADR-2026-05-31-14 §7.3, spirrow-mindwire#482 at
head ``0a4197d``; T-naysayer-codex-backend msg-800 §4 / §7).

The diff cases are the list in §7.3 ("Lexora のテストに、次の diff を入れます"):
each one checks that the line count is unchanged, every line keeps its diff
marker, and no line of the key's body survives. The negative cases are the
same list's "伏せてはならないもの": the output equals the input.
"""

from __future__ import annotations

import base64
import random
import string

import pytest

from lexora.ingress.redact import (
    KIND_ASSIGNMENT,
    KIND_BEARER,
    KIND_ENTROPY,
    KIND_KEY,
    KIND_PEM,
    format_redactions,
    redact,
    shannon_entropy,
)

_RNG = random.Random(20261015)
_BODY = base64.b64encode(bytes(_RNG.getrandbits(8) for _ in range(48 * 6 + 11))).decode()
#: 64-wide PEM body lines with a short last line, like a real key.
ROWS = [_BODY[i : i + 64] for i in range(0, len(_BODY), 64)]
assert len(ROWS[-1]) < 64
BEGIN = "-----BEGIN RSA PRIVATE KEY-----"
END = "-----END RSA PRIVATE KEY-----"
PEM = [BEGIN, *ROWS, END]


def _hunk(lines: list[str], old: int, new: int) -> str:
    return f"@@ -1,{old} +1,{new} @@\n" + "\n".join(lines)


def _assert_redacted(original: str, out: str) -> None:
    """Line count unchanged, markers kept, no body row (or prefix) left."""
    a, b = original.split("\n"), out.split("\n")
    assert len(a) == len(b)
    for before, after in zip(a, b):
        if before[:1] in "+- " and before[:1]:
            assert after[:1] == before[:1], (before, after)
    for row in ROWS:
        assert row[:16] not in out, row


class TestDiffCases:
    def test_pem_only_plus_lines(self) -> None:
        text = _hunk(["+" + line for line in PEM], 0, len(PEM))
        out, counts = redact(text)
        _assert_redacted(text, out)
        assert counts == {KIND_PEM: 1}
        assert out.split("\n")[1] == "+[REDACTED:pem#1]"

    def test_pem_only_minus_lines(self) -> None:
        text = _hunk(["-" + line for line in PEM], len(PEM), 0)
        out, counts = redact(text)
        _assert_redacted(text, out)
        assert counts == {KIND_PEM: 1}

    def test_partially_changed_pem(self) -> None:
        """Begin and end on context lines; middle rows replaced by - and +."""
        lines = [" " + BEGIN, " " + ROWS[0], "-" + ROWS[1], "-" + ROWS[2], "+" + ROWS[2], "+" + ROWS[1]]
        lines += [" " + r for r in ROWS[3:]] + [" " + END]
        old = sum(1 for line in lines if line[0] in " -")
        new = sum(1 for line in lines if line[0] in " +")
        text = _hunk(lines, old, new)
        out, _ = redact(text)
        _assert_redacted(text, out)

    def test_begin_on_context_line_rest_on_plus_lines(self) -> None:
        lines = [" " + BEGIN] + ["+" + r for r in ROWS] + ["+" + END]
        text = _hunk(lines, 1, len(lines))
        out, _ = redact(text)
        _assert_redacted(text, out)

    def test_pem_without_end(self) -> None:
        lines = ["+" + BEGIN] + ["+" + r for r in ROWS[:3]]
        text = _hunk(lines, 0, len(lines))
        out, counts = redact(text)
        _assert_redacted(text, out)
        assert counts == {KIND_PEM: 1}

    def test_pem_without_begin_hunk_starts_midway(self) -> None:
        lines = [" " + r for r in ROWS[2:]] + [" " + END, " trailing = 1"]
        text = _hunk(lines, len(lines), len(lines))
        out, _ = redact(text)
        _assert_redacted(text, out)
        assert out.split("\n")[-1] == " trailing = 1"

    def test_pem_body_only(self) -> None:
        lines = [" " + r for r in ROWS[1:4]]
        text = _hunk(lines, 3, 3)
        out, counts = redact(text)
        _assert_redacted(text, out)
        assert counts == {KIND_PEM: 1}

    @pytest.mark.parametrize("where", ["begin", "body", "end"])
    def test_crlf_pem(self, where: str) -> None:
        lines = ["+" + line + "\r" for line in PEM]
        text = _hunk(lines, 0, len(lines))
        out, _ = redact(text)
        _assert_redacted(text, out)
        # The CR stays on every line: the line ending's form is unchanged.
        assert all(line.endswith("\r") for line in out.split("\n")[1:])

    def test_trailing_whitespace_pem(self) -> None:
        lines = ["+" + line + "   " for line in PEM]
        text = _hunk(lines, 0, len(lines))
        out, _ = redact(text)
        _assert_redacted(text, out)
        assert all(line.endswith("   ") for line in out.split("\n")[1:])

    def test_indented_yaml_block_hunk_midway_with_short_last_line(self) -> None:
        """Kubernetes Secret style: ``key: |`` above the hunk, the hunk holds
        only middle rows and the short last row."""
        lines = [" " + "    " + r for r in ROWS[2:]] + [" " + "  other: value"]
        text = _hunk(lines, len(lines), len(lines))
        out, counts = redact(text)
        _assert_redacted(text, out)
        assert counts == {KIND_PEM: 1}
        assert out.split("\n")[1].startswith("     [REDACTED:pem#1]")
        assert out.split("\n")[-1] == "   other: value"

    def test_yaml_block_opener_then_pem(self) -> None:
        text = "tls.key: |\n  " + "\n  ".join(PEM) + "\nnext: 1"
        out, _ = redact(text)
        _assert_redacted(text, out)
        assert out.split("\n")[0] == "tls.key: |"

    def test_quoted_lines_with_trailing_comma_and_backslash(self) -> None:
        lines = ['+  "' + BEGIN + '\\n" \\'] + ['+  "' + r + '\\n",' for r in ROWS] + ['+  "' + END + '"']
        text = _hunk(lines, 0, len(lines))
        out, _ = redact(text)
        _assert_redacted(text, out)

    def test_begin_in_the_middle_of_a_line(self) -> None:
        lines = ['+private_key: "' + BEGIN] + ["+" + r for r in ROWS] + ["+" + END + '"']
        text = _hunk(lines, 0, len(lines))
        out, _ = redact(text)
        _assert_redacted(text, out)
        assert out.split("\n")[1].startswith('+private_key: "[REDACTED:pem#1]')

    def test_escaped_newline_one_line_json(self) -> None:
        one = '{"private_key": "' + BEGIN + "\\n" + "\\n".join(ROWS) + "\\n" + END + '\\n", "x": 1}'
        assert "\n" not in one
        out, counts = redact(one)
        _assert_redacted(one, out)
        assert counts == {KIND_PEM: 1}
        assert out.startswith('{"private_key": "[REDACTED:pem#1]')

    def test_escaped_newline_without_end(self) -> None:
        one = '"key": "' + BEGIN + "\\n" + ROWS[0] + "\\n" + ROWS[1]
        out, _ = redact(one)
        _assert_redacted(one, out)


class TestMustNotRedact:
    @pytest.mark.parametrize(
        "text",
        [
            '[\n  "read",\n  "write",\n  "admin",\n  "delete"\n]',
            "scopes:\n  - read\n  - write\n  - admin\n  - billing",
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP\nshortline\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU\nxyz",
            "# This comment line has words only\n# and another comment line here\n# third line",
            "@@ -1,3 +1,3 @@\n context line\n-old value = 1\n+new value = 2",
            "sha: 3f2a9c4e1b7d8f6a5e4c3b2a1f0e9d8c7b6a5f4e\nid: 123e4567-e89b-12d3-a456-426614174000",
            "self.token = token_value\npassword: str = Field(...)",
        ],
        ids=["json-array", "yaml-list", "uneven-alnum", "comments", "plain-diff", "hash-shapes", "code"],
    )
    def test_output_equals_input(self, text: str) -> None:
        out, counts = redact(text)
        assert out == text
        assert not counts


class TestSingleLineKinds:
    def test_known_prefix_key(self) -> None:
        out, counts = redact("OPENAI_API_KEY=sk-proj-abcdefghij1234567890KLMNOP")
        assert out == "OPENAI_API_KEY=[REDACTED:key#1]"
        assert counts == {KIND_KEY: 1}

    def test_prefix_inside_a_word_is_not_a_key(self) -> None:
        text = "task-abcdefghijklmnopqrstuvwxyz"
        assert redact(text)[0] == text

    def test_bearer(self) -> None:
        out, counts = redact("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc123")
        assert out == "Authorization: Bearer [REDACTED:bearer#1]"
        assert counts == {KIND_BEARER: 1}

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('password = "hunter2hunter2"', 'password = "[REDACTED:assignment#1]"'),
            ("DB_PASSWORD=s3cr3tvalue", "DB_PASSWORD=[REDACTED:assignment#1]"),
            ("client_secret: 'abcd1234efgh'", "client_secret: '[REDACTED:assignment#1]'"),
        ],
    )
    def test_assignment(self, text: str, expected: str) -> None:
        out, counts = redact(text)
        assert out == expected
        assert counts == {KIND_ASSIGNMENT: 1}

    def test_short_assignment_is_kept(self) -> None:
        assert redact('password = "short"')[0] == 'password = "short"'

    def test_entropy(self) -> None:
        token = "".join(_RNG.choice(string.ascii_letters + string.digits) for _ in range(40))
        assert shannon_entropy(token) >= 4.0
        out, counts = redact(f"value {token} end")
        assert out == "value [REDACTED:entropy#1] end"
        assert counts == {KIND_ENTROPY: 1}

    def test_same_secret_keeps_its_number(self) -> None:
        out, counts = redact("a=sk-proj-abcdefghij1234567890KLMNOP\nb=sk-proj-abcdefghij1234567890KLMNOP")
        assert out == "a=[REDACTED:key#1]\nb=[REDACTED:key#1]"
        assert counts == {KIND_KEY: 2}


class TestInvariants:
    @pytest.mark.parametrize("seed", range(40))
    def test_line_count_never_changes(self, seed: int) -> None:
        """Property: random mixes of PEM pieces, diff markers, CRs, quotes,
        prose and keys never change the number of lines."""
        rng = random.Random(seed)
        pieces = [*PEM, "", "plain words here", "sk-proj-abcdefghij1234567890KLMNOP", '"quoted",', "key: |", "\r"]
        lines = []
        for _ in range(rng.randint(1, 60)):
            piece = rng.choice(pieces)
            lines.append(rng.choice(["", "+", "-", " ", "  "]) + piece + rng.choice(["", "\r", " ", ","]))
        if rng.random() < 0.5:
            lines.insert(0, f"@@ -1,{rng.randint(0, 40)} +1,{rng.randint(0, 40)} @@")
        text = "\n".join(lines)
        out, _ = redact(text)
        assert out.count("\n") == text.count("\n")

    def test_redactions_value_never_contains_the_text(self) -> None:
        secret = "sk-proj-abcdefghij1234567890KLMNOP"
        text = "\n".join(["+" + line for line in PEM]) + f"\nkey={secret}\npassword = 'hunter2hunter2'"
        _, counts = redact(text)
        value = format_redactions(counts)
        assert value == "assignment:1,key:1,pem:1"
        for fragment in (secret, "hunter2", *ROWS, BEGIN):
            assert fragment not in value

    def test_format_redactions_empty_is_none(self) -> None:
        assert format_redactions(redact("nothing here")[1]) is None

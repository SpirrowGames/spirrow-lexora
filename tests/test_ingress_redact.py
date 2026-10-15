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
            # Base64-only lines of uneven width whose joined text has entropy
            # >= 4.0 (so only the body-run *shape* keeps them), while each
            # line alone stays under D-2's 4.0 (15 distinct characters).
            "abcdefghijklmno" * 3 + "\nshortline\n" + "ABCDEFGHIJKLMNO" * 3 + "AB\nxyz",
            "# This comment line has words only\n# and another comment line here\n# third line",
            "@@ -1,3 +1,3 @@\n context line\n-old value = 1\n+new value = 2",
            "sha: 3f2a9c4e1b7d8f6a5e4c3b2a1f0e9d8c7b6a5f4e\nid: 123e4567-e89b-12d3-a456-426614174000",
        ],
        ids=["json-array", "yaml-list", "uneven-alnum", "comments", "plain-diff", "hash-shapes"],
    )
    def test_output_equals_input(self, text: str) -> None:
        assert shannon_entropy("abcdefghijklmno" * 3 + "ABCDEFGHIJKLMNO" * 3) >= 4.0
        out, counts = redact(text)
        assert out == text
        assert not counts


class TestSingleLineKinds:
    def test_known_prefix_key(self) -> None:
        # Not a D-1 key name, so the known-prefix rule is what fires.
        out, counts = redact("OPENAI=sk-proj-abcdefghij1234567890KLMNOP")
        assert out == "OPENAI=[REDACTED:key#1]"
        assert counts == {KIND_KEY: 1}

    @pytest.mark.parametrize(
        "secret",
        [
            "sk-abcdefghijklmnopqrstuvwx",
            "ghp_" + "a1B2c3D4e5" * 4,
            "github_pat_" + "11ABCDEFG0" * 3,
            "AKIAABCDEFGHIJKLMNOP",
            "xoxb-123456789012-abcdefghij",
            "xoxp-123456789012-abcdefghij",
            "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q",
        ],
        ids=["sk-", "ghp_", "github_pat_", "AKIA", "xoxb-", "xoxp-", "AIza"],
    )
    def test_required_known_prefixes(self, secret: str) -> None:
        """msg-836 §3: these prefixes must be in the code."""
        out, counts = redact(f"value {secret} end")
        assert out == "value [REDACTED:key#1] end"
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
            # msg-840: quotes are inside the redacted part.
            ('password = "hunter2hunter2"', "password = [REDACTED:assignment#1]"),
            ("DB_PASSWORD=s3cr3tvalue", "DB_PASSWORD=[REDACTED:assignment#1]"),
            ("client_secret: 'abcd1234efgh'", "client_secret: [REDACTED:assignment#1]"),
        ],
    )
    def test_assignment(self, text: str, expected: str) -> None:
        out, counts = redact(text)
        assert out == expected
        assert counts == {KIND_ASSIGNMENT: 1}

    def test_short_assignment_is_kept(self) -> None:
        """A right-hand side of 7 characters, quotes included (msg-840)."""
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


#: Every must-be-redacted D-1 case of T-naysayer-codex-backend msg-829 to
#: msg-840 that is readable in the thread record, with the fragments that
#: must not survive. (msg-818 to msg-828 were not in the implementer's view.)
_D1_MUST_REDACT = [
    ('API_KEY = """\n}\n"""', ["}"]),
    ("password = '''\n);\n'''", [");"]),
    ("secret: |\n  {\n  }\n  ],", ["{", "}"]),  # "]" is checked by the per-line label assertion
    ('token = "abc\\\n}"', ["}"]),
    ('API_KEY = get_secret_value(\n  "x"\n)', ['"x"', ")"]),
    ("API_KEY = f(\n  g(\n  )\n)\nx = 1", ["g(", "x = 1"]),
    ('API_KEY = get_env( # workaround for )\n  "correcthorsebatterystaple"\n)', ["correcthorse", "get_env"]),
    ('API_KEY = get_env(\n  "VAR",  # )\n  "correcthorsebatterystaple"\n)', ["correcthorse", "VAR"]),
    ('secret = a +  // note\n  "correcthorsebattery"', ["correcthorse", "note"]),
    ('password = "short"\nnext_line', ["next_line"]),
    ('API_KEY = get_env(\n  "VAR"\n)\n  || "correcthorsebatterystaple"', ["correcthorse"]),
    ('API_KEY = get_env(\n  "VAR"\n) /* \n*/ || "correcthorsebatterystaple"', ["correcthorse"]),
    ('API_KEY = "secret_with_escaped_\\"_quote"', ["_quote", "secret_with"]),
    ('token = "abc" + "correcthorsebattery"', ["correcthorse", "abc"]),
    ("password = 'x' # \"correcthorsebattery\"", ["correcthorse"]),
    ('secret = f"{prefix}correcthorsebattery"', ["correcthorse", "prefix"]),
    ("PASSWORD=correcthorsebatterystaple", ["correcthorse"]),
    ("API_KEY=abcdefghijklmnopqrstuv", ["abcdefghij"]),
    ("token = f(x)\na = 1\nb = 2", ["a = 1", "b = 2"]),
    ('password = "abcdef"', ["abcdef"]),
    ("self.token = token_value", ["token_value"]),
    ("password: str = Field(...)", ["Field"]),
]


def _lines_after_the_key_line(text: str) -> list[int]:
    return [n for n, line in enumerate(text.split("\n")) if n > 0 and line.strip()]


class TestD1Assignment:
    @pytest.mark.parametrize(("text", "gone"), _D1_MUST_REDACT)
    def test_must_redact(self, text: str, gone: list[str]) -> None:
        out, counts = redact(text)
        assert out.count("\n") == text.count("\n")
        for fragment in gone:
            assert fragment not in out, (fragment, out)
        assert counts[KIND_ASSIGNMENT] == 1
        # Every non-blank line after the key line (all within 6) is a label.
        for n in _lines_after_the_key_line(text):
            assert "[REDACTED:assignment#1]" in out.split("\n")[n], out

    def test_key_line_keeps_name_operator_and_space(self) -> None:
        out, _ = redact('    "api-key":   "abcdefghijk",  \r')
        assert out == '    "api-key":   [REDACTED:assignment#1]  \r'

    def test_key_line_redacts_from_the_first_operator(self) -> None:
        out, _ = redact("x = {'password': 'abc'}")
        assert out == "x = [REDACTED:assignment#1]"

    @pytest.mark.parametrize("text", ['password = "abcde"', "password = None", "token = 1234567"])
    def test_short_key_line_value_is_kept(self, text: str) -> None:
        assert redact(text) == (text, {})

    def test_password_none_still_opens_the_window(self) -> None:
        out, counts = redact("password = None\nhunter2hunter2")
        assert out == "password = None\n[REDACTED:assignment#1]"
        assert counts == {KIND_ASSIGNMENT: 1}

    def test_window_boundary_counts_non_blank_lines(self) -> None:
        rest = ["l1", "", "l2", "l3", "  ", "l4", "l5", "l6", "", "l7"]
        out = redact("\n".join(["token = abcdefghij", *rest]))[0].split("\n")
        assert out[0] == "token = [REDACTED:assignment#1]"
        for n, name in enumerate(rest, start=1):
            if name.strip() and name != "l7":
                assert out[n] == "[REDACTED:assignment#1]", name
            else:
                assert out[n] == name, name  # blanks untouched; l7 is the 7th

    def test_key_line_inside_the_window_restarts_it(self) -> None:
        lines = ["token = None", "a", "b", "secret = x", "c", "d", "e", "f", "g", "h", "i"]
        out, counts = redact("\n".join(lines))
        o = out.split("\n")
        assert o[0] == "token = None"
        assert all(line == "[REDACTED:assignment#1]" for line in o[1:10])  # key line in full, then 6
        assert o[10] == "i"
        assert counts == {KIND_ASSIGNMENT: 1}

    def test_markers_indentation_and_cr_stay_in_the_window(self) -> None:
        text = "@@ -0,0 +1,3 @@\n+token = abcdefghij\r\n+    'value',\r\n+\t\"other\" \\\r"
        out = redact(text)[0].split("\n")
        assert out[1] == "+token = [REDACTED:assignment#1]\r"
        assert out[2] == "+    '[REDACTED:assignment#1]',\r"
        assert out[3] == '+\t"[REDACTED:assignment#1]" \\\r'

    def test_before_side_key_does_not_redact_after_side(self) -> None:
        text = "@@ -1,1 +1,2 @@\n-password = abcdefghij\n+foo = 1\n+bar = 2"
        out, counts = redact(text)
        assert out == "@@ -1,1 +1,2 @@\n-password = [REDACTED:assignment#1]\n+foo = 1\n+bar = 2"
        assert counts == {KIND_ASSIGNMENT: 1}

    def test_context_key_line_redacts_both_sides(self) -> None:
        text = "@@ -1,2 +1,2 @@\n token = None\n-old_value\n+new_value"
        out, counts = redact(text)
        assert out == "@@ -1,2 +1,2 @@\n token = None\n-[REDACTED:assignment#1]\n+[REDACTED:assignment#2]"
        assert counts == {KIND_ASSIGNMENT: 2}

    def test_context_lines_are_counted_once(self) -> None:
        text = "@@ -1,2 +1,2 @@\n token = abcdefghij\n value_line"
        out, counts = redact(text)
        assert out == "@@ -1,2 +1,2 @@\n token = [REDACTED:assignment#1]\n [REDACTED:assignment#1]"
        assert counts == {KIND_ASSIGNMENT: 1}

    def test_window_stops_at_the_end_of_the_hunk(self) -> None:
        text = "@@ -0,0 +1,1 @@\n+token = None\n@@ -9,0 +10,2 @@\n+plain1\n+plain2"
        assert redact(text) == (text, {})

    def test_window_does_not_run_past_the_hunk_into_prose(self) -> None:
        text = "@@ -0,0 +1,1 @@\n+token = None\ntrailing prose"
        assert redact(text) == (text, {})

    def test_window_stops_at_a_file_header_outside_a_hunk(self) -> None:
        text = "token = None\ndiff --git a/x b/x\n--- a/x\n+++ b/x\nfoo"
        assert redact(text) == (text, {})

    def test_window_line_holding_a_pem_line_is_redacted_whole(self) -> None:
        text = "secret: |\n  " + "\n  ".join(PEM[:3])
        out, counts = redact(text)
        _assert_redacted(text, out)
        assert out.split("\n")[1] == "  [REDACTED:assignment#1]"
        assert counts[KIND_ASSIGNMENT] == 1

    @pytest.mark.parametrize("rows", [ROWS[:2], ROWS], ids=["operator-after-pem", "pad-equals-inside-pem"])
    def test_one_line_pem_before_the_operator_stays_redacted(self, rows: list[str]) -> None:
        """PR-gate on #96: a one-line (escaped ``\\n``) PEM block left of a
        key line's first operator must keep its PEM redaction. With all
        ROWS, the base64 pad ``=`` inside the block is the first operator."""
        one = '"' + BEGIN + "\\n" + "\\n".join(rows) + "\\n" + END + '" token=abcdefghij'
        assert "\n" not in one
        out, counts = redact(one)
        _assert_redacted(one, out)
        assert out.startswith('"[REDACTED:pem#1]')
        assert out.endswith("[REDACTED:assignment#1]")
        assert counts[KIND_PEM] == 1 and counts[KIND_ASSIGNMENT] == 1

    def test_gate_example_pem_value_then_key(self) -> None:
        one = '{"private_key": "' + BEGIN + "\\n" + "\\n".join(ROWS) + "\\n" + END + '", "api_key": "sk-12345"}'
        out, _ = redact(one)
        _assert_redacted(one, out)
        assert "sk-12345" not in out

    def test_multi_line_pem_ending_on_a_key_line(self) -> None:
        lines = ["+" + BEGIN, *("+" + r for r in ROWS), "+" + END + " token=abcdefghij"]
        text = _hunk(lines, 0, len(lines))
        out, counts = redact(text)
        _assert_redacted(text, out)
        assert "abcdefghij" not in out
        assert counts[KIND_PEM] == 1

    def test_left_of_the_operator_still_gets_single_line_rules(self) -> None:
        out, _ = redact("sk-proj-abcdefghij1234567890KLMNOP token: abcdefghij")
        assert out == "[REDACTED:key#1] token: [REDACTED:assignment#1]"


_HIGH = string.ascii_letters  # 52 distinct letters


class TestD2Entropy:
    def test_31_is_kept_32_is_redacted(self) -> None:
        t31, t32 = _HIGH[:31], _HIGH[:32]
        assert shannon_entropy(t31) >= 4.0
        assert redact(f"v {t31} e") == (f"v {t31} e", {})
        assert redact(f"v {t32} e") == ("v [REDACTED:entropy#1] e", {KIND_ENTROPY: 1})

    def test_letters_only_is_redacted(self) -> None:
        """The letters-and-digits check is gone (msg-836 §2)."""
        token = "".join(_RNG.choice(string.ascii_letters) for _ in range(40))
        assert shannon_entropy(token) >= 4.0
        assert redact(f"v {token}")[0] == "v [REDACTED:entropy#1]"

    def test_entropy_threshold(self) -> None:
        below = "abcdefghijklmno" * 3  # 15 distinct: log2(15) < 4.0
        at = "abcdefghijklmnop" * 3  # 16 distinct: exactly 4.0
        assert shannon_entropy(below) < 4.0 <= shannon_entropy(at)
        assert redact(f"v {below}") == (f"v {below}", {})
        assert redact(f"v {at}")[0] == "v [REDACTED:entropy#1]"

    @pytest.mark.parametrize(
        "text",
        [
            "commit 3f2a9c4e1b7d8f6a5e4c3b2a1f0e9d8c7b6a5f4e",
            "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            "image@sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            "digest sha256:9F86D081884C7D659A2FEAA0C55AD015A3BF4F1B2B0B822CD15D6C15B0F00A08",
            "id 123e4567-e89b-12d3-a456-426614174000",
        ],
        ids=["commit-sha", "sha256-hex", "image-digest", "digest-upper", "uuid"],
    )
    def test_hash_shapes_are_excluded(self, text: str) -> None:
        assert redact(text) == (text, {})


# -- diff metadata (mindwire#483 §7.3; T-naysayer-codex-backend msg-856/858) --

AKIA = "AKIAABCDEFGHIJKLMNOP"
GHP = "ghp_" + "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dF3hJ5"


class TestDiffMetadata:
    def test_full_header_block_is_unchanged(self) -> None:
        text = "\n".join(
            [
                "diff --git a/old.py b/new.py",
                "similarity index 87%",
                "rename from old.py",
                "rename to new.py",
                "index 3f2a9c4..9d8c7b6 100644",
                "--- a/old.py",
                "+++ b/new.py",
                "@@ -1,2 +1,2 @@",
                " keep",
                "-a",
                "\\ No newline at end of file",
                "+b",
                "\\ No newline at end of file",
                "diff --git a/m.sh b/m.sh",
                "old mode 100644",
                "new mode 100755",
                "diff --git a/n.txt b/n.txt",
                "new file mode 100644",
                "index 0000000..e69de29",
                "diff --git a/c.txt b/d.txt",
                "dissimilarity index 12%",
                "copy from c.txt",
                "copy to d.txt",
                "diff --git a/gone.txt b/gone.txt",
                "deleted file mode 100644",
                "index e69de29..0000000",
                "diff --git a/img.png b/img.png",
                "index 1234567..89abcde",
                "Binary files a/img.png and b/img.png differ",
                "diff --git a/combo b/combo",
                "index 1234567,89abcde..fedcba9",
            ]
        )
        assert redact(text) == (text, {})
        crlf = text.replace("\n", "\r\n")
        assert redact(crlf) == (crlf, {})

    @pytest.mark.parametrize(
        "text",
        [f"PR body\nindex {AKIA}", f"index {GHP}", f"thread says:\n\nindex {AKIA} here"],
        ids=["pr-body", "first-line", "thread-body"],
    )
    def test_index_outside_the_header_state_is_matched(self, text: str) -> None:
        out, counts = redact(text)
        assert AKIA not in out and GHP not in out
        assert counts[KIND_KEY] == 1

    def test_index_after_a_hunk_started_is_matched(self) -> None:
        text = f"diff --git a/x b/x\nindex 1234567..89abcde 100644\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\nindex {AKIA}"
        out, counts = redact(text)
        assert out.split("\n")[1] == "index 1234567..89abcde 100644"
        assert out.split("\n")[-1] == "index [REDACTED:key#1]"
        assert counts == {KIND_KEY: 1}

    def test_mode_only_diff_then_prose(self) -> None:
        """Einstein msg-857: a header state with no ---/+++/@@ must end."""
        head = "diff --git a/m.sh b/m.sh\nold mode 100644\nnew mode 100755"
        out, counts = redact(f"{head}\n\nindex {AKIA}")
        assert out == f"{head}\n\nindex [REDACTED:key#1]"
        assert counts == {KIND_KEY: 1}

    def test_truncated_diff_rename_line_is_matched(self) -> None:
        out, counts = redact(f"diff --git a/x b/x\nrename to {GHP}")
        assert out == "diff --git a/x b/x\nrename to [REDACTED:key#1]"
        assert counts == {KIND_KEY: 1}

    def test_header_run_broken_by_a_blank_line(self) -> None:
        text = f"diff --git a/x b/x\nindex 1234567..89abcde\n\nindex {AKIA}"
        out, _ = redact(text)
        assert out.split("\n")[1] == "index 1234567..89abcde"
        assert out.split("\n")[3] == "index [REDACTED:key#1]"

    def test_header_run_broken_by_prose(self) -> None:
        text = f"diff --git a/x b/x\nsome prose\nsimilarity index {AKIA}"
        out, counts = redact(text)
        assert out.split("\n")[2] == "similarity index [REDACTED:key#1]"
        assert counts == {KIND_KEY: 1}

    @pytest.mark.parametrize(
        "line",
        [
            'index = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"',
            "index sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
            "old mode sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
            "similarity index sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        ],
        ids=["index-assignment", "index-token", "mode-token", "similarity-token"],
    )
    def test_fixed_shape_headers_are_anchored(self, line: str) -> None:
        """Einstein's advisory on msg-858: a prefix alone does not skip."""
        out, counts = redact(f"diff --git a/x b/x\n{line}")
        assert "sk-ant-api03" not in out
        assert counts[KIND_KEY] == 1

    def test_path_header_path_is_still_matched(self) -> None:
        out, counts = redact(f"diff --git a/x b/x\nrename from a/x\nrename to {GHP}\n--- a/x\n+++ b/y")
        assert out.split("\n")[2] == "rename to [REDACTED:key#1]"
        assert out.split("\n")[1] == "rename from a/x"
        assert counts == {KIND_KEY: 1}

    def test_other_backslash_line_in_a_hunk_is_matched(self) -> None:
        text = f"@@ -1,2 +1,2 @@\n x\n\\ {AKIA}\n-a\n+b"
        out, counts = redact(text)
        assert out.split("\n")[2] == "\\ [REDACTED:key#1]"
        assert counts[KIND_KEY] == 1

    def test_no_newline_marker_with_trailing_whitespace_is_skipped(self) -> None:
        text = "@@ -1 +1 @@\n-a\n\\ No newline at end of file \r\n+b"
        assert redact(text) == (text, {})

    @pytest.mark.parametrize("where", ["prose", "hunk-unknown-line"])
    def test_unmarked_line_keeps_its_first_character(self, where: str) -> None:
        """Regression for the dropped first character (msg-855 §3)."""
        text = AKIA if where == "prose" else f"@@ -1 +1 @@\n-a\n+b\n{AKIA}"
        out, counts = redact(text)
        assert out.split("\n")[-1] == "[REDACTED:key#1]"
        assert counts == {KIND_KEY: 1}

    def test_no_newline_marker_does_not_use_up_the_window(self) -> None:
        body = [f"+line{n}" for n in range(6)]
        text = "@@ -0,0 +1,7 @@\n+api_key = abcdefghij\n\\ No newline at end of file\n" + "\n".join(body)
        out, counts = redact(text)
        rows = out.split("\n")
        assert rows[2] == "\\ No newline at end of file"
        assert rows[3:] == ["+[REDACTED:assignment#1]"] * 6
        assert counts == {KIND_ASSIGNMENT: 1}

    def test_header_lines_do_not_use_up_the_window(self) -> None:
        """M-4: a key line, a skipped header run, then 6 lines -- but the
        ``diff --git`` line itself is file structure and still ends it."""
        text = "token = None\ndiff --git a/x b/x\nindex 1234567..89abcde\nfoo"
        assert redact(text) == (text, {})

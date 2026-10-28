"""Credential redaction at Lexora's ingress (ADR-2026-05-31-14 §7.3).

Source of the rules: spirrow-mindwire#482 (ADR-14 §7.3, "資格情報の伏せ字"),
read at head ``0a4197d``. Lexora-side shape: T-naysayer-codex-backend
msg-800 §4 / msg-802. Where this module and #482's final head disagree, the
ADR wins and this module is wrong.

What the scanner guarantees
---------------------------
* **The line count never changes.** Text is split on ``\\n`` and joined back
  on ``\\n``; a multi-line match is replaced line by line, so every
  ``file:line`` a reviewer cites still points at the same source line.
* **Only the core of a line is replaced.** A line is split into
  ``prefix + core + suffix``: the diff marker, indentation, surrounding
  quotes, a trailing ``,`` or ``\\`` and the line ending (``\\r``) stay where
  they are. A redacted diff line is still a valid diff line
  (``+[REDACTED:pem#1]``).
* **Diff-aware matching.** Inside a unified-diff hunk the line-start marker
  is removed before matching, and multi-line patterns run over two
  sequences per hunk -- *before* (context + ``-`` lines) and *after*
  (context + ``+`` lines) -- so a partially changed secret is matched as one
  piece. A line inside a match in either sequence is redacted whatever its
  marker. Text outside hunks is one plain sequence, matched whole line:
  no character is dropped from a line that has no diff marker.
* **Diff metadata (mindwire#483 §7.3, T-naysayer-codex-backend msg-856 /
  msg-858).** Extended headers are skipped only in an unbroken run right
  after ``diff --git`` (M-2); the first other line ends that state. Inside a
  hunk only the exact ``\\ No newline at end of file`` line is skipped
  (M-3). Neither counts toward, nor stops, a D-1 window (M-4). A backslash
  line never ends a hunk (H-1, msg-861); the context text after a hunk
  header's closing ``@@`` opens both sides of the hunk (H-2, msg-863).
* **When the end (or the start) cannot be seen, redact more -- within a
  bound outside hunks (T-naysayer-codex-backend msg-898 PR-1).** On a hunk
  side (P-3), a PEM begin with no end is redacted to the side's last line
  and an end with no begin from the side's first line: a side is already
  bounded. In the plain sequence (everything outside a hunk's sides), an
  unclosed begin is redacted from the marker to the end of its line and
  then down only while each next line has the shape of a PEM body line
  (P-1, P-2, ``_pem_body_shape``); an orphan end the same way upward. Prose
  ends the run on its first line, so a marker quoted in a message no longer
  wipes the rest of the prompt. A run of PEM body lines with neither marker
  is recognised by shape (see ``_body_runs``).

What gets redacted (kinds, final): ``pem`` (PEM blocks and PEM body runs),
``key`` (known key prefixes), ``assignment`` (rule D-1 below), ``bearer``
(Bearer tokens) and ``entropy`` (rule D-2 below). Each replaced span becomes
``[REDACTED:<kind>#<n>]``; ``n`` numbers distinct secrets per kind within one
request, so the same secret seen twice keeps one number.

D-1, ``assignment`` (T-naysayer-codex-backend msg-834 / msg-836 / msg-840)
--------------------------------------------------------------------------
Purely structural: no decision here reads brackets, quotes, escapes,
comments, continuation characters or YAML.

* **Key line:** ``password|passwd|secret|token|api[_-]?key`` (any case,
  anywhere in an identifier) followed by ``=`` / ``:`` / ``:=``. No
  exclusions.
* **On the key line,** everything from just after the line's *first*
  ``=`` / ``:`` (and the whitespace after it) to the end of the line is
  redacted when it is 8+ characters long, counted with any quotes, after
  trimming surrounding whitespace.
* **Lookahead window:** the next 6 non-blank lines after a key line are
  redacted in full, whatever the key line's own value was (``password =
  None`` still opens a window). A key line inside the window is itself
  redacted in full and restarts the window from that line. The window ends
  early only at diff structure: the end of a hunk, or a hunk / file header
  outside a hunk. Inside a hunk the *before* and *after* sequences are
  followed separately. One window (with its restarts) is one ``assignment``.

D-2, ``entropy``: a token of 32+ characters with Shannon entropy >= 4.0 bits
per character. Hex strings (so commit SHAs), UUIDs and ``sha<N>:<hex>``
digests are excluded.

Accepted limits (msg-834; reported to D8 with the "stricter than D8" list)
------------------------------------------------------------------------
* **(a)** a value that continues more than 6 non-blank lines below its key
  line is not caught;
* **(b)** a secret assembled through names that do not look sensitive
  (``a = "correct"; API_KEY = a + b``) is not caught.

Known prefixes, PEM blocks, Bearer and D-2 still apply to every line D-1
does not cover.

The counts (never the text) are what the ledger records
(``redactions=<kind:count,...>``, see ``format_redactions``), and next to
them the sizes of what was replaced (``redaction_extent=<kind:lines/chars,
...>`` plus ``pem_unclosed:<n>``, see ``format_redaction_extent``; msg-898
P-4). Both hold numbers only.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

KIND_PEM = "pem"
KIND_KEY = "key"
KIND_ASSIGNMENT = "assignment"
KIND_BEARER = "bearer"
KIND_ENTROPY = "entropy"

#: Minimum Shannon entropy (bits per character) for the ``entropy`` kind and
#: for a PEM body run (ADR-14 §7.3).
ENTROPY_THRESHOLD = 4.0
#: A PEM body run: every line but the last has the same core length within
#: these bounds (PEM wraps at 64, MIME base64 at 76; ADR-14 §7.3).
BODY_LINE_MIN = 40
BODY_LINE_MAX = 80
#: Shortest token the entropy rule (D-2) looks at (msg-836 §2).
ENTROPY_MIN_LEN = 32
#: Shortest key-line right-hand side the assignment rule redacts, quotes
#: included (ADR-14 §7.3: 8 文字以上; msg-840).
ASSIGNMENT_MIN_LEN = 8
#: Non-blank lines after a key line that D-1 always redacts (msg-834).
ASSIGNMENT_WINDOW = 6

_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
#: File-header lines outside a hunk: diff structure, never matched. The
#: extended headers (``index`` and the rest) are *not* here: they are skipped
#: only inside the header state that ``diff --git`` opens (M-2).
_FILE_HEADER_PREFIXES = ("diff --git ", "--- ", "+++ ")
_DIFF_GIT = "diff --git "
#: Extended headers with a fixed git shape. mindwire#483 §7.3 names the
#: prefixes; anchoring the whole line to git's shape is Lexora's (Einstein's
#: design advisory), so ``index = "sk-..."`` after ``diff --git`` is matched.
_EXT_HEADER_FIXED_RE = re.compile(
    r"^(?:index [0-9a-f]+(?:,[0-9a-f]+)*\.\.[0-9a-f]+(?: [0-7]{6})?"
    r"|(?:old mode|new mode|deleted file mode|new file mode) [0-7]{6}"
    r"|(?:similarity|dissimilarity) index \d{1,3}%)\r?$"
)
#: Extended headers that carry a path (any text): the prefix is left alone,
#: the path still goes through the single-line rules (it is not a key line,
#: a PEM line or a window line). An ordinary path comes out unchanged.
_EXT_HEADER_PATH_PREFIXES = (
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
    "Binary files ",
)
#: Inside a hunk only this exact line (trailing whitespace ignored) is
#: structure; any other line starting with a backslash is matched (M-3).
_NO_NEWLINE = r"\ No newline at end of file"
_PEM_BEGIN_RE = re.compile(r"-----BEGIN [A-Z0-9 ]+-----")
_PEM_END_RE = re.compile(r"-----END [A-Z0-9 ]+-----")
_BASE64_LINE_RE = re.compile(r"^[A-Za-z0-9+/=]+$")
#: P-2 (msg-898): what is stripped around a plain line before its shape is
#: judged. Leading quote and diff prefixes; trailing whitespace, ``"``, ``,``
#: and a literal two-character ``\n``.
_PEM_BODY_LEAD = " \t\r\f\v>+-#/\"'"
_PEM_BODY_TRAIL = " \t\r\f\v\","
#: P-2: base64 characters only, ``=`` padding allowed, 1-80 characters.
_PEM_BODY_CORE_RE = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
PEM_BODY_SHAPE_MAX = 80
#: P-2: RFC 1421 header lines of an encrypted PEM.
_PEM_HEADER_RE = re.compile(r"^(?:Proc-Type|DEK-Info):")
_SHAPE_BODY = "body"
_SHAPE_HEADER = "header"
_SHAPE_BLANK = "blank"
#: ``key: |`` / ``key: >-`` (also ``- key: |``): the YAML block opener is not
#: content; the indented lines under it are (ADR-14 §7.3).
_YAML_BLOCK_OPENER_RE = re.compile(r"^(?:-\s+)?[\w.\-\"']+\s*:\s*[|>][+-]?\d*\s*$")

#: Known key prefixes. Order matters only for overlap: the first match wins.
_KEY_RES = [
    re.compile(r"(?<![A-Za-z0-9_])" + p)
    for p in (
        r"sk-ant-[A-Za-z0-9_\-]{20,}",
        r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}",
        r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}",
        r"gh[pousr]_[A-Za-z0-9]{36,}",
        r"github_pat_[A-Za-z0-9_]{22,}",
        r"glpat-[A-Za-z0-9_\-]{20,}",
        r"(?:AKIA|ASIA)[0-9A-Z]{16}",
        r"AIza[0-9A-Za-z_\-]{35}",
        r"xox[abposr]-[A-Za-z0-9\-]{10,}",
        r"npm_[A-Za-z0-9]{36}",
    )
]
_BEARER_RE = re.compile(r"(?i)\bBearer\s+([A-Za-z0-9\-._~+/]{8,}=*)")
#: D-1 key line: a credential-named key followed by an assignment operator
#: (msg-836 §1). Nothing about the value is looked at here.
_KEY_LINE_RE = re.compile(
    r"(?i)(?:password|passwd|secret|token|api[_\-]?key)[A-Za-z0-9_.\-]*[\"']?\s*(?::=|=|:)"
)
#: The D-1 key names alone, for structured data (a JSON member name).
_KEY_NAME_RE = re.compile(r"(?i)(?:password|passwd|secret|token|api[_\-]?key)")
#: The first assignment operator of a key line: D-1 redacts from after it.
_OPERATOR_RE = re.compile(r":=|=|:")
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{%d,}" % ENTROPY_MIN_LEN)
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
#: ``sha256:<hex>`` (container / content digests). The token pattern does not
#: span ``:``, so the hex half is already a hex token; the check is explicit
#: so that widening ``_TOKEN_RE`` can never start redacting digests.
_DIGEST_RE = re.compile(r"(?i)^sha(?:1|224|256|384|512):[0-9a-f]+$")


def shannon_entropy(text: str) -> float:
    """Bits per character of ``text`` (0.0 for an empty string)."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


@dataclass
class _Line:
    """One line split as ``prefix + core + suffix``.

    ``content`` is True when the core takes part in matching. A diff header,
    a YAML block opener and an empty core do not.
    """

    prefix: str
    core: str
    suffix: str
    content: bool = True
    #: (start, end, label) spans of ``core`` to replace.
    spans: list[tuple[int, int, str]] = field(default_factory=list)
    #: The diff marker and the line body after it; ``body`` is None for diff
    #: structure (headers, ``\ No newline``), which D-1 never looks at.
    marker: str = ""
    body: str | None = None
    #: D-1 decision: (label, start, end) of ``body`` to replace, or
    #: (label, None, None) for a line redacted in full.
    d1: tuple[str, int | None, int | None] | None = None


def _split_core(body: str) -> tuple[str, str, str]:
    """Split a line body (marker already removed) into prefix, core, suffix.

    Stripped, in this order (ADR-14 §7.3): surrounding whitespace incl. a CR,
    a trailing ``,`` or ``\\``, then quotes at either end of what is left.
    """
    start, end = 0, len(body)
    while start < end and body[start] in " \t\r":
        start += 1
    while end > start and body[end - 1] in " \t\r":
        end -= 1
    while end > start and body[end - 1] in ",\\":
        end -= 1
        while end > start and body[end - 1] in " \t\r":
            end -= 1
    if start < end and body[start] in "\"'":
        start += 1
    if end > start and body[end - 1] in "\"'":
        end -= 1
    return body[:start], body[start:end], body[end:]


def _make_line(marker: str, body: str) -> _Line:
    prefix, core, suffix = _split_core(body)
    content = bool(core) and not _YAML_BLOCK_OPENER_RE.match(core)
    return _Line(marker + prefix, core, suffix, content, marker=marker, body=body)


def _pem_body_shape(body: str | None) -> str | None:
    """P-2 (msg-898): the shape of a plain line next to an unclosed PEM
    marker -- ``"body"``, ``"header"``, ``"blank"``, or None for anything else
    (which ends the run). This only decides the shape; a line that passes is
    redacted whole."""
    if body is None:
        return None
    s = body.lstrip(_PEM_BODY_LEAD)
    while True:
        t = s.rstrip(_PEM_BODY_TRAIL)
        if t.endswith("\\n"):
            t = t[:-2]
        if t == s:
            break
        s = t
    if not s:
        return _SHAPE_BLANK
    if _PEM_HEADER_RE.match(s):
        return _SHAPE_HEADER
    if len(s) <= PEM_BODY_SHAPE_MAX and _PEM_BODY_CORE_RE.match(s):
        return _SHAPE_BODY
    return None


def _kind_of(label: str) -> str:
    """``[REDACTED:<kind>#<n>]`` -> ``<kind>``."""
    return label[len("[REDACTED:") : label.index("#")]


def _is_no_newline(line: str) -> bool:
    return line.rstrip(" \t\r") == _NO_NEWLINE


def is_key_name(name: str) -> bool:
    """True when ``name`` holds a D-1 key name (``max_tokens``, ``API_KEY``,
    ``db_password``, ...): the structured-data form of a key line."""
    return _KEY_NAME_RE.search(name) is not None


def _replace_spans(
    text: str,
    spans: list[tuple[int, int, str]],
    note: Callable[[str, int], None] | None = None,
) -> str:
    """Replace (start, end, label) spans of ``text``; overlapping spans are
    merged and keep the first label. ``note(label, chars)`` is told the size
    of each merged span that is replaced."""
    merged: list[list[object]] = []
    for start, end, label in sorted(spans):
        if merged and start <= merged[-1][1]:  # type: ignore[operator]
            merged[-1][1] = max(merged[-1][1], end)  # type: ignore[call-overload,type-var]
            continue
        merged.append([start, end, label])
    out, last = [], 0
    for start, end, label in merged:
        if note is not None:
            note(label, end - start)  # type: ignore[arg-type,operator]
        out.append(text[last:start])  # type: ignore[misc]
        out.append(label)  # type: ignore[arg-type]
        last = end  # type: ignore[assignment]
    out.append(text[last:])
    return "".join(out)  # type: ignore[arg-type]


def _key_line_value(body: str) -> tuple[int, int] | None:
    """D-1 on a key line: the (start, end) of ``body`` to redact, or None."""
    op = _OPERATOR_RE.search(body)
    if op is None:  # pragma: no cover - a key line always has an operator
        return None
    start = op.end()
    while start < len(body) and body[start] in " \t":
        start += 1
    end = len(body)
    while end > start and body[end - 1] in " \t\r":
        end -= 1
    if end - start < ASSIGNMENT_MIN_LEN:
        return None
    return start, end


class Redactor:
    """Redacts the text of one request; numbers and counts accumulate across
    every text block of that request (``redact`` once per block)."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self._numbers: dict[tuple[str, str], int] = {}
        self._per_kind: Counter[str] = Counter()
        #: P-4 (msg-898): lines touched and characters replaced, per kind,
        #: and the number of unclosed PEM blocks. Sizes only, never text.
        self.extent_lines: Counter[str] = Counter()
        self.extent_chars: Counter[str] = Counter()
        #: Unclosed PEM blocks whose label reached the output. A block whose
        #: every span was superseded (a D-1 line redacted in full, or an
        #: overlapping block's label) replaced nothing, so it is not counted
        #: (PR-gate advisory on #101).
        self.pem_unclosed = 0
        self._line_kinds: set[str] = set()
        self._line_labels: set[str] = set()
        #: This ``redact`` call's unclosed blocks: (label, line indexes).
        self._unclosed: list[tuple[str, list[int]]] = []

    # -- extent ---------------------------------------------------------
    def _note(self, label: str, chars: int) -> None:
        """``chars`` characters of the current line were replaced by
        ``label``."""
        kind = _kind_of(label)
        self.extent_chars[kind] += chars
        self._line_kinds.add(kind)
        self._line_labels.add(label)

    def _end_line(self) -> set[str]:
        """Close the current line; return the labels it was given."""
        for kind in self._line_kinds:
            self.extent_lines[kind] += 1
        labels = self._line_labels
        self._line_kinds = set()
        self._line_labels = set()
        return labels

    def _mark_unclosed(self, lines: list[_Line], span: list[tuple[int, int, int]], secret: str) -> None:
        """``_mark`` an unclosed block and remember it for ``pem_unclosed``."""
        if self._mark(lines, span, secret):
            label = f"[REDACTED:{KIND_PEM}#{self._numbers[(KIND_PEM, secret)]}]"
            self._unclosed.append((label, [k for k, _, _ in span]))

    # -- labels ---------------------------------------------------------
    def _label(self, kind: str, secret: str) -> str:
        """``[REDACTED:<kind>#<n>]``; counts one redaction of ``kind``."""
        key = (kind, secret)
        if key not in self._numbers:
            self._per_kind[kind] += 1
            self._numbers[key] = self._per_kind[kind]
        self.counts[kind] += 1
        return f"[REDACTED:{kind}#{self._numbers[key]}]"

    def label_assignment(self, secret: str) -> str:
        """One ``assignment`` redaction of a whole structured value (a JSON
        member under a D-1 key name); counted like a D-1 run."""
        label = self._label(KIND_ASSIGNMENT, secret)
        self._note(label, len(secret))
        self._end_line()
        return label

    # -- entry point ----------------------------------------------------
    def redact(self, text: str) -> str:
        """Return ``text`` with credentials replaced. Same number of lines."""
        if not text:
            return text
        raw = text.split("\n")
        lines: list[_Line] = []
        sequences: list[list[int]] = []
        plain: list[int] = []
        #: D-1 follows the hunk sequences and the plain text cut at every
        #: hunk or file header: the window never crosses diff structure.
        d1_sequences: list[list[int]] = []
        segment: list[int] = []

        i = 0
        #: M-2 header state: opened by ``diff --git``, kept only by an
        #: unbroken run of extended headers, closed by the first other line.
        #: It can only end too early (that line is then matched: over-
        #: redaction), never run on into prose.
        in_header = False
        while i < len(raw):
            m = _HUNK_RE.match(raw[i])
            if in_header and m is None and not raw[i].startswith(_DIFF_GIT):
                ext = self._ext_header(raw[i])
                if ext is not None:
                    # In no sequence, and neither counted by nor stopping a
                    # D-1 window: ``body`` stays None (M-4).
                    lines.append(ext)
                    i += 1
                    continue
            in_header = False
            if m is None:
                if raw[i].startswith(_FILE_HEADER_PREFIXES):
                    in_header = raw[i].startswith(_DIFF_GIT)
                    lines.append(_Line(raw[i], "", "", content=False))
                    d1_sequences.append(segment)
                    segment = []
                else:
                    lines.append(_make_line("", raw[i]))
                    plain.append(i)
                    segment.append(i)
                i += 1
                continue
            d1_sequences.append(segment)
            segment = []
            # Hunk header. The ``@@ ... @@`` range is structure; the context
            # text git puts after it (the nearest "function" line above the
            # hunk: ``API_KEY = (``, a PEM ``BEGIN`` line, a body row) is
            # content and the first line of *both* sides (H-2, msg-863): it
            # gets the single-line rules and D-1's value rule, a key line
            # there opens a D-1 window into each side (H-2a), and a ``BEGIN``
            # there opens a PEM block into each side (H-2b).
            lines.append(_make_line(raw[i][: m.end()], raw[i][m.end() :]))
            old_left = int(m.group(1)) if m.group(1) is not None else 1
            new_left = int(m.group(2)) if m.group(2) is not None else 1
            before: list[int] = [i]
            after: list[int] = [i]
            i += 1
            while i < len(raw) and (old_left > 0 or new_left > 0):
                line = raw[i]
                # An empty line (or a bare CR) inside a hunk is a context
                # line whose trailing space was stripped by some tool.
                marker = "" if line in ("", "\r") else line[:1]
                if line.startswith("\\"):
                    self._backslash_line(lines, line, i, before, after)
                elif marker in (" ", "") and old_left > 0 and new_left > 0:
                    lines.append(_make_line(marker, line[len(marker):]))
                    before.append(i)
                    after.append(i)
                    old_left -= 1
                    new_left -= 1
                elif marker == "-" and old_left > 0:
                    lines.append(_make_line(marker, line[1:]))
                    before.append(i)
                    old_left -= 1
                elif marker == "+" and new_left > 0:
                    lines.append(_make_line(marker, line[1:]))
                    after.append(i)
                    new_left -= 1
                else:
                    break  # truncated hunk: the line is not part of it
                i += 1
            # The no-newline marker follows the hunk's last line, after its
            # counts are used up: it still belongs to the hunk.
            while i < len(raw) and raw[i].startswith("\\"):
                self._backslash_line(lines, raw[i], i, before, after)
                i += 1
            sequences.extend((before, after))
            d1_sequences.extend((before, after))
        sequences.append(plain)
        d1_sequences.append(segment)

        # Multi-line patterns over each sequence, then D-1, then single-line
        # patterns on what D-1 did not cover.
        for seq in sequences:
            # P-1 bounds unclosed blocks in the plain sequence only; a hunk
            # side keeps running to its end (P-3).
            self._pem_blocks(lines, seq, bounded=seq is plain)
            self._body_runs(lines, seq)
        for seq in d1_sequences:
            self._assignment_windows(lines, seq)
        given: list[set[str]] = []
        for line in lines:
            if line.d1 is not None:
                self._apply_assignment(line)  # applies the multi-line spans too
            else:
                self._apply_multiline(line)
                if line.content:
                    line.core = self._single_line(line.core)
            given.append(self._end_line())
        self.pem_unclosed += sum(1 for label, ks in self._unclosed if any(label in given[k] for k in ks))
        self._unclosed = []

        return "\n".join(line.prefix + line.core + line.suffix for line in lines)

    @staticmethod
    def _backslash_line(lines: list[_Line], line: str, k: int, before: list[int], after: list[int]) -> None:
        """A line starting with a backslash inside a hunk never ends it and
        uses up neither count (H-1, msg-861): git translates the no-newline
        marker in some locales. The exact English marker is structure (M-3,
        M-4); any other one is content of both sides, matched in full (the
        backslash included) and counted by a D-1 window like any line."""
        if _is_no_newline(line):
            lines.append(_Line(line, "", "", content=False))
            return
        lines.append(_make_line("", line))
        before.append(k)
        after.append(k)

    @staticmethod
    def _ext_header(line: str) -> _Line | None:
        """An extended header in header state (M-2), or None when the line
        is not one -- which closes the header state."""
        if _EXT_HEADER_FIXED_RE.match(line):
            return _Line(line, "", "", content=False)
        for prefix in _EXT_HEADER_PATH_PREFIXES:
            if line.startswith(prefix):
                lead, core, suffix = _split_core(line[len(prefix) :])
                return _Line(prefix + lead, core, suffix, content=bool(core))
        return None

    # -- multi-line: PEM ------------------------------------------------
    def _pem_blocks(self, lines: list[_Line], seq: list[int], *, bounded: bool = False) -> None:
        """Mark PEM blocks in one sequence. Begin..end. Unclosed blocks
        (ADR-14 §7.3, bounded by msg-898 PR-1):

        * ``bounded=False`` (a hunk side, P-3): a begin without an end runs
          to the sequence's last line; an end without a begin runs from the
          sequence's first line.
        * ``bounded=True`` (the plain sequence, P-1): a begin without an end
          is redacted from the marker to the end of its line, then down
          while each next line has a PEM body shape (``_pem_body_shape``);
          an end without a begin from the start of its line, then up the
          same way, never into the previous block."""
        idx = [k for k in seq if lines[k].content]
        where = {k: n for n, k in enumerate(seq)}
        pos = 0
        #: First position an orphan end may reach back to: the line after
        #: the previous block, never into it.
        floor = 0
        while pos < len(idx):
            k = idx[pos]
            core = lines[k].core
            begin = _PEM_BEGIN_RE.search(core)
            end = _PEM_END_RE.search(core)
            if begin is not None:
                # Same-line end after the begin (escaped ``\n`` in JSON, or a
                # one-line block): one match inside the line.
                same = _PEM_END_RE.search(core, begin.end())
                if same is not None:
                    self._mark(lines, [(k, begin.start(), same.end())], core[begin.start():same.end()])
                    pos += 1
                    floor = pos
                    continue
                # Look for the end in the following lines of the sequence.
                span = [(k, begin.start(), len(core))]
                closing = None
                for later in range(pos + 1, len(idx)):
                    kk = idx[later]
                    e = _PEM_END_RE.search(lines[kk].core)
                    if e is not None:
                        closing = later
                        span.append((kk, 0, e.end()))
                        break
                    span.append((kk, 0, len(lines[kk].core)))
                if closing is None and bounded:
                    span = [(k, begin.start(), len(core))]
                    span += [(kk, 0, len(lines[kk].core)) for kk in self._run_down(lines, seq, where[k] + 1)]
                    last = where[span[-1][0]]
                    nxt = next((n for n in range(pos + 1, len(idx)) if where[idx[n]] > last), len(idx))
                    self._mark_unclosed(lines, span, "\n".join(lines[a].core for a, _, _ in span))
                    pos = floor = nxt
                    continue
                if closing is None:
                    self._mark_unclosed(lines, span, "\n".join(lines[a].core for a, _, _ in span))
                else:
                    self._mark(lines, span, "\n".join(lines[a].core for a, _, _ in span))
                pos = (closing if closing is not None else len(idx) - 1) + 1
                floor = pos
                continue
            if end is not None:
                if bounded:
                    # Up from this end while the lines have a body shape,
                    # never into the previous block.
                    low = where[idx[floor - 1]] + 1 if floor > 0 else 0
                    above = self._run_up(lines, seq, where[k] - 1, low)
                    span = [(kk, 0, len(lines[kk].core)) for kk in above]
                else:
                    # An end with no begin before it: from the sequence's
                    # first content line -- or the line after the previous
                    # block, whose lines are already redacted -- up to this end.
                    span = [(kk, 0, len(lines[kk].core)) for kk in idx[floor:pos]]
                span.append((k, 0, end.end()))
                self._mark_unclosed(lines, span, "\n".join(lines[a].core for a, _, _ in span))
                floor = pos + 1
            pos += 1

    @staticmethod
    def _run_down(lines: list[_Line], seq: list[int], start: int) -> list[int]:
        """P-1/P-2: the lines of ``seq`` from position ``start`` down that
        have a PEM body shape. One blank line is allowed right after an
        RFC 1421 header; any other blank line, and any other shape, ends the
        run."""
        run: list[int] = []
        after_header = False
        for q in range(start, len(seq)):
            k = seq[q]
            shape = _pem_body_shape(lines[k].body)
            if shape in (_SHAPE_BODY, _SHAPE_HEADER):
                after_header = shape == _SHAPE_HEADER
            elif shape == _SHAPE_BLANK and after_header:
                after_header = False
            else:
                break
            run.append(k)
        return run

    @staticmethod
    def _run_up(lines: list[_Line], seq: list[int], start: int, low: int) -> list[int]:
        """``_run_down`` read upward from position ``start`` to ``low``: a
        blank line is taken only when the line above it is a header."""
        run: list[int] = []
        q = start
        while q >= low:
            shape = _pem_body_shape(lines[seq[q]].body)
            if shape == _SHAPE_BLANK:
                if q - 1 < low or _pem_body_shape(lines[seq[q - 1]].body) != _SHAPE_HEADER:
                    break
            elif shape not in (_SHAPE_BODY, _SHAPE_HEADER):
                break
            run.append(seq[q])
            q -= 1
        run.reverse()
        return run

    # -- multi-line: PEM body run ----------------------------------------
    def _body_runs(self, lines: list[_Line], seq: list[int]) -> None:
        """PEM body lines with neither begin nor end (ADR-14 §7.3, 5th
        REQUEST_CHANGES on #482).

        The run (連なり) is a maximal stretch of 2+ consecutive lines of the
        sequence whose cores are base64 characters only. It is redacted
        whole when every line but the last has the same core length L with
        40 <= L <= 80, the last is no longer than L, and the joined cores
        have entropy >= 4.0. The shape is judged on the whole stretch, not on
        a window of it: otherwise any 40-80 wide line followed by a shorter
        one would pass, and "alphanumeric lines of uneven length" -- a
        must-not-redact case of §7.3 -- could not exist. Lines already inside
        a PEM block are not considered again; a line that fails the shape is
        left to the single-line rules."""
        where = {k: n for n, k in enumerate(seq)}
        cand = [
            k
            for k in seq
            if lines[k].content and not lines[k].spans and _BASE64_LINE_RE.match(lines[k].core)
        ]
        runs: list[list[int]] = []
        for k in cand:
            if runs and where[k] == where[runs[-1][-1]] + 1:
                runs[-1].append(k)
            else:
                runs.append([k])
        for run in runs:
            if len(run) < 2:
                continue
            widths = [len(lines[k].core) for k in run]
            width = widths[0]
            if not BODY_LINE_MIN <= width <= BODY_LINE_MAX:
                continue
            if any(w != width for w in widths[:-1]) or widths[-1] > width:
                continue
            joined = "".join(lines[k].core for k in run)
            if shannon_entropy(joined) >= ENTROPY_THRESHOLD:
                self._mark(lines, [(k, 0, len(lines[k].core)) for k in run], joined)

    def _mark(self, lines: list[_Line], span: list[tuple[int, int, int]], secret: str) -> bool:
        """Mark one ``pem`` block; False when it was already marked."""
        marks = [(k, start, end) for k, start, end in span if end > start]
        number = self._numbers.get((KIND_PEM, secret))
        if number is not None:
            # The other side of the same hunk already marked exactly this
            # (context lines, or a hunk header's context text, belong to both
            # sides): one block, one count.
            label = f"[REDACTED:{KIND_PEM}#{number}]"
            if all((start, end, label) in lines[k].spans for k, start, end in marks):
                return False
        label = self._label(KIND_PEM, secret)
        for k, start, end in marks:
            lines[k].spans.append((start, end, label))
        return True

    def _apply_multiline(self, line: _Line) -> None:
        """Replace the marked spans (merged where two sequences overlap)."""
        if not line.spans:
            return
        line.core = _replace_spans(line.core, line.spans, self._note)
        line.spans = []

    # -- D-1: assignment ------------------------------------------------
    def _assignment_windows(self, lines: list[_Line], seq: list[int]) -> None:
        """Mark D-1 runs in one sequence (rules: module docstring)."""
        runs: list[list[tuple[int, int | None, int | None]]] = []
        current: list[tuple[int, int | None, int | None]] = []
        remaining = 0
        for k in seq:
            body = lines[k].body
            if body is None or not body.strip(" \t\r"):
                continue
            is_key = _KEY_LINE_RE.search(body) is not None
            if remaining > 0:
                current.append((k, None, None))
                remaining = ASSIGNMENT_WINDOW if is_key else remaining - 1
            elif is_key:
                if current:
                    runs.append(current)
                current = []
                value = _key_line_value(body)
                if value is not None:
                    current.append((k, *value))
                remaining = ASSIGNMENT_WINDOW
        if current:
            runs.append(current)

        for run in runs:
            # A context line belongs to both sequences of its hunk: a line the
            # other sequence already marked is not counted again, but a
            # value-only mark is widened to the full line.
            fresh = [
                (k, a, b)
                for k, a, b in run
                if lines[k].d1 is None or (a is None and lines[k].d1[1] is not None)  # type: ignore[index]
            ]
            if not fresh:
                continue
            secret = "\n".join(
                (lines[k].body or "")[a:b] if a is not None else (lines[k].body or "") for k, a, b in run
            )
            label = self._label(KIND_ASSIGNMENT, secret)
            for k, a, b in fresh:
                lines[k].d1 = (label, a, b)

    def _apply_assignment(self, line: _Line) -> None:
        """Render a D-1 line, in place of ``_apply_multiline``. Left of a
        value, the multi-line spans must survive (PR-gate on #96)."""
        label, start, end = line.d1  # type: ignore[misc]
        if start is None:
            # In full: the marker, indentation, outer quotes, a trailing
            # ``,``/``\`` and the CR stay (``_split_core``).
            line.spans = []
            if line.core:
                self._note(label, len(line.core))
                line.core = label
            return
        body = line.body or ""
        # ``core`` starts at this offset of ``body`` (``prefix`` = marker +
        # the stripped lead), so a core span maps to body coordinates.
        offset = len(line.prefix) - len(line.marker)
        left = [
            (offset + a, min(offset + b, start), lab)
            for a, b, lab in line.spans
            if offset + a < start
        ]
        line.spans = []
        line.prefix = line.marker
        self._note(label, end - start)
        line.core = self._single_line(_replace_spans(body[:start], left, self._note)) + label
        line.suffix = body[end:]

    # -- single-line ----------------------------------------------------
    def _single_line(self, core: str) -> str:
        for rx in _KEY_RES:
            core = rx.sub(lambda m: self._replaced(KIND_KEY, m.group(0)), core)
        core = _BEARER_RE.sub(
            lambda m: m.group(0)[: m.start(1) - m.start(0)] + self._replaced(KIND_BEARER, m.group(1)),
            core,
        )
        text = core
        return _TOKEN_RE.sub(lambda m: self._entropy(m, text), core)

    def _entropy(self, m: re.Match[str], text: str) -> str:
        """D-2 (msg-836 §2): 32+ characters and entropy >= 4.0, unless a hash
        shape. There is no letters-and-digits requirement."""
        token = m.group(0)
        if _HEX_RE.match(token) or _UUID_RE.match(token):
            return token  # hex (so commit SHAs) and UUIDs are excluded
        start = m.start()
        while start > 0 and not text[start - 1].isspace() and text[start - 1] not in "\"'=,":
            start -= 1
        if _DIGEST_RE.match(text[start : m.end()]):
            return token  # ``sha256:<digest>``
        if shannon_entropy(token) < ENTROPY_THRESHOLD:
            return token
        return self._replaced(KIND_ENTROPY, token)

    def _replaced(self, kind: str, secret: str) -> str:
        """A single-line match replaced in the current line."""
        label = self._label(kind, secret)
        self._note(label, len(secret))
        return label


def redact(text: str) -> tuple[str, Counter[str]]:
    """Redact one text. Convenience wrapper: ``(text, counts by kind)``."""
    r = Redactor()
    return r.redact(text), r.counts


def format_redaction_extent(redactor: Redactor) -> str | None:
    """The ledger value ``redaction_extent`` (msg-898 P-4): ``kind:lines/chars``
    per kind -- lines touched and characters replaced -- plus
    ``pem_unclosed:<n>`` whenever a ``pem`` was replaced, sorted by name;
    None when nothing was replaced.

    Sizes only -- never the redacted text (ADR-14 §7.3 記録)."""
    items = [
        (k, f"{redactor.extent_lines[k]}/{redactor.extent_chars[k]}")
        for k in redactor.extent_lines
        if redactor.extent_lines[k] > 0
    ]
    if not items:
        return None
    if redactor.extent_lines[KIND_PEM] > 0 or redactor.pem_unclosed:
        items.append(("pem_unclosed", str(redactor.pem_unclosed)))
    return ",".join(f"{k}:{v}" for k, v in sorted(items))


def format_redactions(counts: Counter[str]) -> str | None:
    """The ledger value ``kind:count,...`` sorted by kind; None when empty.

    Counts only -- never the redacted text (ADR-14 §7.3 記録)."""
    items = sorted((k, v) for k, v in counts.items() if v > 0)
    if not items:
        return None
    return ",".join(f"{k}:{v}" for k, v in items)

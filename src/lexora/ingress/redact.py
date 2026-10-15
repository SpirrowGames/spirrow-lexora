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
  marker. Text outside hunks is one plain sequence.
* **When the end (or the start) cannot be seen, redact more.** A PEM begin
  with no end in its sequence is redacted to the sequence's last line; an
  end with no begin from the sequence's first line; a run of PEM body lines
  with neither is recognised by shape (see ``_body_runs``).

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
(``redactions=<kind:count,...>``, see ``format_redactions``).
"""

from __future__ import annotations

import math
import re
from collections import Counter
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
_FILE_HEADER_PREFIXES = ("diff --git", "--- ", "+++ ", "index ")
_PEM_BEGIN_RE = re.compile(r"-----BEGIN [A-Z0-9 ]+-----")
_PEM_END_RE = re.compile(r"-----END [A-Z0-9 ]+-----")
_BASE64_LINE_RE = re.compile(r"^[A-Za-z0-9+/=]+$")
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


def is_key_name(name: str) -> bool:
    """True when ``name`` holds a D-1 key name (``max_tokens``, ``API_KEY``,
    ``db_password``, ...): the structured-data form of a key line."""
    return _KEY_NAME_RE.search(name) is not None


def _replace_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Replace (start, end, label) spans of ``text``; overlapping spans are
    merged and keep the first label."""
    merged: list[list[object]] = []
    for start, end, label in sorted(spans):
        if merged and start <= merged[-1][1]:  # type: ignore[operator]
            merged[-1][1] = max(merged[-1][1], end)  # type: ignore[call-overload,type-var]
            continue
        merged.append([start, end, label])
    out, last = [], 0
    for start, end, label in merged:
        out.append(text[last:start])  # type: ignore[misc]
        out.append(label)  # type: ignore[arg-type]
        last = end  # type: ignore[assignment]
    out.append(text[last:])
    return "".join(out)  # type: ignore[arg-type]


def _key_line_value(body: str) -> tuple[int, int] | None:
    """D-1 on a key line: the (start, end) of ``body`` to redact, or None.

    From just after the line's first ``=`` / ``:`` / ``:=`` and the
    whitespace after it, to the end of the line before trailing whitespace
    and CR. Redacted when that is ``ASSIGNMENT_MIN_LEN``+ characters, quotes
    included (msg-840): quotes, escapes and comments are not looked at, so a
    misjudgement can only redact more."""
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
        return self._label(KIND_ASSIGNMENT, secret)

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
        while i < len(raw):
            m = _HUNK_RE.match(raw[i])
            if m is None:
                if raw[i].startswith(_FILE_HEADER_PREFIXES):
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
            # Hunk header: not content. Consume lines by the header's counts.
            lines.append(_Line(raw[i], "", "", content=False))
            old_left = int(m.group(1)) if m.group(1) is not None else 1
            new_left = int(m.group(2)) if m.group(2) is not None else 1
            before: list[int] = []
            after: list[int] = []
            i += 1
            while i < len(raw) and (old_left > 0 or new_left > 0):
                line = raw[i]
                # An empty line (or a bare CR) inside a hunk is a context
                # line whose trailing space was stripped by some tool.
                marker = "" if line in ("", "\r") else line[:1]
                if line.startswith("\\"):  # "\ No newline at end of file"
                    lines.append(_Line(line, "", "", content=False))
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
            sequences.extend((before, after))
            d1_sequences.extend((before, after))
        sequences.append(plain)
        d1_sequences.append(segment)

        # Multi-line patterns over each sequence, then D-1, then single-line
        # patterns on what D-1 did not cover.
        for seq in sequences:
            self._pem_blocks(lines, seq)
            self._body_runs(lines, seq)
        for seq in d1_sequences:
            self._assignment_windows(lines, seq)
        for line in lines:
            if line.d1 is not None:
                self._apply_assignment(line)  # applies the multi-line spans too
                continue
            self._apply_multiline(line)
            if line.content:
                line.core = self._single_line(line.core)

        return "\n".join(line.prefix + line.core + line.suffix for line in lines)

    # -- multi-line: PEM ------------------------------------------------
    def _pem_blocks(self, lines: list[_Line], seq: list[int]) -> None:
        """Mark PEM blocks in one sequence. Begin..end; a begin without an
        end runs to the sequence's last line; an end without a begin runs
        from the sequence's first line (ADR-14 §7.3)."""
        idx = [k for k in seq if lines[k].content]
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
                self._mark(lines, span, "\n".join(lines[a].core for a, _, _ in span))
                pos = (closing if closing is not None else len(idx) - 1) + 1
                floor = pos
                continue
            if end is not None:
                # An end with no begin before it: from the sequence's first
                # content line -- or the line after the previous block, whose
                # lines are already redacted -- up to this end.
                span = [(kk, 0, len(lines[kk].core)) for kk in idx[floor:pos]]
                span.append((k, 0, end.end()))
                self._mark(lines, span, "\n".join(lines[a].core for a, _, _ in span))
                floor = pos + 1
            pos += 1

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

    def _mark(self, lines: list[_Line], span: list[tuple[int, int, int]], secret: str) -> None:
        label = self._label(KIND_PEM, secret)
        for k, start, end in span:
            if end > start:
                lines[k].spans.append((start, end, label))

    @staticmethod
    def _apply_multiline(line: _Line) -> None:
        """Replace the marked spans (merged where two sequences overlap)."""
        if not line.spans:
            return
        line.core = _replace_spans(line.core, line.spans)
        line.spans = []

    # -- D-1: assignment ------------------------------------------------
    def _assignment_windows(self, lines: list[_Line], seq: list[int]) -> None:
        """Mark D-1 runs in one sequence (msg-834 / msg-836 / msg-840).

        A run starts at a key line outside a window: the key line's value
        (when 8+ characters) plus the next ``ASSIGNMENT_WINDOW`` non-blank
        lines in full. A key line inside the window is one of those lines --
        redacted in full -- and restarts the count, extending the same run.
        Blank lines are left alone and not counted."""
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
        """Render a D-1 line, in place of ``_apply_multiline``.

        In full: the whole core becomes the label, which covers any PEM span
        on it. Value only: the value becomes the label, and the text left of
        it keeps every PEM / body-run span the multi-line pass marked there
        (clipped at the value's start; the part past it is inside the value)
        before the single-line rules run on it. Rebuilding the left side from
        the raw body without those spans would put a PEM block that precedes
        the operator on the same line back into the output (PR-gate on #96).
        """
        label, start, end = line.d1  # type: ignore[misc]
        if start is None:
            # In full: the marker, indentation, outer quotes, a trailing
            # ``,``/``\`` and the CR stay (``_split_core``).
            line.spans = []
            if line.core:
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
        line.core = self._single_line(_replace_spans(body[:start], left)) + label
        line.suffix = body[end:]

    # -- single-line ----------------------------------------------------
    def _single_line(self, core: str) -> str:
        for rx in _KEY_RES:
            core = rx.sub(lambda m: self._label(KIND_KEY, m.group(0)), core)
        core = _BEARER_RE.sub(
            lambda m: m.group(0)[: m.start(1) - m.start(0)] + self._label(KIND_BEARER, m.group(1)),
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
        return self._label(KIND_ENTROPY, token)


def redact(text: str) -> tuple[str, Counter[str]]:
    """Redact one text. Convenience wrapper: ``(text, counts by kind)``."""
    r = Redactor()
    return r.redact(text), r.counts


def format_redactions(counts: Counter[str]) -> str | None:
    """The ledger value ``kind:count,...`` sorted by kind; None when empty.

    Counts only -- never the redacted text (ADR-14 §7.3 記録)."""
    items = sorted((k, v) for k, v in counts.items() if v > 0)
    if not items:
        return None
    return ",".join(f"{k}:{v}" for k, v in items)

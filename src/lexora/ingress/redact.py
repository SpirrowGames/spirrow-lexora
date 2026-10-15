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

What gets redacted (kinds): ``pem`` (PEM blocks and PEM body runs), ``key``
(known key prefixes), ``assignment`` (a credential-named key assigned a value
of 8+ characters), ``bearer`` (Bearer tokens) and ``entropy`` (strings of
Shannon entropy >= 4.0, hash shapes excluded). Each replaced span becomes
``[REDACTED:<kind>#<n>]``; ``n`` numbers distinct secrets per kind within one
request, so the same secret seen twice keeps one number.

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
#: Shortest token the entropy rule looks at.
ENTROPY_MIN_LEN = 20
#: Shortest value the assignment rule redacts (ADR-14 §7.3: 8 文字以上).
ASSIGNMENT_MIN_LEN = 8

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
#: A credential-named key assigned a value. The value is redacted when it is
#: quoted, or unquoted and mixes letters and digits (so ``token = token_value``
#: in code is left alone, ``API_KEY=abc123def456`` is not). ``${...}`` and
#: other references are not values.
_ASSIGN_RE = re.compile(
    r"(?i)([A-Za-z0-9_.\-]*(?:password|passwd|pwd|secret|token|api[_\-]?key|apikey|"
    r"access[_\-]?key|private[_\-]?key|credential|auth)[A-Za-z0-9_.\-]*)"
    r"[\"']?\s*(?::=|=|:)\s*"
    # The closing quote may already sit in the line's suffix (core stripping
    # removes a quote at the end of the core), hence ``\2|$``.
    r"(?:(\"|')([^\"'\s]{%d,})(?:\2|$)|([^\s\"',;()\[\]{}$]{%d,})(?![(\w]))"
    % (ASSIGNMENT_MIN_LEN, ASSIGNMENT_MIN_LEN)
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{%d,}" % ENTROPY_MIN_LEN)
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


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
    return _Line(marker + prefix, core, suffix, content)


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

    # -- entry point ----------------------------------------------------
    def redact(self, text: str) -> str:
        """Return ``text`` with credentials replaced. Same number of lines."""
        if not text:
            return text
        raw = text.split("\n")
        lines: list[_Line] = []
        sequences: list[list[int]] = []
        plain: list[int] = []

        i = 0
        while i < len(raw):
            m = _HUNK_RE.match(raw[i])
            if m is None:
                if raw[i].startswith(_FILE_HEADER_PREFIXES):
                    lines.append(_Line(raw[i], "", "", content=False))
                else:
                    lines.append(_make_line("", raw[i]))
                    plain.append(i)
                i += 1
                continue
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
        sequences.append(plain)

        # Multi-line patterns over each sequence, then single-line patterns.
        for seq in sequences:
            self._pem_blocks(lines, seq)
            self._body_runs(lines, seq)
        for line in lines:
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
        merged: list[list[object]] = []
        for start, end, label in sorted(line.spans):
            if merged and start <= merged[-1][1]:  # type: ignore[operator]
                merged[-1][1] = max(merged[-1][1], end)  # type: ignore[call-overload,type-var]
                continue
            merged.append([start, end, label])
        out, last = [], 0
        for start, end, label in merged:
            out.append(line.core[last:start])  # type: ignore[misc]
            out.append(label)  # type: ignore[arg-type]
            last = end  # type: ignore[assignment]
        out.append(line.core[last:])
        line.core = "".join(out)  # type: ignore[arg-type]
        line.spans = []

    # -- single-line ----------------------------------------------------
    def _single_line(self, core: str) -> str:
        for rx in _KEY_RES:
            core = rx.sub(lambda m: self._label(KIND_KEY, m.group(0)), core)
        core = _BEARER_RE.sub(
            lambda m: m.group(0)[: m.start(1) - m.start(0)] + self._label(KIND_BEARER, m.group(1)),
            core,
        )
        core = _ASSIGN_RE.sub(self._assignment, core)
        core = _TOKEN_RE.sub(self._entropy, core)
        return core

    def _assignment(self, m: re.Match[str]) -> str:
        quoted, bare = m.group(3), m.group(4)
        value = quoted if quoted is not None else bare
        if quoted is None and not (re.search(r"[A-Za-z]", value) and re.search(r"\d", value)):
            return m.group(0)
        if value.startswith("[REDACTED:"):
            return m.group(0)
        start = m.start(3) if quoted is not None else m.start(4)
        head = m.group(0)[: start - m.start(0)]
        tail = m.group(0)[start - m.start(0) + len(value):]
        return head + self._label(KIND_ASSIGNMENT, value) + tail

    def _entropy(self, m: re.Match[str]) -> str:
        token = m.group(0)
        if _HEX_RE.match(token) or _UUID_RE.match(token):
            return token  # hash shapes are excluded (ADR-14 §7.3)
        if not (re.search(r"[A-Za-z]", token) and re.search(r"\d", token)):
            return token
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

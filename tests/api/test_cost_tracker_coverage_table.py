"""The `backend | fills_usage_sink | why` table in `get_cost_tracker.__doc__`, checked.

That table went stale three times at the same site (T-claims-need-detectors
msg-204 §1 rows 3 and 5, msg-1107 §0): it said five backends while the config
allowed seven. This file turns it from prose into an executed claim.

What it detects:

- a backend type added to or removed from ``BackendSettings.type`` while the
  table or ``TYPE_TO_CLASS`` below is not updated to match;
- a table cell (``yes`` / ``no``) that disagrees with the
  ``fills_usage_sink`` declaration of the class it is mapped to;
- a duplicated backend row (same value or not), a second-column value other
  than ``yes`` / ``no``, a line inside the table that is neither a row nor a
  continuation of one, a missing or repeated header, and an empty table.

What it does NOT detect:

- ``TYPE_TO_CLASS`` binding a name to the wrong class. Swapping ``codex`` and
  ``fallback`` passes, because both declare True. Whether each name maps to
  the right class is left to manual review.
- which class the factory actually builds for a type (the construction-site
  enumeration is out of scope, msg-204 §5).
- anything in the ``why`` column. The second column is the class's one-bit
  declaration, not what each stream does: a ``no`` class can still fill the
  sink on some stream (``vllm`` chat), and only ``why`` says so, unchecked.
"""

from __future__ import annotations

import re
import typing

import pytest

from lexora.api.routes import get_cost_tracker
from lexora.backends.anthropic import AnthropicBackend
from lexora.backends.base import Backend
from lexora.backends.claude_code import ClaudeCodeBackend
from lexora.backends.codex import CodexBackend
from lexora.backends.fallback import FallbackBackend
from lexora.backends.gemini import GeminiBackend
from lexora.backends.openai_compatible import OpenAICompatibleBackend
from lexora.backends.vllm import VLLMBackend
from lexora.config import BackendSettings

#: Written by hand on purpose: the factory is an elif chain with no mapping to
#: read. Only its keys are checked (against the ``Literal``); its values are
#: not, see the module docstring.
TYPE_TO_CLASS: dict[str, type[Backend]] = {
    "vllm": VLLMBackend,
    "openai_compatible": OpenAICompatibleBackend,
    "anthropic": AnthropicBackend,
    "claude_code": ClaudeCodeBackend,
    "gemini": GeminiBackend,
    "codex": CodexBackend,
    "fallback": FallbackBackend,
}

_HEADER = re.compile(r"^(?P<indent> *)backend +fills_usage_sink +why *$")
_NAME = re.compile(r"[a-z_]+")


class TableError(AssertionError):
    """The table is missing or malformed."""


def parse_coverage_table(doc: str) -> dict[str, bool]:
    """Parse the table into ``{backend: fills}``, failing on any deviation.

    Bounds: from the one header line to the first blank line after it.
    Inside, every line is either
      - a head row: at the header's indent, ``<name> <yes|no> <why...>`` with
        ``<name>`` matching ``[a-z_]+``; or
      - a continuation: indented deeper than the header's `fills_usage_sink`
        column, with an empty name column, after at least one head row.
    Anything else fails. Duplicates are rejected before building the dict.
    """
    lines = doc.splitlines()
    header_at = [i for i, line in enumerate(lines) if _HEADER.match(line)]
    if len(header_at) != 1:
        raise TableError(f"expected exactly one table header, found {len(header_at)}")
    start = header_at[0]
    header = lines[start]
    indent = len(_HEADER.match(header).group("indent"))  # type: ignore[union-attr]
    fills_col = header.index("fills_usage_sink")

    rows: list[tuple[str, bool]] = []
    for lineno, line in enumerate(lines[start + 1 :], start=start + 2):
        if not line.strip():
            break
        lead = len(line) - len(line.lstrip(" "))
        if lead == indent:
            parts = line.split()
            if len(parts) < 3:
                raise TableError(f"doc line {lineno}: head row needs name, value, why: {line!r}")
            name, value = parts[0], parts[1]
            if not _NAME.fullmatch(name):
                raise TableError(f"doc line {lineno}: bad backend name {name!r}")
            if value not in ("yes", "no"):
                raise TableError(f"doc line {lineno}: cell for {name!r} is {value!r}, not yes/no")
            rows.append((name, value == "yes"))
        elif lead > fills_col:
            if not rows:
                raise TableError(f"doc line {lineno}: continuation line before any row")
        else:
            raise TableError(f"doc line {lineno}: neither a row nor a continuation: {line!r}")

    if not rows:
        raise TableError("the table has no rows")
    names = [name for name, _ in rows]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise TableError(f"duplicated rows: {duplicated}")
    return dict(rows)


def configured_types() -> set[str]:
    return set(typing.get_args(BackendSettings.model_fields["type"].annotation))


def check(doc: str, types: set[str], mapping: dict[str, type[Backend]]) -> None:
    """Every assertion the detector makes, over explicit inputs."""
    table = parse_coverage_table(doc)
    assert set(mapping) == types, "TYPE_TO_CLASS keys differ from BackendSettings.type"
    assert set(table) == types, (
        f"table names {sorted(table)} differ from BackendSettings.type {sorted(types)}"
    )
    for name, fills in table.items():
        declared = mapping[name].fills_usage_sink
        assert declared is True or declared is False, name
        assert declared is fills, (
            f"{name}: table says {'yes' if fills else 'no'}, "
            f"{mapping[name].__name__}.fills_usage_sink is {declared}"
        )


def _doc() -> str:
    doc = get_cost_tracker.__doc__
    assert doc is not None
    return doc


def test_the_coverage_table_agrees_with_the_tree() -> None:
    check(_doc(), configured_types(), TYPE_TO_CLASS)


# --- the detector, shown to fail --------------------------------------------
#
# Each case below is a mutation from msg-1109 A-5 applied to a copy of the real
# inputs. They live here, not only in the PR body, so a later loosening of the
# parser cannot quietly stop catching them.


def _replace_once(doc: str, old: str, new: str) -> str:
    assert doc.count(old) == 1, old
    return doc.replace(old, new)


def _row(doc: str, name: str) -> str:
    """The head line of `name`'s row."""
    (line,) = [ln for ln in doc.splitlines() if ln.split()[:1] == [name] and ("yes" in ln or "no" in ln)]
    return line


def test_mutation_i_a_deleted_row_fails() -> None:
    doc = _replace_once(_doc(), _row(_doc(), "codex") + "\n", "")
    with pytest.raises(AssertionError):
        check(doc, configured_types(), TYPE_TO_CLASS)


def test_mutation_ii_a_flipped_cell_fails() -> None:
    line = _row(_doc(), "gemini")
    doc = _replace_once(_doc(), line, line.replace(" yes ", " no  ", 1))
    with pytest.raises(AssertionError, match="gemini"):
        check(doc, configured_types(), TYPE_TO_CLASS)


def test_mutation_iii_a_changed_declaration_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CodexBackend, "fills_usage_sink", False)
    with pytest.raises(AssertionError, match="codex"):
        check(_doc(), configured_types(), TYPE_TO_CLASS)


def test_mutation_iv_a_new_configured_type_fails() -> None:
    with pytest.raises(AssertionError):
        check(_doc(), configured_types() | {"new_backend"}, TYPE_TO_CLASS)


def test_mutation_v_a_contradicting_duplicate_row_fails() -> None:
    line = _row(_doc(), "codex")
    dup = line.replace(" yes ", " no  ", 1)
    doc = _replace_once(_doc(), line, line + "\n" + dup)
    with pytest.raises(TableError, match="duplicated"):
        check(doc, configured_types(), TYPE_TO_CLASS)


def test_mutation_vi_a_value_other_than_yes_no_fails() -> None:
    line = _row(_doc(), "anthropic")
    doc = _replace_once(_doc(), line, line.replace(" yes  ", " maybe", 1))
    with pytest.raises(TableError, match="maybe"):
        check(doc, configured_types(), TYPE_TO_CLASS)


def test_mutation_vii_a_line_of_neither_shape_fails() -> None:
    line = _row(_doc(), "gemini")
    indent = line[: len(line) - len(line.lstrip(" "))]
    doc = _replace_once(_doc(), line, line + "\n" + indent + "   stray text")
    with pytest.raises(TableError, match="neither"):
        check(doc, configured_types(), TYPE_TO_CLASS)


def test_a_missing_header_fails() -> None:
    doc = _doc().replace("fills_usage_sink why", "fills_usage_sink  why  extra")
    with pytest.raises(TableError, match="found 0"):
        parse_coverage_table(doc)


def test_a_repeated_header_fails() -> None:
    header = next(ln for ln in _doc().splitlines() if _HEADER.match(ln))
    with pytest.raises(TableError, match="found 2"):
        parse_coverage_table(_doc() + "\n" + header + "\n")


def test_an_empty_table_fails() -> None:
    header = next(ln for ln in _doc().splitlines() if _HEADER.match(ln))
    with pytest.raises(TableError, match="no rows"):
        parse_coverage_table(header + "\n\n")


def test_a_continuation_before_any_row_fails() -> None:
    header = next(ln for ln in _doc().splitlines() if _HEADER.match(ln))
    with pytest.raises(TableError, match="before any row"):
        parse_coverage_table(header + "\n" + " " * 60 + "orphan\n")


def test_mutation_viii_the_old_header_fails() -> None:
    """The header used to read `fills the sink`, which overstated a `no` cell
    (msg-1112): restoring it must not parse."""
    doc = _replace_once(_doc(), "fills_usage_sink why", "fills the sink   why")
    with pytest.raises(TableError, match="found 0"):
        parse_coverage_table(doc)

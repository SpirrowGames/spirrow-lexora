"""``parse_trace_id`` (T-cost-row-trace-id msg-627 §2a, amended by msg-629)."""

from unittest.mock import MagicMock

import pytest

from lexora.services import trace
from lexora.services.trace import parse_trace_id

VALID = "01J9Z3K4M5N6P7Q8R9S0T1V2W3"


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(VALID.lower(), id="lowercase"),
        pytest.param(VALID[:-1], id="25-chars"),
        pytest.param(VALID + "0", id="27-chars"),
        pytest.param("01J9Z3K4M5N6P7Q8R9S0T1V2WI", id="contains-I"),
        pytest.param("01J9Z3K4M5N6P7Q8R9S0T1V2WL", id="contains-L"),
        pytest.param("01J9Z3K4M5N6P7Q8R9S0T1V2WO", id="contains-O"),
        pytest.param("01J9Z3K4M5N6P7Q8R9S0T1V2WU", id="contains-U"),
        pytest.param("8" + VALID[1:], id="first-char-8-over-128-bits"),
        pytest.param("", id="empty"),
        pytest.param(f" {VALID} ", id="surrounding-whitespace"),
        pytest.param(VALID + "\n", id="trailing-newline"),
        pytest.param("0" * 100_000, id="very-long"),
    ],
)
def test_invalid_values_are_rejected_and_logged_by_length_only(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(trace, "logger", logger)
    assert parse_trace_id(raw) is None
    logger.warning.assert_called_once_with("trace_id_rejected", length=len(raw))


def test_a_canonical_ulid_is_returned_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(trace, "logger", logger)
    assert parse_trace_id(VALID) == VALID
    assert parse_trace_id("7ZZZZZZZZZZZZZZZZZZZZZZZZZ") == "7ZZZZZZZZZZZZZZZZZZZZZZZZZ"  # max ULID
    logger.warning.assert_not_called()


def test_absent_header_is_none_without_a_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """msg-628 blocking: ``None`` must not reach ``len(raw)``."""
    logger = MagicMock()
    monkeypatch.setattr(trace, "logger", logger)
    assert parse_trace_id(None) is None
    logger.warning.assert_not_called()

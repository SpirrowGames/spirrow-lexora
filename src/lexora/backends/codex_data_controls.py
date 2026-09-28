"""Data-controls attestation for the codex backend (A-15-2b).

T-naysayer-codex-backend msg-535 (A-15-2b, fail-closed on expiry), with
the hot reload and the stateful daily pre-expiry warning that msg-535
proposed, kept by the human's decision after msg-536.

Why this exists: with an API key, "the provider does not train on our
prompts" is a property of the endpoint. With the subscription login that
``codex exec`` uses, it is a setting on the ChatGPT account, which a person
can change or which a re-login under another account can lose. So a human
checks the setting (``deploy/RUNBOOK.md`` section 8) and records WHEN in a
small file; codex may run only while that record is younger than
``DATA_CONTROLS_TTL``.

The file (path: ``codex.data_controls_file``) is YAML, one mapping::

    data_controls_verified_at: "2026-09-29T01:00:00Z"

Fail-closed: a missing file, a file that does not parse, a missing or
naive timestamp, a timestamp in the future (beyond ``FUTURE_SKEW``), and a
timestamp older than ``DATA_CONTROLS_TTL`` all read as NOT verified. There
is no configuration that skips the check and no way to lengthen the TTL
from configuration: both are module constants.

Hot reload: ``state()`` stats the file on every call and re-reads it when
its ``(mtime_ns, size)`` changed, so updating the timestamp takes effect on
the next request (or status read) with no restart. Only this file is hot;
the rest of Lexora's configuration is still read once at start-up.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from lexora.utils.logging import get_logger

logger = get_logger(__name__)

#: How long one human check of the account's data controls stays valid.
DATA_CONTROLS_TTL = timedelta(days=30)
#: How long before expiry the daily warning starts (msg-535).
DATA_CONTROLS_WARN_BEFORE = timedelta(days=7)
#: A recorded time this far ahead of the clock is refused (a typo such as
#: the wrong year would otherwise extend the validity).
FUTURE_SKEW = timedelta(minutes=10)

#: ``codex_disabled_reason`` / fallback ``reason`` when the check fails.
REASON_DATA_CONTROLS_UNVERIFIED = "data_controls_unverified"
#: The key read from the file.
VERIFIED_AT_KEY = "data_controls_verified_at"
#: Where the notices point the operator.
RUNBOOK_POINTER = "deploy/RUNBOOK.md#8"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DataControlsState:
    """One evaluation of the data-controls record.

    ``detail`` is ``None`` when the record is valid; otherwise one of
    ``missing`` / ``unreadable`` / ``invalid`` / ``future`` / ``expired``
    (the operator-facing sub-reason; the gate reason is always
    ``REASON_DATA_CONTROLS_UNVERIFIED``).
    """

    detail: str | None
    verified_at: datetime | None = None
    expires_at: datetime | None = None

    @property
    def valid(self) -> bool:
        return self.detail is None

    def warn_due(self, now: datetime) -> bool:
        """Valid, and inside the pre-expiry warning window."""
        return self.valid and self.expires_at is not None and now >= self.expires_at - DATA_CONTROLS_WARN_BEFORE

    def days_left(self, now: datetime) -> int:
        """Whole days until expiry, rounded up (0 once expired)."""
        if self.expires_at is None or now >= self.expires_at:
            return 0
        remaining = self.expires_at - now
        return remaining.days + (1 if remaining % timedelta(days=1) else 0)


def parse_verified_at(raw: Any) -> datetime | None:
    """A timezone-aware ``datetime`` from the YAML value, else ``None``.

    Accepts an ISO 8601 string (``Z`` allowed) or a YAML timestamp. A
    timestamp without a UTC offset is refused: "when" must be unambiguous.
    """
    if isinstance(raw, datetime):
        value = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


class DataControls:
    """The hot-reloaded data-controls record. See the module docstring."""

    def __init__(self, path: str | Path, clock: Callable[[], datetime] = _utcnow) -> None:
        self.path = Path(path)
        self._clock = clock
        #: ``(inode, mtime_ns, size)`` of the file last read, or ``None``
        #: when the last stat found no file / failed. The inode catches an
        #: atomic replace (write + rename) that keeps mtime and size.
        self._key: tuple[int, int, int] | None = None
        self._verified_at: datetime | None = None
        #: ``None`` = the last read gave a valid timestamp; "unset" only
        #: before the first read (so the first "missing" is logged).
        self._read_detail: str | None = "unset"

    def _reload_if_changed(self) -> None:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            if self._key is not None or self._read_detail != "missing":
                logger.warning("codex_data_controls_missing", path=str(self.path))
            self._key, self._verified_at, self._read_detail = None, None, "missing"
            return
        except OSError as exc:
            self._key, self._verified_at, self._read_detail = None, None, "unreadable"
            logger.warning("codex_data_controls_unreadable", path=str(self.path), error=type(exc).__name__)
            return
        key = (st.st_ino, st.st_mtime_ns, st.st_size)
        if key == self._key:
            return
        self._key = key
        try:
            loaded = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            self._verified_at, self._read_detail = None, "unreadable"
            logger.warning("codex_data_controls_unreadable", path=str(self.path), error=type(exc).__name__)
            return
        raw = loaded.get(VERIFIED_AT_KEY) if isinstance(loaded, dict) else None
        verified_at = parse_verified_at(raw)
        if verified_at is None:
            self._verified_at, self._read_detail = None, "invalid"
            logger.warning("codex_data_controls_invalid", path=str(self.path), key=VERIFIED_AT_KEY)
            return
        self._verified_at, self._read_detail = verified_at, None
        logger.info(
            "codex_data_controls_reloaded",
            path=str(self.path),
            verified_at=verified_at.isoformat(),
            expires_at=(verified_at + DATA_CONTROLS_TTL).isoformat(),
        )

    def state(self) -> DataControlsState:
        """Re-read the file if it changed, then judge it against the clock."""
        self._reload_if_changed()
        if self._read_detail is not None or self._verified_at is None:
            detail = self._read_detail if self._read_detail not in (None, "unset") else "invalid"
            return DataControlsState(detail)
        verified_at = self._verified_at
        expires_at = verified_at + DATA_CONTROLS_TTL
        now = self._clock()
        if verified_at > now + FUTURE_SKEW:
            return DataControlsState("future", verified_at, expires_at)
        if now >= expires_at:
            return DataControlsState("expired", verified_at, expires_at)
        return DataControlsState(None, verified_at, expires_at)

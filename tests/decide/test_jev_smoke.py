"""Smoke test against the REAL Jev (TypeSafe) API. Metered.

Skipped by default (``addopts = "-m 'not smoke'"`` in pyproject.toml) and
never run in CI. Run explicitly, with ``TYPESAFE_API_KEY`` set:

    uv run --extra dev pytest -m smoke tests/decide/test_jev_smoke.py

One problem, one primitive (``noul``). The point is to confirm the wire
shape assumed in :mod:`lexora.decide.jev_client` (marked UNVERIFIED there)
against the live API before any deployment sets ``primary = "jev"``.
"""

from __future__ import annotations

import os

import pytest

from lexora.decide.contract import QuestionSpec
from lexora.decide.providers import JevProvider

pytestmark = pytest.mark.smoke


async def test_real_noul_roundtrip() -> None:
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        pytest.skip("TYPESAFE_API_KEY not set")
    provider = JevProvider(api_key, timeout_ms=30_000)
    answers = await provider.evaluate(
        state="The customer wrote: 'This is the best purchase I have made all year.'",
        questions={
            "positive": QuestionSpec(
                type="noul", instructions="Is the customer's sentiment positive?"
            )
        },
    )
    assert 0.0 <= answers["positive"]["noul"] <= 1.0

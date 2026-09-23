"""Smoke test against the REAL Jev (TypeSafe) API. Metered.

Skipped by default (``addopts = "-m 'not smoke'"`` in pyproject.toml) and
never run in CI. Run explicitly, with ``TYPESAFE_API_KEY`` set:

    uv run --extra dev pytest -m smoke tests/decide/test_jev_smoke.py

One request, one ``noul`` question, one billed call. The point is to
confirm the wire format in :mod:`lexora.decide.jev_client` (taken from
Fermi's quote of docs.typesafe.ai/api.md, T-decide-jev-provider msg-336)
against the live API before any deployment sets ``primary = "jev"``.
Running it is a billing decision and needs sign-off (msg-336 / msg-339
step 3).
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
    result = await provider.evaluate(
        state="The customer wrote: 'This is the best purchase I have made all year.'",
        questions={
            "positive": QuestionSpec(
                type="noul", instructions="Is the customer's sentiment positive?"
            )
        },
    )
    assert 0.0 <= result.answers["positive"]["noul"] <= 1.0
    # The version that actually served the call and its usage: the fields
    # the decision log records (Bohr msg-339 #2 / #5).
    assert result.upstream is not None
    assert result.upstream.model
    assert result.upstream.input_tokens is not None

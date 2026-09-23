"""T-ledger-gemini-thinking-tokens D-1 / D-2: the ledger side.

D-1: two nullable columns, ``tokens_thinking`` and ``tokens_cached_input``,
added by the same ``PRAGMA``-guarded ``ALTER`` path as ``tier`` /
``pricing_known``, with no exception swallowed. Rows written before the
migration keep NULL.

D-2: a pricing entry may carry a ``cached_input`` rate and a ``tiers`` list.
The formula is::

    (prompt - cached) * in + cached * cache_in + (candidates + thinking) * out

and the tier is chosen from ``prompt`` (cached part included). An entry
without either extension prices exactly as before.

The rates used below are TEST rates, picked so each term is separately
visible in the result. They are not Google's prices and say nothing about
them; ``gemini-3.1-pro-preview`` has no entry in ``DEFAULT_PRICING`` (see the
NOTE there for why).
"""

import sqlite3
from pathlib import Path

import pytest

from lexora.services.cost_tracker import DEFAULT_PRICING, CostTracker

MODEL = "tiered-test-model"

# Base rates, and a second tier above 1000 prompt tokens. Each rate is a
# distinct power of ten per MTok so every term of the formula lands in its own
# decimal place and a wrong term cannot be hidden by another.
TIERED: dict[str, dict] = {
    MODEL: {
        "input": 1.0,
        "output": 100.0,
        "cached_input": 0.01,
        "tiers": [
            {
                "above_prompt_tokens": 1000,
                "input": 2.0,
                "output": 200.0,
                "cached_input": 0.02,
            }
        ],
    },
    "flat-test-model": {"input": 3.0, "output": 15.0},
}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "costs.db"


def _cols(db_path: Path) -> dict[str, str]:
    with sqlite3.connect(db_path) as conn:
        return {r[1]: r[2] for r in conn.execute("PRAGMA table_info(request_costs)")}


class TestSchema:
    """D-1."""

    def test_new_db_has_both_columns_as_nullable_integers(self, db_path: Path) -> None:
        CostTracker(db_path=db_path)
        cols = _cols(db_path)
        assert cols["tokens_thinking"] == "INTEGER"
        assert cols["tokens_cached_input"] == "INTEGER"

    def test_pre_migration_db_is_migrated_and_old_rows_stay_null(
        self, db_path: Path
    ) -> None:
        """A DB from before this change: the row it holds must survive as NULL."""
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """CREATE TABLE request_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL, model TEXT NOT NULL, backend TEXT,
                    endpoint TEXT NOT NULL, user_id TEXT,
                    tokens_input INTEGER NOT NULL DEFAULT 0,
                    tokens_output INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    duration_seconds REAL, success INTEGER NOT NULL DEFAULT 1,
                    tier TEXT, pricing_known INTEGER)"""
            )
            conn.execute(
                """INSERT INTO request_costs
                   (timestamp, model, endpoint, tokens_input, tokens_output,
                    cost_usd, pricing_known)
                   VALUES ('2026-09-18T00:00:00+00:00', 'gemini-3.1-pro-preview',
                           '/v1/chat/completions', 500000, 3, 0.0, 0)"""
            )

        CostTracker(db_path=db_path)

        assert {"tokens_thinking", "tokens_cached_input"} <= set(_cols(db_path))
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT tokens_thinking, tokens_cached_input, cost_usd, pricing_known "
                "FROM request_costs"
            ).fetchone()
        # Not back-filled (out of scope: the counts cannot be recovered), and
        # the old unpriced row is still unpriced -- not re-priced to a guess.
        assert row == (None, None, 0.0, 0)

    def test_opening_twice_is_idempotent(self, db_path: Path) -> None:
        CostTracker(db_path=db_path)
        CostTracker(db_path=db_path)
        assert "tokens_thinking" in _cols(db_path)

    def test_record_writes_both_values(self, db_path: Path) -> None:
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        tracker.record(
            model=MODEL,
            endpoint="/v1/chat/completions",
            tokens_input=100,
            tokens_output=0,
            tokens_thinking=16,
            tokens_cached_input=40,
        )
        row = tracker.get_recent(1)[0]
        assert (row["tokens_thinking"], row["tokens_cached_input"]) == (16, 40)
        # The meaning of tokens_output is untouched: thinking is NOT folded in.
        assert row["tokens_output"] == 0

    def test_record_without_them_writes_null(self, db_path: Path) -> None:
        """A backend that does not measure these: NULL, never 0."""
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        tracker.record(
            model="flat-test-model",
            endpoint="/v1/chat/completions",
            tokens_input=10,
            tokens_output=5,
        )
        row = tracker.get_recent(1)[0]
        assert (row["tokens_thinking"], row["tokens_cached_input"]) == (None, None)

    def test_get_costs_sums_both(self, db_path: Path) -> None:
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        for thinking, cached in [(10, 1), (20, 2), (None, None)]:
            tracker.record(
                model=MODEL,
                endpoint="/v1/chat/completions",
                tokens_input=100,
                tokens_output=1,
                tokens_thinking=thinking,
                tokens_cached_input=cached,
            )
        summary = tracker.get_costs(period="all")["summary"]
        assert summary["total_tokens_thinking"] == 30
        assert summary["total_tokens_cached_input"] == 3


class TestFormula:
    """D-2."""

    def test_flat_entry_prices_exactly_as_before(self, db_path: Path) -> None:
        """No extensions, no new counts: the pre-D-2 ``in*in + out*out``."""
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        cost, known = tracker.calculate_cost("flat-test-model", 1_000_000, 1_000_000)
        assert (cost, known) == (18.0, True)

    def test_every_existing_default_entry_is_unchanged(self, db_path: Path) -> None:
        """Each shipped entry, priced through the new code, against the old formula."""
        tracker = CostTracker(db_path=db_path)
        for model, prices in DEFAULT_PRICING.items():
            expected = round(
                (1234 / 1_000_000) * prices["input"]
                + (567 / 1_000_000) * prices["output"],
                8,
            )
            assert tracker.calculate_cost(model, 1234, 567) == (expected, True), model

    def test_thinking_is_charged_at_the_output_rate(self, db_path: Path) -> None:
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        without, _ = tracker.calculate_cost(MODEL, 0, 1_000_000)
        with_thinking, _ = tracker.calculate_cost(
            MODEL, 0, 1_000_000, tokens_thinking=1_000_000
        )
        assert without == 100.0
        assert with_thinking == 200.0

    def test_thinking_alone_is_billed_when_output_is_zero(self, db_path: Path) -> None:
        """The preflight shape: output 0, the whole budget spent thinking."""
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        cost, _ = tracker.calculate_cost(MODEL, 0, 0, tokens_thinking=16)
        assert cost == round(16 / 1_000_000 * 100.0, 8)

    def test_cached_is_moved_to_the_cache_rate_not_added(self, db_path: Path) -> None:
        """F-2: ``promptTokenCount`` already contains the cached part.

        1000 prompt tokens of which 400 cached: 600 at 1.0 + 400 at 0.01.
        Adding the 400 on top (1000 at 1.0 + 400 at 0.01) is the double
        charge this pins against.
        """
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        cost, _ = tracker.calculate_cost(MODEL, 1000, 0, tokens_cached_input=400)
        assert cost == round((600 * 1.0 + 400 * 0.01) / 1_000_000, 8)

    def test_full_formula(self, db_path: Path) -> None:
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        cost, _ = tracker.calculate_cost(
            MODEL, 900, 10, tokens_thinking=30, tokens_cached_input=300
        )
        expected = (600 * 1.0 + 300 * 0.01 + (10 + 30) * 100.0) / 1_000_000
        assert cost == round(expected, 8)

    def test_missing_cache_rate_falls_back_to_input(self, db_path: Path) -> None:
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        plain, _ = tracker.calculate_cost("flat-test-model", 1000, 0)
        split, _ = tracker.calculate_cost(
            "flat-test-model", 1000, 0, tokens_cached_input=400
        )
        assert split == plain

    def test_unpriced_model_is_still_zero_and_unknown(self, db_path: Path) -> None:
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        assert tracker.calculate_cost(
            "no-such-model", 10, 10, tokens_thinking=10, tokens_cached_input=5
        ) == (0.0, False)


class TestTiers:
    """D-2: the tier is chosen from the full prompt count, cached part included."""

    def test_at_the_boundary_is_the_lower_tier(self, db_path: Path) -> None:
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        cost, _ = tracker.calculate_cost(MODEL, 1000, 0)
        assert cost == round(1000 * 1.0 / 1_000_000, 8)

    def test_one_above_the_boundary_is_the_upper_tier(self, db_path: Path) -> None:
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        cost, _ = tracker.calculate_cost(MODEL, 1001, 1, tokens_thinking=1)
        assert cost == round((1001 * 2.0 + 2 * 200.0) / 1_000_000, 8)

    def test_cached_tokens_count_toward_the_tier(self, db_path: Path) -> None:
        """900 fresh + 200 cached = 1100 prompt: upper tier, even though the
        fresh part alone (900) is under the boundary."""
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        cost, _ = tracker.calculate_cost(MODEL, 1100, 0, tokens_cached_input=200)
        assert cost == round((900 * 2.0 + 200 * 0.02) / 1_000_000, 8)

    def test_highest_matching_threshold_wins_regardless_of_order(
        self, db_path: Path
    ) -> None:
        pricing = {
            "m": {
                "input": 1.0,
                "output": 1.0,
                "tiers": [
                    {"above_prompt_tokens": 2000, "input": 3.0, "output": 3.0},
                    {"above_prompt_tokens": 1000, "input": 2.0, "output": 2.0},
                ],
            }
        }
        tracker = CostTracker(db_path=db_path, pricing=pricing)
        assert tracker.calculate_cost("m", 1500, 0)[0] == round(1500 * 2.0 / 1e6, 8)
        assert tracker.calculate_cost("m", 2500, 0)[0] == round(2500 * 3.0 / 1e6, 8)

    def test_recorded_cost_uses_the_tier(self, db_path: Path) -> None:
        tracker = CostTracker(db_path=db_path, pricing=TIERED)
        tracker.record(
            model=MODEL,
            endpoint="/v1/chat/completions",
            tokens_input=2000,
            tokens_output=0,
            tokens_thinking=0,
            tokens_cached_input=0,
        )
        row = tracker.get_recent(1)[0]
        assert row["cost_usd"] == round(2000 * 2.0 / 1_000_000, 8)
        assert row["pricing_known"] == 1

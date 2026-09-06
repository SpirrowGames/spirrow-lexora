"""Tests for the cost tracker.

Covers the D-6 changes from T-frontier-tier (msg-013): the ledger keys pricing
on the resolved model ID rather than the caller's tier alias, and records
``tier`` and ``pricing_known`` as first-class columns so that "tier-alias
requests" and "requests to a model we do not have a price for" are both
queryable rather than silently folded into the ``model`` column.
"""

import sqlite3
from pathlib import Path

import pytest

from lexora.services.cost_tracker import CostTracker, DEFAULT_PRICING


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "costs.db"


class TestCostTrackerPricingResolution:
    """D-6a: pricing keys on the resolved model, not on the tier alias."""

    def test_no_tier_alias_key_in_default_pricing(self) -> None:
        """Tier names must not appear as pricing keys.

        The whole point of D-6a is that pricing follows the resolved model,
        so ``frontier`` (or any other tier) sneaking into ``DEFAULT_PRICING``
        would let a stale row shadow the real answer.
        """
        for tier_name in ("frontier", "naysayer", "heavy", "medium", "light"):
            assert tier_name not in DEFAULT_PRICING, (
                f"tier alias {tier_name!r} must not appear in DEFAULT_PRICING — "
                "pricing keys on resolved model IDs (D-6a)."
            )

    def test_record_stores_resolved_model_not_tier(self, db_path: Path) -> None:
        """The ``model`` column carries the resolved concrete model ID."""
        tracker = CostTracker(
            db_path=db_path,
            pricing={"claude-fable-5-20260101": {"input": 5.0, "output": 25.0}},
        )
        tracker.record(
            model="claude-fable-5-20260101",  # already resolved by the caller
            endpoint="/v1/chat/completions",
            tokens_input=1000,
            tokens_output=500,
            backend="frontier",
            tier="frontier",
        )
        recent = tracker.get_recent(limit=1)
        assert recent[0]["model"] == "claude-fable-5-20260101"
        assert recent[0]["tier"] == "frontier"

    def test_env_swap_reprices(self, db_path: Path) -> None:
        """A model swap (Fable → Opus 5) reprices correctly.

        This is the msg-012 defect regression: previously ``model`` was the
        tier alias, so pricing was fixed to whatever the alias was mapped
        to at compile time. With D-6a the swap is transparent.
        """
        tracker = CostTracker(
            db_path=db_path,
            pricing={
                "claude-fable-5-20260101": {"input": 5.0, "output": 25.0},
                "claude-opus-5-20260601": {"input": 15.0, "output": 75.0},
            },
        )
        cost_fable, _ = tracker.calculate_cost("claude-fable-5-20260101", 1_000_000, 0)
        cost_opus, _ = tracker.calculate_cost("claude-opus-5-20260601", 1_000_000, 0)
        assert cost_fable == pytest.approx(5.0)
        assert cost_opus == pytest.approx(15.0)


class TestCostTrackerTierColumn:
    """D-6b: the tier alias is preserved in its own column."""

    def test_tier_column_is_nullable(self, db_path: Path) -> None:
        """A concrete-model call records tier=None."""
        tracker = CostTracker(db_path=db_path)
        tracker.record(
            model="Qwen3.8-27B",
            endpoint="/v1/chat/completions",
            tokens_input=100,
            tokens_output=50,
            tier=None,
        )
        rows = tracker.get_recent(limit=1)
        assert rows[0]["tier"] is None

    def test_by_tier_group(self, db_path: Path) -> None:
        """``by_tier`` groups rows by tier alias, excluding NULLs."""
        tracker = CostTracker(
            db_path=db_path,
            pricing={"model-x": {"input": 1.0, "output": 2.0}},
        )
        tracker.record(model="model-x", endpoint="/e", tokens_input=1000, tokens_output=1000, tier="frontier")
        tracker.record(model="model-x", endpoint="/e", tokens_input=1000, tokens_output=1000, tier="frontier")
        tracker.record(model="model-x", endpoint="/e", tokens_input=1000, tokens_output=1000, tier="heavy")
        tracker.record(model="model-x", endpoint="/e", tokens_input=1000, tokens_output=1000, tier=None)

        report = tracker.get_costs(period="all")
        by_tier = {row["tier"]: row for row in report["by_tier"]}
        assert set(by_tier) == {"frontier", "heavy"}
        assert by_tier["frontier"]["requests"] == 2
        assert by_tier["heavy"]["requests"] == 1

    def test_tier_filter(self, db_path: Path) -> None:
        """``?tier=frontier`` returns only tier-frontier rows."""
        tracker = CostTracker(db_path=db_path, pricing={"m": {"input": 1.0, "output": 1.0}})
        tracker.record(model="m", endpoint="/e", tokens_input=100, tokens_output=100, tier="frontier")
        tracker.record(model="m", endpoint="/e", tokens_input=100, tokens_output=100, tier="heavy")
        tracker.record(model="m", endpoint="/e", tokens_input=100, tokens_output=100, tier=None)

        report = tracker.get_costs(period="all", tier="frontier")
        assert report["summary"]["total_requests"] == 1


class TestCostTrackerPricingKnown:
    """D-6c: unknown models are distinguished from actually-free models."""

    def test_local_free_model_is_priced_known(self, db_path: Path) -> None:
        """A local model listed at 0.0 records ``pricing_known=1``."""
        tracker = CostTracker(db_path=db_path)
        # Qwen3.8-27B is in DEFAULT_PRICING at 0.0.
        tracker.record(
            model="Qwen3.8-27B",
            endpoint="/v1/chat/completions",
            tokens_input=1000,
            tokens_output=500,
        )
        rows = tracker.get_recent(limit=1)
        assert rows[0]["pricing_known"] == 1
        assert rows[0]["cost_usd"] == 0.0

    def test_unknown_model_marks_unpriced(self, db_path: Path) -> None:
        """An unknown model records ``pricing_known=0`` and surfaces in the summary."""
        tracker = CostTracker(db_path=db_path)
        tracker.record(
            model="claude-fable-5-unknown-id",
            endpoint="/v1/chat/completions",
            tokens_input=1000,
            tokens_output=500,
        )
        rows = tracker.get_recent(limit=1)
        assert rows[0]["pricing_known"] == 0
        assert rows[0]["cost_usd"] == 0.0  # no vendor charge computable

        report = tracker.get_costs(period="all")
        assert report["summary"]["unpriced_requests"] == 1
        assert "claude-fable-5-unknown-id" in report["unpriced_models"]

    def test_free_and_unpriced_are_separable(self, db_path: Path) -> None:
        """Free-known and unpriced must produce different summary counts."""
        tracker = CostTracker(db_path=db_path)
        tracker.record(model="Qwen3.8-27B", endpoint="/e", tokens_input=100, tokens_output=100)
        tracker.record(model="unknown-model", endpoint="/e", tokens_input=100, tokens_output=100)

        report = tracker.get_costs(period="all")
        assert report["summary"]["unpriced_requests"] == 1
        assert report["summary"]["total_requests"] == 2


class TestCostTrackerMigration:
    """D-6b/c: idempotent, non-destructive schema migration."""

    def test_migrates_legacy_schema(self, db_path: Path) -> None:
        """Opening a pre-migration DB adds columns and preserves existing rows."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """CREATE TABLE request_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    backend TEXT,
                    endpoint TEXT NOT NULL,
                    user_id TEXT,
                    tokens_input INTEGER NOT NULL DEFAULT 0,
                    tokens_output INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    duration_seconds REAL,
                    success INTEGER NOT NULL DEFAULT 1
                )"""
            )
            conn.execute(
                """INSERT INTO request_costs
                   (timestamp, model, endpoint, tokens_input, tokens_output, cost_usd, success)
                   VALUES ('2026-01-01T00:00:00Z', 'legacy-model', '/e', 10, 20, 0.001, 1)"""
            )

        tracker = CostTracker(db_path=db_path)
        rows = tracker.get_recent(limit=1)
        assert rows[0]["model"] == "legacy-model"
        # New columns present but NULL for the legacy row — the honest value
        # for "the column did not exist when this row was written."
        assert rows[0]["tier"] is None
        assert rows[0]["pricing_known"] is None

    def test_migration_is_idempotent(self, db_path: Path) -> None:
        """Constructing twice must not fail on ALTER TABLE."""
        CostTracker(db_path=db_path)
        CostTracker(db_path=db_path)  # would raise "duplicate column" if not guarded


class TestSubscriptionBackendIsNotPriced:
    """R-10: the ledger must not price a model it cannot show was billed.

    ``claude-code-opus`` / ``claude-code-sonnet`` are not upstream API model
    IDs. They are Lexora-local names for the ``claude_code`` backend, which
    shells out to the Claude Code CLI (``claude -p``) rather than issuing a
    metered HTTP request. Pricing them at Anthropic's per-MTok API rates put
    a non-zero ``cost_usd`` and ``pricing_known=1`` into every such row.
    """

    def test_cli_backend_ids_absent_from_default_pricing(self) -> None:
        """The two CLI-backend IDs must not be pricing keys."""
        for cli_id in ("claude-code-sonnet", "claude-code-opus"):
            assert cli_id not in DEFAULT_PRICING, (
                f"{cli_id!r} must not appear in DEFAULT_PRICING — it names a "
                "CLI subprocess, not a billable upstream model ID (R-10)."
            )

    def test_cli_backend_records_unpriced_not_free(self, db_path: Path) -> None:
        """``calculate_cost`` returns 0.0 with ``pricing_known`` False.

        Asserting the flag as well as the number is the point: ``0.0`` alone
        is what a genuinely free local model returns, and the distinction
        between "no vendor charge" and "we cannot say" is the subject of
        this requirement.
        """
        tracker = CostTracker(db_path=db_path)
        for cli_id in ("claude-code-sonnet", "claude-code-opus"):
            cost, pricing_known = tracker.calculate_cost(cli_id, 1_000_000, 1_000_000)
            assert (cost, pricing_known) == (0.0, False), (
                f"{cli_id!r} priced at {cost} with pricing_known={pricing_known}; "
                "an unbillable ID must land in the honest unpriced state (R-10)."
            )

    def test_gemini_naysayer_tier_stays_unpriced(self, db_path: Path) -> None:
        """``gemini-3.1-pro-preview`` remains deliberately absent.

        Its published price is a context-size step, which a flat
        {input, output} pair cannot express. Pinning it here so that the
        R-10 deletion is not later "balanced" by filling this one in.
        """
        assert "gemini-3.1-pro-preview" not in DEFAULT_PRICING
        tracker = CostTracker(db_path=db_path)
        assert tracker.calculate_cost("gemini-3.1-pro-preview", 1_000_000, 1_000_000) == (
            0.0,
            False,
        )

    def test_priced_models_are_untouched(self, db_path: Path) -> None:
        """Fence: the models that *are* billable keep their rates.

        Green on both sides of the R-10 change. It exists to catch a
        deletion that takes a neighbouring line with it.
        """
        tracker = CostTracker(db_path=db_path)
        expected = {
            "claude-sonnet-4-20250514": 3.0 + 15.0,
            "claude-opus-4-20250514": 15.0 + 75.0,
            "claude-fable-5": 10.0 + 50.0,
            "claude-opus-5": 5.0 + 25.0,
            "gpt-4": 30.0 + 60.0,
            "gemini-2.5-flash": 0.15 + 0.60,
            "Qwen3-32B": 0.0,
            "Qwen3.8-27B": 0.0,
        }
        for model, total in expected.items():
            cost, pricing_known = tracker.calculate_cost(model, 1_000_000, 1_000_000)
            assert pricing_known is True, f"{model!r} lost its price"
            assert cost == pytest.approx(total), f"{model!r} repriced"

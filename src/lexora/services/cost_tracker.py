"""Cost tracking service for Lexora.

Records per-request token usage and costs, with SQLite persistence
and aggregation queries for daily/monthly cost reports.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lexora.utils.logging import get_logger

logger = get_logger(__name__)

# Default pricing per million tokens (USD).
#
# Keys are concrete upstream model IDs, never tier aliases. The cost tracker
# receives the resolved model (via ``BackendRouter.resolve_model``) at record
# time so that an env override — e.g. ``LEXORA_FRONTIER_MODEL`` swapping
# Fable 5 for Opus 5 — changes the pricing lookup automatically and does not
# leave the ledger stuck on the old price.
#
# Prices with citations:
#   - Anthropic Claude Sonnet 4 / Opus 4: anthropic.com/pricing (accessed
#     2026-08 via config baseline)
#   - Google Gemini 2.5 Flash: ai.google.dev/gemini-api/docs/pricing
#   - Local vLLM (``Qwen3-32B``, ``Qwen3.8-27B``): 0.0 because the marginal
#     cost is our own electricity, and the ledger treats "no vendor charge"
#     as literally zero — distinguished from "we do not know" by the
#     ``pricing_known`` column that ``record`` writes.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # Anthropic — Claude 4 series (Sonnet / Opus, per anthropic.com/pricing)
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    # Anthropic — Claude 5 series (frontier tier candidates, T-frontier-tier).
    # Prices are placeholders taken from the public pricing pages at the
    # date noted here; the frontier tier defaults to Fable 5 but Opus 5
    # is also seeded so an `LEXORA_FRONTIER_MODEL` swap does not silently
    # land in the unpriced bucket. Update these entries alongside the
    # vendor's next price change — recorded rows keep their historical
    # cost, so a mid-life price change never rewrites the past.
    "claude-fable-5-20260101": {"input": 5.0, "output": 25.0},
    "claude-opus-5-20260601": {"input": 15.0, "output": 75.0},
    # Claude Code (uses Anthropic pricing internally)
    "claude-code-sonnet": {"input": 3.0, "output": 15.0},
    "claude-code-opus": {"input": 15.0, "output": 75.0},
    # OpenAI
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    # Google
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    # Local (free)
    "Qwen3-32B": {"input": 0.0, "output": 0.0},
    "Qwen3.8-27B": {"input": 0.0, "output": 0.0},
}


class CostTracker:
    """Tracks API costs with SQLite persistence.

    Args:
        db_path: Path to SQLite database file.
        pricing: Model pricing overrides (per million tokens).
    """

    def __init__(
        self,
        db_path: str | Path = "data/costs.db",
        pricing: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.pricing = {**DEFAULT_PRICING, **(pricing or {})}
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database, create tables, and run idempotent migrations.

        Two columns are added by migration rather than baked into ``CREATE
        TABLE``:

        - ``tier``: the caller-facing tier name (``frontier``, ``naysayer``,
          ...) when the request was routed by tier, else NULL. Storing this
          at record time avoids fragile reverse-mapping later: tier
          configuration can change (a tier can point at a different backend
          tomorrow) but the ledger row records what actually happened.
        - ``pricing_known``: 1 if the resolved model was in the pricing
          table, 0 if it was not (i.e. cost 0.0 is "no vendor charge"),
          NULL for rows written before this migration ran. NULL means "the
          column did not exist when this row was written" — not "unknown
          at the time".

        The DB migration is guarded by ``PRAGMA table_info`` so a DB opened
        twice does not fail; existing rows are preserved with NULL in the
        new columns, which is the honest value for "did not record this".
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_costs (
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
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_costs_timestamp
                ON request_costs(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_costs_model
                ON request_costs(model)
            """)
            # Idempotent migrations (2026-08-31, T-frontier-tier D-6b/D-6c).
            # Existing installs have `model` populated with whatever the
            # caller sent (tier name OR concrete model ID). New writes put
            # only the resolved model ID in `model` and stash the tier name
            # in the new `tier` column. That is a behaviour change for
            # `?model=frontier` filtering (documented in the tier table),
            # but it removes the ambiguity that made costs unqueryable per
            # concrete model when the caller used a tier alias.
            cols = {row[1] for row in conn.execute("PRAGMA table_info(request_costs)")}
            if "tier" not in cols:
                conn.execute("ALTER TABLE request_costs ADD COLUMN tier TEXT")
            if "pricing_known" not in cols:
                conn.execute(
                    "ALTER TABLE request_costs ADD COLUMN pricing_known INTEGER"
                )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_costs_tier
                   ON request_costs(tier)"""
            )

    def calculate_cost(
        self, model: str, tokens_input: int, tokens_output: int
    ) -> tuple[float, bool]:
        """Calculate cost for a request.

        Args:
            model: Resolved (concrete) model name — never a tier alias.
            tokens_input: Number of input tokens.
            tokens_output: Number of output tokens.

        Returns:
            Tuple of (cost in USD, ``pricing_known`` flag). ``pricing_known``
            is False when ``model`` is absent from ``self.pricing``; the
            caller records both, so a subsequent audit can tell a free local
            model from an unpriced one.
        """
        prices = self.pricing.get(model)
        pricing_known = prices is not None
        if prices is None:
            prices = {"input": 0.0, "output": 0.0}
        input_cost = (tokens_input / 1_000_000) * prices["input"]
        output_cost = (tokens_output / 1_000_000) * prices["output"]
        return round(input_cost + output_cost, 8), pricing_known

    def record(
        self,
        model: str,
        endpoint: str,
        tokens_input: int,
        tokens_output: int,
        backend: str | None = None,
        user_id: str | None = None,
        duration: float | None = None,
        success: bool = True,
        tier: str | None = None,
    ) -> float:
        """Record a request's cost.

        The caller is expected to pass the *resolved* model ID in ``model``
        (i.e. ``BackendRouter.resolve_model(requested)``) and the caller's
        tier name in ``tier`` when the request was routed by tier. Callers
        that already have a concrete model in hand pass ``tier=None``.

        Args:
            model: Resolved concrete model name (never a tier alias).
            endpoint: API endpoint.
            tokens_input: Number of input tokens.
            tokens_output: Number of output tokens.
            backend: Backend name.
            user_id: User identifier.
            duration: Request duration in seconds.
            success: Whether the request succeeded.
            tier: Tier alias the caller used, if any (``frontier``,
                ``naysayer``, ...). None when the request specified a
                concrete model directly.

        Returns:
            Calculated cost in USD.
        """
        cost, pricing_known = self.calculate_cost(model, tokens_input, tokens_output)
        if not pricing_known:
            # Best-effort warn. The `record` path is deliberately
            # exception-swallowing (see the except below) so accounting
            # never turns a 200 into a 500; the log line is the durable
            # signal alongside the `pricing_known=0` column write.
            logger.warning(
                "cost_pricing_unknown",
                model=model,
                tier=tier,
                backend=backend,
                endpoint=endpoint,
            )
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO request_costs
                       (timestamp, model, backend, endpoint, user_id,
                        tokens_input, tokens_output, cost_usd,
                        duration_seconds, success, tier, pricing_known)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        timestamp,
                        model,
                        backend,
                        endpoint,
                        user_id,
                        tokens_input,
                        tokens_output,
                        cost,
                        duration,
                        1 if success else 0,
                        tier,
                        1 if pricing_known else 0,
                    ),
                )
        except Exception:
            logger.exception("cost_record_failed", model=model)

        return cost

    def get_costs(
        self,
        period: str = "today",
        model: str | None = None,
        user_id: str | None = None,
        backend: str | None = None,
        tier: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated costs.

        Args:
            period: "today", "month", "all", or ISO date "YYYY-MM-DD".
            model: Filter by resolved model name.
            user_id: Filter by user.
            backend: Filter by backend.
            tier: Filter by tier alias (``frontier``, ``naysayer``, ...).
                Only matches rows recorded via a tier — a request sent as a
                concrete model does not match a tier filter even if that
                tier happens to point at the same model.

        Returns:
            Aggregated cost data.
        """
        now = datetime.now(timezone.utc)

        if period == "today":
            date_filter = now.strftime("%Y-%m-%d")
            where = "timestamp >= ?"
            params: list[Any] = [date_filter]
        elif period == "month":
            date_filter = now.strftime("%Y-%m")
            where = "timestamp >= ?"
            params = [date_filter + "-01"]
        elif period == "all":
            where = "1=1"
            params = []
        else:
            # Assume ISO date
            where = "timestamp >= ? AND timestamp < date(?, '+1 day')"
            params = [period, period]

        if model:
            where += " AND model = ?"
            params.append(model)
        if user_id:
            where += " AND user_id = ?"
            params.append(user_id)
        if backend:
            where += " AND backend = ?"
            params.append(backend)
        if tier:
            where += " AND tier = ?"
            params.append(tier)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total
            row = conn.execute(
                f"""SELECT
                    COUNT(*) as total_requests,
                    COALESCE(SUM(tokens_input), 0) as total_tokens_input,
                    COALESCE(SUM(tokens_output), 0) as total_tokens_output,
                    COALESCE(SUM(cost_usd), 0.0) as total_cost_usd,
                    COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0) as successful_requests,
                    COALESCE(SUM(CASE WHEN pricing_known=0 THEN 1 ELSE 0 END), 0) as unpriced_requests
                FROM request_costs WHERE {where}""",
                params,
            ).fetchone()

            summary = dict(row) if row else {}

            # Per model
            by_model = conn.execute(
                f"""SELECT
                    model,
                    COUNT(*) as requests,
                    COALESCE(SUM(tokens_input), 0) as tokens_input,
                    COALESCE(SUM(tokens_output), 0) as tokens_output,
                    COALESCE(SUM(cost_usd), 0.0) as cost_usd
                FROM request_costs WHERE {where}
                GROUP BY model ORDER BY cost_usd DESC""",
                params,
            ).fetchall()

            # Per tier (tier IS NOT NULL, so requests sent by concrete model
            # ID and pre-migration rows are excluded — the intent of `by_tier`
            # is "what did we spend routing by tier alias", and NULL is not
            # a tier).
            by_tier = conn.execute(
                f"""SELECT
                    tier,
                    COUNT(*) as requests,
                    COALESCE(SUM(tokens_input), 0) as tokens_input,
                    COALESCE(SUM(tokens_output), 0) as tokens_output,
                    COALESCE(SUM(cost_usd), 0.0) as cost_usd
                FROM request_costs WHERE {where} AND tier IS NOT NULL
                GROUP BY tier ORDER BY cost_usd DESC""",
                params,
            ).fetchall()

            # Per day (last 30 days)
            daily = conn.execute(
                f"""SELECT
                    date(timestamp) as date,
                    COUNT(*) as requests,
                    COALESCE(SUM(cost_usd), 0.0) as cost_usd
                FROM request_costs WHERE {where}
                GROUP BY date(timestamp) ORDER BY date DESC LIMIT 30""",
                params,
            ).fetchall()

            # Which models are being recorded without a known price?
            # Deliberately not derived from the current `self.pricing` at
            # read time: if a price is added later, the historical cost
            # column stays 0.0 and would misrepresent the row as "priced,
            # cost zero". Recording `pricing_known` at write time keeps
            # the two apart forever.
            unpriced_rows = conn.execute(
                f"""SELECT DISTINCT model
                    FROM request_costs
                    WHERE {where} AND pricing_known = 0""",
                params,
            ).fetchall()
            unpriced_models = [r["model"] for r in unpriced_rows]

        return {
            "period": period,
            "filters": {
                "model": model,
                "user_id": user_id,
                "backend": backend,
                "tier": tier,
            },
            "summary": summary,
            "by_model": [dict(r) for r in by_model],
            "by_tier": [dict(r) for r in by_tier],
            "daily": [dict(r) for r in daily],
            "unpriced_models": unpriced_models,
            "pricing": self.pricing,
        }

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent request records.

        Args:
            limit: Maximum number of records.

        Returns:
            List of recent request records.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM request_costs
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

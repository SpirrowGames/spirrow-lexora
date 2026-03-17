"""Cost tracking service for Lexora.

Records per-request token usage and costs, with SQLite persistence
and aggregation queries for daily/monthly cost reports.
"""

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lexora.utils.logging import get_logger

logger = get_logger(__name__)

# Default pricing per million tokens (USD)
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
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
        """Initialize SQLite database and create tables."""
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

    def calculate_cost(self, model: str, tokens_input: int, tokens_output: int) -> float:
        """Calculate cost for a request.

        Args:
            model: Model name.
            tokens_input: Number of input tokens.
            tokens_output: Number of output tokens.

        Returns:
            Cost in USD.
        """
        prices = self.pricing.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (tokens_input / 1_000_000) * prices["input"]
        output_cost = (tokens_output / 1_000_000) * prices["output"]
        return round(input_cost + output_cost, 8)

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
    ) -> float:
        """Record a request's cost.

        Args:
            model: Model name.
            endpoint: API endpoint.
            tokens_input: Number of input tokens.
            tokens_output: Number of output tokens.
            backend: Backend name.
            user_id: User identifier.
            duration: Request duration in seconds.
            success: Whether the request succeeded.

        Returns:
            Calculated cost in USD.
        """
        cost = self.calculate_cost(model, tokens_input, tokens_output)
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO request_costs
                       (timestamp, model, backend, endpoint, user_id,
                        tokens_input, tokens_output, cost_usd,
                        duration_seconds, success)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
    ) -> dict[str, Any]:
        """Get aggregated costs.

        Args:
            period: "today", "month", "all", or ISO date "YYYY-MM-DD".
            model: Filter by model.
            user_id: Filter by user.
            backend: Filter by backend.

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

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total
            row = conn.execute(
                f"""SELECT
                    COUNT(*) as total_requests,
                    COALESCE(SUM(tokens_input), 0) as total_tokens_input,
                    COALESCE(SUM(tokens_output), 0) as total_tokens_output,
                    COALESCE(SUM(cost_usd), 0.0) as total_cost_usd,
                    COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0) as successful_requests
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

        return {
            "period": period,
            "filters": {
                "model": model,
                "user_id": user_id,
                "backend": backend,
            },
            "summary": summary,
            "by_model": [dict(r) for r in by_model],
            "daily": [dict(r) for r in daily],
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

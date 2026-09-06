"""R-12: an endpoint that computes token counts must open a ledger row.

At `develop` (`00eafd5`) `cost_tracker.record` had exactly two call sites in
the whole tree -- `routes.py:591` (`/v1/chat/completions`) and
`routes.py:1989` (`/v1/messages`) -- while `usage.get("prompt_tokens", 0)` had
six. The four handlers in between computed the numbers the ledger wants and
dropped them:

    endpoint              handler        computes tokens   records a row
    /v1/completions       completions    routes.py:938     no
    /v1/embeddings        embeddings     routes.py:1079    no
    /generate             generate       routes.py:1468    no
    /chat                 chat           routes.py:1632    no

All four are live `@router.post` routes and all four call the backend directly
rather than delegating to `chat_completions`, so "recorded upstream" was not
the reason. None carried a comment explaining the omission, and none even took
`cost_tracker` as a dependency. `/stats/costs` and `/stats/costs/recent` served
the sum of the two recorded routes without saying so anywhere.

Still not covered after this file, deliberately: the three streaming paths
(`stream_generator` at `00eafd5`'s `routes.py:428` / `:812` / `:1883`), which
relay the bytes `backend.*_stream` yields without decoding them. See
`get_cost_tracker`'s docstring.

Why these detectors drive the routes instead of calling a helper: R-11 shipped
five green unit detectors over the token parser while the ledger itself stayed
wrong, because nothing asserted the parser was reached. The requirement here is
"a row is opened", so that is what is measured -- at the route.

Mutation, so the detectors are measured rather than asserted. Counts are this
file alone, 16 cases:

- Against `develop`: 6 red / 10 green. Red = the four
  `TestLedgerCoversEveryNonStreamingRoute` cases (`record` never called) plus
  the two `TestUnpricedModelSaysSo` cases that read the row back out of SQLite.
  Green, and therefore fences rather than detectors, = both
  `TestExistingRoutesAreUnchanged` cases, all seven `TestGuard` cases, and
  `test_no_embedding_model_is_priced_as_chat`. The other 566 tests are green
  against `develop` too: the suite was blind to this.
- Half-fix, `/v1/completions` alone (block and dependency added there and
  nowhere else): 5 red / 11 green. `[completions]` goes green while
  `[embeddings]`, `[generate]` and `[chat]` stay red -- four independent
  detectors, not one assertion counted four times.
- Removing only `generate`'s `Depends(get_cost_tracker)`, leaving its `record`
  block: 3 red / 13 green. Of the four detectors only `[generate]` reds, on
  `NameError` -- so it is bound to that handler's wiring, not to the block
  existing somewhere in the file. The other two reds drive the same broken
  route.
- Weakening the guard to a bare `if cost_tracker:` at the four new call sites
  only: 4 red / 12 green. All four detectors stay green and only `TestGuard`'s
  four new-route cases red; its two existing-route cases stay green, which is
  what shows the mutation reached only the new sites.
"""

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexora.api.routes import (
    get_backend,
    get_backend_router,
    get_cost_tracker,
    get_metrics_collector,
    get_rate_limiter,
    get_retry_handler,
    get_stats_collector,
    is_rate_limit_enabled,
    router,
)
from lexora.services.cost_tracker import DEFAULT_PRICING, CostTracker
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector

# The caller always says `heavy` (a tier alias); the router always resolves it
# to `claude-fable-5`. Keeping the two distinct is what lets the assertions
# below tell the `model` column (concrete) from the `tier` column (alias).
REQUESTED = "heavy"
RESOLVED = "claude-fable-5"
BACKEND_NAME = "frontier"

USAGE = {"prompt_tokens": 11, "completion_tokens": 5}

COMPLETION_RESPONSE: dict[str, Any] = {
    "id": "cmpl-1",
    "object": "text_completion",
    "choices": [{"index": 0, "text": "Hi", "finish_reason": "stop"}],
    "usage": USAGE,
}
CHAT_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": USAGE,
}
EMBEDDINGS_RESPONSE: dict[str, Any] = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": RESOLVED,
    "usage": {"prompt_tokens": 11, "total_tokens": 11},
}

COMPLETIONS_BODY = {"model": REQUESTED, "prompt": "Hi"}
EMBEDDINGS_BODY = {"model": REQUESTED, "input": "Hi"}
GENERATE_BODY = {"model": REQUESTED, "prompt": "Hi"}
CHAT_BODY = {"model": REQUESTED, "messages": [{"role": "user", "content": "Hi"}]}
CHAT_COMPLETIONS_BODY = {
    "model": REQUESTED,
    "messages": [{"role": "user", "content": "Hi"}],
}
MESSAGES_BODY = {
    "model": REQUESTED,
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hi"}],
}

# The four routes R-12 adds, and the two that already recorded. `tokens_output`
# is 0 for embeddings because the handler hard-codes it (embeddings have no
# completion side); that 0 is the reason the guard is `input > 0 OR output > 0`
# and not `and`.
NEW_ROUTES = [
    ("/v1/completions", COMPLETIONS_BODY, 11, 5),
    ("/v1/embeddings", EMBEDDINGS_BODY, 11, 0),
    ("/generate", GENERATE_BODY, 11, 5),
    ("/chat", CHAT_BODY, 11, 5),
]
EXISTING_ROUTES = [
    ("/v1/chat/completions", CHAT_COMPLETIONS_BODY, 11, 5),
    ("/v1/messages", MESSAGES_BODY, 11, 5),
]


def _backend(usage: dict[str, int] | None = USAGE) -> MagicMock:
    """A backend whose three non-streaming methods return `usage`.

    `usage=None` strips the block entirely, which is what the guard fence
    drives; the response stays otherwise well-formed so the handlers still
    reach their recording site rather than erroring out earlier.
    """

    def _with(payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        if usage is None:
            body.pop("usage", None)
        else:
            body["usage"] = usage
        return body

    backend = MagicMock()
    backend.completions = AsyncMock(return_value=_with(COMPLETION_RESPONSE))
    backend.chat_completions = AsyncMock(return_value=_with(CHAT_RESPONSE))
    backend.embeddings = AsyncMock(return_value=_with(EMBEDDINGS_RESPONSE))
    backend.error_passthrough = False
    return backend


def _client(
    backend: MagicMock, cost_tracker: Any, resolved: str = RESOLVED
) -> TestClient:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=backend)
    backend_router.resolve_model = MagicMock(return_value=resolved)
    backend_router.get_backend_name_for_model = MagicMock(return_value=BACKEND_NAME)
    backend_router.is_tier = MagicMock(return_value=True)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: backend
    app.dependency_overrides[get_backend_router] = lambda: backend_router
    app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
    app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
        max_retries=1, base_delay=0.01, max_delay=0.1, jitter=False
    )
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
        default_rate=1000.0, default_burst=1000
    )
    app.dependency_overrides[is_rate_limit_enabled] = lambda: False
    app.dependency_overrides[get_metrics_collector] = lambda: None
    app.dependency_overrides[get_cost_tracker] = lambda: cost_tracker
    return TestClient(app)


class TestLedgerCoversEveryNonStreamingRoute:
    """D-1..D-4: the four routes that computed tokens now open a row."""

    @pytest.mark.parametrize(
        ("endpoint", "body", "tokens_input", "tokens_output"),
        NEW_ROUTES,
        ids=["completions", "embeddings", "generate", "chat"],
    )
    def test_route_records_exactly_one_row(
        self, endpoint: str, body: dict, tokens_input: int, tokens_output: int
    ) -> None:
        tracker = MagicMock()
        response = _client(_backend(), tracker).post(endpoint, json=body)

        assert response.status_code == 200
        assert tracker.record.call_count == 1
        kwargs = tracker.record.call_args.kwargs
        assert kwargs["model"] == RESOLVED
        assert kwargs["endpoint"] == endpoint
        assert kwargs["tokens_input"] == tokens_input
        assert kwargs["tokens_output"] == tokens_output
        assert kwargs["backend"] == BACKEND_NAME
        # The alias goes to its own column, never into `model` -- the contract
        # `CostTracker.record` states in its docstring.
        assert kwargs["tier"] == REQUESTED


class TestExistingRoutesAreUnchanged:
    """F-1: the two routes that already recorded still record once, not twice."""

    @pytest.mark.parametrize(
        ("endpoint", "body", "tokens_input", "tokens_output"),
        EXISTING_ROUTES,
        ids=["chat_completions", "messages"],
    )
    def test_no_double_counting(
        self, endpoint: str, body: dict, tokens_input: int, tokens_output: int
    ) -> None:
        tracker = MagicMock()
        response = _client(_backend(), tracker).post(endpoint, json=body)

        assert response.status_code == 200
        assert tracker.record.call_count == 1
        kwargs = tracker.record.call_args.kwargs
        assert kwargs["tokens_input"] == tokens_input
        assert kwargs["tokens_output"] == tokens_output


class TestGuard:
    """F-2: no `usage` means no row. The pre-R-12 rule, kept."""

    @pytest.mark.parametrize(
        ("endpoint", "body"),
        [(e, b) for e, b, _, _ in NEW_ROUTES + EXISTING_ROUTES],
        ids=[
            "completions",
            "embeddings",
            "generate",
            "chat",
            "chat_completions",
            "messages",
        ],
    )
    def test_no_usage_opens_no_row(self, endpoint: str, body: dict) -> None:
        tracker = MagicMock()
        response = _client(_backend(usage=None), tracker).post(endpoint, json=body)

        assert response.status_code == 200
        assert tracker.record.call_count == 0

    def test_absent_cost_tracker_is_not_an_error(self) -> None:
        """`cost_tracker` is `None` whenever the app never built one."""
        client = _client(_backend(), None)
        for endpoint, body, _, _ in NEW_ROUTES:
            assert client.post(endpoint, json=body).status_code == 200


class TestUnpricedModelSaysSo:
    """F-3: bringing embeddings in lands on `pricing_known=0`, not a silent 0.0.

    No embedding model is in `DEFAULT_PRICING` (measured 2026-09-06 at
    `05f6827`: nine entries, all chat -- four Claude, two GPT, one Gemini, two
    local Qwen at 0.0). Recording them therefore writes cost 0.0 with
    `pricing_known=0`, which is R-10's "we cannot price this" state and is
    distinguishable from the local Qwen rows' priced 0.0. Not recording them
    at all said neither.
    """

    def test_no_embedding_model_is_priced_as_chat(self) -> None:
        assert not [m for m in DEFAULT_PRICING if "embed" in m.lower()]

    def test_row_is_written_with_pricing_known_zero(self, tmp_path: Path) -> None:
        db = tmp_path / "costs.db"
        tracker = CostTracker(db_path=db)
        client = _client(_backend(), tracker, resolved="text-embedding-3-small")

        assert client.post("/v1/embeddings", json=EMBEDDINGS_BODY).status_code == 200

        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT model, endpoint, tokens_input, tokens_output, cost_usd,"
                " pricing_known, tier FROM request_costs"
            ).fetchall()
        assert rows == [
            ("text-embedding-3-small", "/v1/embeddings", 11, 0, 0.0, 0, REQUESTED)
        ]

    def test_priced_model_is_distinguishable(self, tmp_path: Path) -> None:
        """The same route with a priced model writes `pricing_known=1`."""
        db = tmp_path / "costs.db"
        tracker = CostTracker(db_path=db)
        client = _client(_backend(), tracker, resolved="Qwen3-32B")

        assert client.post("/v1/embeddings", json=EMBEDDINGS_BODY).status_code == 200

        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT cost_usd, pricing_known FROM request_costs"
            ).fetchone()
        assert row == (0.0, 1)

"""T-ledger-gemini-thinking-tokens D-4: every record site passes the two columns.

Three streaming sites and six non-streaming sites (``/v1/messages`` included)
call ``cost_tracker.record``. Each spreads ``_ledger_token_extras(...)`` into
its kwargs; these cases drive every one of them and read the kwargs back.

The route doubles and constants are reused from the two files that already
fence these sites, so this file tests only the new pass-through.
"""

import sqlite3
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from lexora.backends.gemini import GeminiBackend
from lexora.services.cost_tracker import CostTracker
from tests.api import test_ledger_coverage as nonstream
from tests.api import test_streaming_ledger_row as stream

GEMINI_USAGE: dict[str, Any] = {
    "prompt_tokens": 11,
    "completion_tokens": 5,
    "total_tokens": 16,
    "prompt_tokens_details": {"cached_tokens": 4},
    "lexora_thinking_tokens": 7,
}

ALL_NON_STREAMING = nonstream.NEW_ROUTES + nonstream.EXISTING_ROUTES
ALL_NON_STREAMING_IDS = [
    "completions",
    "embeddings",
    "generate",
    "chat",
    "chat_completions",
    "messages",
]


class TestNonStreamingSites:
    @pytest.mark.parametrize(
        ("endpoint", "body", "_in", "_out"), ALL_NON_STREAMING, ids=ALL_NON_STREAMING_IDS
    )
    def test_gemini_usage_reaches_record(
        self, endpoint: str, body: dict, _in: int, _out: int
    ) -> None:
        tracker = MagicMock()
        response = nonstream._client(nonstream._backend(usage=GEMINI_USAGE), tracker).post(
            endpoint, json=body
        )
        assert response.status_code == 200
        kwargs = tracker.record.call_args.kwargs
        assert kwargs["tokens_thinking"] == 7
        assert kwargs["tokens_cached_input"] == 4

    @pytest.mark.parametrize(
        ("endpoint", "body", "_in", "_out"), ALL_NON_STREAMING, ids=ALL_NON_STREAMING_IDS
    )
    def test_other_backends_record_null(
        self, endpoint: str, body: dict, _in: int, _out: int
    ) -> None:
        """No ``lexora_thinking_tokens`` key: not measured, so None -> NULL.

        An upstream-relayed ``prompt_tokens_details`` alone does not count as
        measured (D-1: NULL = "this backend is not measured here").
        """
        usage = {**nonstream.USAGE, "prompt_tokens_details": {"cached_tokens": 3}}
        tracker = MagicMock()
        nonstream._client(nonstream._backend(usage=usage), tracker).post(endpoint, json=body)
        kwargs = tracker.record.call_args.kwargs
        assert kwargs["tokens_thinking"] is None
        assert kwargs["tokens_cached_input"] is None


def _filling_backend(thinking: int | None, cached: int | None) -> MagicMock:
    def _factory(_request: dict, usage_sink: Any = None) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            yield stream.CHUNK
            if usage_sink is not None:
                usage_sink.prompt_tokens = stream.PROMPT_TOKENS
                usage_sink.completion_tokens = stream.COMPLETION_TOKENS
                usage_sink.thinking_tokens = thinking
                usage_sink.cached_input_tokens = cached

        return gen()

    backend = MagicMock()
    backend.chat_completions_stream = MagicMock(side_effect=_factory)
    backend.completions_stream = MagicMock(side_effect=_factory)
    backend.error_passthrough = False
    return backend


class TestStreamingSites:
    @pytest.mark.parametrize(("endpoint", "body"), stream.ROUTES, ids=stream.ROUTE_IDS)
    def test_sink_values_reach_record(self, endpoint: str, body: dict) -> None:
        tracker = MagicMock()
        response = stream._client(_filling_backend(9, 2), tracker).post(endpoint, json=body)
        assert response.status_code == 200
        _ = response.content
        kwargs = tracker.record.call_args.kwargs
        assert (kwargs["tokens_thinking"], kwargs["tokens_cached_input"]) == (9, 2)

    @pytest.mark.parametrize(("endpoint", "body"), stream.ROUTES, ids=stream.ROUTE_IDS)
    def test_unfilled_sink_records_null(self, endpoint: str, body: dict) -> None:
        tracker = MagicMock()
        stream._client(_filling_backend(None, None), tracker).post(endpoint, json=body)
        kwargs = tracker.record.call_args.kwargs
        assert (kwargs["tokens_thinking"], kwargs["tokens_cached_input"]) == (None, None)


class TestPreflightProbeRowIsUnchanged:
    """mindwire ``preflight.py`` finds its probe by a ledger row with ``tier``
    and ``backend`` and never reads the body. Through the real Gemini response
    conversion and a real ``CostTracker``: the probe's thinking-exhausted
    answer still opens exactly one row, with ``tokens_output = 0`` (the same
    ``completion_tokens`` the probe got before this change) and the thinking
    on its own column.
    """

    def test_probe_row(self, tmp_path: Path) -> None:
        gemini_resp = GeminiBackend(name="naysayer")._to_openai_response(
            {
                "candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "thoughtsTokenCount": 16,
                    "totalTokenCount": 19,
                },
            },
            "gemini-3.1-pro-preview",
        )
        backend = MagicMock()
        backend.chat_completions = AsyncMock(return_value=gemini_resp)
        backend.error_passthrough = False
        tracker = CostTracker(db_path=tmp_path / "costs.db")

        response = nonstream._client(backend, tracker, resolved="gemini-3.1-pro-preview").post(
            "/v1/chat/completions",
            json={
                "model": nonstream.REQUESTED,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 16,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["usage"]["completion_tokens"] == 0
        assert body["choices"][0]["finish_reason"] == "length"
        assert "completion_tokens_details" not in body["usage"]

        with sqlite3.connect(tmp_path / "costs.db") as conn:
            rows = conn.execute(
                "SELECT tier, backend, tokens_input, tokens_output, "
                "tokens_thinking, tokens_cached_input FROM request_costs"
            ).fetchall()
        assert rows == [
            (nonstream.REQUESTED, nonstream.BACKEND_NAME, 3, 0, 16, 0),
        ]

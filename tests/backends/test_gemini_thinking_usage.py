"""T-ledger-gemini-thinking-tokens D-3: thinking and cached counts off Gemini.

Both read sites (``_to_openai_response`` and the streaming loop) now read
``thoughtsTokenCount`` and ``cachedContentTokenCount`` as well as the two
counts they already read. What is fixed here:

- the OpenAI-facing ``usage`` keeps ``completion_tokens`` =
  ``candidatesTokenCount`` (thinking EXCLUDED). mindwire's
  ``naysayer/preflight.py`` probes with ``max_tokens=16`` and gets back a
  thinking-exhausted answer as ``completion_tokens=0``; that must stay 0;
- there is NO ``completion_tokens_details``: OpenAI defines its
  ``reasoning_tokens`` as a part of ``completion_tokens``, and here it would
  exceed it. Thinking rides ``usage.lexora_thinking_tokens`` instead;
- ``prompt_tokens_details.cached_tokens`` <= ``prompt_tokens``, because
  ``promptTokenCount`` already includes the cached part;
- streaming uses the same presence rule as the existing two fields
  (assigned, never summed; a later event without the key does not clear
  it), and a stream that saw any usage turns "not stated" into 0.

The wire shape is a premise, as in ``test_gemini_stream_usage.py`` (no
Gemini key reaches this tree). The key names are the ones Google documents
for ``UsageMetadata``.
"""

from typing import Any

from lexora.backends.base import UsageSink
from lexora.backends.gemini import GeminiBackend
from tests.backends.test_gemini_stream_usage import _backend, _chunk, _collect

MODEL = "gemini-3.1-pro-preview"


def _metadata(
    prompt: int | None = 500,
    candidates: int | None = 0,
    thoughts: int | None = 16,
    cached: int | None = 200,
) -> dict[str, Any]:
    md: dict[str, Any] = {}
    for key, value in [
        ("promptTokenCount", prompt),
        ("candidatesTokenCount", candidates),
        ("thoughtsTokenCount", thoughts),
        ("cachedContentTokenCount", cached),
    ]:
        if value is not None:
            md[key] = value
    return md


def _non_streaming(metadata: dict[str, Any], finish: str = "MAX_TOKENS") -> dict:
    return GeminiBackend(name="naysayer")._to_openai_response(
        {
            "candidates": [{"content": {"parts": [{"text": ""}]}, "finishReason": finish}],
            "usageMetadata": metadata,
        },
        MODEL,
    )


class TestNonStreamingUsage:
    def test_preflight_shape_keeps_completion_tokens_zero(self) -> None:
        """The mindwire preflight probe: thinking ate the whole budget.

        ``completion_tokens`` must be exactly ``candidatesTokenCount`` (0),
        and ``finish_reason`` must still say ``length``. Folding thinking
        into ``completion_tokens`` would make this 16.
        """
        resp = _non_streaming(_metadata(prompt=3, candidates=0, thoughts=16, cached=0))
        assert resp["usage"]["completion_tokens"] == 0
        assert resp["usage"]["prompt_tokens"] == 3
        assert resp["usage"]["lexora_thinking_tokens"] == 16
        assert resp["choices"][0]["finish_reason"] == "length"

    def test_no_completion_tokens_details(self) -> None:
        resp = _non_streaming(_metadata())
        assert "completion_tokens_details" not in resp["usage"]

    def test_cached_is_reported_in_the_standard_field_and_is_a_subset(self) -> None:
        resp = _non_streaming(_metadata(prompt=500, cached=200))
        usage = resp["usage"]
        assert usage["prompt_tokens_details"] == {"cached_tokens": 200}
        assert usage["prompt_tokens_details"]["cached_tokens"] <= usage["prompt_tokens"]
        # promptTokenCount is passed through, not re-derived as fresh + cached.
        assert usage["prompt_tokens"] == 500

    def test_missing_keys_are_zero_on_a_whole_response(self) -> None:
        resp = _non_streaming(_metadata(thoughts=None, cached=None))
        assert resp["usage"]["lexora_thinking_tokens"] == 0
        assert resp["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}


class TestStreamingSink:
    async def test_both_new_counts_land(self) -> None:
        sink = UsageSink()
        await _collect(_backend([_chunk("", finish="MAX_TOKENS", usage=_metadata())]), sink)
        assert (sink.thinking_tokens, sink.cached_input_tokens) == (16, 200)
        # The existing two are unchanged by the new reads.
        assert (sink.prompt_tokens, sink.completion_tokens) == (500, 0)

    async def test_no_usage_at_all_leaves_none(self) -> None:
        """No usage block ever seen: not measured, so NULL -- not 0."""
        sink = UsageSink()
        await _collect(_backend([_chunk("Hi", finish="STOP")]), sink)
        assert (sink.thinking_tokens, sink.cached_input_tokens) == (None, None)

    async def test_usage_without_the_keys_becomes_zero(self) -> None:
        """Gemini omits the keys when the count is zero."""
        sink = UsageSink()
        md = _metadata(thoughts=None, cached=None)
        await _collect(_backend([_chunk("Hi", finish="STOP", usage=md)]), sink)
        assert (sink.thinking_tokens, sink.cached_input_tokens) == (0, 0)

    async def test_a_running_total_is_assigned_not_summed(self) -> None:
        sink = UsageSink()
        await _collect(
            _backend(
                [
                    _chunk("a", usage=_metadata(thoughts=10, cached=50)),
                    _chunk("b", finish="STOP", usage=_metadata(thoughts=30, cached=50)),
                ]
            ),
            sink,
        )
        assert (sink.thinking_tokens, sink.cached_input_tokens) == (30, 50)

    async def test_a_later_block_without_the_keys_does_not_clear_them(self) -> None:
        sink = UsageSink()
        await _collect(
            _backend(
                [
                    _chunk("a", usage=_metadata(thoughts=30, cached=50)),
                    _chunk("b", finish="STOP", usage=_metadata(thoughts=None, cached=None)),
                ]
            ),
            sink,
        )
        assert (sink.thinking_tokens, sink.cached_input_tokens) == (30, 50)


class TestTheTwoReadSitesDoNotDrift:
    """The existing one-fixture/both-readers check, extended to all four counts."""

    async def test_all_four_counts_agree(self) -> None:
        metadata = _metadata(prompt=321, candidates=7, thoughts=99, cached=123)
        sink = UsageSink()
        await _collect(_backend([_chunk("x", finish="STOP", usage=metadata)]), sink)
        usage = _non_streaming(metadata, finish="STOP")["usage"]

        assert (
            sink.prompt_tokens,
            sink.completion_tokens,
            sink.thinking_tokens,
            sink.cached_input_tokens,
        ) == (
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["lexora_thinking_tokens"],
            usage["prompt_tokens_details"]["cached_tokens"],
        )

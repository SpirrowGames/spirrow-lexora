"""T-streaming-ledger-row PR-B: the gemini stream fills the `UsageSink`.

At `develop` (`d8b9698`) `GeminiBackend.chat_completions_stream` decoded every
upstream event with `json.loads` and read `candidates[0].content.parts[*].text`,
`candidates[0].finishReason` and `promptFeedback.blockReason` off it. It dropped
`usageMetadata` -- the same key `_to_openai_response` reads on the non-streaming
path, four hundred lines up in the same file. The streamed request was not
missing a count; it was discarding one, and the docstring said so and called it
PR-B.

WHAT IS MEASURED AND WHAT IS ASSUMED, stated first because the two are not the
same strength here. No Gemini API key exists in this environment -- checked, not
supposed -- so nothing in this repo can reach `generativelanguage.googleapis.com`
and the *wire shape* below is a premise, on the same footing as
`test_anthropic_stream_usage.py`'s event fixtures and the cache-field premise at
`anthropic.py:235`. What these cases do measure is the part that was wrong: that
the backend reads whatever that shape carries instead of discarding it.

Because the shape is a premise, the read is written to depend on as little of it
as possible, and three cases exist only to pin that:

- usage may arrive on one event or on every event -- `test_a_running_total
  _is_not_summed` pins the second, `test_the_count_survives_a_later_event
  _without_usage` the first;
- an event may carry one of the two counts and not the other --
  `test_a_partial_usage_block_does_not_clear_the_other_side`;
- a prompt-level block yields no candidate at all, and the handler `continue`s
  past it -- `test_a_blocked_prompt_still_records_what_it_was_billed` pins that
  the read happens before that `continue`, because a blocked prompt is still a
  prompt that was sent and billed.

The two field names are read in two places in `gemini.py` and deliberately not
factored into a shared helper: the non-streaming path sees one whole response,
where a missing key genuinely means zero, so the streaming presence check would
be wrong there. `test_the_two_read_sites_agree_on_the_field_names` drives both
sites off one fixture instead, so the names cannot drift apart without a red.

Mutation, so the detectors are measured rather than asserted. Counts below are
this file alone, 11 cases, run on win32 / CPython 3.12. Each row is a
single-site edit against the finished tree, restored before the next. The router
file (`tests/api/test_streaming_ledger_row.py`, 16 cases) was run under every
one and stayed **16 green throughout**, which is what shows these measure the
backend and not the wiring.

Three of the seven rows came back different from what I predicted before running
them; the measured numbers are what is written here and the two cases whose
stated reasons the measurement contradicted were rewritten rather than reworded.
The corrections are noted inline, because a mutation table that agrees with its
author's expectations is the one worth distrusting.

- Deleting the whole `usage_sink` block: **9 red / 2 green**. Green =
  `test_bytes_are_identical_with_and_without_a_sink` and
  `test_no_sink_is_still_a_working_stream`, both fences on the unchanged stream
  rather than detectors on the count. (Predicted 8/3: I expected
  `test_the_two_read_sites_agree_on_the_field_names` to survive on its
  non-streaming half. It does not and should not -- it compares the two halves,
  so a dead streaming half is exactly what it is for.)
- Deleting the `promptTokenCount` branch only: **7 red / 4 green** --
  `test_input_side_is_read`, `test_both_sides_land`, `test_a_blocked_prompt
  _still_records_what_it_was_billed`, `test_sink_is_not_shared_between_calls`,
  `test_the_count_survives_a_later_event_without_usage`, `test_a_partial_usage
  _block_does_not_clear_the_other_side`, `test_the_two_read_sites_agree_on_the
  _field_names`. `test_output_side_is_read` stays green, so the two sides of
  the bill are separately detected.
- Deleting the `candidatesTokenCount` branch only: **7 red / 4 green** --
  `test_output_side_is_read`, `test_both_sides_land`, `test_a_running_total_is
  _not_summed`, `test_the_count_survives_a_later_event_without_usage`,
  `test_a_partial_usage_block_does_not_clear_the_other_side`, `test_sink_is_not
  _shared_between_calls`, `test_the_two_read_sites_agree_on_the_field_names`.
  `test_input_side_is_read` and `test_a_blocked_prompt_still_records_what_it
  _was_billed` stay green here, the mirror of the row above.
- `candidatesTokenCount` accumulating (`+=`) instead of assigning: **1 red / 10
  green** -- `test_a_running_total_is_not_summed`, the only case sending two
  events that both carry usage. Every single-usage case stays green, which is
  why that case exists: without it the summation bug ships green and multiplies
  a bill.
- Dropping the two `in metadata` presence checks for an unconditional
  `.get(key, 0)`, block-level `isinstance` guard left in place: **1 red / 10
  green** -- `test_a_partial_usage_block_does_not_clear_the_other_side`, alone.
  (Predicted 2/9. `test_the_count_survives_a_later_event_without_usage` stays
  green, and the reason is worth having: an event with no `usageMetadata` at
  all gives `metadata is None`, which the `isinstance` guard already rejects
  before either field is read. So the two guards are separate mechanisms, not
  one, and the row below isolates the other.)
- An absent `usageMetadata` treated as an empty one (`event.get(...) or {}`
  plus the unconditional `.get`): **2 red / 9 green** --
  `test_the_count_survives_a_later_event_without_usage` and
  `test_a_partial_usage_block_does_not_clear_the_other_side`. This is the
  block-level guard's own detector, added after the row above showed the first
  case was measuring something else than I had claimed.
- Moving the block below the `if not candidates: ... continue`: **1 red / 10
  green** -- `test_a_blocked_prompt_still_records_what_it_was_billed`. One
  case, and it is the only one that can see placement at all.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from lexora.backends.base import UsageSink
from lexora.backends.gemini import GeminiBackend

PROMPT_TOKENS = 137
CANDIDATE_TOKENS = 42

REQUEST: dict[str, Any] = {
    "model": "gemini-3.1-pro-preview",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 16,
}


def _usage(
    prompt: int | None = PROMPT_TOKENS, candidates: int | None = CANDIDATE_TOKENS
) -> dict[str, Any]:
    """A `usageMetadata` block, with either count omittable.

    `None` means "this event does not state that number", which is the case the
    presence checks exist for -- not "that number is zero".
    """
    metadata: dict[str, Any] = {}
    if prompt is not None:
        metadata["promptTokenCount"] = prompt
    if candidates is not None:
        metadata["candidatesTokenCount"] = candidates
    if prompt is not None and candidates is not None:
        metadata["totalTokenCount"] = prompt + candidates
    return metadata


def _chunk(
    text: str = "",
    finish: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One `streamGenerateContent` SSE payload."""
    candidate: dict[str, Any] = {"content": {"parts": [{"text": text}]}}
    if finish is not None:
        candidate["finishReason"] = finish
    event: dict[str, Any] = {"candidates": [candidate]}
    if usage is not None:
        event["usageMetadata"] = usage
    return event


def _blocked(usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """A prompt-level block: no candidate, and the handler `continue`s on it."""
    event: dict[str, Any] = {"promptFeedback": {"blockReason": "SAFETY"}}
    if usage is not None:
        event["usageMetadata"] = usage
    return event


FULL_STREAM = [
    _chunk("Hi"),
    _chunk(" there", finish="STOP", usage=_usage()),
]


class _FakeStreamResponse:
    """A 200 whose `aiter_lines` replays canned upstream SSE lines."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._events = events

    async def aread(self) -> bytes:  # pragma: no cover - success path only
        return b""

    async def aiter_lines(self):
        for event in self._events:
            # `alt=sse` interleaves blank lines; the handler skips them, and
            # including them keeps this fixture from quietly depending on a
            # tidier stream than exists.
            yield f"data: {json.dumps(event)}"
            yield ""


class _FakeStreamCM:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


def _backend(events: list[dict[str, Any]]) -> GeminiBackend:
    backend = GeminiBackend(name="naysayer")
    backend._client.stream = MagicMock(
        return_value=_FakeStreamCM(_FakeStreamResponse(events))
    )
    return backend


async def _collect(
    backend: GeminiBackend, usage_sink: UsageSink | None
) -> list[bytes]:
    return [
        chunk
        async for chunk in backend.chat_completions_stream(
            dict(REQUEST), usage_sink=usage_sink
        )
    ]


class TestTheSinkIsFilled:
    """D-1: the count is produced inside the backend, from what it already parsed."""

    @pytest.mark.asyncio
    async def test_input_side_is_read(self) -> None:
        sink = UsageSink()
        await _collect(_backend(FULL_STREAM), sink)
        assert sink.prompt_tokens == PROMPT_TOKENS

    @pytest.mark.asyncio
    async def test_output_side_is_read(self) -> None:
        sink = UsageSink()
        await _collect(_backend(FULL_STREAM), sink)
        assert sink.completion_tokens == CANDIDATE_TOKENS

    @pytest.mark.asyncio
    async def test_both_sides_land(self) -> None:
        """Asserted together as well as apart: refuses a half-filled sink."""
        sink = UsageSink()
        await _collect(_backend(FULL_STREAM), sink)
        assert (sink.prompt_tokens, sink.completion_tokens) == (
            PROMPT_TOKENS,
            CANDIDATE_TOKENS,
        )

    @pytest.mark.asyncio
    async def test_a_running_total_is_not_summed(self) -> None:
        """`usageMetadata` states the total so far, it does not add to it.

        Two events carrying usage, `candidatesTokenCount` 12 then 42: the bill
        is 42, the last value the upstream stated. Summing to 54 would inflate
        the invoice on any stream that reports usage more than once, and no
        single-usage case can see that.
        """
        sink = UsageSink()
        await _collect(
            _backend(
                [
                    _chunk("Hi", usage=_usage(candidates=12)),
                    _chunk(" there", finish="STOP", usage=_usage(candidates=42)),
                ]
            ),
            sink,
        )
        assert sink.completion_tokens == 42

    @pytest.mark.asyncio
    async def test_the_count_survives_a_later_event_without_usage(self) -> None:
        """Usage on a middle event, nothing on the last one.

        Whether the upstream reports usage on every event or only on the final
        one is exactly the part of the wire shape this repo cannot measure, so
        the read must not depend on which it is.

        What this pins is the BLOCK-level guard, not the per-field checks --
        measured, and not what I first wrote here. An event with no
        `usageMetadata` gives `metadata is None`, and `isinstance(metadata,
        dict)` rejects it before either field is looked at, so dropping the two
        `in metadata` checks leaves this case green. What reds it is treating an
        absent block as an empty one (`event.get(...) or {}` with unconditional
        `.get`), which would clear a count that had already arrived and bill the
        request as free. `test_a_partial_usage_block_does_not_clear_the_other
        _side` is the per-field detector; this is the block one.
        """
        sink = UsageSink()
        await _collect(
            _backend([_chunk("Hi", usage=_usage()), _chunk(" there", finish="STOP")]),
            sink,
        )
        assert (sink.prompt_tokens, sink.completion_tokens) == (
            PROMPT_TOKENS,
            CANDIDATE_TOKENS,
        )

    @pytest.mark.asyncio
    async def test_a_partial_usage_block_does_not_clear_the_other_side(self) -> None:
        """One event states both counts, the next restates only the prompt side.

        The same defect one level finer than the case above: the block is
        present, so a presence check on the block alone would pass, and only a
        per-field check keeps `candidatesTokenCount` from being zeroed by an
        event that never mentioned it.
        """
        sink = UsageSink()
        await _collect(
            _backend(
                [
                    _chunk("Hi", usage=_usage()),
                    _chunk(
                        " there",
                        finish="STOP",
                        usage=_usage(prompt=PROMPT_TOKENS, candidates=None),
                    ),
                ]
            ),
            sink,
        )
        assert (sink.prompt_tokens, sink.completion_tokens) == (
            PROMPT_TOKENS,
            CANDIDATE_TOKENS,
        )

    @pytest.mark.asyncio
    async def test_a_blocked_prompt_still_records_what_it_was_billed(self) -> None:
        """A safety block yields no candidate, and the handler `continue`s.

        The prompt was still sent and still billed, so the read has to happen
        before that `continue` or the one case where the client receives no
        content is also the one case that bills nothing. This is the only case
        in the file that can see where the block sits.
        """
        sink = UsageSink()
        chunks = await _collect(
            _backend([_blocked(usage=_usage(candidates=0))]), sink
        )
        assert sink.prompt_tokens == PROMPT_TOKENS
        # And the block is still surfaced as `content_filter`, unchanged.
        assert any(b"content_filter" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_sink_is_not_shared_between_calls(self) -> None:
        """D-2: the carrier is per request, so two calls cannot collide.

        The backend instance is reused on purpose -- that is the production
        shape, one cached instance serving every request -- while the sinks are
        separate, and each keeps its own numbers. An attribute on the backend
        would not have this property, so the reason it was rejected lives in
        the suite and not only in a comment.
        """
        backend = GeminiBackend(name="naysayer")
        first, second = UsageSink(), UsageSink()

        backend._client.stream = MagicMock(
            return_value=_FakeStreamCM(
                _FakeStreamResponse(
                    [_chunk("a", finish="STOP", usage=_usage(11, 5))]
                )
            )
        )
        await _collect(backend, first)

        backend._client.stream = MagicMock(
            return_value=_FakeStreamCM(
                _FakeStreamResponse(
                    [_chunk("b", finish="STOP", usage=_usage(999, 888))]
                )
            )
        )
        await _collect(backend, second)

        assert (first.prompt_tokens, first.completion_tokens) == (11, 5)
        assert (second.prompt_tokens, second.completion_tokens) == (999, 888)


class TestTheTwoReadSitesDoNotDrift:
    """The streaming read and `_to_openai_response` must name the same keys."""

    @pytest.mark.asyncio
    async def test_the_two_read_sites_agree_on_the_field_names(self) -> None:
        """One `usageMetadata` object, both readers, compared.

        `gemini.py` reads `promptTokenCount` / `candidatesTokenCount` twice --
        once per whole response, once per streamed event -- with deliberately
        different missing-key semantics, so they are not one helper. This is
        what stands in for that: rename a key at either site and the two sides
        stop matching.

        It is a comparison and not a bracket, so it reds for *any* divergence
        between the two readers, including a streaming half that reads nothing
        at all -- 0/0 against 137/42. That is why it appears in the red set of
        every mutation that touches the count and not only in the renaming one.
        A case that only detects renames would have to name the keys itself,
        which is the third copy this exists to avoid.
        """
        metadata = _usage()
        sink = UsageSink()
        await _collect(
            _backend([_chunk("Hi", finish="STOP", usage=metadata)]), sink
        )

        non_streaming = GeminiBackend(name="naysayer")._to_openai_response(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": metadata,
            },
            "gemini-3.1-pro-preview",
        )

        assert (sink.prompt_tokens, sink.completion_tokens) == (
            non_streaming["usage"]["prompt_tokens"],
            non_streaming["usage"]["completion_tokens"],
        )


class TestTheStreamItselfIsUnchanged:
    """F: reading the count must not alter one byte the client receives."""

    @pytest.mark.asyncio
    async def test_bytes_are_identical_with_and_without_a_sink(self) -> None:
        """The same canned upstream relayed twice, once with a sink.

        Compared as whole byte lists rather than by spot checks, so a chunk
        added, dropped or reordered by the usage read shows up here.
        """
        with_sink = _backend(FULL_STREAM)
        without_sink = _backend(FULL_STREAM)

        # `chunk_id` is a fresh uuid4 per call and `created` a wall-clock
        # second, so the two runs are normalised on exactly those two fields
        # and on nothing else.
        def _normalise(chunks: list[bytes]) -> list[Any]:
            out: list[Any] = []
            for chunk in chunks:
                text = chunk.decode()
                if not text.startswith("data: ") or text.strip() == "data: [DONE]":
                    out.append(text)
                    continue
                payload = json.loads(text[6:])
                payload.pop("id", None)
                payload.pop("created", None)
                out.append(payload)
            return out

        assert _normalise(await _collect(with_sink, UsageSink())) == _normalise(
            await _collect(without_sink, None)
        )

    @pytest.mark.asyncio
    async def test_no_sink_is_still_a_working_stream(self) -> None:
        """`usage_sink=None`, the default every pre-existing caller uses.

        Driven explicitly: that default is what keeps every non-router caller
        working untouched.
        """
        chunks = await _collect(_backend(FULL_STREAM), None)
        assert chunks[-1] == b"data: [DONE]\n\n"
        assert any(b'"content": "Hi"' in c or b'"content":"Hi"' in c for c in chunks)

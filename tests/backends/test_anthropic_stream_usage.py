"""T-streaming-ledger-row PR-A: the anthropic stream fills the `UsageSink`.

At `develop` (`c8a97dc`) `AnthropicBackend.chat_completions_stream` decoded
every upstream event with `json.loads`, read three fields off the result --
`type`, `delta.text`, `delta.stop_reason` -- and dropped the rest. Two of the
dropped fields are the whole bill: `message_start` carries
`message.usage.input_tokens`, `message_delta` carries `usage.output_tokens`,
and the non-streaming twin reads those two numbers off that shape already
(`anthropic.py`, "# Map usage"). The streamed request was not missing a count;
it was discarding one.

Anthropic goes first of the three parsing backends because its usage arrives
split across *two* events. A design assuming one terminal frame carries the
pair would work for `claude_code` and `gemini` and fail here.

The event fixtures are upstream wire shape, which this repo cannot measure --
nothing here reaches api.anthropic.com. They are a premise, on the same
footing as the cache-field premise at `anthropic.py:235`, and their nesting
asymmetry is cross-checked against `anthropic_compat.py`'s own
`message_start`. What these cases do measure is the part that was wrong: that
the backend reads whatever that shape carries instead of discarding it.

Mutation, so the detectors are measured rather than asserted. Counts are this
file alone, 7 cases, run on win32 / CPython 3.12.

Against `develop` this file does not red -- it fails to *collect*, on
`ImportError: cannot import name 'UsageSink'`, so 0 of the 7 cases run. That
is a weak red and is recorded as one: it says the symbol is absent, not that
any count is read. What binds each detector to a line is the table below, run
against the finished tree, one single-site edit at a time, restored before the
next. The router file (`tests/api/test_streaming_ledger_row.py`, 16 cases)
stayed 16 green under every one of them, which is what shows these four are
measuring the backend and not the wiring:

- Deleting the `message_start` block: 3 red / 4 green -- `test_input_side_is
  _read`, `test_both_sides_land`, `test_sink_is_not_shared_between_calls`.
  The output-side case, the running-total case and both fences stay green, so
  the two sides of the bill are separately detected.
- Deleting the `message_delta` block: 4 red / 3 green -- `test_output_side_is
  _read`, `test_both_sides_land`, `test_a_running_total_is_not_summed`,
  `test_sink_is_not_shared_between_calls`. Overlaps the row above only at the
  cases that read both sides on purpose: `test_input_side_is_read` stays green
  here and `test_output_side_is_read` stays green there.
- `message_delta` accumulating (`+=`) instead of assigning: 1 red / 6 green --
  `test_a_running_total_is_not_summed`, the only case sending two
  `message_delta` events. Every single-delta case stays green, which is why
  that case exists: without it the summation bug ships green.
- Reading `event_data.get("usage")` on `message_start`, i.e. losing the
  nesting: 3 red / 4 green, the same red set as deleting the block, because
  the key is absent at the top level. The fixture's nesting is load-bearing.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from lexora.backends.anthropic import AnthropicBackend
from lexora.backends.base import UsageSink

INPUT_TOKENS = 137
OUTPUT_TOKENS = 42

REQUEST: dict[str, Any] = {
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 16,
}


def _message_start(input_tokens: int = INPUT_TOKENS) -> dict[str, Any]:
    """The upstream's opening event.

    `usage` sits under `message` here and at the top level in `message_delta`;
    the asymmetry is the vendor's, not this file's. `anthropic_compat.py`
    builds the same event with the same nesting.
    """
    return {
        "type": "message_start",
        "message": {
            "id": "msg_x",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }


def _content_delta(text: str) -> dict[str, Any]:
    return {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    }


def _message_delta(output_tokens: int = OUTPUT_TOKENS) -> dict[str, Any]:
    return {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }


MESSAGE_STOP: dict[str, Any] = {"type": "message_stop"}


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
            # The real transport interleaves `event:` lines and blank lines;
            # both are skipped by the handler, and including them keeps this
            # fixture from quietly depending on a tidier stream than exists.
            yield f"event: {event['type']}"
            yield f"data: {json.dumps(event)}"
            yield ""


class _FakeStreamCM:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


def _backend(events: list[dict[str, Any]]) -> AnthropicBackend:
    backend = AnthropicBackend(api_key="sk-ant-test", name="frontier")
    backend._client.stream = MagicMock(
        return_value=_FakeStreamCM(_FakeStreamResponse(events))
    )
    return backend


async def _collect(
    backend: AnthropicBackend, usage_sink: UsageSink | None
) -> list[bytes]:
    return [
        chunk
        async for chunk in backend.chat_completions_stream(
            dict(REQUEST), usage_sink=usage_sink
        )
    ]


FULL_STREAM = [
    _message_start(),
    _content_delta("Hi"),
    _message_delta(),
    MESSAGE_STOP,
]


class TestTheSinkIsFilled:
    """D-1: the count is produced inside the backend, from what it already parsed."""

    @pytest.mark.asyncio
    async def test_input_side_is_read(self) -> None:
        sink = UsageSink()
        await _collect(_backend(FULL_STREAM), sink)
        assert sink.prompt_tokens == INPUT_TOKENS

    @pytest.mark.asyncio
    async def test_output_side_is_read(self) -> None:
        sink = UsageSink()
        await _collect(_backend(FULL_STREAM), sink)
        assert sink.completion_tokens == OUTPUT_TOKENS

    @pytest.mark.asyncio
    async def test_both_sides_land(self) -> None:
        """Asserted together as well as apart: refuses a half-filled sink."""
        sink = UsageSink()
        await _collect(_backend(FULL_STREAM), sink)
        assert (sink.prompt_tokens, sink.completion_tokens) == (
            INPUT_TOKENS,
            OUTPUT_TOKENS,
        )

    @pytest.mark.asyncio
    async def test_a_running_total_is_not_summed(self) -> None:
        """The upstream's `message_delta` count is cumulative, not incremental.

        Two `message_delta` events, 40 then 42: the bill is 42, the last value
        the upstream stated. Summing them to 82 would nearly double the
        invoice on any stream that emits more than one, and no single-delta
        case can see that.
        """
        sink = UsageSink()
        await _collect(
            _backend(
                [
                    _message_start(),
                    _content_delta("Hi"),
                    _message_delta(40),
                    _message_delta(42),
                    MESSAGE_STOP,
                ]
            ),
            sink,
        )
        assert sink.completion_tokens == 42

    @pytest.mark.asyncio
    async def test_sink_is_not_shared_between_calls(self) -> None:
        """D-2: the carrier is per request, so two calls cannot collide.

        The backend instance is reused on purpose -- that is the production
        shape, one cached instance serving every request -- while the sinks
        are separate, and each keeps its own numbers. An attribute on the
        backend would not have this property, so the reason it was rejected
        lives in the suite and not only in a comment.
        """
        backend = AnthropicBackend(api_key="sk-ant-test", name="frontier")
        first, second = UsageSink(), UsageSink()

        backend._client.stream = MagicMock(
            return_value=_FakeStreamCM(
                _FakeStreamResponse([_message_start(11), _message_delta(5), MESSAGE_STOP])
            )
        )
        await _collect(backend, first)

        backend._client.stream = MagicMock(
            return_value=_FakeStreamCM(
                _FakeStreamResponse(
                    [_message_start(999), _message_delta(888), MESSAGE_STOP]
                )
            )
        )
        await _collect(backend, second)

        assert (first.prompt_tokens, first.completion_tokens) == (11, 5)
        assert (second.prompt_tokens, second.completion_tokens) == (999, 888)


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

        Driven explicitly: that default is what keeps the other 617 tests, and
        every non-router caller, working untouched.
        """
        chunks = await _collect(_backend(FULL_STREAM), None)
        assert chunks[-1] == b"data: [DONE]\n\n"
        assert any(b'"content": "Hi"' in c or b'"content":"Hi"' in c for c in chunks)

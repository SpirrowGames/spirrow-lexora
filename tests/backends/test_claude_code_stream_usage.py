"""T-streaming-ledger-row PR-B: the claude_code stream fills the `UsageSink`.

At `develop` (`d8b9698`) `ClaudeCodeBackend.chat_completions_stream` read two
fields off the CLI's `result` event -- `type` and `stop_reason` -- and dropped
the `usage` object sitting beside them. That object is the same one
`_tokens_from_result` reads on the non-streaming path (`chat_completions`, "the
result JSON"), so the streamed request was not missing a count; it was
discarding one, and the docstring said so and called it PR-B.

THIS ONE IS MEASURED, which is why it reads differently from its two siblings.
`test_anthropic_stream_usage.py` and `test_gemini_stream_usage.py` both rest on
an unmeasured wire shape -- nothing in this repo reaches api.anthropic.com and
no Gemini key exists here. The Claude Code CLI is a local subprocess, so its
shape can be, and was, read directly. Run 2026-09-07 on CLI **2.1.263** (the
same build the `usage` samples in `test_claude_code.py` were taken on) with the
command `_build_command` assembles for THIS path, `stream=True`:

    claude -p --model sonnet --output-format stream-json --verbose
           --no-session-persistence --max-turns 1

Four events came back: `rate_limit_event`, `system`, `assistant`, `result`. The
`result` event's top-level `usage` was

    input_tokens 2, cache_creation_input_tokens 14495,
    cache_read_input_tokens 23732, output_tokens 4

-- the same four keys, at the same nesting, that `--output-format json` gives
the non-streaming path. So "the `result` event is the same object
`_tokens_from_result` already reads" is now a measurement rather than a premise,
and SAMPLE_D below is that reading verbatim.

Two things that run also settled, both of which shape the code:

1. The `assistant` event carries a `message.usage` of its own, with the same
   keys. On this single-turn call it agreed with the result event. On a
   multi-turn call there is one assistant event per turn and only `result`
   carries the invocation's total, so filling from assistant events is exactly
   the "second summation" this work is forbidden to write.
   `test_assistant_event_usage_is_not_the_source` drives that difference.
2. `modelUsage` again listed two models (`claude-sonnet-5` **and**
   `claude-haiku-4-5-20251001`, 8 output tokens of its own) while the top-level
   `usage.output_tokens` was 4 -- the third independent confirmation of the note
   at `_tokens_from_result`, that `completion_tokens` describes one completion
   and not the whole invocation. Unchanged here on purpose: that is a question
   about what the field means, and this work does not answer it.

`_tokens_from_result` is called, not re-implemented. Its three-field input-side
sum is measured and documented at its own definition and there must not be two
copies of that arithmetic;
`test_the_streamed_count_is_the_non_streaming_reader_s_own_answer` is the
detector that keeps it one.

Mutation, so the detectors are measured rather than asserted. Counts below are
this file alone, 8 cases, run on win32 / CPython 3.12. Each row is a single-site
edit against the finished tree, restored before the next. The router file
(`tests/api/test_streaming_ledger_row.py`, 16 cases) was run under every one and
stayed **16 green throughout**, which is what shows these measure the backend
and not the wiring.

Two of the four rows came back different from what I predicted. The measured
numbers are what is written here, with the reason each prediction was wrong,
because a mutation table that agrees with its author's expectations is the one
worth distrusting.

- Deleting the whole `usage_sink` block: **6 red / 2 green**. Green =
  `test_bytes_are_identical_with_and_without_a_sink` and
  `test_no_sink_is_still_a_working_stream`, both fences on the unchanged stream
  rather than detectors on the count.
- Assigning `usage["input_tokens"]` to `prompt_tokens` instead of calling
  `_tokens_from_result` -- i.e. re-implementing it, wrongly, in the way the
  non-streaming path was wrong until 2026-09-06: **3 red / 5 green** --
  `test_the_result_event_fills_both_sides`, `test_the_cached_prompt_is_part_of
  _the_bill`, `test_the_streamed_count_is_the_non_streaming_reader_s_own
  _answer`. (Predicted 4/3, counting `test_sink_is_not_shared_between_calls`.
  It stays green, and the reason is a real limit on that case: its two vectors
  carry no cache fields at all, so the three-field sum and `input_tokens` alone
  give the same number. It measures that two sinks do not collide, and it
  cannot also measure what is put in them.)
- Accumulating (`+=`) instead of assigning: **1 red / 7 green** --
  `test_a_second_result_event_is_not_summed`, the only case sending two `result`
  events. Every single-result case stays green, which is why that case exists.
- Filling from the `assistant` event's `message.usage` instead of the `result`
  event's: **4 red / 4 green** -- `test_assistant_event_usage_is_not_the
  _source`, `test_a_second_result_event_is_not_summed`, `test_the_streamed
  _count_is_the_non_streaming_reader_s_own_answer`, `test_sink_is_not_shared
  _between_calls`. (Predicted 1/7.) The relocation is one edit but two effects:
  it removes the authoritative source *and* installs a wrong one, so three of
  those four red at zero, merely because their streams carry no assistant usage
  to find. Only `test_assistant_event_usage_is_not_the_source` reds on a
  *different non-zero number* -- 7 against the result event's 4 -- and it is
  therefore the only case here that distinguishes the two sources rather than
  the presence of one.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lexora.backends.base import UsageSink
from lexora.backends.claude_code import ClaudeCodeBackend

# Measured 2026-09-07, CLI 2.1.263, the streaming command above. The input side
# sums to 38,229 while `input_tokens` alone is 2 -- a factor of nineteen
# thousand, which is the defect `_tokens_from_result` was written to fix and
# which this path would have re-introduced had it summed for itself.
SAMPLE_D_USAGE: dict[str, Any] = {
    "input_tokens": 2,
    "cache_creation_input_tokens": 14495,
    "cache_read_input_tokens": 23732,
    "output_tokens": 4,
}
SAMPLE_D_PROMPT_TOKENS = 2 + 14495 + 23732  # 38,229
SAMPLE_D_COMPLETION_TOKENS = 4

REQUEST: dict[str, Any] = {
    "model": "claude-code-sonnet",
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
}


def _result_event(usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """The CLI's terminal event, trimmed to the keys this handler reads.

    The real event carries twenty-odd more (`duration_ms`, `total_cost_usd`,
    `modelUsage`, ...); they are dropped here because the handler reads none of
    them and carrying them would suggest it did.
    """
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "ok",
        "stop_reason": "end_turn",
        "usage": dict(SAMPLE_D_USAGE if usage is None else usage),
    }


def _assistant_event(text: str = "ok", usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """A per-turn assistant event. It carries usage of its own -- measured."""
    message: dict[str, Any] = {
        "role": "assistant",
        "type": "message",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }
    if usage is not None:
        message["usage"] = dict(usage)
    return {"type": "assistant", "message": message}


SYSTEM_EVENT: dict[str, Any] = {"type": "system", "subtype": "init", "model": "sonnet"}
RATE_LIMIT_EVENT: dict[str, Any] = {"type": "rate_limit_event", "rate_limit_info": {}}

# The four events the measured run produced, in the order it produced them.
FULL_STREAM: list[dict[str, Any]] = [
    RATE_LIMIT_EVENT,
    SYSTEM_EVENT,
    _assistant_event(usage=SAMPLE_D_USAGE),
    _result_event(),
]


def _process(events: list[dict[str, Any]]) -> MagicMock:
    """A fake `asyncio` subprocess that replays canned stream-json lines.

    `readline` returns `b""` once the events are exhausted, which is the EOF
    the handler breaks on. No `claude` binary is invoked.
    """
    lines = [(json.dumps(event) + "\n").encode("utf-8") for event in events]
    lines.append(b"")

    process = MagicMock()
    process.returncode = 0
    process.stdin = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdout = MagicMock()
    process.stdout.readline = AsyncMock(side_effect=lines)
    process.wait = AsyncMock(return_value=0)
    return process


async def _collect(
    events: list[dict[str, Any]], usage_sink: UsageSink | None
) -> list[bytes]:
    backend = ClaudeCodeBackend(model="sonnet")
    with patch(
        "asyncio.create_subprocess_exec", new=AsyncMock(return_value=_process(events))
    ):
        return [
            chunk
            async for chunk in backend.chat_completions_stream(
                dict(REQUEST), usage_sink=usage_sink
            )
        ]


class TestTheSinkIsFilled:
    """D-1: the count is produced inside the backend, from what it already parsed."""

    @pytest.mark.asyncio
    async def test_the_result_event_fills_both_sides(self) -> None:
        sink = UsageSink()
        await _collect(FULL_STREAM, sink)
        assert (sink.prompt_tokens, sink.completion_tokens) == (
            SAMPLE_D_PROMPT_TOKENS,
            SAMPLE_D_COMPLETION_TOKENS,
        )

    @pytest.mark.asyncio
    async def test_the_cached_prompt_is_part_of_the_bill(self) -> None:
        """The uncached remainder alone is not the count.

        Asserted as an inequality as well as an equality because the failure
        this guards against is not "wrong number" in general -- it is one
        specific wrong number, `usage.input_tokens`, which reads 2 for a prompt
        of 38,229. That is the shape the non-streaming path shipped until
        2026-09-06, and a fresh read written here rather than a call to
        `_tokens_from_result` is how it would come back.
        """
        sink = UsageSink()
        await _collect(FULL_STREAM, sink)
        assert sink.prompt_tokens == SAMPLE_D_PROMPT_TOKENS
        assert sink.prompt_tokens != SAMPLE_D_USAGE["input_tokens"]

    @pytest.mark.asyncio
    async def test_a_second_result_event_is_not_summed(self) -> None:
        """A `result` event restates the total; it does not add to it.

        Two of them, output side 4 then 9: the bill is 9. Summing to 13 would
        inflate the invoice, and no single-result case can see that.
        """
        sink = UsageSink()
        await _collect(
            [
                _assistant_event(),
                _result_event(),
                _result_event({**SAMPLE_D_USAGE, "output_tokens": 9}),
            ],
            sink,
        )
        assert sink.completion_tokens == 9

    @pytest.mark.asyncio
    async def test_the_streamed_count_is_the_non_streaming_reader_s_own_answer(
        self,
    ) -> None:
        """There must be exactly one summation, and this is which one.

        The requirement is not "the streamed number is correct" -- it is that
        the streamed number is produced by `_tokens_from_result` itself, so the
        two paths cannot come to disagree about what one prompt cost. Compared
        against that function's own return value on the same event rather than
        against a literal, so a change to the summation moves both sides
        together and only a *second* implementation reds this.
        """
        event = _result_event()
        sink = UsageSink()
        await _collect([_assistant_event(), event], sink)
        assert (
            sink.prompt_tokens,
            sink.completion_tokens,
        ) == ClaudeCodeBackend._tokens_from_result(event)

    @pytest.mark.asyncio
    async def test_assistant_event_usage_is_not_the_source(self) -> None:
        """Measured: `assistant` events carry a `message.usage` of their own.

        On a single-turn call the two agree, which is why a stream built only
        from the measured run cannot tell them apart. Here they are made to
        disagree -- the assistant event reports 7 output tokens, the result
        event 4 -- and the invocation total is the one that must be billed. On
        a genuine multi-turn call there is one assistant event per turn, so
        reading them instead would bill either a fragment or, summed, a
        multiple.
        """
        sink = UsageSink()
        await _collect(
            [
                _assistant_event(usage={**SAMPLE_D_USAGE, "output_tokens": 7}),
                _result_event(),
            ],
            sink,
        )
        assert sink.completion_tokens == SAMPLE_D_COMPLETION_TOKENS

    @pytest.mark.asyncio
    async def test_sink_is_not_shared_between_calls(self) -> None:
        """D-2: the carrier is per request, so two calls cannot collide.

        One backend instance serves both calls -- the production shape, since
        instances are cached and shared -- while the sinks are separate and
        each keeps its own numbers. An attribute on the backend would not have
        this property, so the reason it was rejected lives in the suite and not
        only in a comment.
        """
        backend = ClaudeCodeBackend(model="sonnet")
        first, second = UsageSink(), UsageSink()

        async def _drive(events: list[dict[str, Any]], sink: UsageSink) -> None:
            with patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=_process(events)),
            ):
                async for _ in backend.chat_completions_stream(
                    dict(REQUEST), usage_sink=sink
                ):
                    pass

        await _drive(
            [_result_event({"input_tokens": 11, "output_tokens": 5})], first
        )
        await _drive(
            [_result_event({"input_tokens": 999, "output_tokens": 888})], second
        )

        assert (first.prompt_tokens, first.completion_tokens) == (11, 5)
        assert (second.prompt_tokens, second.completion_tokens) == (999, 888)


class TestTheStreamItselfIsUnchanged:
    """F: reading the count must not alter one byte the client receives."""

    @pytest.mark.asyncio
    async def test_bytes_are_identical_with_and_without_a_sink(self) -> None:
        """The same canned CLI output replayed twice, once with a sink.

        Compared as whole byte lists rather than by spot checks, so a chunk
        added, dropped or reordered by the usage read shows up here.
        """

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

        assert _normalise(await _collect(FULL_STREAM, UsageSink())) == _normalise(
            await _collect(FULL_STREAM, None)
        )

    @pytest.mark.asyncio
    async def test_no_sink_is_still_a_working_stream(self) -> None:
        """`usage_sink=None`, the default every pre-existing caller uses.

        Driven explicitly: that default is what keeps every non-router caller
        working untouched.
        """
        chunks = await _collect(FULL_STREAM, None)
        assert chunks[-1] == b"data: [DONE]\n\n"
        assert any(b'"content": "ok"' in c or b'"content":"ok"' in c for c in chunks)

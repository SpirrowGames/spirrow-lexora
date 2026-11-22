"""T-frontier-tier-to-fable-5-2: the upstream's own `model`, and thinking stays out of `content`.

Two things are fenced here (msg-1145 Obj-1-2 / Obj-1-3, msg-1147 Obj-2-1):

(i) `anthropic_upstream_response` -- the info line every Anthropic-backend
    response emits. Lexora's own response carries the model it *sent*, so
    this line is the only record of what the upstream says ran. Three
    conditions, non-stream and stream each:
      * match    -- upstream reports the sent ID; logged verbatim.
      * mismatch -- upstream reports another ID; logged verbatim (the
                    comparison is the reader's; nothing is suppressed).
      * missing  -- upstream omits `model`; logged as None, no exception.
                    None is "not confirmed", never a match.

(ii) A response mixing `thinking` and `text` blocks: non-stream `content` is
     the text only; in a stream, `thinking_delta` / `signature_delta` produce
     no chunk. This is the evidence that thinking does not leak into
     `content` -- a live request can only show that the upstream *returned*
     a thinking block (its `block_types`), not that it was removed.

The upstream wire shapes are fixtures (nothing here reaches
api.anthropic.com), on the same footing as
`tests/backends/test_anthropic_stream_usage.py`.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from lexora.backends import anthropic as anthropic_mod
from lexora.backends.anthropic import AnthropicBackend
from lexora.services.trace import set_current_trace_id

SENT = "claude-fable-5-2"
TRACE = "01JD0000000000000000000000"
THINKING_TEXT = "SECRET-REASONING-should-never-surface"
SIGNATURE = "SIG-should-never-surface"

REQUEST: dict[str, Any] = {
    "model": SENT,
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 64,
}


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the module logger so the info line is read as a call, not text."""
    fake = MagicMock()
    monkeypatch.setattr(anthropic_mod, "logger", fake)
    set_current_trace_id(TRACE)
    yield fake
    set_current_trace_id(None)


def _upstream_lines(fake: MagicMock) -> list[dict[str, Any]]:
    return [
        c.kwargs
        for c in fake.info.call_args_list
        if c.args and c.args[0] == "anthropic_upstream_response"
    ]


# --- non-stream ------------------------------------------------------------


def _body(model: object = SENT, *, with_thinking: bool = False) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if with_thinking:
        content.append(
            {"type": "thinking", "thinking": THINKING_TEXT, "signature": SIGNATURE}
        )
    content.append({"type": "text", "text": "Hello"})
    body: dict[str, Any] = {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "content": content,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }
    if model is not _ABSENT:
        body["model"] = model
    return body


_ABSENT = object()


def _nonstream_backend(body: dict[str, Any]) -> AnthropicBackend:
    backend = AnthropicBackend(api_key="sk-ant-test", name="frontier")
    response = httpx.Response(
        200,
        json=body,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )

    async def _post(*_a: Any, **_k: Any) -> httpx.Response:
        return response

    backend._client.post = _post  # type: ignore[method-assign]
    return backend


class TestNonStreamUpstreamModel:
    @pytest.mark.asyncio
    async def test_match_is_logged(self, log: MagicMock) -> None:
        result = await _nonstream_backend(_body(SENT)).chat_completions(dict(REQUEST))
        [line] = _upstream_lines(log)
        assert line["upstream_model"] == SENT
        assert line["sent_model"] == SENT
        assert line["trace_id"] == TRACE
        assert line["backend"] == "frontier"
        assert line["block_types"] == ["text"]
        # Lexora's own response shape is unchanged: it still names what it sent.
        assert result["model"] == SENT

    @pytest.mark.asyncio
    async def test_mismatch_is_logged_verbatim(self, log: MagicMock) -> None:
        result = await _nonstream_backend(_body("claude-fable-5-1")).chat_completions(
            dict(REQUEST)
        )
        [line] = _upstream_lines(log)
        assert line["upstream_model"] == "claude-fable-5-1"
        assert line["sent_model"] == SENT
        assert result["model"] == SENT

    @pytest.mark.asyncio
    async def test_missing_is_none_not_an_exception(self, log: MagicMock) -> None:
        result = await _nonstream_backend(_body(_ABSENT)).chat_completions(dict(REQUEST))
        [line] = _upstream_lines(log)
        assert line["upstream_model"] is None
        assert result["choices"][0]["message"]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_non_string_model_is_none(self, log: MagicMock) -> None:
        await _nonstream_backend(_body({"weird": 1})).chat_completions(dict(REQUEST))
        [line] = _upstream_lines(log)
        assert line["upstream_model"] is None


class TestNonStreamThinkingStaysOut:
    @pytest.mark.asyncio
    async def test_content_is_text_only(self, log: MagicMock) -> None:
        result = await _nonstream_backend(
            _body(SENT, with_thinking=True)
        ).chat_completions(dict(REQUEST))
        content = result["choices"][0]["message"]["content"]
        assert content == "Hello"
        assert THINKING_TEXT not in json.dumps(result)
        assert SIGNATURE not in json.dumps(result)
        [line] = _upstream_lines(log)
        assert line["block_types"] == ["thinking", "text"]
        # The log carries types only, never the body.
        assert THINKING_TEXT not in json.dumps(line)


# --- stream ----------------------------------------------------------------


class _FakeStreamResponse:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._events = events

    async def aread(self) -> bytes:  # pragma: no cover - success path only
        return b""

    async def aiter_lines(self):
        for event in self._events:
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


def _events(model: object = SENT, *, with_thinking: bool = False) -> list[dict[str, Any]]:
    message: dict[str, Any] = {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }
    if model is not _ABSENT:
        message["model"] = model
    events: list[dict[str, Any]] = [{"type": "message_start", "message": message}]
    idx = 0
    if with_thinking:
        events += [
            {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "thinking_delta", "thinking": THINKING_TEXT},
            },
            {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "signature_delta", "signature": SIGNATURE},
            },
            {"type": "content_block_stop", "index": idx},
        ]
        idx += 1
    events += [
        {
            "type": "content_block_start",
            "index": idx,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {"type": "content_block_stop", "index": idx},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ]
    return events


async def _stream(events: list[dict[str, Any]]) -> list[bytes]:
    backend = AnthropicBackend(api_key="sk-ant-test", name="frontier")
    backend._client.stream = MagicMock(  # type: ignore[method-assign]
        return_value=_FakeStreamCM(_FakeStreamResponse(events))
    )
    return [c async for c in backend.chat_completions_stream(dict(REQUEST))]


def _deltas(chunks: list[bytes]) -> list[dict[str, Any]]:
    out = []
    for raw in chunks:
        text = raw.decode()
        assert text.startswith("data: ")
        payload = text[6:].strip()
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload)["choices"][0]["delta"])
    return out


class TestStreamUpstreamModel:
    @pytest.mark.asyncio
    async def test_match_is_logged(self, log: MagicMock) -> None:
        chunks = await _stream(_events(SENT))
        [line] = _upstream_lines(log)
        assert line["upstream_model"] == SENT
        assert line["sent_model"] == SENT
        assert line["trace_id"] == TRACE
        assert line["block_types"] == ["text"]
        assert all(
            json.loads(c.decode()[6:])["model"] == SENT
            for c in chunks
            if b"[DONE]" not in c
        )

    @pytest.mark.asyncio
    async def test_mismatch_is_logged_verbatim(self, log: MagicMock) -> None:
        await _stream(_events("claude-fable-5-1"))
        [line] = _upstream_lines(log)
        assert line["upstream_model"] == "claude-fable-5-1"
        assert line["sent_model"] == SENT

    @pytest.mark.asyncio
    async def test_missing_is_none_not_an_exception(self, log: MagicMock) -> None:
        chunks = await _stream(_events(_ABSENT))
        [line] = _upstream_lines(log)
        assert line["upstream_model"] is None
        assert chunks[-1] == b"data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_no_message_stop_leaves_no_line(self, log: MagicMock) -> None:
        """A truncated stream leaves no line -- read as 'not confirmed'."""
        await _stream(_events(SENT)[:-1])
        assert _upstream_lines(log) == []


class TestStreamThinkingStaysOut:
    @pytest.mark.asyncio
    async def test_thinking_and_signature_deltas_yield_no_chunk(
        self, log: MagicMock
    ) -> None:
        chunks = await _stream(_events(SENT, with_thinking=True))
        joined = b"".join(chunks).decode()
        assert THINKING_TEXT not in joined
        assert SIGNATURE not in joined
        text = "".join(d.get("content", "") for d in _deltas(chunks))
        assert text == "Hello"
        [line] = _upstream_lines(log)
        assert line["block_types"] == ["thinking", "text"]

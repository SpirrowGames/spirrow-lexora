"""T-streaming-ledger-row PR-C: vLLM's chat stream fills the ``UsageSink``.

Design: msg-879 (Bohr), approved msg-880 (Einstein), GO msg-883 (human).

* ``chat_completions_stream`` always sends ``stream_options.include_usage:
  true`` upstream, on a copy of the request, whatever the caller sent.
* The relay yields every upstream chunk byte-for-byte *before* the side-copy
  reader sees it; the reader splits on any SSE delimiter, caps its buffer at
  64 KiB, and catches its own exceptions.
* ``completions_stream`` is out of scope: it sends no ``stream_options``.

``PROBE_BODY`` is the exact ``probe.body`` the operator captured from vLLM
0.17.1 on 2026-10-20 (msg-874), rebuilt here byte for byte and pinned by its
recorded sha256 so a stray edit cannot turn it into a fixture of our own
invention. It is *measured*. The CRLF variant below is *synthetic*.
"""

import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from lexora.backends import vllm as vllm_module
from lexora.backends.base import UsageSink
from lexora.backends.vllm import USAGE_PARSE_BUFFER_CAP, VLLMBackend
from tests.api import test_streaming_ledger_row as route_harness

# --------------------------------------------------------------------------
# The measured fixture
# --------------------------------------------------------------------------

_PREFIX = (
    '{"id":"chatcmpl-af0fbeedbf1b853c","object":"chat.completion.chunk",'
    '"created":1792455711,"model":"Qwen3.8-27B",'
)
_EVENTS = [
    _PREFIX + '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
    '"logprobs":null,"finish_reason":null}],"prompt_token_ids":null}',
    *(
        _PREFIX + '"choices":[{"index":0,"delta":{"reasoning":"%s"},'
        '"logprobs":null,"finish_reason":null,"token_ids":null}]}' % word
        for word in ["The", " user", " wants", " me", " to", " reply", " with"]
    ),
    _PREFIX + '"choices":[{"index":0,"delta":{"reasoning":" the"},'
    '"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}]}',
    _PREFIX + '"choices":[],"usage":{"prompt_tokens":59,"total_tokens":67,'
    '"completion_tokens":8}}',
    "[DONE]",
]
PROBE_BODY = "".join(f"data: {e}\n\n" for e in _EVENTS).encode()
PROBE_SHA256 = "0f657071d249a1e0bd7ebc4a93acffc29ade72af335a86107fc8f57d93ebb9ab"
USAGE_EVENT = ("data: " + _EVENTS[-2] + "\n\n").encode()

#: Synthetic, not measured: the same events with CRLF delimiters.
CRLF_BODY = "".join(f"data: {e}\r\n\r\n" for e in _EVENTS).encode()


def test_fixture_is_the_measured_probe_body() -> None:
    assert len(PROBE_BODY) == 2285
    assert hashlib.sha256(PROBE_BODY).hexdigest() == PROBE_SHA256


# --------------------------------------------------------------------------
# A fake upstream that records what it was sent and when it was read
# --------------------------------------------------------------------------


class FakeUpstream:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.sent: dict[str, Any] | None = None
        self.log: list[tuple[str, int]] = []

    def stream(self, method: str, path: str, json: dict[str, Any]) -> Any:
        self.sent = json
        self.path = path
        upstream = self

        class _Response:
            status_code = 200

            async def __aenter__(self) -> "_Response":
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

            async def aiter_bytes(self) -> AsyncIterator[bytes]:
                for index, chunk in enumerate(upstream.chunks):
                    upstream.log.append(("read", index))
                    yield chunk

        return _Response()


def _backend(upstream: FakeUpstream) -> VLLMBackend:
    backend = VLLMBackend(name="vllm-test")
    backend._client = MagicMock()
    backend._client.stream = upstream.stream
    return backend


async def _relay(
    chunks: list[bytes], request: dict[str, Any] | None = None
) -> tuple[bytes, UsageSink, FakeUpstream]:
    upstream = FakeUpstream(chunks)
    sink = UsageSink()
    received = []
    async for chunk in _backend(upstream).chat_completions_stream(
        request or {"model": "Qwen3.8-27B", "messages": [], "stream": True}, sink
    ):
        received.append(chunk)
    return b"".join(received), sink, upstream


# --------------------------------------------------------------------------
# Reading the count
# --------------------------------------------------------------------------


class TestSinkIsFilled:
    @pytest.mark.asyncio
    async def test_probe_body_fills_59_and_8(self) -> None:
        relayed, sink, _ = await _relay([PROBE_BODY])
        assert (sink.prompt_tokens, sink.completion_tokens) == (59, 8)
        # Reasoning is inside completion_tokens; no split is reported, so
        # nothing is inferred.
        assert sink.thinking_tokens is None
        assert sink.cached_input_tokens is None

    @pytest.mark.asyncio
    async def test_crlf_variant_fills_the_same(self) -> None:
        """Synthetic fixture (msg-876 edge-case): CRLF delimiters."""
        relayed, sink, _ = await _relay([CRLF_BODY])
        assert relayed == CRLF_BODY
        assert (sink.prompt_tokens, sink.completion_tokens) == (59, 8)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [PROBE_BODY, CRLF_BODY], ids=["lf", "crlf"])
    async def test_every_split_offset_gives_the_same_sink(self, body: bytes) -> None:
        # One backend for every offset: building an httpx client per offset
        # costs about a minute here.
        upstream = FakeUpstream([])
        backend = _backend(upstream)
        for offset in range(len(body) + 1):
            upstream.chunks = [body[:offset], body[offset:]]
            sink = UsageSink()
            relayed = b"".join(
                [c async for c in backend.chat_completions_stream({"model": "m"}, sink)]
            )
            assert relayed == body, offset
            assert (sink.prompt_tokens, sink.completion_tokens) == (59, 8), offset

    @pytest.mark.asyncio
    async def test_one_byte_chunks_give_the_same_sink(self) -> None:
        chunks = [PROBE_BODY[i : i + 1] for i in range(len(PROBE_BODY))]
        relayed, sink, _ = await _relay(chunks)
        assert relayed == PROBE_BODY
        assert (sink.prompt_tokens, sink.completion_tokens) == (59, 8)


# --------------------------------------------------------------------------
# The relay never depends on the reader
# --------------------------------------------------------------------------


class TestRelayIsUntouched:
    @pytest.mark.asyncio
    async def test_bytes_are_identical_and_each_chunk_is_yielded_before_the_next_read(
        self,
    ) -> None:
        events = [f"data: {e}\n\n".encode() for e in _EVENTS]
        upstream = FakeUpstream(events)
        received: list[bytes] = []
        async for chunk in _backend(upstream).chat_completions_stream(
            {"model": "m", "messages": []}, UsageSink()
        ):
            # The chunk we just got was the last one read: nothing is held.
            assert upstream.log[-1] == ("read", len(received))
            received.append(chunk)
        assert received == events
        assert b"".join(received) == PROBE_BODY

    @pytest.mark.asyncio
    async def test_usage_chunk_reaches_the_client_before_the_sink_is_filled(
        self,
    ) -> None:
        """Yield first, read after: the consumer holds the usage chunk while
        the sink is still empty."""
        events = [f"data: {e}\n\n".encode() for e in _EVENTS]
        sink = UsageSink()
        seen_at_usage_chunk: tuple[int, int] | None = None
        async for chunk in _backend(FakeUpstream(events)).chat_completions_stream(
            {"model": "m", "messages": []}, sink
        ):
            if chunk == USAGE_EVENT:
                seen_at_usage_chunk = (sink.prompt_tokens, sink.completion_tokens)
        assert seen_at_usage_chunk == (0, 0)
        assert (sink.prompt_tokens, sink.completion_tokens) == (59, 8)

    @pytest.mark.asyncio
    async def test_a_raising_reader_cannot_reach_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(self: Any, event: bytes) -> None:
            raise RuntimeError("parser bug")

        monkeypatch.setattr(vllm_module._StreamUsageReader, "_read_event", boom)
        relayed, sink, _ = await _relay([PROBE_BODY])
        assert relayed == PROBE_BODY
        assert (sink.prompt_tokens, sink.completion_tokens) == (0, 0)

    @pytest.mark.asyncio
    async def test_malformed_usage_json_is_relayed_and_bills_nothing(self) -> None:
        body = PROBE_BODY.replace(b'"completion_tokens":8}}', b'"completion_tokens":8}')
        assert body != PROBE_BODY
        relayed, sink, _ = await _relay([body])
        assert relayed == body
        assert (sink.prompt_tokens, sink.completion_tokens) == (0, 0)

    @pytest.mark.asyncio
    async def test_one_bad_event_does_not_cost_the_later_count(self) -> None:
        """A non-JSON event earlier in the stream is skipped, not fatal: the
        reader keeps going and still reads the usage event after it."""
        body = b"data: {not json\n\n" + PROBE_BODY
        relayed, sink, _ = await _relay([body])
        assert relayed == body
        assert (sink.prompt_tokens, sink.completion_tokens) == (59, 8)

    @pytest.mark.asyncio
    async def test_buffer_cap_stops_reading_and_relays_everything(self) -> None:
        """An incomplete event over the cap ends reading for the request, so
        the valid usage event that follows it is not read."""
        oversize = b"data: " + b"x" * (USAGE_PARSE_BUFFER_CAP + 1)
        body = oversize + b"\n\n" + PROBE_BODY
        relayed, sink, _ = await _relay([oversize, b"\n\n" + PROBE_BODY])
        assert relayed == body
        assert (sink.prompt_tokens, sink.completion_tokens) == (0, 0)

    @pytest.mark.asyncio
    async def test_no_sink_still_relays(self) -> None:
        upstream = FakeUpstream([PROBE_BODY])
        received = [
            chunk
            async for chunk in _backend(upstream).chat_completions_stream(
                {"model": "m", "messages": []}
            )
        ]
        assert b"".join(received) == PROBE_BODY


# --------------------------------------------------------------------------
# Request injection
# --------------------------------------------------------------------------


class TestIncludeUsageIsAlwaysRequested:
    @pytest.mark.asyncio
    async def test_absent_is_injected(self) -> None:
        request = {"model": "m", "messages": []}
        _, _, upstream = await _relay([PROBE_BODY], request)
        assert upstream.sent is not None
        assert upstream.sent["stream_options"] == {"include_usage": True}
        assert "stream_options" not in request

    @pytest.mark.asyncio
    async def test_explicit_false_is_overridden(self) -> None:
        """msg-879: a caller cannot opt the request out of the ledger."""
        options = {"include_usage": False, "continuous_usage_stats": False}
        request = {"model": "m", "messages": [], "stream_options": options}
        _, sink, upstream = await _relay([PROBE_BODY], request)
        assert upstream.sent is not None
        assert upstream.sent["stream_options"] == {
            "include_usage": True,
            "continuous_usage_stats": False,
        }
        assert options == {"include_usage": False, "continuous_usage_stats": False}
        assert request["stream_options"] is options
        assert (sink.prompt_tokens, sink.completion_tokens) == (59, 8)

    @pytest.mark.asyncio
    async def test_explicit_true_is_kept(self) -> None:
        options = {"include_usage": True}
        request = {"model": "m", "messages": [], "stream_options": options}
        _, _, upstream = await _relay([PROBE_BODY], request)
        assert upstream.sent is not None
        assert upstream.sent["stream_options"] == {"include_usage": True}
        assert options == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_completions_stream_sends_no_stream_options(self) -> None:
        """Scope limit: only /v1/chat/completions was probed (msg-874)."""
        upstream = FakeUpstream([b"data: [DONE]\n\n"])
        sink = UsageSink()
        async for _ in _backend(upstream).completions_stream(
            {"model": "m", "prompt": "Hi"}, sink
        ):
            pass
        assert upstream.path == "/v1/completions"
        assert upstream.sent is not None
        assert "stream_options" not in upstream.sent
        assert (sink.prompt_tokens, sink.completion_tokens) == (0, 0)


# --------------------------------------------------------------------------
# Through the route
# --------------------------------------------------------------------------


class TestThroughTheChatRoute:
    def test_client_stream_options_reach_vllm_and_the_row_is_written(self) -> None:
        """`ChatCompletionRequest` has `extra: allow` and the handler sends
        `model_dump(exclude_none=True)`, so a caller's `stream_options` reach
        the backend; vLLM gets them with `include_usage` forced on, the
        client receives vLLM's bytes unchanged, and one row is written."""
        upstream = FakeUpstream([PROBE_BODY])
        backend = _backend(upstream)
        tracker = MagicMock()
        body = {
            **route_harness.CHAT_BODY,
            "stream_options": {"include_usage": False, "continuous_usage_stats": False},
        }
        response = route_harness._client(backend, tracker).post(
            "/v1/chat/completions", json=body
        )
        assert response.status_code == 200
        assert response.content == PROBE_BODY
        assert upstream.sent is not None
        assert upstream.sent["stream_options"] == {
            "include_usage": True,
            "continuous_usage_stats": False,
        }
        assert tracker.record.call_count == 1
        kwargs = tracker.record.call_args.kwargs
        assert (kwargs["tokens_input"], kwargs["tokens_output"]) == (59, 8)
        assert kwargs["tokens_thinking"] is None
        # The usage event the client did not ask for is in what it received.
        assert json.loads(USAGE_EVENT[6:].strip())["choices"] == []
        assert USAGE_EVENT in response.content

"""``redact_request`` beyond message text: tool calls and non-text blocks.

PR-gate on #96 (T-naysayer-codex-backend): a credential inside an OpenAI
``tool_calls[*].function.arguments`` string or an Anthropic ``tool_use``
``input`` reached the backend unredacted, because only ``content`` text was
scanned. The codex backend refuses both, but a ``redact: true`` tier also
answers through Gemini (``unrouted``, ``naysayer-gemini``, fallback).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from lexora.ingress.gate import redact_request

PREFIXED = "sk-ant-api03-1234567890abcdefghij"
PLAIN = "correcthorsebattery"


def _assert_gone(request: dict[str, Any], *secrets: str) -> None:
    text = json.dumps(request, ensure_ascii=False)
    for secret in secrets:
        assert secret not in text, secret


class TestOpenAIToolCalls:
    def _request(self, arguments: str) -> dict[str, Any]:
        return {
            "model": "naysayer",
            "messages": [
                {"role": "user", "content": "log in"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "login_service", "arguments": arguments},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
        }

    def test_gate_example_prefixed_key(self) -> None:
        request = self._request(json.dumps({"password": PREFIXED}))
        counts = redact_request(request)
        _assert_gone(request, PREFIXED)
        assert counts

    def test_plain_password_under_a_key_name(self) -> None:
        request = self._request(json.dumps({"user": "bob", "password": PLAIN}))
        redact_request(request)
        _assert_gone(request, PLAIN)
        call = request["messages"][1]["tool_calls"][0]
        args = json.loads(call["function"]["arguments"])  # still valid JSON
        assert args["user"] == "bob"
        assert args["password"].startswith("[REDACTED:assignment#")
        # Structural fields a later message must match are untouched.
        assert call["id"] == "call_1" and call["function"]["name"] == "login_service"
        assert request["messages"][2]["tool_call_id"] == "call_1"

    def test_nested_value_under_a_key_name(self) -> None:
        request = self._request(json.dumps({"auth": {"api_key": {"primary": PLAIN, "n": [PLAIN]}}}))
        redact_request(request)
        _assert_gone(request, PLAIN)

    def test_invalid_json_arguments_are_redacted_as_text(self) -> None:
        request = self._request('{"password": "' + PLAIN + '"')  # truncated JSON
        redact_request(request)
        _assert_gone(request, PLAIN)

    def test_legacy_function_call(self) -> None:
        request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "f", "arguments": json.dumps({"x": PREFIXED})},
                }
            ]
        }
        redact_request(request)
        _assert_gone(request, PREFIXED)
        assert request["messages"][0]["function_call"]["name"] == "f"

    def test_other_message_text_fields(self) -> None:
        request = {"messages": [{"role": "assistant", "content": "", "reasoning_content": f"key {PREFIXED}"}]}
        redact_request(request)
        _assert_gone(request, PREFIXED)


class TestAnthropicBlocks:
    def test_tool_use_input(self) -> None:
        request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "login_service",
                            "input": {"password": PLAIN, "note": f"use {PREFIXED}", "id": PREFIXED},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": f"t {PREFIXED}"}],
                },
            ]
        }
        redact_request(request)
        _assert_gone(request, PLAIN, PREFIXED)
        block = request["messages"][0]["content"][1]
        assert (block["type"], block["id"], block["name"]) == ("tool_use", "toolu_1", "login_service")
        assert request["messages"][1]["content"][0]["tool_use_id"] == "toolu_1"

    @pytest.mark.parametrize(
        "block",
        [
            {"type": "thinking", "thinking": f"the key is {PREFIXED}", "signature": "sig"},
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": f"k={PREFIXED}"}},
            {"type": "some_future_block", "payload": {"deep": [f"x {PREFIXED}"]}},
        ],
        ids=["thinking", "text-document", "unknown-type"],
    )
    def test_non_text_blocks_fail_closed(self, block: dict[str, Any]) -> None:
        request = {"messages": [{"role": "assistant", "content": [block]}]}
        redact_request(request)
        _assert_gone(request, PREFIXED)
        assert request["messages"][0]["content"][0]["type"] == block["type"]

    def test_binary_media_is_passed_through(self) -> None:
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD" * 20}}
        request = {"messages": [{"role": "user", "content": [image]}]}
        assert not redact_request(request)
        assert request["messages"][0]["content"][0] == image


class TestNothingToRedact:
    def test_tool_call_without_secrets_is_unchanged(self) -> None:
        arguments = '{"path":"src/app.py",  "max_tokens":1024}'  # odd spacing kept verbatim
        request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c", "type": "function", "function": {"name": "read", "arguments": arguments}}],
                }
            ]
        }
        before = json.dumps(request)
        assert not redact_request(request)
        assert json.loads(request["messages"][0]["tool_calls"][0]["function"]["arguments"]) == json.loads(arguments)
        assert json.dumps(request) == before

"""Tests for the Gemini backend (native generateContent + governance gate)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lexora.backends.base import (
    BackendError,
    BackendRateLimitError,
    BackendUnavailableError,
)
from lexora.backends.gemini import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    GEMINI_API_VERSION,
    GeminiBackend,
    GeminiGovernanceError,
)


class TestGeminiBackendInit:
    """Tests for GeminiBackend initialization."""

    def test_default_values(self):
        backend = GeminiBackend()
        assert backend.base_url == "https://generativelanguage.googleapis.com"
        assert backend.api_key is None
        assert backend.model_mapping == {}
        assert backend.name is None

    def test_custom_url_trailing_slash_stripped(self):
        backend = GeminiBackend(base_url="https://example.com/")
        assert backend.base_url == "https://example.com"

    def test_openai_compat_suffix_stripped(self):
        # The shared config historically pointed at the OpenAI-compat shim;
        # the native adapter must strip it to build /v1beta/models/... paths.
        backend = GeminiBackend(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai"
        )
        assert backend.base_url == "https://generativelanguage.googleapis.com"

    def test_with_api_key_sets_header(self):
        backend = GeminiBackend(api_key="gm-test", paid_key_acknowledged=True)
        assert backend.api_key == "gm-test"
        assert backend._client.headers["x-goog-api-key"] == "gm-test"

    def test_without_api_key_no_header(self):
        backend = GeminiBackend()
        assert "x-goog-api-key" not in backend._client.headers

    def test_with_model_mapping(self):
        mapping = {"naysayer": "gemini-2.5-flash"}
        backend = GeminiBackend(model_mapping=mapping)
        assert backend._map_model("naysayer") == "gemini-2.5-flash"
        assert backend._map_model("unmapped") == "unmapped"

    def test_model_path(self):
        backend = GeminiBackend()
        assert backend._model_path("gemini-2.5-flash", "generateContent") == (
            f"/{GEMINI_API_VERSION}/models/gemini-2.5-flash:generateContent"
        )


class TestPaidKeyGuarantee:
    """ADR-14 D-4: fail-closed paid-key affirmation when a key is configured."""

    def test_key_without_ack_refused(self):
        with pytest.raises(GeminiGovernanceError, match="paid-key guarantee"):
            GeminiBackend(api_key="gm-test")

    def test_key_with_ack_constructs(self):
        backend = GeminiBackend(api_key="gm-test", paid_key_acknowledged=True)
        assert backend.api_key == "gm-test"
        assert backend.paid_key_acknowledged is True

    def test_keyless_construction_allowed_without_ack(self):
        # No key configured -> nothing to mis-bill; construction is allowed
        # (e.g. for unit-testing conversion helpers).
        backend = GeminiBackend()
        assert backend.api_key is None
        assert backend.paid_key_acknowledged is False

    def test_paid_key_guarantee_error_is_backend_error(self):
        assert issubclass(GeminiGovernanceError, BackendError)


class TestGovernanceGate:
    """Tests for the data-governance gate (plain generateContent only)."""

    @pytest.fixture
    def backend(self):
        return GeminiBackend(
            name="naysayer", api_key="gm-test", paid_key_acknowledged=True
        )

    @pytest.mark.parametrize(
        "forbidden_key",
        [
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "parallel_tool_calls",
            "cached_content",
            "cachedContent",
        ],
    )
    def test_forbidden_request_keys_rejected(self, backend, forbidden_key):
        request = {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            forbidden_key: "anything",
        }
        with pytest.raises(GeminiGovernanceError, match="data-governance gate"):
            backend._enforce_governance_gate(request)

    def test_non_text_content_part_rejected(self):
        backend = GeminiBackend()
        request = {
            "model": "gemini-2.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x"}}
                    ],
                }
            ],
        }
        with pytest.raises(GeminiGovernanceError, match="text-only"):
            backend._enforce_governance_gate(request)

    def test_plain_text_request_allowed(self, backend):
        request = {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "be skeptical"},
                {"role": "user", "content": "review this"},
                {"role": "user", "content": [{"type": "text", "text": "more"}]},
            ],
        }
        # Should not raise.
        backend._enforce_governance_gate(request)

    def test_governance_error_is_backend_error(self):
        # Subclass so generic backend error handling still catches it, while
        # tests/callers can assert specifically.
        assert issubclass(GeminiGovernanceError, BackendError)

    @pytest.mark.asyncio
    async def test_embeddings_blocked(self, backend):
        with pytest.raises(GeminiGovernanceError, match="embedContent"):
            await backend.embeddings({"input": "x"})

    @pytest.mark.asyncio
    async def test_completions_blocked(self, backend):
        with pytest.raises(GeminiGovernanceError, match="text completions"):
            await backend.completions({"prompt": "x"})

    @pytest.mark.asyncio
    async def test_completions_stream_blocked(self, backend):
        with pytest.raises(GeminiGovernanceError, match="text completions"):
            async for _ in backend.completions_stream({"prompt": "x"}):
                pass

    @pytest.mark.asyncio
    async def test_chat_completions_gate_runs_before_network(self, backend):
        # A forbidden request must be refused without any HTTP call.
        backend._client.post = AsyncMock(
            side_effect=AssertionError("network must not be reached")
        )
        request = {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
        }
        with pytest.raises(GeminiGovernanceError):
            await backend.chat_completions(request)


class TestRequestConversion:
    """Tests for OpenAI -> Gemini request conversion."""

    @pytest.fixture
    def backend(self):
        return GeminiBackend(name="test")

    def test_basic_conversion(self, backend):
        request = {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        result = backend._to_gemini_request(request)
        assert result["contents"] == [
            {"role": "user", "parts": [{"text": "Hello"}]}
        ]
        assert result["generationConfig"]["maxOutputTokens"] == 100

    def test_default_max_output_tokens(self, backend):
        request = {"messages": [{"role": "user", "content": "Hi"}]}
        result = backend._to_gemini_request(request)
        assert (
            result["generationConfig"]["maxOutputTokens"]
            == DEFAULT_MAX_OUTPUT_TOKENS
        )

    def test_configured_default_max_tokens_used_when_omitted(self):
        backend = GeminiBackend(name="test", default_max_tokens=8000)
        request = {"messages": [{"role": "user", "content": "Hi"}]}
        result = backend._to_gemini_request(request)
        assert result["generationConfig"]["maxOutputTokens"] == 8000

    def test_request_max_tokens_overrides_configured_default(self):
        backend = GeminiBackend(name="test", default_max_tokens=8000)
        request = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 42}
        result = backend._to_gemini_request(request)
        assert result["generationConfig"]["maxOutputTokens"] == 42

    def test_none_default_falls_back_to_module_constant(self):
        backend = GeminiBackend(name="test", default_max_tokens=None)
        assert backend.default_max_tokens == DEFAULT_MAX_OUTPUT_TOKENS

    def test_system_instruction_extraction(self, backend):
        request = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        }
        result = backend._to_gemini_request(request)
        assert result["systemInstruction"] == {
            "parts": [{"text": "You are helpful."}]
        }
        assert all(c["role"] != "system" for c in result["contents"])

    def test_assistant_role_mapped_to_model(self, backend):
        request = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "More"},
            ]
        }
        result = backend._to_gemini_request(request)
        roles = [c["role"] for c in result["contents"]]
        assert roles == ["user", "model", "user"]

    def test_sampling_params_passed(self, backend):
        request = {
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["END"],
        }
        cfg = backend._to_gemini_request(request)["generationConfig"]
        assert cfg["temperature"] == 0.2
        assert cfg["topP"] == 0.9
        assert cfg["stopSequences"] == ["END"]

    def test_stop_string_wrapped_in_list(self, backend):
        request = {
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "END",
        }
        cfg = backend._to_gemini_request(request)["generationConfig"]
        assert cfg["stopSequences"] == ["END"]


class TestResponseConversion:
    """Tests for Gemini -> OpenAI response conversion."""

    @pytest.fixture
    def backend(self):
        return GeminiBackend(name="test")

    def test_basic_response(self, backend):
        gemini_resp = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Answer"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            },
        }
        result = backend._to_openai_response(gemini_resp, "gemini-2.5-flash")
        assert result["object"] == "chat.completion"
        assert result["model"] == "gemini-2.5-flash"
        choice = result["choices"][0]
        assert choice["message"]["content"] == "Answer"
        assert choice["message"]["role"] == "assistant"
        assert choice["finish_reason"] == "stop"
        assert result["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }

    def test_multi_part_text_concatenated(self, backend):
        gemini_resp = {
            "candidates": [
                {"content": {"parts": [{"text": "a"}, {"text": "b"}]}}
            ]
        }
        result = backend._to_openai_response(gemini_resp, "m")
        assert result["choices"][0]["message"]["content"] == "ab"

    def test_max_tokens_finish_reason_mapped(self, backend):
        gemini_resp = {
            "candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]
        }
        result = backend._to_openai_response(gemini_resp, "m")
        assert result["choices"][0]["finish_reason"] == "length"

    def test_safety_finish_reason_mapped(self, backend):
        gemini_resp = {
            "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]
        }
        result = backend._to_openai_response(gemini_resp, "m")
        assert result["choices"][0]["finish_reason"] == "content_filter"

    def test_total_tokens_fallback(self, backend):
        gemini_resp = {
            "candidates": [{"content": {"parts": [{"text": "x"}]}}],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 4,
            },
        }
        result = backend._to_openai_response(gemini_resp, "m")
        assert result["usage"]["total_tokens"] == 6

    def test_empty_candidates(self, backend):
        result = backend._to_openai_response({}, "m")
        assert result["choices"][0]["message"]["content"] == ""
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_prompt_level_block_is_content_filter(self, backend):
        # No candidates but a prompt-level block must surface as content_filter
        # (symmetric with the streaming path), not silently "stop".
        gemini_resp = {"promptFeedback": {"blockReason": "SAFETY"}}
        result = backend._to_openai_response(gemini_resp, "m")
        assert result["choices"][0]["finish_reason"] == "content_filter"


class TestHealthCheckModel:
    """B-1: health_check must probe a configurable model, not a hard-coded one."""

    def test_default_health_model(self):
        backend = GeminiBackend()
        assert backend.health_check_model == "gemini-2.5-flash"

    def test_configured_health_model(self):
        backend = GeminiBackend(health_check_model="gemini-3.1-pro")
        assert backend.health_check_model == "gemini-3.1-pro"

    @pytest.mark.asyncio
    async def test_health_check_uses_configured_model(self):
        backend = GeminiBackend(
            api_key="gm-test",
            health_check_model="gemini-3.1-pro",
            paid_key_acknowledged=True,
        )
        backend._client.post = AsyncMock(return_value=_mock_response(200, {}))
        await backend.health_check()
        called_path = backend._client.post.call_args[0][0]
        assert "gemini-3.1-pro:generateContent" in called_path

    @pytest.mark.asyncio
    async def test_health_model_respects_mapping(self):
        backend = GeminiBackend(
            api_key="gm-test",
            health_check_model="naysayer",
            model_mapping={"naysayer": "gemini-2.5-flash"},
            paid_key_acknowledged=True,
        )
        backend._client.post = AsyncMock(return_value=_mock_response(200, {}))
        await backend.health_check()
        called_path = backend._client.post.call_args[0][0]
        assert "gemini-2.5-flash:generateContent" in called_path


def _mock_response(status_code=200, json_body=None, headers=None):
    """Build a MagicMock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.headers = headers or {}
    resp.text = json.dumps(json_body) if json_body else ""
    return resp


class TestChatCompletions:
    """Tests for chat_completions HTTP behavior."""

    @pytest.fixture
    def backend(self):
        return GeminiBackend(
            name="naysayer", api_key="gm-test", paid_key_acknowledged=True
        )

    @pytest.mark.asyncio
    async def test_successful_request(self, backend):
        gemini_resp = {
            "candidates": [
                {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
                "totalTokenCount": 2,
            },
        }
        backend._client.post = AsyncMock(
            return_value=_mock_response(200, gemini_resp)
        )
        result = await backend.chat_completions(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert result["choices"][0]["message"]["content"] == "ok"
        # Posted to the native generateContent path.
        called_path = backend._client.post.call_args[0][0]
        assert called_path.endswith("gemini-2.5-flash:generateContent")

    @pytest.mark.asyncio
    async def test_rate_limit_raises(self, backend):
        backend._client.post = AsyncMock(
            return_value=_mock_response(429, headers={"Retry-After": "5"})
        )
        with pytest.raises(BackendRateLimitError) as exc:
            await backend.chat_completions(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            )
        assert exc.value.retry_after == 5.0

    @pytest.mark.asyncio
    async def test_unavailable_raises(self, backend):
        backend._client.post = AsyncMock(return_value=_mock_response(503))
        with pytest.raises(BackendUnavailableError):
            await backend.chat_completions(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            )

    @pytest.mark.asyncio
    async def test_generic_error_raises(self, backend):
        backend._client.post = AsyncMock(
            return_value=_mock_response(
                400, {"error": {"message": "bad request"}}
            )
        )
        with pytest.raises(BackendError, match="bad request"):
            await backend.chat_completions(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            )


class TestListAndClose:
    """Tests for list_models and close."""

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        backend = GeminiBackend()
        result = await backend.list_models()
        assert result == {"object": "list", "data": []}

    @pytest.mark.asyncio
    async def test_close(self):
        backend = GeminiBackend()
        backend._client.aclose = AsyncMock()
        await backend.close()
        backend._client.aclose.assert_awaited_once()

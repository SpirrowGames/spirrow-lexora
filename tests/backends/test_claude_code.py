"""Tests for the Claude Code CLI backend's token accounting.

The `usage` vectors below are shapes taken from real
`claude -p --model sonnet --output-format json --no-session-persistence
--max-turns 1` output (CLI 2.1.263, measured 2026-09-06), which is the command
`_build_command` assembles for the non-streaming path.

The *totals* are not invariants of the CLI. Three independent samples of the
same trivial prompt produced input-side totals of 50,169 / 50,169 / 40,958: the
cache split varies with session state, and so does the sum. Do not read the
constants below as properties of the CLI — they are properties of the samples.

What *is* stable across all three samples, and what these tests pin, is the
structure: `input_tokens` carries only the uncached remainder of the prompt
(2 in every sample), while the bulk of the prompt sits in the sibling
`cache_creation_input_tokens` and `cache_read_input_tokens` fields of the same
`usage` object. Tokens read back from the cache are still tokens the model was
prompted with, so all three belong in `prompt_tokens`.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from lexora.backends.claude_code import ClaudeCodeBackend

# Sample A (msg-109 §1(2), measured 2026-09-06). The whole prompt was served
# from cache, so `cache_creation_input_tokens` is 0 and one field carries
# everything. Input side sums to 50,169.
SAMPLE_A_USAGE = {
    "input_tokens": 2,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 50167,
    "output_tokens": 4,
}

# Sample B (msg-103 §3, measured 2026-09-06). Same prompt, same CLI, same day,
# but the split lands across both cache fields. Input side sums to 50,169 as
# well — the same total reached from a different split, which is what makes the
# A/B pair a discriminating test and not a duplicated one.
SAMPLE_B_USAGE = {
    "input_tokens": 2,
    "cache_creation_input_tokens": 26435,
    "cache_read_input_tokens": 23732,
    "output_tokens": 4,
}

# Sample C, measured 2026-09-06 while implementing this change. Both cache
# fields non-zero again, but the input side sums to 40,958 rather than 50,169:
# the total is session state, not a constant. Carried here for its `modelUsage`
# block, which the top-level `usage` does not include (see the fence below).
SAMPLE_C_RESULT = {
    "type": "result",
    "result": "ok",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 2,
        "cache_creation_input_tokens": 17224,
        "cache_read_input_tokens": 23732,
        "output_tokens": 4,
    },
    "modelUsage": {
        "claude-sonnet-5": {
            "inputTokens": 2,
            "outputTokens": 4,
            "cacheReadInputTokens": 23732,
            "cacheCreationInputTokens": 17224,
        },
        "claude-haiku-4-5-20251001": {
            "inputTokens": 899,
            "outputTokens": 8,
            "cacheReadInputTokens": 0,
            "cacheCreationInputTokens": 0,
        },
    },
}


class TestTokensFromResult:
    """`_tokens_from_result` must count the cached prompt as prompt."""

    def test_cached_prompt_counts_as_prompt_tokens_when_all_cache_is_read(self):
        """Detector, sample A. Before this change the parser returned 2."""
        prompt_tokens, completion_tokens = ClaudeCodeBackend._tokens_from_result(
            {"usage": dict(SAMPLE_A_USAGE)}
        )
        assert prompt_tokens == 50169
        assert completion_tokens == 4

    def test_cached_prompt_counts_as_prompt_tokens_when_the_split_differs(self):
        """Detector, sample B — the same total reached from a different split.

        This is not a copy of the sample-A case. A parser that adds only
        `cache_read_input_tokens` and overlooks `cache_creation_input_tokens`
        satisfies sample A exactly (2 + 50,167) and fails here (2 + 23,732 =
        23,734, not 50,169). Requiring one total from two splits is what makes
        a half-read of the `usage` object detectable.
        """
        prompt_tokens, completion_tokens = ClaudeCodeBackend._tokens_from_result(
            {"usage": dict(SAMPLE_B_USAGE)}
        )
        assert prompt_tokens == 50169
        assert completion_tokens == 4

    def test_absent_cache_fields_are_treated_as_zero(self):
        """Fence. Output without the cache keys must not raise or collapse.

        Older CLI builds, and the non-JSON fallback path, need not supply the
        cache fields. Missing means "no cached tokens", not "unknown", and must
        not turn the counted prompt into 0 either.
        """
        assert ClaudeCodeBackend._tokens_from_result(
            {"usage": {"input_tokens": 7, "output_tokens": 3}}
        ) == (7, 3)

    def test_absent_usage_block_yields_zeroes(self):
        """Fence. A result with no `usage` at all stays at the previous 0/0."""
        assert ClaudeCodeBackend._tokens_from_result({"result": "ok"}) == (0, 0)

    def test_completion_tokens_ignore_the_second_model_in_model_usage(self):
        """Fence pinning a decision that was deliberately NOT made here.

        Sample C's invocation billed two models: the requested `claude-sonnet-5`
        (4 output tokens) and `claude-haiku-4-5-20251001` (8 more). The
        top-level `usage` reports only the first. Whether `completion_tokens`
        should describe one completion or the whole invocation is a separate
        question about what the field means, and this change does not answer
        it — it corrects the input side only, where the missing tokens sit in
        the same `usage` object that is already being read.

        This assertion exists so that folding the second model in becomes a
        visible decision rather than a silent tidy-up: it fails the moment
        someone makes `completion_tokens` 12.
        """
        _, completion_tokens = ClaudeCodeBackend._tokens_from_result(SAMPLE_C_RESULT)
        assert completion_tokens == 4


class TestChatCompletionsUsageWiring:
    """The parsed counts must reach the response the ledger is written from."""

    async def test_response_usage_carries_the_summed_prompt_tokens(self):
        """Detector on the call site, not on the parser.

        `api/routes.py` records `usage.prompt_tokens` from this response, so a
        correct `_tokens_from_result` that nothing calls would leave the ledger
        exactly as wrong as before. Sample B is used because it is the vector a
        half-read of the cache fields fails.

        Hermetic: the subprocess is replaced, so no `claude` binary is invoked.
        """
        backend = ClaudeCodeBackend(model="sonnet")
        payload = {
            "type": "result",
            "result": "ok",
            "stop_reason": "end_turn",
            "usage": dict(SAMPLE_B_USAGE),
        }
        process = MagicMock()
        process.returncode = 0
        process.communicate = AsyncMock(
            return_value=(json.dumps(payload).encode("utf-8"), b"")
        )

        with patch(
            "asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)
        ):
            response = await backend.chat_completions(
                {
                    "model": "claude-code-sonnet",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )

        assert response["usage"]["prompt_tokens"] == 50169
        assert response["usage"]["completion_tokens"] == 4
        assert response["usage"]["total_tokens"] == 50173


class TestListModels:
    """`list_models` returns an empty catalogue, and that is the behaviour.

    Until this class, `ClaudeCodeBackend.list_models` had no test at all
    while its docstring said "Return configured model in OpenAI format" --
    a claim the body contradicts and nothing checked. The empty list is
    correct: this backend shells out to the CLI and has no upstream
    catalogue to report, and the `claude-code-*` names reach `/v1/models`
    through `BackendRouter.list_all_models`, which advertises declared
    names on the gateway's own authority. Pinning the empty list here
    makes "so do not re-derive the config here" a checked statement rather
    than a comment: a well-meaning change that returned the configured
    models would double-emit rows into a loop that already deduplicates,
    and it now goes red instead of shipping.

    Hermetic: no subprocess, no `claude` binary.
    """

    async def test_returns_empty_list(self):
        backend = ClaudeCodeBackend(model="sonnet")
        assert await backend.list_models() == {"object": "list", "data": []}

"""Claude Code non-interactive backend implementation.

Routes LLM requests through Claude Code CLI (`claude -p`) as a subprocess,
translating OpenAI-compatible requests/responses.
"""

import asyncio
import json
import shutil
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from lexora.backends.base import (
    Backend,
    BackendError,
    BackendTimeoutError,
)
from lexora.utils.logging import get_logger

logger = get_logger(__name__)


class ClaudeCodeBackend(Backend):
    """Claude Code CLI non-interactive backend.

    Executes `claude -p` as a subprocess to handle LLM requests.
    Supports both non-streaming (--output-format json) and
    streaming (--output-format stream-json --verbose) modes.

    Args:
        model: Claude Code model to use (e.g. 'sonnet', 'opus').
        timeout: Subprocess timeout in seconds.
        allowed_tools: Tools to allow (e.g. ["Bash(git:*)", "Read"]).
        system_prompt: Optional system prompt to append.
        working_dir: Working directory for Claude Code subprocess.
        max_turns: Maximum number of agentic turns.
        name: Optional backend name for logging.
    """

    def __init__(
        self,
        model: str = "sonnet",
        timeout: float = 300.0,
        allowed_tools: list[str] | None = None,
        system_prompt: str | None = None,
        working_dir: str | None = None,
        max_turns: int | None = None,
        model_mapping: dict[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.allowed_tools = allowed_tools
        self.system_prompt = system_prompt
        self.working_dir = working_dir
        self.max_turns = max_turns
        self.model_mapping = model_mapping or {}
        self.name = name

    def _build_command(self, prompt: str, stream: bool = False) -> list[str]:
        """Build the claude CLI command.

        Args:
            prompt: The user prompt to send.
            stream: Whether to use streaming output.

        Returns:
            Command as list of strings.
        """
        # Resolve CLI model name via model_mapping
        cli_model = self.model_mapping.get(self.model, self.model)
        # Use full path to avoid PATH issues in systemd units
        claude_bin = shutil.which("claude") or "/home/sgadmin/.local/bin/claude"
        cmd = [claude_bin, "-p", "--model", cli_model]

        if stream:
            cmd.extend(["--output-format", "stream-json", "--verbose"])
        else:
            cmd.extend(["--output-format", "json"])

        cmd.append("--no-session-persistence")

        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        if self.system_prompt:
            cmd.extend(["--append-system-prompt", self.system_prompt])

        if self.max_turns is not None:
            cmd.extend(["--max-turns", str(self.max_turns)])

        return cmd

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        """Convert OpenAI-style messages to a single prompt string.

        System messages are prepended, then user/assistant messages
        are formatted as a conversation.

        Args:
            messages: List of OpenAI-style message dicts.

        Returns:
            Combined prompt string.
        """
        system_parts: list[str] = []
        conversation_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Handle content blocks (extract text only)
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)

            if role == "system":
                system_parts.append(content)
            elif role == "user":
                conversation_parts.append(content)
            elif role == "assistant":
                conversation_parts.append(f"[Previous assistant response]\n{content}")

        parts = []
        if system_parts:
            parts.append("[System Instructions]\n" + "\n\n".join(system_parts))
        if conversation_parts:
            parts.append("\n\n".join(conversation_parts))

        return "\n\n".join(parts)

    def _make_openai_response(
        self,
        content: str,
        model: str,
        finish_reason: str = "stop",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible chat completion response.

        Args:
            content: Response text.
            model: Model name.
            finish_reason: Finish reason string.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            OpenAI-compatible response dict.
        """
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send chat completion request via Claude Code CLI.

        Args:
            request: OpenAI-compatible chat completion request.

        Returns:
            OpenAI-compatible chat completion response.
        """
        messages = request.get("messages", [])
        prompt = self._messages_to_prompt(messages)
        requested_model = request.get("model", self.model)

        # Temporarily set self.model to the requested model for _build_command
        original_model = self.model
        self.model = requested_model
        cmd = self._build_command(prompt, stream=False)
        self.model = original_model

        logger.info(
            "claude_code_request",
            model=requested_model,
            backend=self.name,
            prompt_length=len(prompt),
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode("utf-8")),
                timeout=self.timeout,
            )

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                logger.error(
                    "claude_code_process_error",
                    returncode=process.returncode,
                    stderr=error_msg[:500],
                    backend=self.name,
                )
                raise BackendError(
                    f"Claude Code process failed (exit {process.returncode}): {error_msg[:200]}"
                )

            output = stdout.decode("utf-8").strip()

            try:
                result = json.loads(output)
            except json.JSONDecodeError:
                # Non-JSON output — treat as plain text response
                return self._make_openai_response(output, requested_model)

            # Parse Claude Code JSON result format
            content = result.get("result", output)
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            stop_reason = result.get("stop_reason", "end_turn")
            finish_reason = "stop" if stop_reason == "end_turn" else "length"

            return self._make_openai_response(
                content=content,
                model=requested_model,
                finish_reason=finish_reason,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )

        except asyncio.TimeoutError:
            logger.error(
                "claude_code_timeout",
                timeout=self.timeout,
                backend=self.name,
            )
            raise BackendTimeoutError(
                f"Claude Code subprocess timed out after {self.timeout}s"
            )
        except BackendError:
            raise
        except Exception as e:
            logger.error(
                "claude_code_unexpected_error",
                error=str(e),
                backend=self.name,
            )
            raise BackendError(f"Claude Code request failed: {e}") from e

    async def chat_completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Send streaming chat completion request via Claude Code CLI.

        Uses `--output-format stream-json --verbose` to get streaming events.
        Translates to OpenAI SSE format.

        Args:
            request: OpenAI-compatible chat completion request.

        Yields:
            SSE data chunks in OpenAI format.
        """
        messages = request.get("messages", [])
        prompt = self._messages_to_prompt(messages)
        requested_model = request.get("model", self.model)

        original_model = self.model
        self.model = requested_model
        cmd = self._build_command(prompt, stream=True)
        self.model = original_model
        model = requested_model

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        logger.info(
            "claude_code_stream_request",
            model=model,
            backend=self.name,
            prompt_length=len(prompt),
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )

            # Send prompt via stdin and close it
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

            # Send initial role chunk
            initial_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(initial_chunk)}\n\n".encode()

            # Read stdout line by line
            while True:
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error("claude_code_stream_timeout", backend=self.name)
                    break

                if not line:
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    event = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "assistant":
                    # Extract content from assistant message
                    message = event.get("message", {})
                    content_blocks = message.get("content", [])
                    for block in content_blocks:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                openai_chunk = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(openai_chunk)}\n\n".encode()

                elif event_type == "result":
                    # Final result — send finish chunk
                    stop_reason = event.get("stop_reason", "end_turn")
                    finish_reason = "stop" if stop_reason == "end_turn" else "length"
                    final_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n".encode()
                    yield b"data: [DONE]\n\n"

            await process.wait()

        except BackendError:
            raise
        except Exception as e:
            logger.error(
                "claude_code_stream_error",
                error=str(e),
                backend=self.name,
            )
            raise BackendError(f"Claude Code streaming failed: {e}") from e

    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Not supported — use chat_completions instead."""
        raise BackendError("Text completions are not supported by Claude Code backend")

    async def completions_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Not supported — use chat_completions_stream instead."""
        raise BackendError("Text completions are not supported by Claude Code backend")
        yield b""  # Make this an async generator

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Not supported by Claude Code."""
        raise BackendError("Embeddings are not supported by Claude Code backend")

    async def list_models(self) -> dict[str, Any]:
        """Return configured model in OpenAI format."""
        return {"object": "list", "data": []}

    async def health_check(self) -> bool:
        """Check if Claude Code CLI is available.

        Runs `claude --version` to verify the CLI is installed and accessible.
        """
        try:
            claude_bin = shutil.which("claude") or "/home/sgadmin/.local/bin/claude"
            process = await asyncio.create_subprocess_exec(
                claude_bin, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0,
            )
            return process.returncode == 0
        except Exception:
            return False

    async def close(self) -> None:
        """No persistent connections to close."""
        pass

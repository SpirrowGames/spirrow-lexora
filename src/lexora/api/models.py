"""API request and response models."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model."""

    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str | None = None
    name: str | None = None
    function_call: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool = False
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    # vLLM specific parameters
    best_of: int | None = None
    top_k: int | None = None
    ignore_eos: bool | None = None
    use_beam_search: bool | None = None

    model_config = {"extra": "allow"}


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""

    model: str
    prompt: str | list[str]
    suffix: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool = False
    logprobs: int | None = None
    echo: bool | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    best_of: int | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None

    model_config = {"extra": "allow"}


class EmbeddingsRequest(BaseModel):
    """OpenAI-compatible embeddings request."""

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = None
    user: str | None = None

    model_config = {"extra": "allow"}


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy", "degraded"]
    backends: dict[str, Literal["healthy", "unhealthy"]]
    version: str
    # Legacy field for backward compatibility
    vllm_status: Literal["healthy", "unhealthy", "unknown"] | None = None


class StatsResponse(BaseModel):
    """Statistics response."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    total_tokens_input: int
    total_tokens_output: int
    total_retries: int
    average_duration_seconds: float
    requests_per_endpoint: dict[str, int]
    requests_per_model: dict[str, int]
    errors_by_type: dict[str, int]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: dict[str, Any] = Field(
        ...,
        description="Error details",
        examples=[{"message": "Internal server error", "type": "server_error"}],
    )

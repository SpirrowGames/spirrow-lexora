"""API request and response models."""

from typing import Any, Literal

from pydantic import BaseModel, Field


# Convenience endpoint models for simple text generation


class GenerateRequest(BaseModel):
    """Simple text generation request for /generate endpoint."""

    prompt: str = Field(description="The prompt to generate text from")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    model: str | None = Field(default=None, description="Model to use (optional)")
    user: str | None = Field(default=None, description="User identifier for rate limiting")

    model_config = {"extra": "allow"}


class GenerateResponse(BaseModel):
    """Response for /generate endpoint."""

    text: str = Field(description="Generated text")


class SimpleChatMessage(BaseModel):
    """Simple chat message for /chat endpoint."""

    role: Literal["system", "user", "assistant"] = Field(description="Message role")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    """Simple chat request for /chat endpoint."""

    messages: list[SimpleChatMessage] = Field(description="Chat messages")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    model: str | None = Field(default=None, description="Model to use (optional)")
    user: str | None = Field(default=None, description="User identifier for rate limiting")

    model_config = {"extra": "allow"}


class ChatResponse(BaseModel):
    """Response for /chat endpoint."""

    response: str = Field(description="Assistant response text")


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


# Models for /v1/models/capabilities endpoint


class ModelCapabilityInfo(BaseModel):
    """Model information with capabilities."""

    id: str = Field(description="Model identifier")
    backend: str = Field(description="Backend name serving this model")
    backend_type: str = Field(description="Backend type (vllm or openai_compatible)")
    capabilities: list[str] = Field(description="List of capability tags")
    description: str | None = Field(default=None, description="Model description")


class ModelCapabilitiesResponse(BaseModel):
    """Response for /v1/models/capabilities endpoint."""

    models: list[ModelCapabilityInfo] = Field(description="List of available models")
    available_capabilities: list[str] = Field(
        description="List of all available capability tags"
    )
    default_model_for_unknown_task: str | None = Field(
        default=None, description="Default model for unknown task types"
    )


# Models for /v1/classify-task endpoint


class ClassifyTaskRequest(BaseModel):
    """Request for /v1/classify-task endpoint."""

    task_description: str = Field(
        description="Description of the task to classify",
        min_length=1,
        max_length=10000,
    )


class ModelAlternative(BaseModel):
    """Alternative model recommendation."""

    model: str = Field(description="Model identifier")
    score: float = Field(description="Relevance score (0.0-1.0)")


class ClassifyTaskResponse(BaseModel):
    """Response for /v1/classify-task endpoint."""

    recommended_model: str = Field(description="Recommended model for the task")
    task_type: str = Field(description="Classified task type/capability")
    confidence: float = Field(description="Classification confidence (0.0-1.0)")
    reasoning: str = Field(description="Explanation for the classification")
    alternatives: list[ModelAlternative] = Field(
        default_factory=list, description="Alternative model recommendations"
    )

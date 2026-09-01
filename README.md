# Spirrow-Lexora

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)](pyproject.toml)

**[日本語版はこちら](README.ja.md)**

LLM Gateway / Router for the Spirrow Platform

## Overview

Spirrow-Lexora is a proxy/gateway that sits in front of vLLM inference servers. It provides OpenAI API-compatible endpoints while adding operational features such as queuing, rate limiting, statistics collection, and multi-backend routing.

**Key Features:**
- OpenAI API-compatible endpoints (Chat Completions, Completions, Embeddings)
- Streaming response support (SSE)
- Per-user rate limiting (Token Bucket algorithm)
- Automatic retry with exponential backoff and Retry-After header support
- Multi-backend routing with automatic model-based routing
- OpenAI-compatible API backend support (OpenAI, Azure OpenAI, etc.)
- 429 rate limit handling with Retry-After header respect
- Model capabilities API (list models with their capabilities)
- Task classification API (LLM-based optimal model recommendation)
- Prometheus metrics export
- Structured logging (structlog)

## Architecture

### Single Backend Mode
```
Client → Lexora (Gateway) → vLLM (Inference Engine) → GPU
            :8001              :8000
```

### Multi-Backend Mode
```
                              ┌→ vLLM-1 (model-a, model-b) → GPU
Client → Lexora (Gateway) ────┼→ vLLM-2 (model-c, model-d) → GPU
            :8001             └→ OpenAI-compatible API (gpt-4, etc.)
```

## Requirements

- Python 3.11+
- vLLM (inference backend)

## Installation

```bash
# Clone
git clone https://github.com/SpirrowGames/spirrow-lexora.git
cd spirrow-lexora

# Install
pip install -e ".[dev]"
```

## Quick Start

```bash
# Development mode
uvicorn lexora.main:app --reload --port 8001

# Production mode
python -m lexora.main
```

## Configuration

Configuration is applied in the following priority order (later takes precedence):

1. Default values
2. Configuration file (`config/lexora_config.yaml`)
3. Environment variables (`LEXORA_` prefix)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LEXORA_VLLM__URL` | vLLM server URL | `http://localhost:8000` |
| `LEXORA_VLLM__TIMEOUT` | Request timeout (seconds) | `120.0` |
| `LEXORA_VLLM__CONNECT_TIMEOUT` | Connection timeout (seconds) | `5.0` |
| `LEXORA_SERVER__HOST` | Server bind host | `0.0.0.0` |
| `LEXORA_SERVER__PORT` | Server bind port | `8001` |
| `LEXORA_RATE_LIMIT__ENABLED` | Enable rate limiting | `true` |
| `LEXORA_RATE_LIMIT__DEFAULT_RATE` | Requests per second | `10.0` |
| `LEXORA_RATE_LIMIT__DEFAULT_BURST` | Burst capacity | `20` |
| `LEXORA_RETRY__MAX_RETRIES` | Max retry attempts | `3` |
| `LEXORA_RETRY__BASE_DELAY` | Base retry delay (seconds) | `1.0` |
| `LEXORA_RETRY__RESPECT_RETRY_AFTER` | Respect Retry-After header | `true` |
| `LEXORA_RETRY__MAX_RETRY_AFTER` | Max Retry-After delay (seconds) | `60.0` |
| `LEXORA_FRONTIER_MODEL` | Concrete model ID served by the `frontier` tier (overrides the shipped default; updates both the routing tier and the advertised model of the backend that tier points at, in lockstep — a config where it cannot update both refuses to start) | shipped `frontier` tier default |
| `LEXORA_LOGGING__LEVEL` | Log level | `INFO` |
| `LEXORA_LOGGING__FORMAT` | Log format (`console`/`json`) | `console` |

Note: nested-delimiter env variables such as
`LEXORA_ROUTING__TIERS__FRONTIER__MODEL` are **not** honoured — the loader
builds `RoutingSettings` as an init kwarg and pydantic-settings gives init
kwargs precedence over env, so the nested-name override is silently
ignored. Use the flat `LEXORA_FRONTIER_MODEL` for the frontier tier.

### Configuration File

`config/lexora_config.yaml`:

```yaml
vllm:
  url: "http://localhost:8000"
  timeout: 120.0
  connect_timeout: 5.0

server:
  host: "0.0.0.0"
  port: 8001

rate_limit:
  enabled: true
  default_rate: 10.0    # requests per second
  default_burst: 20

retry:
  max_retries: 3
  base_delay: 1.0       # seconds
  max_delay: 30.0       # seconds
  exponential_base: 2.0
  respect_retry_after: true   # Respect Retry-After header from 429 responses
  max_retry_after: 60.0       # Max Retry-After delay

logging:
  level: "INFO"
  format: "console"     # "json" for production
```

### Multi-Backend Routing Configuration

You can distribute models across multiple backends (vLLM, OpenAI-compatible APIs):

```yaml
routing:
  enabled: true
  default_backend: "main"
  default_model_for_unknown_task: "qwen3-32b"  # Default model when task classification fails

  # Task classification settings (for /v1/classify-task)
  classifier:
    enabled: true
    model: "mistral-7b"      # Model for classification (lightweight model recommended)
    backend: "secondary"     # Backend for classification

  backends:
    main:
      type: "vllm"                    # vLLM backend (default)
      url: "http://localhost:8000"
      timeout: 120.0
      connect_timeout: 5.0
      models:
        # New format: with capabilities (available via /v1/models/capabilities)
        - name: "qwen3-32b"
          capabilities: ["code", "reasoning", "analysis", "general"]
          description: "For code generation and complex reasoning"
        - name: "llama3-70b"
          capabilities: ["reasoning", "general"]
          description: "General-purpose large model"

    secondary:
      type: "vllm"
      url: "http://localhost:8010"
      timeout: 120.0
      connect_timeout: 5.0
      models:
        - name: "mistral-7b"
          capabilities: ["summarization", "translation", "simple_qa"]
          description: "Fast model for lightweight tasks"
        - "embedding-model"           # Legacy format also supported (capabilities=["general"])

    openai_backup:
      type: "openai_compatible"       # OpenAI-compatible API
      url: "https://api.openai.com"
      api_key_env: "OPENAI_API_KEY"   # API key from environment variable
      # api_key: "sk-..."             # Or direct API key (not recommended)
      timeout: 60.0
      connect_timeout: 5.0
      models:
        - "gpt-4"
        - "gpt-4-turbo"
      model_mapping:                  # Map requested model to actual model
        "gpt-4": "gpt-4-0125-preview"
        "gpt-4-turbo": "gpt-4-turbo-preview"
```

**Behavior:**
- Requests are routed to the appropriate backend based on the `model` parameter
- Unregistered model names are routed to `default_backend`
- `/v1/models` aggregates models from all backends and appends the configured tier aliases as `{"type": "tier", "id": "<tier>", "resolved_model": "<concrete-id>"}` entries so callers can see both the concrete model IDs the backend serves and the tier names the router accepts
- `/health` returns health status of all backends (`healthy`, `degraded`, `unhealthy`)

### Tier Reference (shipped `config/lexora_config.yaml`)

Tiers are named entry points that the router resolves to a concrete `(backend, model)` pair. A caller sends `model: "<tier>"` on any OpenAI-compat request; the router picks the right backend, sends the concrete model upstream, and the cost tracker records both the tier alias and the resolved model separately (so pricing follows the actual upstream, not the alias).

| Tier | Backend | Concrete model | Purpose |
|------|---------|----------------|---------|
| `light` | `light` (vLLM, local GPU) | `Qwen3.8-27B` (no-think) | Lightweight tasks (summarisation, translation, simple QA) |
| `medium` | `heavy` (vLLM, local GPU) | `Qwen3.8-27B` (think, `reasoning_effort=medium`) | Standard tasks with thinking |
| `heavy` | `deep` (vLLM, local GPU) | `Qwen3.8-27B` (think, `reasoning_effort=xhigh`) | Complex reasoning (marginal cost zero — do not confuse "heavy" with "expensive"; heavy means "runs the biggest local reasoning budget", not "the paid frontier model") |
| `naysayer` | `gemini` (Google Gemini API, paid) | `gemini-3.1-pro-preview` | Independent-distribution reviewer (data-governance gate: plain `generateContent` only) |
| `frontier` | `frontier` (Anthropic API, paid) | `claude-fable-5` (env-configurable via `LEXORA_FRONTIER_MODEL`) | Top-of-line paid model; distinct entry so cost/latency/decline behaviour are chosen deliberately |

### Frontier Tier — Operational Notes

The `frontier` tier exists so a caller expressing "I want the smartest paid model, and I am willing to pay for it" gets a distinct entry point instead of that intent being buried in `heavy`. Its behaviour differs from every other tier on three axes:

- **No silent fallback.** Not because the tier opts out, but because the gateway has no fallback mechanism at all (removed 2026-08-31 — see below). If the upstream fails, the caller gets that failure — the point of picking `frontier` was to know which model produced the result. A config that tries to reintroduce a fallback target fails startup rather than downgrading a request.
- **No silent retry.** Two mechanisms, and it is worth knowing which does what. A 429 or a safety-classifier decline now leaves a passthrough backend as the upstream's own answer (`BackendUpstreamError`), a class the retry handler was never willing to retry — so those never cost a second call regardless of retry configuration. Separately, the route passes `retryable_exceptions=()` for a passthrough backend, which suppresses retry for the classes that *are* retryable by default (connection / timeout). With `retry.max_retries: 3` the result is one billed call rather than four.
- **Upstream errors pass through verbatim.** Instead of collapsing every 4xx/5xx into a 502 with a stringified detail, the route forwards the upstream status code and the parsed body as-is. A Fable/Opus classifier decline arrives as HTTP 400 with a structured `{"error": {"type": "refusal", ...}}` body; a `stop_reason: refusal` on a 200 response is mapped to OpenAI `finish_reason: "content_filter"` rather than being rounded off to `"stop"`.

Streaming caveat: for streaming requests the route pre-flights the first chunk from the backend so a refusal / auth failure still surfaces as a proper HTTP status. Once bytes have started flowing the caller can only see SSE-body passthrough — that is a protocol limit, and there is no way for a gateway to change an already-sent 200 into an HTTP 400.

**Swapping the frontier model (Fable 5 → Opus 5):**
```bash
export LEXORA_FRONTIER_MODEL=claude-opus-5-20260601
```
Both the tier resolution and `/v1/models{,/capabilities}` update together, so the surface never disagrees with what the router sends upstream. The advertised model is found through `routing.tiers.frontier.backend`, so renaming that backend in the YAML does not quietly leave half the override behind; if the variable cannot be applied to both places — no `frontier` tier, a tier naming a backend that does not exist, or that backend having no `models` — the gateway refuses to start and names which of the three it hit. Being billed for Opus 5 under a config still advertising Fable 5 is the failure this trades a startup error for. If you swap to a model ID that is not in `DEFAULT_PRICING`, cost records write `pricing_known=0` and log a `cost_pricing_unknown` warning — the ledger stays honest, but you should add the price entry before relying on `/stats/costs?tier=frontier` for reconciliation.

**Data governance:** Fable 5 has 30-day retention and a safety classifier. Using the frontier tier is an operator affirmation that this policy is acceptable for the traffic you route through it (analogous to `paid_key_acknowledged` on the Gemini backend and `ANTHROPIC_API_KEY` on the Claude backend — the config keeps these as owner decisions rather than hidden defaults). See the `frontier` backend comment in `config/lexora_config.yaml`.

**Rate limiting:** the general per-user rate limit (default 10 rps / burst 20) applies to frontier as it does to every other tier, but there is deliberately no daily cost cap in the initial version. The gateway sits on tailnet and has no authentication in front of it, so a runaway caller can bill the frontier account until it is stopped externally — this is a known limit, not an oversight; a per-tier daily cap is a candidate for a follow-up when a real user needs it.

**Backend Types:**
| Type | Description |
|------|-------------|
| `vllm` | vLLM backend (default) |
| `openai_compatible` | OpenAI-compatible API (OpenAI, Azure OpenAI, etc.) |

**No fallback mechanism (removed 2026-08-31):** a backend failure is returned to the
caller as an error; requests are never re-routed to a different backend. The
`FallbackService` that used to be configured here was never wired into the request
path, so the settings it read (`fallback:`, `fallback_backends:`) promised operators a
failover that never happened; both the code and the settings were removed rather than
wired. `BackendSettings` forbids unknown keys, so a `fallback_backends:` left in a
config file now fails startup loudly instead of being silently ignored.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI compatible) |
| `/v1/completions` | POST | Text completion (OpenAI compatible) |
| `/v1/embeddings` | POST | Text embeddings (OpenAI compatible) |
| `/v1/models` | GET | List available models |
| `/v1/models/capabilities` | GET | List models with capabilities |
| `/v1/classify-task` | POST | Classify task and recommend model |
| `/generate` | POST | Simple generation endpoint (for Magickit integration) |
| `/chat` | POST | Simple chat endpoint (for Magickit integration) |
| `/health` | GET | Health check |
| `/stats` | GET | Statistics |
| `/metrics` | GET | Prometheus metrics |

### Example Requests

```bash
# Health check
curl http://localhost:8001/health

# Statistics
curl http://localhost:8001/stats

# Chat completion
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# With user ID for rate limiting
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "messages": [{"role": "user", "content": "Hello"}],
    "user": "user-123"
  }'

# Streaming chat completion
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

# Streaming completion
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "prompt": "Once upon a time",
    "stream": true
  }'

# Embeddings
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-model",
    "input": "Hello, world!"
  }'

# Get model capabilities
curl http://localhost:8001/v1/models/capabilities | jq

# Classify task and get recommended model
curl -X POST http://localhost:8001/v1/classify-task \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Implement quicksort in Python"
  }' | jq

# Simple generation (for Magickit integration)
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "model": "qwen3-32b",
    "max_tokens": 500
  }'
```

## Production Deployment (Ubuntu)

### systemd Service

1. Create service file `/etc/systemd/system/lexora.service`:

```ini
[Unit]
Description=Lexora LLM Gateway
After=network.target

[Service]
Type=simple
User=lexora
Group=lexora
WorkingDirectory=/opt/lexora
Environment="LEXORA_VLLM__URL=http://localhost:8000"
Environment="LEXORA_LOGGING__FORMAT=json"
Environment="LEXORA_LOGGING__LEVEL=INFO"
ExecStart=/opt/lexora/venv/bin/python -m lexora.main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

2. Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable lexora
sudo systemctl start lexora
```

3. Check status:

```bash
sudo systemctl status lexora
sudo journalctl -u lexora -f
```

### Installation Script

```bash
# Create user and directory
sudo useradd -r -s /bin/false lexora
sudo mkdir -p /opt/lexora
sudo chown lexora:lexora /opt/lexora

# Clone and install
cd /opt/lexora
sudo -u lexora git clone https://github.com/SpirrowGames/spirrow-lexora.git .
sudo -u lexora python3 -m venv venv
sudo -u lexora ./venv/bin/pip install -e .

# Copy config
sudo -u lexora cp config/lexora_config.yaml /opt/lexora/config/
# Edit config as needed
sudo -u lexora vim /opt/lexora/config/lexora_config.yaml

# Install and start service
sudo cp deploy/lexora.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lexora
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| vLLM Proxy | ✅ | Chat Completions, Completions API |
| Embeddings | ✅ | Text Embeddings API |
| Streaming | ✅ | SSE-based streaming responses |
| Health Check | ✅ | Backend monitoring, degraded detection |
| Statistics Collection | ✅ | Request statistics, token aggregation |
| Rate Limiting | ✅ | Token Bucket algorithm |
| Auto Retry | ✅ | Exponential Backoff + Retry-After support |
| Priority Queue | ✅ | Priority-based request queue |
| Prometheus Metrics | ✅ | Metrics export |
| Multi-Backend Routing | ✅ | Automatic model-based routing |
| OpenAI-Compatible Backend | ✅ | OpenAI, Azure OpenAI, etc. support |
| Fallback Support | ❌ | Removed 2026-08-31 — was never wired; errors are returned, not re-routed |
| 429 Rate Limit Handling | ✅ | Retry-After header respect |
| Model Capabilities API | ✅ | Model list with capability information |
| Task Classification | ✅ | LLM-based task classification and model recommendation |

## Development

```bash
# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=lexora

# Type check
mypy src/lexora

# Lint
ruff check src/lexora
```

## Project Structure

```
spirrow-lexora/
├── pyproject.toml
├── config/
│   └── lexora_config.yaml
├── deploy/
│   └── lexora.service
├── src/lexora/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Pydantic Settings
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── models.py        # Request/Response models
│   ├── services/
│   │   ├── queue.py         # Priority queue
│   │   ├── rate_limiter.py  # Token bucket rate limiter
│   │   ├── retry_handler.py # Exponential backoff retry + Retry-After
│   │   ├── router.py        # Multi-backend routing
│   │   ├── metrics.py       # Prometheus metrics
│   │   ├── stats.py         # Statistics collection
│   │   ├── model_registry.py    # Model capabilities registry
│   │   └── task_classifier.py   # LLM-based task classification
│   ├── backends/
│   │   ├── base.py          # Backend ABC + exceptions
│   │   ├── vllm.py          # vLLM client
│   │   ├── openai_compatible.py  # OpenAI-compatible API client
│   │   └── factory.py       # Backend factory
│   └── utils/
│       └── logging.py       # structlog config
└── tests/
```

## Roadmap

- [x] OpenAI-compatible API backend support
- [x] 429 rate limit handling (Retry-After)
- [x] Model capabilities API
- [x] Task classification with automatic model recommendation
- [ ] WebSocket support
- [ ] Authentication and authorization
- [ ] Caching functionality
- [ ] Grafana dashboard templates

## License

MIT

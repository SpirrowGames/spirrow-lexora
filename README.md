# Spirrow-Lexora

LLM Gateway / Router for Spirrow Platform

## Overview

vLLMの前段に立つプロキシ/ゲートウェイ。OpenAI API互換のエンドポイントを提供しつつ、キューイング・レート制限・統計収集などの運用機能を追加する。

## Architecture

```
Client → Lexora (Gateway) → vLLM (推論エンジン) → GPU
            :8001              :8000
```

## Requirements

- Python 3.11+
- vLLM (推論バックエンド)

## Installation

```bash
# Clone
git clone https://github.com/your-org/spirrow-lexora.git
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

設定は以下の優先順位で適用されます（後のものが優先）：

1. デフォルト値
2. 設定ファイル (`config/lexora_config.yaml`)
3. 環境変数 (`LEXORA_` プレフィックス)

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
| `LEXORA_LOGGING__LEVEL` | Log level | `INFO` |
| `LEXORA_LOGGING__FORMAT` | Log format (`console`/`json`) | `console` |

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

logging:
  level: "INFO"
  format: "console"     # "json" for production
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI compatible) |
| `/v1/completions` | POST | Text completion (OpenAI compatible) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/stats` | GET | Statistics |

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
sudo -u lexora git clone https://github.com/your-org/spirrow-lexora.git .
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

| Feature | Status |
|---------|--------|
| vLLM Proxy (chat/completions) | ✅ |
| Health Check | ✅ |
| Statistics Collection | ✅ |
| Rate Limiting (Token Bucket) | ✅ |
| Auto Retry (Exponential Backoff) | ✅ |
| Priority Queue | ✅ (component ready) |
| Streaming | ❌ (Phase 2) |
| Embeddings | ❌ (Phase 2) |

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
│   │   ├── retry_handler.py # Exponential backoff retry
│   │   └── stats.py         # Statistics collection
│   ├── backends/
│   │   ├── base.py          # Backend ABC
│   │   └── vllm.py          # vLLM client
│   └── utils/
│       └── logging.py       # structlog config
└── tests/
```

## License

MIT

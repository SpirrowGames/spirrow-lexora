# Spirrow-Lexora

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)](pyproject.toml)

LLM Gateway / Router for Spirrow Platform

## Overview

vLLMの前段に立つプロキシ/ゲートウェイ。OpenAI API互換のエンドポイントを提供しつつ、キューイング・レート制限・統計収集などの運用機能を追加する。

**主要機能:**
- OpenAI API互換エンドポイント（Chat Completions, Completions, Embeddings）
- ストリーミングレスポンス対応
- ユーザー別レート制限（Token Bucket）
- 自動リトライ（Exponential Backoff + Retry-After対応）
- 複数バックエンド対応・自動ルーティング
- **OpenAI互換APIバックエンド対応**（OpenAI, Azure OpenAI等）
- **フォールバック機能**（プライマリ失敗時に代替バックエンドへ自動切替）
- **429レート制限エラー対応**（Retry-Afterヘッダー尊重）
- Prometheusメトリクス
- 構造化ログ（structlog）

## Architecture

### Single Backend Mode
```
Client → Lexora (Gateway) → vLLM (推論エンジン) → GPU
            :8001              :8000
```

### Multi-Backend Mode
```
                              ┌→ vLLM-1 (model-a, model-b) → GPU
Client → Lexora (Gateway) ────┼→ vLLM-2 (model-c, model-d) → GPU
            :8001             └→ OpenAI API (gpt-4, etc.) [fallback]
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
| `LEXORA_RETRY__RESPECT_RETRY_AFTER` | Respect Retry-After header | `true` |
| `LEXORA_RETRY__MAX_RETRY_AFTER` | Max Retry-After delay (seconds) | `60.0` |
| `LEXORA_FALLBACK__ENABLED` | Enable fallback to alternative backends | `true` |
| `LEXORA_FALLBACK__ON_RATE_LIMIT` | Allow fallback on 429 rate limit | `true` |
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
  respect_retry_after: true   # Respect Retry-After header from 429 responses
  max_retry_after: 60.0       # Max Retry-After delay

fallback:
  enabled: true
  on_rate_limit: true   # Allow fallback on 429 errors

logging:
  level: "INFO"
  format: "console"     # "json" for production
```

### Multi-Backend Routing Configuration

複数のバックエンド（vLLM、OpenAI互換API）にモデルを分散配置し、フォールバックを設定できます：

```yaml
routing:
  enabled: true
  default_backend: "main"
  backends:
    main:
      type: "vllm"                    # vLLM backend (default)
      url: "http://localhost:8000"
      timeout: 120.0
      connect_timeout: 5.0
      models:
        - "qwen3-32b"
        - "llama3-70b"
      fallback_backends:              # Fallback on failure
        - "openai_backup"

    secondary:
      type: "vllm"
      url: "http://localhost:8010"
      timeout: 120.0
      connect_timeout: 5.0
      models:
        - "mistral-7b"
        - "embedding-model"
      fallback_backends:
        - "openai_backup"

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

**動作説明:**
- リクエストの`model`パラメータに基づいて、適切なバックエンドにルーティング
- 未登録のモデル名は`default_backend`にルーティング
- `/v1/models`は全バックエンドのモデルを集約して返却
- `/health`は全バックエンドのヘルス状態を返却（`healthy`, `degraded`, `unhealthy`）

**バックエンドタイプ:**
| Type | Description |
|------|-------------|
| `vllm` | vLLMバックエンド（デフォルト） |
| `openai_compatible` | OpenAI互換API（OpenAI, Azure OpenAI等） |

**フォールバック機能:**
- `fallback_backends`で代替バックエンドを指定
- プライマリバックエンド失敗時（接続エラー、タイムアウト、503等）に自動切替
- 429レート制限時もフォールバック可能（`fallback.on_rate_limit: true`で有効）
- 複数のフォールバックを順番に試行

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI compatible) |
| `/v1/completions` | POST | Text completion (OpenAI compatible) |
| `/v1/embeddings` | POST | Text embeddings (OpenAI compatible) |
| `/v1/models` | GET | List available models |
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

| Feature | Status | Description |
|---------|--------|-------------|
| vLLM Proxy | ✅ | Chat Completions, Completions API |
| Embeddings | ✅ | Text Embeddings API |
| Streaming | ✅ | SSE対応ストリーミング |
| Health Check | ✅ | バックエンド監視、degraded検知 |
| Statistics Collection | ✅ | リクエスト統計、トークン集計 |
| Rate Limiting | ✅ | Token Bucketアルゴリズム |
| Auto Retry | ✅ | Exponential Backoff + Retry-After対応 |
| Priority Queue | ✅ | 優先度付きリクエストキュー |
| Prometheus Metrics | ✅ | メトリクスエクスポート |
| Multi-Backend Routing | ✅ | モデル別自動ルーティング |
| OpenAI-Compatible Backend | ✅ | OpenAI, Azure OpenAI等のAPI対応 |
| Fallback Support | ✅ | プライマリ失敗時の自動切替 |
| 429 Rate Limit Handling | ✅ | Retry-Afterヘッダー尊重 |

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
│   │   ├── fallback.py      # Fallback service
│   │   ├── metrics.py       # Prometheus metrics
│   │   └── stats.py         # Statistics collection
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

- [x] OpenAI互換APIバックエンド対応
- [x] フォールバック機能
- [x] 429レート制限対応（Retry-After）
- [ ] WebSocket対応
- [ ] 認証・認可機能
- [ ] キャッシュ機能
- [ ] プロンプト解析による自動モデル選択
- [ ] Grafanaダッシュボードテンプレート

## License

MIT

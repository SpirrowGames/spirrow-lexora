# Spirrow-Lexora 設計ドキュメント

## 概要

| 項目 | 内容 |
|------|------|
| **名前** | Spirrow-Lexora |
| **世界観** | 語る辞書 |
| **役割** | LLM Gateway / Router |
| **コンセプト** | 「問いかけると答えてくれる知識の基盤」 |

```
Lexora = Lexi (辞書) + Ora (語る)

「賢くないが頼れるインフラ」
- LLMを安定して呼べる状態を維持
- キューイング、レート制限でチーム利用をサポート
- 何をどう聞くかは知らない（それはCognilens等の仕事）
```

## 配置

- **場所**: AIサーバ
- **理由**: GPU (RTX 5090) を活用したLLM推論

## 設計方針

### vLLMとの役割分担

```
┌─────────────────────────────────────────────────────────────┐
│ Lexora = vLLM (OpenAI API互換) + 運用機能ラッパー           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  vLLMに任せる:                                             │
│  ├── OpenAI API互換エンドポイント                          │
│  ├── 推論エンジン                                          │
│  ├── KVキャッシュ                                          │
│  └── Continuous Batching                                   │
│                                                             │
│  Lexoraで追加:                                             │
│  ├── キューイング（優先度付き）                            │
│  ├── ユーザー別レート制限                                  │
│  ├── 統計収集・メトリクス                                  │
│  ├── 自動リトライ・フェイルオーバー                        │
│  ├── WatchDog連携用ヘルスチェック                          │
│  └── (将来) マルチモデルルーティング                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### OpenAI API互換について

**自前実装しない**。vLLMが提供するOpenAI互換サーバをそのまま活用。

```bash
# vLLMだけでOpenAI API互換サーバが立つ
python -m vllm.entrypoints.openai.api_server \
    --model /models/qwen3-32b \
    --port 8000
```

Lexoraはその**前段にプロキシ/ゲートウェイとして立つ**。

## アーキテクチャ

```
                    ┌─────────────────────────────────────────┐
                    │           Spirrow-Lexora                │
   クライアント     │         (LLM Gateway)                   │
        │           ├─────────────────────────────────────────┤
        ▼           │                                         │
   ┌─────────┐      │  ┌─────────────────────────────────┐   │
   │ Request │─────▶│  │     API Layer (FastAPI)         │   │
   └─────────┘      │  │  - OpenAI互換エンドポイント     │   │
                    │  │  - /v1/chat/completions         │   │
                    │  │  - /v1/completions              │   │
                    │  │  - /v1/embeddings               │   │
                    │  └─────────────────────────────────┘   │
                    │                  │                      │
                    │  ┌─────────────────────────────────┐   │
                    │  │     Service Layer               │   │
                    │  │  - RequestQueue (優先度キュー)  │   │
                    │  │  - RateLimiter (レート制限)     │   │
                    │  │  - RetryHandler (リトライ)      │   │
                    │  │  - StatsCollector (統計)        │   │
                    │  │  - ModelRouter (将来)           │   │
                    │  └─────────────────────────────────┘   │
                    │                  │                      │
                    │  ┌─────────────────────────────────┐   │
                    │  │     Backend Layer               │   │
                    │  │  - vLLMBackend (メイン)         │   │
                    │  │  - OllamaBackend (開発用)       │   │
                    │  │  - LlamaCppBackend (軽量)       │   │
                    │  └─────────────────────────────────┘   │
                    │                  │                      │
                    └──────────────────│──────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │              vLLM Server                │
                    │         (OpenAI API互換)                │
                    │              :8000                      │
                    └─────────────────────────────────────────┘
                                       │
                    ┌─────────────────────────────────────────┐
                    │           Hardware Layer                │
                    │  - NVIDIA RTX 5090                      │
                    │  - CUDA 12.x                            │
                    └─────────────────────────────────────────┘
```

## 主要機能

### Phase 1: 基本機能（現在のスコープ）

#### 1. プロキシ/ゲートウェイ

vLLMの前段に立ち、リクエストを中継。

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 1. レート制限チェック
    # 2. キューに追加
    # 3. vLLMに転送
    # 4. 統計記録
    # 5. レスポンス返却
    pass
```

#### 2. キューイング（優先度付き）

```python
class RequestQueue:
    async def enqueue(
        self,
        request: Request,
        priority: str = "normal",  # high / normal / low
        timeout: float = 60.0
    ) -> Response:
        pass
```

#### 3. レート制限

```python
class RateLimiter:
    def check(
        self,
        user_id: str,
        requests_per_minute: int = 30,
        tokens_per_minute: int = 50000
    ) -> bool:
        pass
```

#### 4. 自動リトライ

```python
class RetryHandler:
    async def execute_with_retry(
        self,
        func: Callable,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_multiplier: float = 2.0
    ) -> Any:
        pass
```

#### 5. 統計収集

```python
class StatsCollector:
    def record_request(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        user_id: str
    ) -> None:
        pass
    
    def get_stats(self) -> dict:
        # {
        #   "total_requests": 15234,
        #   "total_tokens": 4523122,
        #   "avg_latency_ms": 245,
        #   "requests_per_minute": 12.5,
        #   "errors_last_hour": 2
        # }
        pass
```

#### 6. ヘルスチェック

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "qwen3-32b",
        "gpu_memory_used": "24GB",
        "gpu_memory_total": "32GB",
        "uptime": "3d 4h 12m",
        "vllm_status": "connected"
    }
```

### Phase 2: 複数モデル対応（将来）

モデル指定でルーティング。

```python
# クライアントからの呼び出し
response = await lexora.generate(
    prompt="...",
    model="code"  # code / japanese / reasoning
)
```

### Phase 3: 自動ルーティング（将来）

プロンプト内容から最適モデルを自動選択。

```
┌─────────────────────────────────────────────────────────────┐
│              Request Analyzer (将来)                        │
│  「このプロンプト、どのモデルが得意？」                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  プロンプト特性        →  最適モデル                       │
│  ─────────────────────────────────────                      │
│  Pythonコード生成      →  Qwen-Coder                       │
│  日本語ドキュメント    →  ELYZA/Swallow                    │
│  設計相談・推論        →  Llama3/Qwen3                     │
│  簡単なタスク          →  小型モデル(7B)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## API設計

### エンドポイント

OpenAI API互換 + Lexora独自エンドポイント。

```
# OpenAI互換（vLLMにプロキシ）
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
GET  /v1/models

# Lexora独自
GET  /health          # ヘルスチェック（WatchDog用）
GET  /stats           # 統計情報
GET  /metrics         # Prometheus形式メトリクス
POST /admin/reload    # 設定リロード
```

### リクエスト拡張

OpenAI互換リクエストに追加パラメータ。

```json
{
  "model": "qwen3-32b",
  "messages": [...],
  "max_tokens": 1000,
  
  // Lexora拡張
  "x_priority": "high",
  "x_user_id": "takahito",
  "x_timeout": 30
}
```

## 設定ファイル

```yaml
# lexora_config.yaml

server:
  host: "0.0.0.0"
  port: 8001

vllm:
  url: "http://localhost:8000"
  timeout: 120

default_model: "qwen3-32b"

# 将来のマルチモデル対応用
backends:
  vllm:
    enabled: true
    url: "http://localhost:8000"
    models:
      - name: "qwen3-32b"
        path: "/models/qwen3-32b"
        gpu_memory_utilization: 0.9

  ollama:
    enabled: false
    url: "http://localhost:11434"

queue:
  max_size: 100
  default_timeout: 60
  priorities:
    high: 1
    normal: 5
    low: 10

rate_limits:
  default:
    requests_per_minute: 30
    tokens_per_minute: 50000
  users:
    takahito:
      requests_per_minute: 60
      tokens_per_minute: 100000

retry:
  max_attempts: 3
  delay_seconds: 1.0
  backoff_multiplier: 2.0

logging:
  level: "INFO"
  format: "json"

metrics:
  enabled: true
  port: 9090
```

## systemd サービス設定

### vLLMサービス

```ini
# /etc/systemd/system/vllm.service

[Unit]
Description=vLLM OpenAI Compatible Server
After=network.target

[Service]
Type=simple
User=spirrow
WorkingDirectory=/opt/spirrow/vllm
ExecStart=/opt/spirrow/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /models/qwen3-32b \
    --port 8000 \
    --host 127.0.0.1
Restart=always
RestartSec=10
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

### Lexoraサービス

```ini
# /etc/systemd/system/spirrow-lexora.service

[Unit]
Description=Spirrow-Lexora LLM Gateway
After=network.target vllm.service
Requires=vllm.service

[Service]
Type=simple
User=spirrow
WorkingDirectory=/opt/spirrow/lexora
ExecStart=/opt/spirrow/lexora/venv/bin/python -m lexora.main
Restart=always
RestartSec=5
Environment="LEXORA_CONFIG=/opt/spirrow/lexora/config.yaml"

[Install]
WantedBy=multi-user.target
```

## プロジェクト構成

```
spirrow-lexora/
├── docs/
│   ├── DESIGN.md           # この設計書
│   └── API.md              # API仕様書
├── src/
│   └── lexora/
│       ├── __init__.py
│       ├── main.py         # エントリーポイント
│       ├── config.py       # 設定読み込み
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py   # エンドポイント定義
│       │   └── models.py   # リクエスト/レスポンスモデル
│       ├── services/
│       │   ├── __init__.py
│       │   ├── queue.py    # RequestQueue
│       │   ├── rate_limiter.py
│       │   ├── retry_handler.py
│       │   └── stats.py    # StatsCollector
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py     # Backend基底クラス
│       │   ├── vllm.py     # vLLMBackend
│       │   └── ollama.py   # OllamaBackend
│       └── utils/
│           ├── __init__.py
│           └── logging.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_queue.py
│   └── test_rate_limiter.py
├── config/
│   ├── lexora_config.yaml
│   └── lexora_config.example.yaml
├── scripts/
│   ├── install.sh
│   └── setup_systemd.sh
├── pyproject.toml
├── README.md
└── CLAUDE.md               # Claude Code用
```

## 責務の明確化

### Lexoraがやること
- vLLMへのプロキシ/ゲートウェイ
- キューイング（優先度付き）
- ユーザー別レート制限
- 統計収集・メトリクス
- 自動リトライ
- ヘルスチェック（WatchDog連携）
- (将来) マルチモデルルーティング

### Lexoraがやらないこと
- LLM推論そのもの（→ vLLM）
- OpenAI API互換の実装（→ vLLM）
- プロンプトエンジニアリング（→ Cognilens）
- 知識検索（→ Prismind）
- タスク管理（→ Magickit）

## 連携サービス

| サービス | 連携内容 |
|----------|----------|
| vLLM | 実際のLLM推論 |
| Cognilens | 圧縮処理のためのLLM呼び出し |
| Prismind | RAG用の埋め込み生成 |
| Magickit | オーケストレーション |
| WatchDog | ヘルスチェック、障害監視 |

## ハードウェア要件

| 項目 | 最小要件 | 推奨要件 |
|------|----------|----------|
| GPU | RTX 3090 (24GB) | RTX 5090 (32GB) |
| RAM | 32GB | 64GB |
| Storage | 100GB SSD | 500GB NVMe |

## 今後の拡張

- [ ] Phase 2: 複数モデル対応
- [ ] Phase 3: 自動ルーティング（プロンプト解析）
- [ ] マルチGPU対応（Tensor Parallelism）
- [ ] モデルの動的ロード/アンロード
- [ ] A/Bテスト機能
- [ ] Claude API フォールバック
- [ ] Prometheus/Grafanaダッシュボード

---

*Document Version: 2.0*
*Last Updated: 2026-01-16*

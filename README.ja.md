# Spirrow-Lexora

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)](pyproject.toml)

**[English version](README.md)**

Spirrow Platform向けLLMゲートウェイ/ルーター

## 概要

Spirrow-Lexoraは、vLLM推論サーバーの前段に配置するプロキシ/ゲートウェイです。OpenAI API互換のエンドポイントを提供しながら、キューイング、レート制限、統計収集、マルチバックエンドルーティングなどの運用機能を追加します。

**主要機能:**
- OpenAI API互換エンドポイント（Chat Completions, Completions, Embeddings）
- ストリーミングレスポンス対応（SSE）
- ユーザー別レート制限（Token Bucketアルゴリズム）
- 自動リトライ（Exponential Backoff + Retry-Afterヘッダー対応）
- マルチバックエンドルーティング（モデルベースの自動ルーティング）
- OpenAI互換APIバックエンド対応（OpenAI、Azure OpenAI等）
- フォールバック機能（代替バックエンドへの自動切替）
- 429レート制限対応（Retry-Afterヘッダー尊重）
- モデル能力情報API（モデル一覧と各モデルのcapabilities取得）
- タスク分類API（LLMによる最適モデル推奨）
- Prometheusメトリクスエクスポート
- 構造化ログ（structlog）

## アーキテクチャ

### シングルバックエンドモード
```
Client → Lexora (Gateway) → vLLM (推論エンジン) → GPU
            :8001              :8000
```

### マルチバックエンドモード
```
                              ┌→ vLLM-1 (model-a, model-b) → GPU
Client → Lexora (Gateway) ────┼→ vLLM-2 (model-c, model-d) → GPU
            :8001             └→ OpenAI API (gpt-4等) [フォールバック]
```

## 必要条件

- Python 3.11以上
- vLLM（推論バックエンド）

## インストール

```bash
# クローン
git clone https://github.com/SpirrowGames/spirrow-lexora.git
cd spirrow-lexora

# インストール
pip install -e ".[dev]"
```

## クイックスタート

```bash
# 開発モード
uvicorn lexora.main:app --reload --port 8001

# 本番モード
python -m lexora.main
```

## 設定

設定は以下の優先順位で適用されます（後のものが優先）：

1. デフォルト値
2. 設定ファイル (`config/lexora_config.yaml`)
3. 環境変数 (`LEXORA_` プレフィックス)

### 環境変数

| 変数名 | 説明 | デフォルト値 |
|--------|------|--------------|
| `LEXORA_VLLM__URL` | vLLMサーバーURL | `http://localhost:8000` |
| `LEXORA_VLLM__TIMEOUT` | リクエストタイムアウト（秒） | `120.0` |
| `LEXORA_VLLM__CONNECT_TIMEOUT` | 接続タイムアウト（秒） | `5.0` |
| `LEXORA_SERVER__HOST` | サーバーバインドホスト | `0.0.0.0` |
| `LEXORA_SERVER__PORT` | サーバーバインドポート | `8001` |
| `LEXORA_RATE_LIMIT__ENABLED` | レート制限を有効化 | `true` |
| `LEXORA_RATE_LIMIT__DEFAULT_RATE` | 秒間リクエスト数 | `10.0` |
| `LEXORA_RATE_LIMIT__DEFAULT_BURST` | バースト容量 | `20` |
| `LEXORA_RETRY__MAX_RETRIES` | 最大リトライ回数 | `3` |
| `LEXORA_RETRY__BASE_DELAY` | リトライ基本遅延（秒） | `1.0` |
| `LEXORA_RETRY__RESPECT_RETRY_AFTER` | Retry-Afterヘッダーを尊重 | `true` |
| `LEXORA_RETRY__MAX_RETRY_AFTER` | Retry-After最大遅延（秒） | `60.0` |
| `LEXORA_FALLBACK__ENABLED` | 代替バックエンドへのフォールバックを有効化 | `true` |
| `LEXORA_FALLBACK__ON_RATE_LIMIT` | 429レート制限時のフォールバックを許可 | `true` |
| `LEXORA_LOGGING__LEVEL` | ログレベル | `INFO` |
| `LEXORA_LOGGING__FORMAT` | ログフォーマット（`console`/`json`） | `console` |

### 設定ファイル

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
  default_rate: 10.0    # 秒間リクエスト数
  default_burst: 20

retry:
  max_retries: 3
  base_delay: 1.0       # 秒
  max_delay: 30.0       # 秒
  exponential_base: 2.0
  respect_retry_after: true   # 429レスポンスのRetry-Afterヘッダーを尊重
  max_retry_after: 60.0       # Retry-After最大遅延

fallback:
  enabled: true
  on_rate_limit: true   # 429エラー時のフォールバックを許可

logging:
  level: "INFO"
  format: "console"     # 本番環境では"json"
```

### マルチバックエンドルーティング設定

複数のバックエンド（vLLM、OpenAI互換API）にモデルを分散配置し、フォールバックを設定できます：

```yaml
routing:
  enabled: true
  default_backend: "main"
  default_model_for_unknown_task: "qwen3-32b"  # タスク分類失敗時のデフォルトモデル

  # タスク分類設定（/v1/classify-task用）
  classifier:
    enabled: true
    model: "mistral-7b"      # 分類に使うモデル（軽量モデル推奨）
    backend: "secondary"     # 分類用バックエンド

  backends:
    main:
      type: "vllm"                    # vLLMバックエンド（デフォルト）
      url: "http://localhost:8000"
      timeout: 120.0
      connect_timeout: 5.0
      models:
        # 新形式: capabilities付き（/v1/models/capabilitiesで取得可能）
        - name: "qwen3-32b"
          capabilities: ["code", "reasoning", "analysis", "general"]
          description: "コード生成・複雑な推論向け"
        - name: "llama3-70b"
          capabilities: ["reasoning", "general"]
          description: "汎用大規模モデル"
      fallback_backends:              # 失敗時のフォールバック
        - "openai_backup"

    secondary:
      type: "vllm"
      url: "http://localhost:8010"
      timeout: 120.0
      connect_timeout: 5.0
      models:
        - name: "mistral-7b"
          capabilities: ["summarization", "translation", "simple_qa"]
          description: "軽量タスク向け高速モデル"
        - "embedding-model"           # 従来形式もサポート（capabilities=["general"]）
      fallback_backends:
        - "openai_backup"

    openai_backup:
      type: "openai_compatible"       # OpenAI互換API
      url: "https://api.openai.com"
      api_key_env: "OPENAI_API_KEY"   # 環境変数からAPIキーを取得
      # api_key: "sk-..."             # または直接APIキーを指定（非推奨）
      timeout: 60.0
      connect_timeout: 5.0
      models:
        - "gpt-4"
        - "gpt-4-turbo"
      model_mapping:                  # リクエストされたモデルを実際のモデルにマッピング
        "gpt-4": "gpt-4-0125-preview"
        "gpt-4-turbo": "gpt-4-turbo-preview"
```

**動作説明:**
- リクエストの`model`パラメータに基づいて、適切なバックエンドにルーティング
- 未登録のモデル名は`default_backend`にルーティング
- `/v1/models`は全バックエンドのモデルを集約して返却
- `/health`は全バックエンドのヘルス状態を返却（`healthy`, `degraded`, `unhealthy`）

**バックエンドタイプ:**
| タイプ | 説明 |
|--------|------|
| `vllm` | vLLMバックエンド（デフォルト） |
| `openai_compatible` | OpenAI互換API（OpenAI、Azure OpenAI等） |

**フォールバック機能:**
- `fallback_backends`で代替バックエンドを指定
- プライマリバックエンド失敗時（接続エラー、タイムアウト、503等）に自動切替
- 429レート制限時もフォールバック可能（`fallback.on_rate_limit: true`で有効）
- 複数のフォールバックを順番に試行

## APIエンドポイント

| エンドポイント | メソッド | 説明 |
|----------------|----------|------|
| `/v1/chat/completions` | POST | チャット補完（OpenAI互換） |
| `/v1/completions` | POST | テキスト補完（OpenAI互換） |
| `/v1/embeddings` | POST | テキスト埋め込み（OpenAI互換） |
| `/v1/models` | GET | 利用可能なモデル一覧 |
| `/v1/models/capabilities` | GET | capabilities付きモデル一覧 |
| `/v1/classify-task` | POST | タスク分類とモデル推奨 |
| `/generate` | POST | シンプル生成エンドポイント（Magickit連携用） |
| `/chat` | POST | シンプルチャットエンドポイント（Magickit連携用） |
| `/health` | GET | ヘルスチェック |
| `/stats` | GET | 統計情報 |
| `/metrics` | GET | Prometheusメトリクス |

### リクエスト例

```bash
# ヘルスチェック
curl http://localhost:8001/health

# 統計情報
curl http://localhost:8001/stats

# チャット補完
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "messages": [{"role": "user", "content": "こんにちは"}]
  }'

# ユーザーID付き（レート制限用）
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "messages": [{"role": "user", "content": "こんにちは"}],
    "user": "user-123"
  }'

# ストリーミングチャット補完
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "messages": [{"role": "user", "content": "こんにちは"}],
    "stream": true
  }'

# ストリーミング補完
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "prompt": "むかしむかし",
    "stream": true
  }'

# 埋め込み
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-model",
    "input": "こんにちは、世界！"
  }'

# モデルcapabilities取得
curl http://localhost:8001/v1/models/capabilities | jq

# タスク分類と推奨モデル取得
curl -X POST http://localhost:8001/v1/classify-task \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Pythonでクイックソートを実装して"
  }' | jq

# シンプル生成（Magickit連携用）
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "量子コンピューティングについて説明して",
    "model": "qwen3-32b",
    "max_tokens": 500
  }'
```

## 本番デプロイ（Ubuntu）

### systemdサービス

1. サービスファイル `/etc/systemd/system/lexora.service` を作成：

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

2. 有効化と起動：

```bash
sudo systemctl daemon-reload
sudo systemctl enable lexora
sudo systemctl start lexora
```

3. ステータス確認：

```bash
sudo systemctl status lexora
sudo journalctl -u lexora -f
```

### インストールスクリプト

```bash
# ユーザーとディレクトリ作成
sudo useradd -r -s /bin/false lexora
sudo mkdir -p /opt/lexora
sudo chown lexora:lexora /opt/lexora

# クローンとインストール
cd /opt/lexora
sudo -u lexora git clone https://github.com/SpirrowGames/spirrow-lexora.git .
sudo -u lexora python3 -m venv venv
sudo -u lexora ./venv/bin/pip install -e .

# 設定ファイルコピー
sudo -u lexora cp config/lexora_config.yaml /opt/lexora/config/
# 必要に応じて設定を編集
sudo -u lexora vim /opt/lexora/config/lexora_config.yaml

# サービスのインストールと起動
sudo cp deploy/lexora.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lexora
```

## 機能一覧

| 機能 | 状態 | 説明 |
|------|------|------|
| vLLMプロキシ | ✅ | Chat Completions, Completions API |
| 埋め込み | ✅ | Text Embeddings API |
| ストリーミング | ✅ | SSEベースのストリーミングレスポンス |
| ヘルスチェック | ✅ | バックエンド監視、degraded検知 |
| 統計収集 | ✅ | リクエスト統計、トークン集計 |
| レート制限 | ✅ | Token Bucketアルゴリズム |
| 自動リトライ | ✅ | Exponential Backoff + Retry-After対応 |
| 優先度キュー | ✅ | 優先度付きリクエストキュー |
| Prometheusメトリクス | ✅ | メトリクスエクスポート |
| マルチバックエンドルーティング | ✅ | モデルベースの自動ルーティング |
| OpenAI互換バックエンド | ✅ | OpenAI、Azure OpenAI等のAPI対応 |
| フォールバック対応 | ✅ | プライマリ失敗時の自動切替 |
| 429レート制限対応 | ✅ | Retry-Afterヘッダー尊重 |
| モデルcapabilities API | ✅ | モデル一覧と能力情報の取得 |
| タスク分類 | ✅ | LLMによるタスク分類・モデル推奨 |

## 開発

```bash
# テスト実行
pytest tests/ -v

# カバレッジ付きテスト
pytest tests/ -v --cov=lexora

# 型チェック
mypy src/lexora

# Lint
ruff check src/lexora
```

## プロジェクト構成

```
spirrow-lexora/
├── pyproject.toml
├── config/
│   └── lexora_config.yaml
├── deploy/
│   └── lexora.service
├── src/lexora/
│   ├── __init__.py
│   ├── main.py              # FastAPIアプリエントリーポイント
│   ├── config.py            # Pydantic Settings
│   ├── api/
│   │   ├── routes.py        # APIエンドポイント
│   │   └── models.py        # Request/Responseモデル
│   ├── services/
│   │   ├── queue.py         # 優先度キュー
│   │   ├── rate_limiter.py  # Token Bucketレートリミッター
│   │   ├── retry_handler.py # Exponential Backoffリトライ + Retry-After
│   │   ├── router.py        # マルチバックエンドルーティング
│   │   ├── fallback.py      # フォールバックサービス
│   │   ├── metrics.py       # Prometheusメトリクス
│   │   ├── stats.py         # 統計収集
│   │   ├── model_registry.py    # モデルcapabilitiesレジストリ
│   │   └── task_classifier.py   # LLMベースのタスク分類
│   ├── backends/
│   │   ├── base.py          # Backend ABC + 例外
│   │   ├── vllm.py          # vLLMクライアント
│   │   ├── openai_compatible.py  # OpenAI互換APIクライアント
│   │   └── factory.py       # バックエンドファクトリ
│   └── utils/
│       └── logging.py       # structlog設定
└── tests/
```

## ロードマップ

- [x] OpenAI互換APIバックエンド対応
- [x] フォールバック機能
- [x] 429レート制限対応（Retry-After）
- [x] モデルcapabilities API
- [x] タスク分類による自動モデル推奨
- [ ] WebSocket対応
- [ ] 認証・認可機能
- [ ] キャッシュ機能
- [ ] Grafanaダッシュボードテンプレート

## ライセンス

MIT

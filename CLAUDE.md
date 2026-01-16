# Spirrow-Lexora

LLM Gateway / Router for Spirrow Platform

## 概要

vLLMの前段に立つプロキシ/ゲートウェイ。OpenAI API互換のエンドポイントを提供しつつ、キューイング・レート制限・統計収集などの運用機能を追加する。

## アーキテクチャ

```
Client → Lexora (Gateway) → vLLM (推論エンジン) → GPU
            :8001              :8000
```

**重要**: OpenAI API互換はvLLMが提供。Lexoraは運用機能に集中。

## 技術スタック

- Python 3.11+
- FastAPI
- httpx (非同期HTTPクライアント)
- Pydantic v2

## プロジェクト構成

```
src/lexora/
├── main.py              # FastAPIアプリ、エントリーポイント
├── config.py            # 設定読み込み (Pydantic Settings)
├── api/
│   ├── routes.py        # エンドポイント定義
│   └── models.py        # Request/Responseモデル
├── services/
│   ├── queue.py         # RequestQueue (優先度付き)
│   ├── rate_limiter.py  # ユーザー別レート制限
│   ├── retry_handler.py # リトライロジック
│   └── stats.py         # 統計収集
├── backends/
│   ├── base.py          # Backend ABC
│   └── vllm.py          # vLLM httpxクライアント
└── utils/
    └── logging.py       # structlog設定
```

## 開発ルール

### コーディング規約

- 型ヒント必須
- docstring必須（Google style）
- 非同期処理は async/await
- エラーは適切な例外クラスで

### 命名規則

- クラス: PascalCase
- 関数/変数: snake_case
- 定数: UPPER_SNAKE_CASE

### テスト

- pytest + pytest-asyncio
- カバレッジ80%以上目標
- `tests/` にミラー構成

## 主要コンポーネント

### 1. API Layer (`api/routes.py`)

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # vLLMにプロキシ
    pass

@app.get("/health")
async def health():
    # WatchDog用ヘルスチェック
    pass

@app.get("/stats")
async def stats():
    # 統計情報
    pass
```

### 2. RequestQueue (`services/queue.py`)

優先度付きキュー。asyncio.PriorityQueueベース。

```python
class RequestQueue:
    async def enqueue(request, priority="normal", timeout=60) -> Response
    async def process() -> None  # ワーカーループ
```

### 3. RateLimiter (`services/rate_limiter.py`)

Token Bucket アルゴリズム。ユーザー別に制限。

```python
class RateLimiter:
    def check(user_id: str) -> bool
    def consume(user_id: str, tokens: int) -> None
```

### 4. vLLMBackend (`backends/vllm.py`)

httpxでvLLMに非同期リクエスト。

```python
class VLLMBackend:
    async def chat_completions(request) -> Response
    async def completions(request) -> Response
    async def health_check() -> bool
```

## 設定

`config/lexora_config.yaml` を参照。環境変数でオーバーライド可能。

```bash
LEXORA_VLLM_URL=http://localhost:8000
LEXORA_PORT=8001
```

## 起動方法

```bash
# 開発
uvicorn lexora.main:app --reload --port 8001

# 本番
python -m lexora.main
```

## Phase 1 スコープ

1. vLLMへのプロキシ（/v1/chat/completions, /v1/completions）
2. ヘルスチェック（/health）
3. 統計収集（/stats）
4. 優先度付きキューイング
5. レート制限
6. 自動リトライ

## 将来の拡張（Phase 2以降）

- 複数モデル対応
- 自動ルーティング（プロンプト解析）
- Prometheus メトリクス

## 参照ドキュメント

- `docs/DESIGN.md` - 詳細設計
- `docs/API.md` - API仕様（未作成）

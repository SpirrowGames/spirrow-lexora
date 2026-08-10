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

## ヘルスチェック (`GET /health`)

`BackendRouter.health_check()` が各 backend を**並行に**叩き、`routes.py` が集計する。

**backend ごとに `health_check: false` で probe を外せる。** 外した backend は
`"skipped"` として**表示され続ける** — 名前ごと落とすと「設定されていない」と読めるが、
skipped な backend は普通にトラフィックを捌いているので別物。集計 (`status`) からは除外する。

| 値 | 意味 |
|---|---|
| `healthy` / `unhealthy` | 訊いて答えが返った |
| `skipped` | `health_check: false`。**訊いていない**。down でも未設定でもない |
| `status: unknown` | probe 対象が 1 つも無かった。`all([])` は True なので空を先に弾く必要がある |

**なぜ外すか (2026-08-11 の調査)** — probe の中身は backend 型ごとに全く違う:

- `vllm` — ローカル `GET /health` (0.00s)
- `claude_code` — `claude --version` のサブプロセス起動 (0.05s)
- `anthropic` — **本物の `/v1/messages` 推論リクエスト** (0.20s)
- `gemini` — **本物の `generateContent` 推論リクエスト** (1.2s)
- `openai_compatible` — `GET /v1/models`

∴ `/health` の 1 回ごとにリモート API の課金対象呼び出しが走り、そのプロバイダのレイテンシを
そのまま被る。gemini / claude は `health_check: false` にした — 異常が出ても運用側に打てる手が
無いため。**backend 自体は有効** (naysayer は gemini で動いている)。

`anthropic` の probe には別の問題もある: 「5xx でなければ到達可能」と判定するため、
`ANTHROPIC_API_KEY` 未設定でも 401 を **healthy と報告していた** = リクエストを処理できない
backend が緑に見えていた。

**`openai_gpt4` backend は 2026-08-11 に削除した。** `OPENAI_API_KEY` が無く常に 401 =
常に unhealthy でありながら、遠端 (`api.openai.com`) が不定期に停止して `/health` を
**20〜40s ブロック**していた (素の curl でも再現。`time_connect` / `time_appconnect` は常に高速で
`time_starttransfer` だけが伸びる ∴ こちら側の問題ではない)。`heavy` / `light` の fallback 先でも
あったが、キーが無いのでフォールバックしても 401 になるだけだった。再導入するならキー設定が先。

結果: `/health` は 2.5s (最大 40s) → **0.056s、スパイクなし** (20 回連続で計測)。

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

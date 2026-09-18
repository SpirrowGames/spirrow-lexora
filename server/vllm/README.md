# vllm

Lexora の `vllm` バックエンドが向いている推論サーバ（`http://localhost:8000`）の起動設定。
**正本は Git、`~/services/vllm/` の実体は複製。**

## 正本はこのディレクトリ、サーバー上の実体は手で配る複製

ズレは `drift-check.sh` が検出し、`--apply` で解消する（`server/docs-host/` と同じ方式）。

| このリポジトリ | sg-ai-server-01 の実体 | 読む主体 |
|---|---|---|
| `start-qwen38-27b.sh` | `~/services/vllm/start-qwen38-27b.sh` | `vllm-32b.service` の `ExecStart`（**稼働中**） |
| `start-32b.sh` | `~/services/vllm/start-32b.sh` | 同上（`qwen38.conf` を消したときの切り戻し先） |
| `start-35b.sh` `start-14b.sh` `start-1.7b.sh` `start-1.5b.sh` | `~/services/vllm/start-*.sh` | 各 `vllm-*.service`（いずれも disabled） |
| `drift-check.sh` `README.md` | （配らない。repo でだけ読む） | 人間 |

disabled のスクリプトも正本に含めている。部分的に載せると「どれが正本か」という
新しい曖昧さが増えるだけで、ファイルはどれも小さい。

    server/vllm/drift-check.sh          # 差分を見る
    server/vllm/drift-check.sh --apply  # 配る

## systemd unit はここに入れない

`vllm-32b.service` と drop-in（`memory-limit.conf` / `qwen38.conf`）は `/etc/systemd/system/`
にあり root 所有。版管理されたコピーを置くと「repo にはあるが実体は別」という嘘になりやすい
——**一次ソースは `systemctl cat vllm-32b.service`。**

これは仮定の話ではない。このリポジトリの `deploy/lexora.service` は、実在しない unit 名
（`lexora.service`）・実在しない user（`lexora`）・実在しない WorkingDirectory（`/opt/lexora`）を
記述したまま出荷されている。**unit を repo に置いた結果がそれである。**

なお `ExecStart` がどのスクリプトを指すかは drop-in `qwen38.conf` が決めている
（`ExecStart=` で一度空にしてから再定義）。`vllm-32b` という unit 名は互換のために
残っているだけで、実際に起動するのは Qwen3.8-27B である。

## 配っても再起動するまで効かない

これが `docs-host` との一番大きな違い。docs-host は timer が毎 tick で `sync.py` を
exec し直すので配れば済むが、こちらは**スクリプトが systemd unit の `ExecStart` そのもの**で、
走っているプロセスは起動時の引数を持ち続ける。

    sudo systemctl restart vllm-32b.service

**再起動は自動化しない。** GPU を落とすので Lexora の light / medium / heavy が全部止まる
（モデルロード込みで実測 40 秒）。1 コマンドを人間が打つ形に留める。

反映されたかは、unit ではなく**プロセスの引数**で確かめる:

    ps -eo args | grep -m1 api_server

## 稼働中の設定で理由のあるもの

詳細は `start-qwen38-27b.sh` のコメントに書いてある。ここは「なぜ触ってはいけないか」だけ。

- `--max-num-seqs 8` — **無いと起動時に必ず落ちる。** Qwen3.8 は 64 層中 48 層が
  linear attention（gated delta net）で、既定の cudagraph capture（最大 512 バッチ）が
  GDN の conv state 行数を超えて `causal_conv1d_update` の assert に当たる。
- `--max-model-len 32768` / `--gpu-memory-utilization 0.89` — KV は full attention 16 層分
  だけだが mamba page size に合わせて 1 トークン約 258KB と重い。**65536 以上は物理的に載らない。**
  この値は GPU を vLLM が実質専有する前提で、他プロセスを戻すなら下げること。
- `--reasoning-parser qwen3`（2026-09-18 追加） — 無いと thinking 系ティア（medium / heavy）の
  本文が `<思考文>\n</think>\n\n<答え>` になり、思考文が `content` に混ざったまま
  呼び出し元へ届く。Qwen3.8 の chat template は `<think>` を prompt 側で開くので、
  出力には閉じタグだけが残るという壊れ方をする。
- `--quantization` は**指定しない**。`config.json` の `quantization_config` から自動判別される。
  `awq` を明示すると誤る。

### reasoning parser を足したことで変わった契約

- 思考文のフィールド名は **`reasoning`**。`reasoning_content` ではない（vLLM 0.17.1）。
  `VLLMBackend.chat_completions` は純パススルーなのでそのまま呼び出し元へ届く。
- light ティアは**挙動不変**。パーサはリクエスト毎に `chat_template_kwargs` 付きで生成される
  （`entrypoints/openai/chat_completion/serving.py`）ので、`_apply_thinking_controls` が送る
  `enable_thinking=false` がそのまま尊重される。
- ⚠️ **打ち切り時の意味論が変わった。** thinking 有効で `max_tokens` が `</think>` より手前で
  尽きると **`content: null` / `finish_reason: "length"`**（思考文は `reasoning` 側に入る）。
  従来は思考文が `content` に入っていた。`.content` を str 前提で扱う呼び出し元は None を踏みうる。

## 切り戻し

    # 起動設定だけ戻す
    cp ~/services/vllm/start-qwen38-27b.sh{.bak-<timestamp>,}   # 手で取った控えがあれば
    git -C <このrepo> revert <commit> && server/vllm/drift-check.sh --apply
    sudo systemctl restart vllm-32b.service

    # モデルごと Qwen3-32B-AWQ へ戻す
    sudo rm /etc/systemd/system/vllm-32b.service.d/qwen38.conf
    sudo systemctl daemon-reload && sudo systemctl restart vllm-32b.service

# Lexora 運用手順書 — naysayer の codex backend

対象: naysayer ティアの裏にある codex backend（`type: codex`）とフォールバックのラッパー（`type: fallback`）。
出典: `T-naysayer-codex-backend` の msg-403 / msg-408 / msg-441 / msg-448 / msg-450、および Einstein の最後の承認。
コードの正本: `src/lexora/backends/codex.py`、`src/lexora/backends/fallback.py`、`src/lexora/services/process_lock.py`。

以下の例では Lexora が `http://127.0.0.1:8001` で動いているものとします。

---

## 0. 前提: Lexora はワーカー 1 つでしか動かさない（msg-450 B-7）

- codex か fallback の backend が設定されていると、Lexora は起動時に、codex の state DB の隣にある `codex.lock` を OS の排他ロックで取ります。
- **ロックが取れなければ起動しません。** ログに `ProcessLockError: another process holds the codex state` と出ます。
  - このときは、古い Lexora のプロセスが残っていないかを確かめてください（`systemctl status lexora`、`pgrep -af lexora.main`）。
  - `--workers` を増やしたり、2 つ目のインスタンスを起動したりしないでください。
- ロックは、プロセスが落ちれば OS が外します。古いロックが残ることはないので、`codex.lock` を消す必要はありません。
- 下の `inflight_runs == 0` の確認が系全体の確認として成り立つのは、このロックがあるからです。

## 1. デプロイ・再起動の前に（msg-421 S-2、msg-441 の 4）

1. status を見ます。

   ```sh
   curl -s http://127.0.0.1:8001/v1/naysayer/status | jq
   ```

2. `codex.inflight_runs` が **0** であることを確かめてから、`systemctl restart lexora` を実行します。
   - 0 でないまま再起動すると、実行中の codex の run（shadow の run を含む）が打ち切られます。その run は `run_unfinished` として latch され、人が解除するまで codex は止まります（下の 3）。
   - systemd の再起動は、古いプロセスが止まってから新しいプロセスを起動するので、B-7 のロックはぶつかりません。

> **注意（確認と再起動の間の競合）**: `inflight_runs` が 0 と読めてから `systemctl restart` を打つまでの数秒の間に、新しいリクエストが来ることがあります。その run は再起動で打ち切られ、`run_unfinished` の latch になります。
> **手順どおりに作業しても、まれに latch します。これは侵害の兆候ではありません。** 下の 3 の手順で解除してください。解除の前に、latch した run の開始時刻（state DB の `codex_gate_log`）が再起動の直前であることを確かめてください。

## 2. `codex_disabled_reason` の読み方

`GET /v1/naysayer/status` の `codex.codex_disabled_reason` は、リクエストの経路と同じ判定（`codex_availability()`）の結果です。

| 値 | 意味 | 対処 |
|---|---|---|
| `null` | codex が使える | なし |
| `quota_hold` | 枠切れのあと、`quota_hold_until` まで codex を止めている | 待つ（その間は Gemini にフォールバック） |
| `launch_failed` | `codex --version` が動かない | CLI のインストールと PATH を確かめる |
| `verification_missing` / `verification_stale` | verification が無いか、設定や CLI の版が変わった | `python -m lexora.tools.verify_codex` を通す |
| `tool_use_violation` / `run_unfinished` | latch 中 | 下の 3 |
| `schema_outdated` | state DB が古い schema | 下の 4 |
| `state_unreadable` / `state_unwritable` | state DB が読めない・書けない | ディスクと権限を確かめ、Lexora を再起動 |

## 3. latch の解除（`tool_use_violation` / `run_unfinished`）

1. `data/codex.db`（設定の `codex.state_db_path`）の違反と、未完了の run を調べます。原因を確かめてから次に進んでください。
2. 解除します。

   ```sh
   python -m lexora.tools.verify_codex --clear-violation <ID> --reason "<何が起きたか>"
   ```

   - `<ID>` は違反の ID、または未完了の run の ID です。latch した run は、違反を解除すれば run の側も解除されます。
3. **解除しただけでは codex は開きません。** `python -m lexora.tools.verify_codex` をもう一度通して、新しい `pass` を記録してください。
   - status は再起動しなくても変わります（clearance の直後は `verification_missing`、`pass` の後は `null`）。

## 4. `schema_outdated` のとき（msg-408 の 5）

- state DB が、run log（D-1e'）より前の schema で作られています。migration はありません。
- **違反の履歴が無いことを確かめてから**、state DB を作り直します。

  ```sh
  sqlite3 data/codex.db "SELECT COUNT(*) FROM codex_runtime_violation;"   # 0 であること
  systemctl stop lexora
  mv data/codex.db data/codex.db.old
  systemctl start lexora
  python -m lexora.tools.verify_codex
  ```

- 違反の行があるときは、作り直す前に、その内容を記録に残してください。

## 5. `codex_home` と `ro_binds` に symlink を使わない（msg-448 の #58 レビュー）

- `codex_home` と `ro_binds` は、絶対パスで、`.`・`..`・空の要素（`//`）を含まない形しか受け付けません。
- ただし symlink は解決しません。symlink を使うと、sandbox を広げる設定の検査をすり抜けることがあります。どちらにも、実体のディレクトリのパスを書いてください。

## 6. フォールバックの通知（msg-448 B-3）

### 設定

- Discord の webhook の URL を `/etc/lexora/environment` の `LEXORA_FALLBACK_WEBHOOK_URL` に書きます（`deploy/environment.example` を参照）。
  - URL は資格情報として扱ってください。Lexora はログに URL を出しません。
- 未設定のときは、起動時に `fallback_webhook_unset` の WARNING が 1 回出るだけで、通知は送られません。

### 届く通知

1 行の固定の書式で、500 字以内です。

```
[Lexora naysayer] fallback STARTED: reason=quota fallback_since=... gemini_calls=0 gemini_cost_usd=0.0000 quota_hold_until=...
```

| event | いつ |
|---|---|
| `STARTED` | codex から Gemini に切り替わった最初のリクエスト |
| `CONTINUING` | フォールバックが続いている間、6 時間ごと（10 分ごとに確かめる） |
| `ENDED` | フォールバックの後、codex が最初に答えたとき |

- 再起動をまたいでフォールバックが続くと、`STARTED` がもう一度届きます（仕様どおり）。
- webhook への送信に失敗すると、`fallback_notice_failed` の WARNING が出て、次の 10 分の周期で送り直されます。

### 通知が来たときに見る場所

1. `GET /v1/naysayer/status` の次の値。
   - `mode`: 次のリクエストが `codex` / `fallback` / `shadow` のどれで処理されるか。
   - `fallback_since`、`fallback_calls`、`fallback_cost_usd`。
   - `codex.codex_disabled_reason`、`codex.quota_hold_until`。
2. `codex_disabled_reason` が `quota_hold` 以外なら、上の 2 の表に従って対処します。`quota_hold` なら、待てば戻ります。
3. 期間中の費用は、台帳でも確かめられます。

   ```sh
   sqlite3 data/costs.db "SELECT COUNT(*), SUM(cost_usd) FROM request_costs WHERE answered_by='gemini-fallback' AND timestamp >= '<fallback_since>';"
   ```

## 7. shadow モードと `shadow_report`（msg-448 B-5）

- `mode: shadow` のとき、答えを返すのは常に Gemini です。codex は、同じリクエストを裏で 1 本まで走らせます。
- 結果は `data/costs.db` の `shadow_comparisons` に、verdict と所要時間だけが残ります。本文は残りません。
- 集計を出すには、次を実行します。

  ```sh
  python -m lexora.tools.shadow_report --db data/costs.db [--since 2026-09-24T00:00:00+00:00]
  ```

- 出力の見方は次のとおりです。

  | 項目 | 意味 |
  |---|---|
  | `compared` | 両方の verdict が読めた件数 |
  | `agreement_rate` / `disagreements` | `compared` のうち、一致した割合と、不一致の件数 |
  | `unparsed` | verdict を読めなかった件数（両側それぞれ） |
  | `codex_failures` | codex が失敗した件数を、理由ごとに数えたもの |
  | `gemini_errors` | Gemini の側の失敗の件数 |
  | `median_seconds` | 所要時間の中央値 |

- status の `shadow_skipped` は、裏の codex がまだ走っていたために飛ばした件数です。
- shadow の run も `inflight_runs` に数えられます。デプロイの前の確認（上の 1）は、そのまま当てはまります。
- codex が枠切れで hold 中の間は、shadow の run は走りません。そのリクエストは比較の行を残しません。

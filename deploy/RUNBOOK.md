# Lexora 運用手順書 — naysayer の codex backend

対象: naysayer ティアの裏にある codex backend（`type: codex`）とフォールバックのラッパー（`type: fallback`）。
出典: `T-naysayer-codex-backend` の msg-403 / msg-408 / msg-441 / msg-448 / msg-450、および Einstein の最後の承認。
コードの正本: `src/lexora/backends/codex.py`、`src/lexora/backends/fallback.py`、`src/lexora/services/process_lock.py`。

以下の例では Lexora が `http://127.0.0.1:8001` で動いているものとします。

> **2026-11-10 以降の出荷 mode は `codex_only` です**（Takahito 判断。`config/lexora_config.yaml` の `naysayer-codex.fallback.mode`）。
> codex が答えられないときは Gemini に回さず、そのリクエストは **502（`NaysayerCodexUnavailable`）で断られます**。Gemini の backend も `naysayer-gemini` ティアも設定から外しました。
> 以下で「Gemini にフォールバック」「Gemini が答える」とある箇所は、`codex_only` では「断られる（naysayer はその間止まる）」と読み替えてください。通知（6）は `codex unavailable STARTED / CONTINUING / ENDED` として同じ webhook に届きます。
> deploy の再起動で codex が閉じたままにならないよう、リリースの手順に codex の復帰を組み込みました（下の 11）。
> 下の 9 は、2026-10-13〜11-04 に行った `fallback` mode への切替の記録です。いまのリリースの手順は 11 です。

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
   - magickit の deploy runner 経由のリリースでは、この確認と再起動後の復帰を runner が `scripts/pre_restart.sh` / `scripts/post_restart.sh` で行います（下の 11）。手で再起動するときだけ、この手順をそのまま使います。
   - 0 でないまま再起動すると、実行中の codex の run（shadow の run を含む）が打ち切られます。その run は `run_unfinished` として latch され、人が解除するまで codex は止まります（下の 3）。
   - systemd の再起動は、古いプロセスが止まってから新しいプロセスを起動するので、B-7 のロックはぶつかりません。

> **注意（確認と再起動の間の競合）**: `inflight_runs` が 0 と読めてから `systemctl restart` を打つまでの数秒の間に、新しいリクエストが来ることがあります。その run は再起動で打ち切られ、`run_unfinished` の latch になります。
> **手順どおりに作業しても、まれに latch します。これは侵害の兆候ではありません。** 下の 3 の手順で解除してください。解除の前に、latch した run の開始時刻（state DB の `codex_gate_log`）が再起動の直前であることを確かめてください。

## 2. `codex_disabled_reason` の読み方

`GET /v1/naysayer/status` の `codex.codex_disabled_reason` は、リクエストの経路と同じ判定（`codex_availability()`）の結果です。

| 値 | 意味 | 対処 |
|---|---|---|
| `null` | codex が使える | なし |
| `quota_hold` | 枠切れのあと、`quota_hold_until` まで codex を止めている | 待つ（その間 naysayer は断られる。`fallback` mode なら Gemini） |
| `failure_hold` | 分類できない失敗（`CodexFailed`）のあと、5 分 codex を止めている（msg-765 L-2a） | 待つ（その間 naysayer は断られる。`fallback` mode なら Gemini）。繰り返すなら、ログの `naysayer_codex_failed_fallback` の snippet を見る（下の 6 の 4） |
| `launch_failed` | `codex --version` が動かない | CLI のインストールと PATH を確かめる |
| `verification_missing` / `verification_stale` | verification が無いか、設定・CLI の版・モデルカタログの中身が変わった | CLI の版が変わったなら先に `python -m lexora.tools.codex_catalog`（下の 10）。それから `python -m lexora.tools.verify_codex` を通す |
| `tool_use_violation` / `run_unfinished` | latch 中 | 下の 3 |
| `schema_outdated` | state DB が古い schema | 下の 4 |
| `state_unreadable` / `state_unwritable` | state DB が読めない・書けない | ディスクと権限を確かめ、Lexora を再起動 |
| `data_controls_unverified` | ChatGPT アカウントのデータ設定（学習に使わない設定）を人が確かめた記録が無い・読めない・30 日を過ぎた | 下の 8 |

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

- Discord の webhook の URL を、unit の `EnvironmentFile` に `LEXORA_FALLBACK_WEBHOOK_URL` として書きます（`deploy/environment.example` を参照）。
  - `deploy/lexora.service` では `/etc/lexora/environment` ですが、実際に動いている unit はこれと違うことがあります。パスは `systemctl cat <unit>` で確かめてください。
  - 読み込まれるのは起動時です。書いた後は再起動が要ります（上の 1 の手順で）。
  - webhook を読むのは `type: fallback` の backend だけです。`codex` の backend しか無い構成では、設定しても通知は届きません（下の 8 の期限前の予告も同じ）。
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
2. `codex_disabled_reason` が `quota_hold` / `failure_hold` 以外なら、上の 2 の表に従って対処します。`quota_hold` / `failure_hold` なら、待てば戻ります。
3. 通知の `reason=auth` は、codex の device login が期限切れか失効していることを表します（msg-498 B-1'）。この理由は `codex_disabled_reason` には出ません。{{HOST_SERVICES}}（[[platform:infra-registry]]）で `codex login --device-auth` をやり直してください。やり直すまでは、リクエストのたびに codex が起動して auth で失敗し、Gemini が答えます。
4. 通知の `reason=codex_failed` は、codex が分類できない失敗で終わったことを表します（msg-763 / msg-765 L-2）。そのリクエストは Gemini が答え、codex は 5 分止まります（`failure_hold`）。
   - ログの `naysayer_codex_failed_fallback`（WARNING）に、例外の class 名と stderr の先頭 300 字（`error`）が出ます。
   - 枠切れの文言（usage limit / rate limit など）が出ているなら、それは `_QUOTA_RE` が拾えなかった本物の枠切れです。その snippet を添えて、`_QUOTA_RE` と `parse_reset_at` を直すスレッドを切ってください。それまでは 5 分ごとに 1 回 codex が落ちて Gemini が答えるだけで、答えは止まりません。
   - 枠切れでない文言が繰り返すなら、snippet をもとに原因を調べます。
5. 期間中の費用は、台帳でも確かめられます。

   ```sh
   sqlite3 data/costs.db "SELECT COUNT(*), SUM(cost_usd) FROM request_costs WHERE answered_by='gemini-fallback' AND timestamp >= '<fallback_since>';"
   ```

## 7. shadow モードと `shadow_report`（msg-448 B-5）

- **出荷時の mode は `codex_only` です**（2026-11-10。それ以前は `fallback`、2026-10-13 msg-762 / msg-763 L-1）。`shadow` で比較を取り直すには、git 履歴から `gemini` backend を戻し、`fallback: "gemini"` を書き足す必要があります（`codex_only` は `fallback` を受け付けません）。
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
  | `codex_not_run` | ゲートが閉じていたために codex を走らせなかった件数を、ゲートの理由ごとに数えたもの（msg-498 B-5'）。`tool_use_violation` や `run_unfinished` があれば、比較期間の途中で latch していたということなので、上の 3 に従って解除します |
  | `gemini_errors` | Gemini の側の失敗の件数 |
  | `median_seconds` | 所要時間の中央値 |

- status の `shadow_skipped` は、裏の codex がまだ走っていたために飛ばした件数です。
- shadow の run も `inflight_runs` に数えられます。デプロイの前の確認（上の 1）は、そのまま当てはまります。
- codex が枠切れで hold 中の間は、shadow の run は走りません。そのリクエストは比較の行を残しません。

## 8. データ設定の確認と記録（`data_controls_unverified`、msg-535 A-15-2b）

codex はサブスクの login で動くため、「OpenAI が私たちのプロンプトを学習に使わない」ことは API 鍵の性質ではなく、ChatGPT アカウントの設定です。そこで、人が設定を確かめた日時を記録し、**記録から 30 日を過ぎると codex を止めます**（fail-closed）。記録が無い・読めない・形式が違う・未来の日時のときも止まります。止まっている間は Gemini にフォールバックし、通知は `reason=data_controls_unverified runbook=deploy/RUNBOOK.md#8` で届きます。

### 記録の場所と形式

- ファイル: 設定の `codex.data_controls_file`（既定は `data/codex_data_controls.yaml`）。
- 中身は 1 行です。UTC オフセット付きの ISO 8601 で書きます（オフセットの無い時刻は受け付けません）。

  ```yaml
  data_controls_verified_at: "2026-09-29T01:00:00Z"
  ```

### 手順（30 日ごと、および再ログインのたび）

1. ChatGPT に codex と同じアカウントでログインし、データの設定で「モデルの改善に使う」がオフであることを確かめます。
2. 確かめた時刻を書きます。書きかけのファイルを読まれないよう、別名で書いてから置き換えます。

   ```sh
   printf 'data_controls_verified_at: "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > data/codex_data_controls.yaml.tmp
   mv data/codex_data_controls.yaml.tmp data/codex_data_controls.yaml
   ```

3. **再起動は要りません。** Lexora はリクエストと status のたびにファイルの変化を見て読み直します。`GET /v1/naysayer/status` の `codex.codex_disabled_reason` が `data_controls_unverified` でなくなったことを確かめます。ログには `codex_data_controls_reloaded`（`expires_at` 付き）が出ます。
4. フォールバック中だった場合は、次に codex が答えたときに `fallback ENDED` の通知が届きます。

`codex login --device-auth` をやり直したとき（上の 6 の `reason=auth` の対処）も、同じアカウントかどうかを含めてこの手順を必ず踏んでください。

### 期限前の予告

- 期限の 7 日前から、フォールバック通知と同じ webhook に、UTC の 1 日 1 回、`data controls EXPIRING: codex stops in N day(s)` が届きます。
- 送ったことはメモリに持っています。その日のうちに Lexora を再起動すると、同じ日にもう 1 回届くことがあります。送信に失敗したときは、10 分後の確認でもう一度送ります。
- 期限が切れた後は予告は止まり、上のフォールバック通知（STARTED、6 時間ごとの CONTINUING、ENDED）に替わります。shadow モードでは Gemini が答えるのでフォールバック通知は出ません。止まっている間は、`shadow_report` の `codex_not_run` に `data_controls_unverified` として数えられます。

### 再発防止: 2026-11-01〜11-04 の停止（msg-953 / msg-974 / msg-975 / msg-976）

- 下の 9 の切替のときに書いた記録が、**30 日で期限切れ**になりました。更新が間に合わず、2026-11-01〜11-04 の間、codex は `data_controls_unverified` で止まっていました（当時の `fallback` mode では、その間の naysayer は Gemini が答えた）。
- **`codex_only` の今は、期限が切れると naysayer は止まります**（Gemini に回らず、502 で断られる）。deploy でも直りません（下の 11 の `post_restart.sh` は `codex_disabled_reason` が `null` でなければ exit 5 で終わるだけです）。
- したがって、**期限の前に**上の手順で確かめ直し、記録を書き直します。
  - 7 日前から届く `data controls EXPIRING` の予告を受けたら、その日のうちに行います。予告は `LEXORA_FALLBACK_WEBHOOK_URL` が設定されていないと届きません（上の 6）。
  - 期限の日時は、記録の `data_controls_verified_at` に 30 日を足したものです。予告とログの `codex_data_controls_reloaded` に `expires_at` として出ます。
- 期限が切れてしまったときも、手順は同じです（上の「手順」の 1〜3。再起動は要りません）。

## 9. codex を primary にした切替の記録（2026-10-13〜11-04、fallback mode。msg-953 / msg-974 / msg-975 / msg-976）

> **この節は、2026-10-13〜11-04 に行った `fallback` mode（codex が答え、閉じている間は Gemini が答える）への切替の記録です。** 実際に行った順序に合わせて書き直しました（2026-11-10、msg-976 の 1 件目）。
> **2026-11-10 以降の出荷 mode は `codex_only` です**（冒頭の注記と下の 11）。いまのリリースと codex の復帰は 11 の手順で行い、この節の手順は繰り返しません。
> この節で `gemini-fallback`・`unrouted`（S-2）・「Gemini が答える」と書いてあるのは、**当時の `fallback` mode の挙動**です。`codex_only` では Gemini の backend も `naysayer-gemini` ティアも `unrouted` も設定に無く、codex が答えられないリクエストは 502（`NaysayerCodexUnavailable`）で断られます。

出典: `T-naysayer-codex-backend` の msg-762（Takahito の Tier-C 判断）、msg-763（手順）、msg-765（`failure_hold` と C-3 の注）、msg-778（条件 1〜3）、msg-950（Takahito の裁定: shadow を挟まない）、msg-953 / msg-974 / msg-975 / msg-976（実際に行った順序と、この書き直しの範囲）。
ホスト上の作業は Takahito か、ホスト上の operator lane が {{HOST_SERVICES}}（[[platform:infra-registry]]）で行いました。

> 注: 下の 10 の変更以降、`verify_codex` の `pass` は **deploy されたコードと設定でしか取れません**。config hash に `cli_overrides`・`ro_binds`・カタログの中身が入っているためです。deploy の前の Lexora で取った `pass` は、deploy の後に `verification_stale` になります。したがって `verify_codex` はフェーズ 0 ではなく、フェーズ 1 の C-2′ で撃ちました。

### 実際に行った順序

- 順序は **mindwire が先（#463/#464 の main merge と mindwire の deploy）、Lexora の deploy が後** でした。
  - 当初の計画（msg-778 の順序 (a)）は逆で、Lexora の deploy を先にするものでした。この節は実際の順序に合わせて書き直したものです。
- **shadow のリリースは挟みませんでした**（Takahito の裁定、msg-950。msg-777 の (b) も採っていません）。
- deploy（C-2）から `verify_codex` の `pass`（C-2′）までの間は codex が閉じているので、当時の `fallback` mode では Gemini が `gemini-fallback` で答えました（`codex_only` の今は、この間のリクエストは断られます）。窓を短くするため、フェーズ 0 はすべて切り替えの前に済ませ、フェーズ 1 は 1 回の作業で続けて行い、naysayer の呼び出しが流れていない時間帯を選びました。

### フェーズ 0: 切り替えの前

- **H-1**（msg-267）: ChatGPT Pro $100 を契約し、ChatGPT のセキュリティ設定で device code 認証を有効にします。
- **H-2**: Lexora の**サービスユーザ**として、設定の `codex_home` にログインします。`~/.codex` にログインしても Lexora からは見えません。

  ```sh
  CODEX_HOME="<config/lexora_config.yaml の codex.codex_home>" codex login --device-auth
  codex --version   # 0.160.0 であること（CODEX_MODEL_PIN と同梱カタログの前提）
  ```

  - `auth.json` はパスワードと同じ扱いです（リポジトリ・ログ・チャットに出さない）。
- **H-3**（上の 8）: データ設定を確かめ、`data/codex_data_controls.yaml` に記録します。**30 日で期限が切れる記録です。** 切替の後に更新が間に合わず、2026-11-01〜11-04 に codex が `data_controls_unverified` で止まりました（上の 8 の「再発防止」）。
- **H-4′**（下の 10）: リリースのコードの**新しいチェックアウト**から `python -m lexora.tools.codex_catalog` を撃ち、`codex_home` にカタログを置いておきます（`--check` で exit 0 になること）。
  - deploy の**前**に書きます。deploy の後で書き直すと、C-2′ で取る `pass` の hash とずれます。
  - 再起動は要りません。カタログの sha256 は `config_hash()` を呼ぶたびにファイルから計算し直しており、起動時にキャッシュしていません（`backends/codex.py` の `model_catalog_digest`）。
- **H-4″**（下の 10「sandbox の中の DNS」）: このホストの `EnvironmentFile` に `LEXORA_CODEX_RO_BINDS` を書きます。{{HOST_SERVICES}}（[[platform:infra-registry]]）では `LEXORA_CODEX_RO_BINDS=/run/systemd/resolve/stub-resolv.conf` です。
  - 共有の `config/lexora_config.yaml` にはこのパスを書きません（ホストに依存するため）。
- **H-4‴ は不要でした**（旧版では、ホストの設定ファイルに `redact:` / `unrouted:` を書く手順でした）。
  - `config/lexora_config.yaml` はリリースに同梱されています。`load_yaml_config`（`src/lexora/config.py`）は、`config/lexora_config.yaml` → `lexora_config.yaml` → `/etc/lexora/config.yaml` の順に**作業ディレクトリから**探し、最初に見つかったものを読みます。サービスはリリースのディレクトリで動くので、同梱の設定が先に見つかり、`/etc/lexora/config.yaml` は使われません。
  - `redact` / `unrouted` はリリースの設定に入っているので、ホストでの作業はありません。
- **H-5**（上の 6）: `systemctl cat lexora` で `EnvironmentFile` に `LEXORA_FALLBACK_WEBHOOK_URL` があること、起動ログに `fallback_webhook_unset` が出ていないことを確かめます。
- **H-6′**（msg-778）: deploy の前に、本番の `codex_home` で `CodexBackend.chat_completions` 経由の本物の run を 1 本流し、`tool_use_violation` が記録されないこと、`agent_message` と `turn.completed` が届くこと、usage が読めることを確かめます（2026-10-13 に operator が実施済み、msg-777）。

### フェーズ 1: 切り替え（1 回の作業で続けて行った）

ステップの番号は msg-763 / msg-778 のものを残し、実際に行った順に並べています（C-4 が先頭）。

- **C-4**（実際には最初）: mindwire の #463/#464 を main に merge し、mindwire を deploy します（attestation が `{codex, gemini-fallback}` を受け入れる版）。
- **C-1**: Lexora のリリース PR（`naysayer-codex` を `mode: fallback` にし、下の 10 の設定を含むもの）を main に merge します。
- **C-2**（上の 1）: `codex.inflight_runs == 0` を確かめてから、Lexora の deploy request を出します。
- **C-2′**（msg-778 の条件 2）: deploy の直後、`codex.inflight_runs == 0` を確かめて `python -m lexora.tools.verify_codex` を撃ち、`pass` を取ります。
  - **サービスと同じ `EnvironmentFile` を読み込んだシェルで撃ちます**（`set -a; . /etc/lexora/environment; set +a`。パスは `systemctl cat lexora` で確かめる）。`LEXORA_CODEX_RO_BINDS` が無いと `ro_binds` が違う hash になり、その `pass` はサービスから見て `verification_stale` です。
  - 今のリリースでは、この再検証は deploy runner の `scripts/post_restart.sh`（`codex_restore`）が行います（下の 11）。
- **C-3**: status が `mode: codex`、`codex.codex_disabled_reason: null` になることを確かめます。
  - 注: status の `mode` は設定の値ではなく、**次のリクエストの経路**を返します（`fallback.py` の `status_fields`）。設定が `fallback` で codex が開いていれば `codex` になります。
- **C-5**（msg-778 の条件 3）: 最初の数本の naysayer の呼び出しで、次をすべて確かめます。
  - 台帳に `answered_by='codex'` の行ができ、tokens が 0 でないこと。
  - mindwire の attestation が通ること。
  - status の `codex.codex_disabled_reason` が `failure_hold` を繰り返していないこと。
  - `tool_use_violation` が出ていないこと。

  ```sh
  sqlite3 data/costs.db "SELECT id, backend, answered_by, tokens_input, tokens_output, redactions FROM request_costs ORDER BY id DESC LIMIT 5;"
  journalctl -u lexora --since '-15 min' | grep -E 'n3_route_unrouted|ingress_redaction_failed'
  ```

  - **当時の挙動（`fallback` mode の設定）**: `X-Mindwire-N3-Route: v1` の無い naysayer の要求は codex を試さず `unrouted`（Gemini）が答え、その行は `backend='gemini'`・`answered_by='gemini'`、ジャーナルに `n3_route_unrouted`（`n3_route=absent`）が出ました（S-2）。ヘッダを付ける mindwire（#463）を先に deploy したので、Lexora の deploy の時点で mindwire はすでにヘッダを付けていました。
    - 2026-11-10 以降は `unrouted` を置かず（S-2 は任意化）、ヘッダの有無にかかわらず codex に送ります。
  - 伏せ字をした行は `redactions` に `kind:count,…`（数だけ）が入ります。

### ロールバック（mindwire の #463/#464 が merge 済みである前提）

mindwire の #463/#464 はすでに main に merge され、deploy されています。旧版にあった「#463/#464 を merge せずに Lexora を `dbe1170`（shadow）に戻す」ロールバックは、もう取れません（`dbe1170` と組み合わせる前提だった旧 mindwire は残っていません）。いまのロールバックは次の 2 つで、**どちらを取るかで手順がまったく違います**。

#### A. 今のリリース（`codex_only`）の中でのロールバック

- codex が答えられないだけなら、ロールバックは要りません。下の 11 の手順と、上の 2・3・8 で codex を開けます（`codex_only` では、その間 naysayer は断られます）。
- 新しいリリースが原因なら、直前の `codex_only` のリリースに戻します。いつもの deploy と同じ経路で、deploy runner が下の 11 の `pre_restart.sh` / `post_restart.sh` を実行します。mindwire 側の作業はありません。

#### B. Gemini に戻す（`fallback` mode を復元する）

`codex_only` は `fallback` を受け付けないので、**`mode` の値を `fallback` に変えるだけでは戻りません**（`fallback` を書かずに `mode: fallback` にすると、`fallback_floor_s` の既定 600 が足されて時間の予算を超え、起動しません）。Gemini を naysayer の経路に戻すのは 2026-11-10 の Takahito 判断を覆すことなので、その判断を取ってから行います。

1. git 履歴から設定を戻します。`config/lexora_config.yaml` の、lexora#110（commit `c5bf320`）より前の版（例: `git show c5bf320^:config/lexora_config.yaml`）にある次をすべて戻します。
   - `backends.gemini`（`type: "gemini"`、`api_key_env: "GEMINI_API_KEY"`、`paid_key_acknowledged: true` ほか）。
   - `naysayer-codex.fallback` の `fallback: "gemini"` と `mode: "fallback"`。
   - 時間の予算: `slot_wait_s + codex_timeout_s + fallback_floor_s <= caller_budget_s`（900）を満たす値（当時は 150 + 150 + 600）。今の 600 + 270 のままでは起動しません。
   - `naysayer` ティアの `model`（当時は `gemini-3.1-pro-preview`。fallback 先の解決 `resolve_model("naysayer")` に使う）。`unrouted: "gemini"` と `naysayer-gemini` ティアは、要るかどうかをそのときに決めます（S-2 は任意化済み）。
2. `GEMINI_API_KEY` がサービスの `EnvironmentFile` に残っていることを確かめます（`config/lexora_config.yaml` のコメントでは `/etc/spirrow-lexora.env` に残置。パスは `systemctl cat lexora` で確かめる）。
3. 上の 6 の通知の設定（`LEXORA_FALLBACK_WEBHOOK_URL`）が残っていることを確かめます。
4. mindwire 側が `answered_by='gemini-fallback'` の行を受け入れる版であることを確かめます。#463/#464 は `{codex, gemini-fallback}` を受け入れる版でしたが、2026-11-10 の変更（ADR-14 §10）で mindwire 側の期待値が変わっていれば、その戻しも要ります。
5. リリース PR を出し、いつもの経路で deploy します（下の 11 の復帰はそのまま動きます）。deploy の後、status の `mode` と `codex_disabled_reason` が想定どおりであることを確かめます。

### 実際に枠が切れるまで確かめられないこと

- 枠切れのときの stderr の文言と、リセット時刻の書式です。文言が想定と違っても、その失敗は `CodexFailed` として扱われ、通知は `reason=codex_failed` で届きます（上の 6 の 4。当時は Gemini に落ち、`codex_only` では断られる）。そうなったら、ログの snippet をもとに `_QUOTA_RE` と `parse_reset_at` を直す別スレッドを切ります。

## 10. ツールを全部無効にする設定とモデルカタログ（2026-10-13 実測、codex-cli 0.160.0）

naysayer の codex はツール無しで動かします（Takahito の方針、spirrow-mindwire D8 msg-6050。`auth.json` が同じ sandbox にあるので、ファイルを読めるツールがあると読めてしまいます）。

### 設定の中身

- `config/lexora_config.yaml` の `codex.cli_overrides` に、フラグ（`features.*=false` と `tools.experimental_request_user_input.enabled=false`）と `model_catalog_json` を入れています。
  - 何がどのツールを消すかの根拠は `backends/codex.py` の `TOOL_DISABLE_FEATURE_OVERRIDES` と `CATALOG_KEYS_*` にあります。
  - フラグだけでは、code mode の `exec`、`apply_patch`、`collaboration.*`（`spawn_agent` など）、`clock.*` が残ります。そこで、pin したモデルの項目からそれらを外したモデルカタログを `model_catalog_json` で渡します。
  - `features.apps=false` が無いと、ログイン済みの `codex_home` では ChatGPT のコネクタ（MCP）のツールが付きます。空の `codex_home` で試すと出ないので注意してください。
- これらのキーは `TOOL_DISABLE_OVERRIDE_KEYS` と同じ集合です（`tests/test_shipped_config.py` が検査します）。V-2′-control はこのキーをすべて外して走るので、control は CLI の既定（code mode の `exec_command`）でツールを実行し、本番の設定では宣言されるツールがゼロであることを `V-2/3_no_tools_declared` が確かめます。

### カタログを書く

```sh
python -m lexora.tools.codex_catalog            # codex_home にカタログを書く（600）
python -m lexora.tools.codex_catalog --check    # 書かずに比べる。違えば exit 1
python -m lexora.tools.verify_codex             # カタログを書いたら必ずやり直す
```

- カタログは、インストールされた CLI の同梱カタログ（`codex debug models`、空の `CODEX_HOME` で実行）から pin したモデルだけを取り出し、ツールの項目を編集したものです。context window が `CODEX_MODEL_PIN` と合わなければ書きません（pin を読み直してください）。
- **カタログの中身の sha256 は config hash に入ります。** カタログを書き直すと `verification_stale` になり、`verify_codex` の `pass` までゲートは閉じます。
- CLI を上げたときは、`codex_catalog` → `verify_codex` の順にやり直します。

### sandbox の中の DNS

- このホストの `/etc/resolv.conf` は systemd-resolved の `stub-resolv.conf` への symlink です。bwrap は `/run` を入れないので、そのままでは sandbox の中で名前解決ができず、`codex exec` は `workspace routing discovery failed` で落ちます。
- そのため `stub-resolv.conf` のファイル 1 つだけを sandbox に読み取り専用で入れます（同じディレクトリにある resolved の制御ソケットは入れません）。
- **このパスはホストに依存するので、共有の `config/lexora_config.yaml` には書かず、ホストごとの deploy の手順で書きます。** unit の `EnvironmentFile` に次を書きます（`deploy/environment.example` を参照）。

  ```sh
  # 複数書くときは ":" で区切る。説明は必ずこのように別の行に書く
  LEXORA_CODEX_RO_BINDS=/run/systemd/resolve/stub-resolv.conf
  ```

  - 値は YAML の `codex.ro_binds` に**足され**、YAML の項目と同じ検査（絶対パス・正規化・sandbox を広げないこと、上の 5）を通り、config hash に入ります。書いた・変えた・消したときは再起動のあと `verify_codex` をやり直します。
  - **代入の行の後ろにコメントを書かないでください。** systemd の `EnvironmentFile` は行の途中の `#` をコメントとして扱わず、値に含めます（シェルで `.` したときは捨てられるので、サービスと `verify_codex` で値が食い違います）。引用符も付けません。
  - 空の要素（先頭・末尾・連続した `:`）、空白や `#` を含む要素、`codex` の backend が無い構成での設定は、起動を拒否します。
  - `verify_codex` は同じ `EnvironmentFile` を読み込んだシェルで撃ちます（上の 9 の C-2′）。
- 別のホストに移すときは、そのホストの `/etc/resolv.conf` の実体を確かめ、そのホストの `EnvironmentFile` に合うパスを書きます（`/etc` は sandbox に読み取り専用で入るので、`/etc/resolv.conf` が実ファイルならこの変数は要りません。`/etc` の外を指す symlink なら、その先の実ファイルのパスを書きます）。

## 11. リリース（deploy）と codex の復帰（2026-11-10、Takahito）

`codex_only` では codex が閉じている間 naysayer が止まるので、**リリースは codex が開いた状態で終わるところまで**を含めます。
きっかけは 2026-11-08 の lexora#108 の deploy です。再起動が 3 秒前に始まった codex の run を殺し（`aborted:signal`、violation 1017）、codex は 2 日間閉じたままでした（その間の naysayer 332 件は、当時の `fallback` mode で全部 Gemini が答えた）。

magickit の deploy runner は、spirrow-lexora の再起動の前後で次の 2 つを実行します（`magickit/deploy/registry.py` の `pre_restart_script` / `post_restart_script`）。

1. **再起動の前: `scripts/pre_restart.sh`** — `codex.inflight_runs == 0` になるまで最大 300 秒待ちます。待ちきれなくても deploy は止めません（取りこぼしは次の 2 が片付けます）。上の 1 の手順を runner が代わりに行うものです。
2. **再起動の後: `scripts/post_restart.sh`** — 動いているサービスの環境（`/proc/<MainPID>/environ`。`EnvironmentFile` の `LEXORA_CODEX_RO_BINDS` を含む）で `python -m lexora.tools.codex_restore` を実行します。runner は再起動の直前と直後の時刻を `MAGICKIT_DEPLOY_RESTART_STARTED_AT` / `_FINISHED_AT` で渡します。
   - **再起動が原因の latch だけを解除します。** 条件は、violation の detail が `["aborted:signal"]` か `["aborted:cancelled"]` で、再起動の時間帯（前後 5 秒）に書かれたもの。または、未完了の run で、再起動の時刻より `codex_timeout_s` 以内に始まったもの（再起動のときに走っていた run）。
   - **それ以外の latch が 1 つでもあれば、何も解除しません**（exit 3）。本物のツール使用、timeout の latch、以前のクラッシュの残りなどは、上の 3 の手順で人が調べます。
   - 解除したとき、またはサービスが `verification_missing` / `verification_stale` を返すとき（設定やカタログが変わった deploy）は、`verify_codex` と同じ検査を実行して記録します。`fail` なら exit 4。
   - 最後に、動いているサービスの `/v1/naysayer/status` を読みます。`codex_disabled_reason` が `null` でなければ exit 5 で、理由を出力します（`quota_hold` や `data_controls_unverified` は deploy では直せません。上の 2 と 8）。
   - **0 以外で終わると、runner はその deploy を `failed` として記録します**（サービスは新しいコードで動いています。残っているのは codex の復帰だけで、`error` に理由が書かれます）。

手で確かめるとき（何も書き込まない）:

```sh
python -m lexora.tools.codex_restore --restart-started-at <ISO> --restart-finished-at <ISO> --dry-run
```

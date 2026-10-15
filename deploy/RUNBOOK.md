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
| `failure_hold` | 分類できない失敗（`CodexFailed`）のあと、5 分 codex を止めている（msg-765 L-2a） | 待つ（その間は Gemini にフォールバック）。繰り返すなら、ログの `naysayer_codex_failed_fallback` の snippet を見る（下の 6 の 4） |
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

- **出荷時の mode は `fallback` です**（2026-10-13、msg-762 / msg-763 L-1）。codex が答え、codex が閉じている間だけ Gemini が答えます。`shadow` は、比較を取り直すときにだけ使います（`config/lexora_config.yaml` の `naysayer-codex.fallback.mode`）。
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

## 9. codex を primary に切り替える手順（msg-763 / msg-765 / msg-778）

出典: `T-naysayer-codex-backend` の msg-762（Takahito の Tier-C 判断）、msg-763（手順）、msg-765（`failure_hold` と C-3 の注）、msg-778（順序 (a) と条件 1〜3。下の 10 の変更で前提が変わったための書き直し）、Einstein の承認。
ホスト上の作業は Takahito か、ホスト上の operator lane が {{HOST_SERVICES}}（[[platform:infra-registry]]）で行います。

> 注: 下の 10 の変更以降、`verify_codex` の `pass` は **deploy されたコードと設定でしか取れません**。config hash に `cli_overrides`・`ro_binds`・カタログの中身が入っているためです。deploy の前の Lexora で取った `pass` は、deploy の後に `verification_stale` になります。したがって `verify_codex` はフェーズ 0 ではなく、フェーズ 1 の C-2′ で撃ちます。

### 順序と fail-closed の窓

- 順序は **Lexora の deploy が先、mindwire の #463/#464 の main merge と mindwire の deploy が後** です（#463 の HOLD 条件 (i′) が「本番 Lexora が fallback で動き、status が `codex_disabled_reason: null`」のため）。
- その間は、codex が答えた行が旧 mindwire（`expected=gemini`）の attestation で落ち、naysayer の呼び出しが fail closed になります。間違った verdict が通ることはなく、gate や design レビューが止まって後でやり直されるだけです。
- deploy（C-2）から `verify_codex` の `pass`（C-2′）までの間（約 1 分）は codex が閉じているので、Gemini が `gemini-fallback` で答えます。これも旧 mindwire が弾くので、同じ種類の fail-closed の窓です。
- 窓を短くするため、フェーズ 0 はすべて切り替えの前に済ませ、フェーズ 1 は 1 回の作業で続けて行い、naysayer の呼び出しが流れていない時間帯を選びます。shadow のリリースを挟む案（msg-777 の (b)）は採りません（msg-778）。

### フェーズ 0: 切り替えの前（今の deploy のまま。費用は変わらない）

- **H-1**（msg-267）: ChatGPT Pro $100 を契約し、ChatGPT のセキュリティ設定で device code 認証を有効にします。済んでいれば飛ばします。
- **H-2**: Lexora の**サービスユーザ**として、設定の `codex_home` にログインします。`~/.codex` にログインしても Lexora からは見えません。

  ```sh
  CODEX_HOME="<config/lexora_config.yaml の codex.codex_home>" codex login --device-auth
  codex --version   # 0.160.0 であること（CODEX_MODEL_PIN と同梱カタログの前提）
  ```

  - `auth.json` はパスワードと同じ扱いです（リポジトリ・ログ・チャットに出さない）。
- **H-3**（上の 8）: データ設定を確かめ、`data/codex_data_controls.yaml` に記録します。30 日ごとに繰り返す作業で、期限の 7 日前から webhook に予告が届きます。
- **H-4′**（下の 10）: リリースのコードの**新しいチェックアウト**から `python -m lexora.tools.codex_catalog` を撃ち、`codex_home` にカタログを置いておきます（`--check` で exit 0 になること）。
  - deploy の**前**に書きます。deploy の後で書き直すと、C-2′ で取る `pass` の hash とずれます。
  - 再起動は要りません。カタログの sha256 は `config_hash()` を呼ぶたびにファイルから計算し直しており、起動時にキャッシュしていません（`backends/codex.py` の `model_catalog_digest`）。新しいカタログを読むのは C-2 で起動する新しいサービスです。
  - 今動いている旧サービス（`dbe1170`）には影響しません。旧設定には `cli_overrides` が無く（`model_catalog_json` も無い）、旧コードの config hash はカタログの中身を含まないので、`codex_home` にファイルが 1 つ増えても CLI も hash も変わりません。
- **H-4″**（下の 10「sandbox の中の DNS」）: このホストの `EnvironmentFile` に `LEXORA_CODEX_RO_BINDS` を書きます。{{HOST_SERVICES}}（[[platform:infra-registry]]）では `LEXORA_CODEX_RO_BINDS=/run/systemd/resolve/stub-resolv.conf` です。
  - 共有の `config/lexora_config.yaml` にはこのパスを書きません（ホストに依存するため）。
  - 旧サービス（`dbe1170`）はこの変数を読まないので、deploy の前に書いても影響しません。読むのは C-2 で起動する新しいサービスです。
- **H-4‴**（ADR-14 §7.3、spirrow-mindwire#482。T-naysayer-codex-backend msg-800 §1 / msg-802）: サービスが共有の `config/lexora_config.yaml` ではなくホストの設定ファイル（例 `/etc/lexora/config.yaml`）を読んでいる場合は、**どの deploy よりも先に**、その中のすべてのティアに `redact:` を書きます。`redact` には既定値が無く、書いていないティアがあると新しいサービスは起動しません。
  - 値は共有の YAML と同じにします: `naysayer` と `naysayer-gemini` は `true`、ほかは `false`。
  - 連鎖に codex を含むティア（`naysayer`）には `unrouted: "gemini"` も書きます。書かないと起動しません（S-2）。codex を含まないティアに `unrouted` を書いても起動しません（S-4）。
  - 旧サービス（`dbe1170`）はこの 2 つのキーを知らず、`extra="forbid"` で起動を拒否します。したがって、ホストの設定ファイルを書き換えるのは C-2 の直前です。共有の YAML を使っている場合は、リリースのコードに入っているので作業はありません。
- **H-5**（上の 6）: `systemctl cat lexora` で `EnvironmentFile` に `LEXORA_FALLBACK_WEBHOOK_URL` があること、起動ログに `fallback_webhook_unset` が出ていないことを確かめます。書き足したら上の 1 の手順で再起動します。
- **H-6′**（msg-778。shadow_report による旧 H-6 の置き換え）: deploy の前に、本番の `codex_home` で `CodexBackend.chat_completions` 経由の本物の run を 1 本流し、`tool_use_violation` が記録されないこと、`agent_message` と `turn.completed` が届くこと、usage が読めることを確かめます。shadow_report は使いません（2026-10-13 に operator が実施済み、msg-777）。

### フェーズ 1: 切り替え（1 回の作業で続けて行う）

- **C-1**: Lexora のリリース PR（`naysayer-codex` を `mode: fallback` にし、下の 10 の設定を含むもの）を main に merge します。
- **C-2**（上の 1）: `codex.inflight_runs == 0` を確かめてから、Lexora の deploy request を出します（前回の deploy と同じ経路）。
- **C-2′**（msg-778 の条件 2）: deploy の直後、`codex.inflight_runs == 0` を確かめて `python -m lexora.tools.verify_codex` を撃ち、`pass` を取ります。
  - **サービスと同じ `EnvironmentFile` を読み込んだシェルで撃ちます**（`set -a; . /etc/lexora/environment; set +a`。パスは `systemctl cat lexora` で確かめる）。`LEXORA_CODEX_RO_BINDS` が無いと `ro_binds` が違う hash になり、その `pass` はサービスから見て `verification_stale` です（DNS も引けず control が落ちます）。
  - **deploy から 1 分以内に `pass` が取れなければ、窓を延ばさずに下のロールバックに進みます。**
- **C-3**: status が `mode: codex`、`codex.codex_disabled_reason: null` になることを確かめます。ならなければ、下のロールバックに進みます。
  - 注: status の `mode` は設定の値ではなく、**次のリクエストの経路**を返します（`fallback.py` の `status_fields`）。設定が `fallback` で codex が開いていれば `codex` になります。`codex` なら成功、`fallback` なら codex が閉じています（`codex_disabled_reason` を見る）。
- **C-4**: mindwire の #463/#464 を main に merge し、mindwire を deploy します。
- **C-5**（msg-778 の条件 3 を含む）: 最初の数本の naysayer の呼び出しで、次をすべて確かめます。
  - 台帳に `answered_by='codex'` の行ができ、tokens が 0 でないこと。
  - mindwire の attestation が通ること。
  - status の `codex.codex_disabled_reason` が `failure_hold` を繰り返していないこと。
  - `tool_use_violation` が出ていないこと。

  ```sh
  sqlite3 data/costs.db "SELECT id, backend, answered_by, tokens_input, tokens_output FROM request_costs ORDER BY id DESC LIMIT 5;"
  ```

  - C-4 の前に `failure_hold` の繰り返しが見えたら、#463/#464 は merge しません。
  - **N-3 の経路の宣言と伏せ字**（ADR-14 §7.3）: `X-Mindwire-N3-Route: v1` の無い naysayer の要求は codex を試さず、`unrouted`（Gemini）が答えます。その行は `backend='gemini'`・`answered_by='gemini'` で、ジャーナルに `n3_route_unrouted`（`n3_route=absent`）が出ます。ヘッダを付ける mindwire（#463）が deploy されるまでは、すべての naysayer の要求がこの形になります。C-5 で `answered_by='codex'` の行が出ない場合は、まずこれを疑います。
  - 伏せ字をした行は `redactions` に `kind:count,…`（数だけ）が入ります。

  ```sh
  sqlite3 data/costs.db "SELECT id, backend, answered_by, redactions FROM request_costs ORDER BY id DESC LIMIT 5;"
  journalctl -u lexora --since '-15 min' | grep -E 'n3_route_unrouted|ingress_redaction_failed'
  ```

### ロールバック

- C-2′ で 1 分以内に `pass` が取れない、C-3 で `mode: codex` かつ `codex_disabled_reason: null` にならない、または C-5 で `failure_hold` を繰り返す場合は、#463/#464 は merge しません。
- Lexora を直前の deploy（`dbe1170`、shadow）に戻します。戻せば旧 mindwire と組み合わせて元どおり動きます。

### 実際に枠が切れるまで確かめられないこと

- 枠切れのときの stderr の文言と、リセット時刻の書式です。文言が想定と違っても、その失敗は `CodexFailed` として Gemini に落ち、通知は `reason=codex_failed` で届きます（上の 6 の 4）。そうなったら、ログの snippet をもとに `_QUOTA_RE` と `parse_reset_at` を直す別スレッドを切ります。

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

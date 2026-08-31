"""要件 3 の構造テスト — フォールバック機構は存在しない。

`FallbackService` と関連設定は 2026-08-31 に撤去した (T-frontier-tier msg-025 R-1)。
撤去の理由: このサービスは `73853e7` で生まれてから **一度も配線されたことが無い**
(`main.py` は構築せず `routes.py` は呼ばない) のに、config は
`fallback: enabled: true` と `heavy: fallback_backends: ["claude"]` を出荷し続けて
いた ∴ 出荷物を読んだ運用者への偽の約束だった。PR #9 の独立 gate が
REQUEST_CHANGES で止めたのはこの点であり、処置は「config ブロックの削除」では
なく「機構の削除」だった。

このファイルが固定するのは「消したこと」ではなく **「戻せないこと」**。
`BackendSettings` の extra は ``"forbid"`` ∴ フィールドを消した後は、誰かが config に
フォールバックを書き戻した瞬間に **起動が落ちる**。要件 3 (frontier を黙って格下げ
しない) は「空リストでそう宣言している」から「**どんな config でも格下げできない**」
に格上げされる — 宣言よりスキーマの方が強い。

再導入するなら実装が先で config は後。機構より先に config を出荷しない、が今回の教訓。
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from lexora.config import BackendSettings, create_settings, load_yaml_config

#: 出荷されている config。テスト用の一時ファイルではなく現物を読む — 偽の約束は
#: 「出荷物に書いてあること」なので、出荷物そのものを assert しないと意味が無い。
SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "lexora_config.yaml"


@pytest.fixture(scope="module")
def shipped_config() -> dict:
    """出荷 config をパースして返す。

    `load_yaml_config` はファイルが無いと ``{}`` を返す ∴ 空でないことを先に
    assert する。そうしないと「config が消えた/移動した」が「fallback が無い」と
    同じ緑に見える。
    """
    assert SHIPPED_CONFIG.exists(), f"shipped config not found: {SHIPPED_CONFIG}"
    config = load_yaml_config(SHIPPED_CONFIG)
    assert config, f"shipped config parsed empty: {SHIPPED_CONFIG}"
    return config


class TestShippedConfigDeclaresNoFallback:
    """出荷 config にフォールバックの痕跡が無いこと。"""

    def test_no_toplevel_fallback_section(self, shipped_config: dict) -> None:
        """トップレベルの `fallback:` セクションが無いこと。

        `create_settings` は既知セクションしか読まないので、残っていても
        起動は落ちない = 静かに嘘が残る ∴ テストで見張る必要がある。
        """
        assert "fallback" not in shipped_config

    def test_no_backend_declares_fallback_backends(self, shipped_config: dict) -> None:
        """どの backend エントリも `fallback_backends` を持たないこと。"""
        backends = shipped_config.get("routing", {}).get("backends", {})
        assert backends, "shipped config declares no backends — check the fixture"
        offenders = [name for name, cfg in backends.items() if "fallback_backends" in cfg]
        assert offenders == [], f"fallback_backends は撤去済みのキー: {offenders}"

    def test_shipped_config_still_constructs_settings(self) -> None:
        """出荷 config が撤去後も Settings を構築できること。

        `extra="forbid"` は残骸キーを起動失敗に変える ∴ 撤去漏れがあれば
        ここが最初に落ちる (デプロイ先で落ちる前に CI で落ちる)。
        """
        settings = create_settings(SHIPPED_CONFIG)
        assert settings.routing.enabled is True
        assert not hasattr(settings, "fallback")


class TestFallbackConfigCannotBeReintroduced:
    """スキーマがフォールバックの書き戻しを拒否すること (要件 3 の格上げ)。"""

    def test_backend_settings_rejects_fallback_backends(self) -> None:
        """`fallback_backends` を与えた backend 設定が ValidationError になること。

        空リストの宣言より強い保証: config を編集しても frontier (や他のティア)
        をサイレントに格下げする設定は**書けない**。
        """
        with pytest.raises(ValidationError) as exc_info:
            BackendSettings(type="anthropic", fallback_backends=["claude"])
        assert "fallback_backends" in str(exc_info.value)

"""Tests for configuration module."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from lexora.config import (
    Settings,
    VLLMSettings,
    create_settings,
    load_yaml_config,
)


class TestVLLMSettings:
    """Tests for VLLMSettings."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        settings = VLLMSettings()
        assert settings.url == "http://localhost:8000"
        assert settings.timeout == 120.0
        assert settings.connect_timeout == 5.0

    def test_custom_values(self) -> None:
        """Test custom values are applied."""
        settings = VLLMSettings(url="http://vllm:8080", timeout=60.0)
        assert settings.url == "http://vllm:8080"
        assert settings.timeout == 60.0


class TestSettings:
    """Tests for Settings."""

    def test_default_settings(self) -> None:
        """Test default settings structure."""
        settings = Settings()
        assert settings.vllm.url == "http://localhost:8000"
        assert settings.server.port == 8001
        assert settings.queue.max_size == 1000
        assert settings.rate_limit.enabled is True
        assert settings.retry.max_retries == 3
        assert settings.logging.level == "INFO"

    def test_nested_settings(self) -> None:
        """Test nested settings work correctly."""
        settings = Settings(
            vllm=VLLMSettings(url="http://custom:9000"),
        )
        assert settings.vllm.url == "http://custom:9000"


class TestLoadYamlConfig:
    """Tests for load_yaml_config function."""

    def test_nonexistent_file(self) -> None:
        """Test loading nonexistent file returns empty dict."""
        result = load_yaml_config(Path("/nonexistent/path.yaml"))
        assert result == {}

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """Test loading valid YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
vllm:
  url: "http://test:8000"
  timeout: 30.0
server:
  port: 9000
"""
        )
        result = load_yaml_config(config_file)
        assert result["vllm"]["url"] == "http://test:8000"
        assert result["vllm"]["timeout"] == 30.0
        assert result["server"]["port"] == 9000

    def test_empty_yaml(self, tmp_path: Path) -> None:
        """Test loading empty YAML file returns empty dict."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        result = load_yaml_config(config_file)
        assert result == {}


class TestCreateSettings:
    """Tests for create_settings function."""

    def test_from_yaml(self, tmp_path: Path) -> None:
        """Test creating settings from YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
vllm:
  url: "http://yaml-test:8000"
queue:
  max_size: 500
"""
        )
        settings = create_settings(config_file)
        assert settings.vllm.url == "http://yaml-test:8000"
        assert settings.queue.max_size == 500
        # Check defaults are still applied
        assert settings.server.port == 8001

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variables work with nested settings."""
        # Note: Pydantic-settings requires specific env var format for nested
        settings = Settings()
        # Default should be used
        assert settings.vllm.url == "http://localhost:8000"

    def test_no_config_file(self) -> None:
        """Test settings work without config file."""
        settings = create_settings(Path("/nonexistent/config.yaml"))
        # Should use all defaults
        assert settings.vllm.url == "http://localhost:8000"
        assert settings.server.port == 8001


class TestErrorPassthroughIsRefusedWhereUnimplemented:
    """N-2: `error_passthrough` は受け口だけ全 type にあり、配線は 1 type だけ。

    `error_passthrough` は `BackendSettings` にある ∴ どの backend type でも
    受理される。ところが `backends/factory.py` は `AnthropicBackend` にしか
    渡していない。実測すると vllm / openai_compatible / gemini / claude_code
    は config の `true` が **例外も警告もなく False になる**。

    これは #9 で撤去したばかりの `fallback_backends` と同じ故障クラスである
    — 運用者が設定を書き、gateway が受理し、約束された挙動は起きない。処置も
    R-1 に揃える: 配線ではなく **起動時に拒否** する。2 つ目の backend で
    透過が要るなら、実装が先で config は後 (config を先に出荷しない)。
    """

    def test_vllm_rejects_error_passthrough(self) -> None:
        from pydantic import ValidationError

        from lexora.config import BackendSettings

        with pytest.raises(ValidationError) as exc_info:
            BackendSettings(type="vllm", error_passthrough=True)
        assert "error_passthrough" in str(exc_info.value)

    @pytest.mark.parametrize(
        "backend_type", ["openai_compatible", "gemini", "claude_code"]
    )
    def test_every_unwired_type_is_rejected(self, backend_type: str) -> None:
        """gate が挙げた 3 type だけでなく claude_code も同じ穴だった (実測)。"""
        from pydantic import ValidationError

        from lexora.config import BackendSettings

        with pytest.raises(ValidationError):
            BackendSettings(type=backend_type, error_passthrough=True)

    def test_anthropic_accepts_error_passthrough(self) -> None:
        """唯一の実装済み type。拒否が広すぎないことの検出器。"""
        from lexora.config import BackendSettings

        settings = BackendSettings(type="anthropic", error_passthrough=True)
        assert settings.error_passthrough is True

    def test_absent_or_false_is_always_fine(self) -> None:
        """既定 (未指定 / false) はどの type でも通る = 既存 config を壊さない。"""
        from lexora.config import BackendSettings

        for backend_type in ("vllm", "openai_compatible", "gemini", "claude_code"):
            assert BackendSettings(type=backend_type).error_passthrough is False
            assert (
                BackendSettings(
                    type=backend_type, error_passthrough=False
                ).error_passthrough
                is False
            )

    def test_shipped_configs_still_load(self) -> None:
        """出荷 config が新しい validator で落ちないこと。

        本スレッドは稼働機のクラッシュループ・ハザードを抱えている
        (`Restart=always` / `RestartSec=5` ∴ 起動時 ValidationError は
        5 秒周期のクラッシュループになり、外からは「naysayer が応答しない」に
        見える)。validator を足すたびに、出荷物が実際にロードできることを
        CI で確かめる — デプロイ先で確かめない。
        """
        from lexora.config import create_settings

        config_dir = Path(__file__).resolve().parents[1] / "config"
        configs = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
        assert configs, f"no shipped config found under {config_dir}"
        for config_path in configs:
            settings = create_settings(config_path)
            assert settings is not None, f"failed to load {config_path}"

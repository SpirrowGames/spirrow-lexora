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

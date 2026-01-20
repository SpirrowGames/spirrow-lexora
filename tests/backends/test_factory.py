"""Tests for backend factory."""

import os
import pytest
from unittest.mock import patch

from lexora.backends.factory import create_backend, resolve_api_key
from lexora.backends.vllm import VLLMBackend
from lexora.backends.openai_compatible import OpenAICompatibleBackend
from lexora.config import BackendSettings


class TestResolveApiKey:
    """Tests for API key resolution."""

    def test_direct_api_key(self):
        """Test resolving API key from direct setting."""
        settings = BackendSettings(api_key="sk-direct-key")
        result = resolve_api_key(settings)
        assert result == "sk-direct-key"

    def test_env_var_api_key(self):
        """Test resolving API key from environment variable."""
        settings = BackendSettings(api_key_env="TEST_API_KEY")

        with patch.dict(os.environ, {"TEST_API_KEY": "sk-env-key"}):
            result = resolve_api_key(settings)
            assert result == "sk-env-key"

    def test_direct_key_takes_precedence(self):
        """Test that direct API key takes precedence over env var."""
        settings = BackendSettings(api_key="sk-direct", api_key_env="TEST_API_KEY")

        with patch.dict(os.environ, {"TEST_API_KEY": "sk-env"}):
            result = resolve_api_key(settings)
            assert result == "sk-direct"

    def test_missing_env_var(self):
        """Test handling missing environment variable."""
        settings = BackendSettings(api_key_env="NONEXISTENT_KEY")

        # Make sure the env var doesn't exist
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_api_key(settings)
            assert result is None

    def test_no_api_key_configured(self):
        """Test when no API key is configured."""
        settings = BackendSettings()
        result = resolve_api_key(settings)
        assert result is None


class TestCreateBackend:
    """Tests for backend creation."""

    def test_create_vllm_backend(self):
        """Test creating vLLM backend."""
        settings = BackendSettings(
            type="vllm",
            url="http://localhost:8000",
            timeout=60.0,
            connect_timeout=3.0,
        )

        backend = create_backend("test_vllm", settings)

        assert isinstance(backend, VLLMBackend)
        assert backend.base_url == "http://localhost:8000"
        assert backend.name == "test_vllm"

    def test_create_openai_compatible_backend(self):
        """Test creating OpenAI-compatible backend."""
        settings = BackendSettings(
            type="openai_compatible",
            url="https://api.openai.com",
            api_key="sk-test-key",
            timeout=30.0,
            model_mapping={"gpt-4": "gpt-4-turbo"},
        )

        backend = create_backend("openai_prod", settings)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == "https://api.openai.com"
        assert backend.api_key == "sk-test-key"
        assert backend.model_mapping == {"gpt-4": "gpt-4-turbo"}
        assert backend.name == "openai_prod"

    def test_create_openai_compatible_with_env_key(self):
        """Test creating OpenAI-compatible backend with env var API key."""
        settings = BackendSettings(
            type="openai_compatible",
            url="https://api.openai.com",
            api_key_env="OPENAI_API_KEY",
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}):
            backend = create_backend("openai_env", settings)

            assert isinstance(backend, OpenAICompatibleBackend)
            assert backend.api_key == "sk-from-env"

    def test_create_backend_default_type_is_vllm(self):
        """Test that default backend type is vllm."""
        settings = BackendSettings(url="http://localhost:8000")

        backend = create_backend("default", settings)

        assert isinstance(backend, VLLMBackend)

    def test_create_backend_unknown_type_rejected_by_pydantic(self):
        """Test that unknown backend type is rejected by Pydantic validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            BackendSettings(type="unknown")  # type: ignore

        assert "type" in str(exc_info.value)
        assert "literal_error" in str(exc_info.value)

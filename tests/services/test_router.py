"""Tests for backend router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lexora.backends.base import BackendError
from lexora.config import BackendSettings, RoutingSettings, VLLMSettings
from lexora.services.router import BackendRouter


class TestBackendRouterSingleMode:
    """Tests for BackendRouter in single backend mode."""

    def test_single_backend_mode(self) -> None:
        """Test router initializes in single backend mode when routing disabled."""
        routing_settings = RoutingSettings(enabled=False)
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        assert not router.routing_enabled
        assert "default" in router.backends
        assert router.default_backend is router.backends["default"]

    def test_get_backend_for_model_returns_default(self) -> None:
        """Test that any model returns the default backend."""
        routing_settings = RoutingSettings(enabled=False)
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        backend = router.get_backend_for_model("any-model")
        assert backend is router.default_backend


class TestBackendRouterMultiMode:
    """Tests for BackendRouter in multi-backend mode."""

    def test_multi_backend_mode(self) -> None:
        """Test router initializes in multi-backend mode when routing enabled."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="backend1",
            backends={
                "backend1": BackendSettings(
                    url="http://localhost:8001",
                    models=["model-a", "model-b"],
                ),
                "backend2": BackendSettings(
                    url="http://localhost:8002",
                    models=["model-c"],
                ),
            },
        )
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        assert router.routing_enabled
        assert len(router.backends) == 2
        assert "backend1" in router.backends
        assert "backend2" in router.backends

    def test_get_backend_for_mapped_model(self) -> None:
        """Test that mapped models return correct backend."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="backend1",
            backends={
                "backend1": BackendSettings(
                    url="http://localhost:8001",
                    models=["model-a"],
                ),
                "backend2": BackendSettings(
                    url="http://localhost:8002",
                    models=["model-b"],
                ),
            },
        )
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        backend1 = router.get_backend_for_model("model-a")
        backend2 = router.get_backend_for_model("model-b")

        assert backend1 is router.backends["backend1"]
        assert backend2 is router.backends["backend2"]

    def test_get_backend_for_unmapped_model_returns_default(self) -> None:
        """Test that unmapped models return default backend."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="backend1",
            backends={
                "backend1": BackendSettings(
                    url="http://localhost:8001",
                    models=["model-a"],
                ),
                "backend2": BackendSettings(
                    url="http://localhost:8002",
                    models=["model-b"],
                ),
            },
        )
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        backend = router.get_backend_for_model("unknown-model")
        assert backend is router.backends["backend1"]

    def test_get_backend_for_invalid_default_raises_error(self) -> None:
        """Test that invalid default backend raises error."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="nonexistent",
            backends={
                "backend1": BackendSettings(
                    url="http://localhost:8001",
                    models=["model-a"],
                ),
            },
        )
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        with pytest.raises(BackendError):
            router.get_backend_for_model("unknown-model")


class TestBackendRouterHealthCheck:
    """Tests for BackendRouter health check."""

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self) -> None:
        """Test health check when all backends are healthy."""
        routing_settings = RoutingSettings(enabled=False)
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        # Mock the backend's health_check
        router.backends["default"].health_check = AsyncMock(return_value=True)

        health = await router.health_check()

        assert health == {"default": True}

    @pytest.mark.asyncio
    async def test_health_check_some_unhealthy(self) -> None:
        """Test health check when some backends are unhealthy."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="backend1",
            backends={
                "backend1": BackendSettings(
                    url="http://localhost:8001",
                    models=["model-a"],
                ),
                "backend2": BackendSettings(
                    url="http://localhost:8002",
                    models=["model-b"],
                ),
            },
        )
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        # Mock the backends' health_check
        router.backends["backend1"].health_check = AsyncMock(return_value=True)
        router.backends["backend2"].health_check = AsyncMock(return_value=False)

        health = await router.health_check()

        assert health["backend1"] is True
        assert health["backend2"] is False


class TestBackendRouterListModels:
    """Tests for BackendRouter list models."""

    @pytest.mark.asyncio
    async def test_list_all_models(self) -> None:
        """Test listing models from all backends."""
        routing_settings = RoutingSettings(enabled=False)
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        # Mock the backend's list_models
        router.backends["default"].list_models = AsyncMock(
            return_value={
                "object": "list",
                "data": [
                    {"id": "model-a", "object": "model"},
                    {"id": "model-b", "object": "model"},
                ],
            }
        )

        models = await router.list_all_models()

        assert models["object"] == "list"
        assert len(models["data"]) == 2
        # Check that backend name is added
        assert all(m.get("backend") == "default" for m in models["data"])

    @pytest.mark.asyncio
    async def test_list_all_models_multi_backend(self) -> None:
        """Test listing models from multiple backends."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="backend1",
            backends={
                "backend1": BackendSettings(
                    url="http://localhost:8001",
                    models=["model-a"],
                ),
                "backend2": BackendSettings(
                    url="http://localhost:8002",
                    models=["model-b"],
                ),
            },
        )
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        # Mock the backends' list_models
        router.backends["backend1"].list_models = AsyncMock(
            return_value={
                "object": "list",
                "data": [{"id": "model-a", "object": "model"}],
            }
        )
        router.backends["backend2"].list_models = AsyncMock(
            return_value={
                "object": "list",
                "data": [{"id": "model-b", "object": "model"}],
            }
        )

        models = await router.list_all_models()

        assert models["object"] == "list"
        assert len(models["data"]) == 2

        model_ids = {m["id"] for m in models["data"]}
        assert model_ids == {"model-a", "model-b"}

    @pytest.mark.asyncio
    async def test_list_all_models_handles_backend_error(self) -> None:
        """Test that backend errors are handled gracefully."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="backend1",
            backends={
                "backend1": BackendSettings(
                    url="http://localhost:8001",
                    models=["model-a"],
                ),
                "backend2": BackendSettings(
                    url="http://localhost:8002",
                    models=["model-b"],
                ),
            },
        )
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        # Mock backend1 to succeed, backend2 to fail
        router.backends["backend1"].list_models = AsyncMock(
            return_value={
                "object": "list",
                "data": [{"id": "model-a", "object": "model"}],
            }
        )
        router.backends["backend2"].list_models = AsyncMock(
            side_effect=BackendError("Connection failed")
        )

        models = await router.list_all_models()

        # Should still return models from backend1
        assert len(models["data"]) == 1
        assert models["data"][0]["id"] == "model-a"


class TestBackendRouterClose:
    """Tests for BackendRouter close."""

    @pytest.mark.asyncio
    async def test_close_all_backends(self) -> None:
        """Test that close() closes all backends."""
        routing_settings = RoutingSettings(
            enabled=True,
            default_backend="backend1",
            backends={
                "backend1": BackendSettings(
                    url="http://localhost:8001",
                    models=["model-a"],
                ),
                "backend2": BackendSettings(
                    url="http://localhost:8002",
                    models=["model-b"],
                ),
            },
        )
        vllm_settings = VLLMSettings(url="http://localhost:8000")

        router = BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

        # Mock the backends' close
        router.backends["backend1"].close = AsyncMock()
        router.backends["backend2"].close = AsyncMock()

        await router.close()

        router.backends["backend1"].close.assert_called_once()
        router.backends["backend2"].close.assert_called_once()

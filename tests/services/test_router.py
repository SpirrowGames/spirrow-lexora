"""Tests for backend router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lexora.backends.base import BackendError
from lexora.config import BackendSettings, RoutingSettings, TierSettings, VLLMSettings
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


class TestBackendRouterTierRouting:
    """Tests for BackendRouter tier-based routing."""

    def _make_router(
        self,
        tiers: dict[str, TierSettings] | None = None,
    ) -> BackendRouter:
        """Create a router with tier configuration."""
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
            tiers=tiers or {},
        )
        vllm_settings = VLLMSettings(url="http://localhost:8000")
        return BackendRouter(
            routing_settings=routing_settings,
            vllm_settings=vllm_settings,
        )

    def test_tier_routes_to_correct_backend(self) -> None:
        """Test that tier names route to the configured backend."""
        router = self._make_router(
            tiers={
                "light": TierSettings(backend="backend1"),
                "heavy": TierSettings(backend="backend2"),
            }
        )

        assert router.get_backend_for_model("light") is router.backends["backend1"]
        assert router.get_backend_for_model("heavy") is router.backends["backend2"]

    def test_tier_takes_precedence_over_model(self) -> None:
        """Test that tier mapping takes precedence when name collides with model."""
        router = self._make_router(
            tiers={
                "model-a": TierSettings(backend="backend2"),
            }
        )

        # "model-a" is registered as both a model (→backend1) and a tier (→backend2)
        # Tier should win
        assert router.get_backend_for_model("model-a") is router.backends["backend2"]

    def test_unknown_name_falls_to_default(self) -> None:
        """Test that names matching neither tier nor model fall to default."""
        router = self._make_router(
            tiers={
                "light": TierSettings(backend="backend1"),
            }
        )

        backend = router.get_backend_for_model("unknown")
        assert backend is router.backends["backend1"]  # default

    def test_tier_with_invalid_backend_is_skipped(self) -> None:
        """Test that tiers referencing non-existent backends are not registered."""
        router = self._make_router(
            tiers={
                "broken": TierSettings(backend="nonexistent"),
                "valid": TierSettings(backend="backend2"),
            }
        )

        # "broken" tier should not be registered
        assert router.get_backend_for_model("broken") is router.backends["backend1"]  # falls to default
        # "valid" tier should work
        assert router.get_backend_for_model("valid") is router.backends["backend2"]

    def test_get_backend_name_for_model_resolves_tier(self) -> None:
        """Test that get_backend_name_for_model also resolves tiers."""
        router = self._make_router(
            tiers={
                "light": TierSettings(backend="backend2"),
            }
        )

        assert router.get_backend_name_for_model("light") == "backend2"


class TestBackendRouterHealthCheckOptOut:
    """`health_check: false` — a backend that serves traffic but is not probed.

    Added 2026-08-11. Two of the configured backends (gemini, anthropic) probe
    by sending a real inference request to a remote API, so every GET /health
    billed a call and inherited that provider's latency; a third
    (openai_compatible, unauthenticated) intermittently stalled the endpoint
    for 20-40s. Skipping has to stay distinguishable from "unhealthy" and from
    "not configured", which is what these tests pin.
    """

    @staticmethod
    def _router(**flags: bool) -> BackendRouter:
        backends = {
            name: BackendSettings(
                type="vllm",
                url="http://localhost:8000",
                models=[f"model-{name}"],
                health_check=checked,
            )
            for name, checked in flags.items()
        }
        return BackendRouter(
            routing_settings=RoutingSettings(enabled=True, backends=backends),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )

    @pytest.mark.asyncio
    async def test_skipped_backend_reports_none_not_false(self) -> None:
        """False would mean "we asked and it is down"."""
        router = self._router(probed=True, quiet=False)
        router.backends["probed"].health_check = AsyncMock(return_value=True)
        router.backends["quiet"].health_check = AsyncMock(return_value=True)

        health = await router.health_check()

        assert health == {"probed": True, "quiet": None}

    @pytest.mark.asyncio
    async def test_a_skipped_backend_is_never_probed(self) -> None:
        """The whole point is not paying for the call."""
        router = self._router(quiet=False)
        probe = AsyncMock(return_value=True)
        router.backends["quiet"].health_check = probe

        await router.health_check()

        probe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_backends_default_to_being_checked(self) -> None:
        """Opting out must be explicit; silence means "probe me"."""
        router = self._router(a=True)
        router.backends["a"].health_check = AsyncMock(return_value=True)

        assert await router.health_check() == {"a": True}

    @pytest.mark.asyncio
    async def test_a_probe_that_raises_is_not_confused_with_a_skip(self) -> None:
        """gather(return_exceptions=True) must not leak an exception object
        into a field typed as a health verdict."""
        router = self._router(boom=True)
        router.backends["boom"].health_check = AsyncMock(
            side_effect=RuntimeError("probe blew up")
        )

        assert await router.health_check() == {"boom": False}

    @pytest.mark.asyncio
    async def test_probes_run_concurrently(self) -> None:
        """Serially, one slow remote set the latency of the whole endpoint."""
        import asyncio

        async def slow() -> bool:
            await asyncio.sleep(0.15)
            return True

        router = self._router(a=True, b=True, c=True)
        for name in ("a", "b", "c"):
            router.backends[name].health_check = slow

        start = asyncio.get_event_loop().time()
        await router.health_check()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.3, f"probes look serial: {elapsed:.2f}s for 3x0.15s"


class TestBackendRouterIsTier:
    """T-frontier-tier D-6b: is_tier() disambiguates aliases from model IDs."""

    def _router_with_tiers(self) -> BackendRouter:
        return BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(url="http://localhost:1", models=["m1"]),
                    "b2": BackendSettings(url="http://localhost:2", models=["m2"]),
                },
                tiers={
                    "frontier": TierSettings(backend="b2", model="m2"),
                },
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )

    def test_is_tier_true_for_registered(self) -> None:
        assert self._router_with_tiers().is_tier("frontier") is True

    def test_is_tier_false_for_concrete_model(self) -> None:
        assert self._router_with_tiers().is_tier("m1") is False

    def test_is_tier_false_for_unknown(self) -> None:
        assert self._router_with_tiers().is_tier("no-such-thing") is False


class TestBackendRouterListModelsTierEntries:
    """T-frontier-tier D-5: /v1/models exposes tier aliases with resolved model."""

    @pytest.mark.asyncio
    async def test_tier_alias_appears_with_resolved_model(self) -> None:
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(url="http://localhost:1", models=["real-model"]),
                },
                tiers={
                    "frontier": TierSettings(backend="b1", model="real-model"),
                },
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        router.backends["b1"].list_models = AsyncMock(
            return_value={"object": "list", "data": [{"id": "real-model", "object": "model"}]}
        )

        listing = await router.list_all_models()
        tier_entries = [m for m in listing["data"] if m.get("type") == "tier"]
        assert len(tier_entries) == 1
        entry = tier_entries[0]
        assert entry["id"] == "frontier"
        assert entry["resolved_model"] == "real-model"
        assert entry["backend"] == "b1"

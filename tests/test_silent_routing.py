"""T-silent-routing: routing decisions must not be made silently.

The tests in this file cover the requirements the T-silent-routing spec
introduces (msg-078 through msg-083):

* R-5 — a YAML file with duplicate mapping keys is rejected at load time.
  PyYAML's SafeLoader silently keeps the last value; the strict loader
  this branch installs raises ``DuplicateYamlKeyError`` instead, so a
  config where the file text disagrees with the loaded dict does not
  reach the application.
* R-1b — a config where a tier name is also declared as a backend model
  name is rejected at ``RoutingSettings`` construction. Tier lookup wins
  over by-name at request time, so the model declaration would be
  unreachable-but-present, which is the class of lie this branch is
  against.
* R-1a — a model name declared by two or more backends emits a boot-time
  WARNING and is refused with HTTP 404 at request time, rather than
  silently resolving to whichever backend happened to be the last writer
  into ``_model_to_backend``.
* R-1a-5 — the ambiguity check applies only to the requested name. A
  tier alias whose concrete model happens to be declared by three
  backends must still route: the request field is the tier name, not the
  concrete model.
* R-2 — a model name that matches no tier and no declared backend model
  is refused with HTTP 404 rather than silently falling through to
  ``default_backend``. The refusal is logged at WARNING (not DEBUG) so
  the operational surface for fall-through requests survives the
  ``INFO``-level log filter that runs in production.
* W-3 — ``/v1/models`` never enumerates a name the router will 404.
  Ambiguous names are filtered out of the listing so the advertised set
  is a subset of the routable set.
* 404 body shape — clients see the OpenAI standard ``model_not_found``
  code; the anthropic-shaped ``/v1/messages`` endpoint sees the
  Anthropic ``{"type": "error", ...}`` envelope. The distinction between
  "unknown" and "ambiguous" is a server-side concern (log event names)
  and does not appear in the API code.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from lexora.backends.base import ModelNotFoundError
from lexora.config import (
    BackendSettings,
    DuplicateYamlKeyError,
    RoutingSettings,
    TierSettings,
    VLLMSettings,
    create_settings,
    load_yaml_config,
)
from lexora.main import create_app
from lexora.services.router import BackendRouter


# --------------------------------------------------------------------------
# R-5 — strict YAML loader
# --------------------------------------------------------------------------


class TestStrictYamlLoader:
    """R-5: reject duplicate keys at parse time.

    The bug this catches is subtle by construction: PyYAML's SafeLoader
    returns a Python dict that Python then sees as single-keyed and
    correct. Neither Pydantic nor any downstream validator can see that
    the source file disagreed with the loaded value — the disagreement
    only exists in the text, and the text is gone by the time anyone
    else looks. That is exactly the class this branch is against: a
    config the operator wrote and the application never saw.
    """

    def test_duplicate_top_level_key_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text("a: 1\na: 2\n")
        with pytest.raises(DuplicateYamlKeyError) as exc_info:
            load_yaml_config(path)
        # The error must name the key and both lines so an operator can
        # find the collision without re-parsing the file.
        message = str(exc_info.value)
        assert "'a'" in message
        assert "line 1" in message
        assert "line 2" in message

    def test_duplicate_nested_key_rejected(self, tmp_path: Path) -> None:
        """Nesting is not a loophole: R-5 is applied to every mapping in the file.

        A duplicate at ``routing.tiers.frontier`` was one of Einstein's
        cited failure modes (msg-080 V-1) — the last-writer-wins would
        happen at parse time and no app-level tier validator could see
        it. This test is the general form of that concern.
        """
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "routing:\n"
            "  tiers:\n"
            "    frontier:\n"
            "      backend: gemini\n"
            "    frontier:\n"
            "      backend: heavy\n"
        )
        with pytest.raises(DuplicateYamlKeyError) as exc_info:
            load_yaml_config(path)
        assert "'frontier'" in str(exc_info.value)

    def test_non_duplicate_yaml_still_loads(self, tmp_path: Path) -> None:
        """R-5 is scope-preserving: legal YAML keeps loading.

        A regression here would be worse than not having R-5 at all:
        every config in every environment would start refusing to load.
        """
        path = tmp_path / "cfg.yaml"
        path.write_text("a: 1\nb: 2\nc:\n  d: 3\n  e: 4\n")
        result = load_yaml_config(path)
        assert result == {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}


# --------------------------------------------------------------------------
# R-1b — tier / model name collision
# --------------------------------------------------------------------------


class TestTierBackendCollision:
    """R-1b: a tier name declared as a backend model must not silently disappear.

    ``get_backend_for_model`` looks up tiers first, so if a tier
    ``frontier`` is also declared as a model on some backend, the model
    declaration is present but unreachable. Rather than accept the
    unreachable declaration, ``RoutingSettings`` refuses to construct.
    """

    def test_collision_at_boot_is_rejected(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            RoutingSettings(
                enabled=True,
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1",
                        # `frontier` also appears as a tier below
                        models=[{"name": "frontier"}],
                    ),
                },
                tiers={
                    "frontier": TierSettings(backend="b1", model="frontier"),
                },
            )
        # The message must name both sides so the operator can fix it.
        message = str(exc_info.value)
        assert "frontier" in message
        assert "b1" in message

    def test_no_collision_still_constructs(self) -> None:
        """R-1b applies narrowly; a config with disjoint tier and model
        names must keep constructing without incident."""
        settings = RoutingSettings(
            enabled=True,
            backends={
                "b1": BackendSettings(
                    url="http://localhost:1", models=[{"name": "real-model"}]
                ),
            },
            tiers={
                "light": TierSettings(backend="b1", model="real-model"),
            },
        )
        assert "light" in settings.tiers
        assert settings.backends["b1"].models[0].name == "real-model"


# --------------------------------------------------------------------------
# R-1a — model-name ambiguity: boot WARNING + request-time 404
# --------------------------------------------------------------------------


def _shared_model_router() -> BackendRouter:
    """Router configured so ``dup-model`` is declared by three backends.

    Kept as a helper because R-1a is the requirement most easily tested
    by symmetry: the same fixture drives the boot-time behavior, the
    request-time refusal, and (with the same name in the tier position)
    the R-1a-5 invariant.
    """
    return BackendRouter(
        routing_settings=RoutingSettings(
            enabled=True,
            default_backend="b1",
            backends={
                "b1": BackendSettings(
                    url="http://localhost:1", models=[{"name": "dup-model"}]
                ),
                "b2": BackendSettings(
                    url="http://localhost:2", models=[{"name": "dup-model"}]
                ),
                "b3": BackendSettings(
                    url="http://localhost:3", models=[{"name": "dup-model"}]
                ),
            },
            tiers={
                "light": TierSettings(backend="b1", model="dup-model"),
                "medium": TierSettings(backend="b2", model="dup-model"),
                "heavy": TierSettings(backend="b3", model="dup-model"),
            },
        ),
        vllm_settings=VLLMSettings(url="http://localhost:8000"),
    )


class TestModelAmbiguity:
    """R-1a: same model name declared by multiple backends."""

    def test_boot_emits_warning_with_all_backends(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The WARNING must list every colliding backend, not just the winner."""
        with caplog.at_level(logging.WARNING):
            _shared_model_router()
        matching = [
            r for r in caplog.records if "model_declaration_ambiguous" in r.message
        ]
        assert len(matching) == 1, (
            "Expected exactly one ambiguity WARNING for 'dup-model' "
            "(three backends collided, one WARNING). Got: "
            f"{[r.message for r in matching]}"
        )
        text = matching[0].message
        for backend in ("b1", "b2", "b3"):
            assert backend in text, (
                f"WARNING must name every colliding backend so the "
                f"operator does not have to grep INFO lines; missing {backend}"
            )

    def test_ambiguous_by_name_request_refused_with_404_reason(self) -> None:
        """Raw-name request for the ambiguous model raises ModelNotFoundError."""
        router = _shared_model_router()
        with pytest.raises(ModelNotFoundError) as exc_info:
            router.get_backend_for_model("dup-model")
        assert exc_info.value.reason == "ambiguous"
        assert exc_info.value.model_name == "dup-model"

    def test_ambiguity_logged_at_warning_on_refusal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The refusal itself must be at WARNING so operators see it in
        production (default log level is INFO, so DEBUG would vanish)."""
        router = _shared_model_router()
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ModelNotFoundError):
                router.get_backend_for_model("dup-model")
        matching = [
            r for r in caplog.records if "model_ambiguous_refused" in r.message
        ]
        assert matching, (
            "Every ambiguous-name refusal must log 'model_ambiguous_refused' "
            "at WARNING so the request appears in production logs. Got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_get_backend_name_for_ambiguous_also_raises(self) -> None:
        """The stats-collector path must not silently attribute an
        ambiguous request to the last-writer-wins backend."""
        router = _shared_model_router()
        with pytest.raises(ModelNotFoundError):
            router.get_backend_name_for_model("dup-model")


# --------------------------------------------------------------------------
# R-1a-5 — the ambiguity check is on the REQUESTED name, not the resolved one
# --------------------------------------------------------------------------


class TestTierRoutingSurvivesAmbiguousConcreteModel:
    """R-1a-5: tier routing must not be collateral damage of R-1a.

    The failure this pins is that a naive R-1a implementation would take
    the resolved model name after tier lookup and apply the ambiguity
    check to it — killing every tier whose concrete model was shared.
    That would make the shipping config's ``light`` / ``medium`` /
    ``heavy`` tiers all 404 because they all resolve to
    ``Qwen3.8-27B``, which is exactly what the routing spec calls the
    "shortest path to tier-routing failure and gate death" (msg-083
    W-1).
    """

    def test_each_tier_still_routes_despite_shared_concrete_model(self) -> None:
        router = _shared_model_router()
        # All three tiers resolve to 'dup-model' and each points at a
        # different backend. The ambiguity of 'dup-model' as a raw name
        # is unrelated to whether the tiers route.
        assert router.get_backend_for_model("light") is router.backends["b1"]
        assert router.get_backend_for_model("medium") is router.backends["b2"]
        assert router.get_backend_for_model("heavy") is router.backends["b3"]

    def test_tier_resolves_to_concrete_model(self) -> None:
        """The resolved model must still be the concrete name, even
        though that name would 404 as a raw request."""
        router = _shared_model_router()
        assert router.resolve_model("light") == "dup-model"
        assert router.resolve_model("heavy") == "dup-model"


# --------------------------------------------------------------------------
# R-2 — unknown model refused with 404 (no fall-through to default_backend)
# --------------------------------------------------------------------------


class TestUnknownModelRefused:
    """R-2: names that match neither tier nor declared model must 404.

    The old behavior fell through to ``default_backend`` with a single
    DEBUG line, which vanished under the INFO-level production filter.
    The point of R-2 is that "silently routed somewhere" is worse than
    "loudly refused" — even (especially) when the somewhere happens to
    be a real backend that accepts the request.
    """

    def _router(self) -> BackendRouter:
        return BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1",
                        models=[{"name": "model-a"}],
                    ),
                },
                tiers={"light": TierSettings(backend="b1", model="model-a")},
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )

    def test_unknown_name_raises_not_fall_through(self) -> None:
        router = self._router()
        with pytest.raises(ModelNotFoundError) as exc_info:
            router.get_backend_for_model("no-such-model")
        assert exc_info.value.reason == "unknown"
        assert exc_info.value.model_name == "no-such-model"

    def test_refusal_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        router = self._router()
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ModelNotFoundError):
                router.get_backend_for_model("no-such-model")
        matching = [
            r for r in caplog.records if "model_unknown_refused" in r.message
        ]
        assert matching, (
            "R-2's contract is 'do not silently fall through' — the "
            "refusal must be logged at WARNING (not DEBUG), because the "
            "shipping log level is INFO. Got records: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_known_tier_and_model_still_route(self) -> None:
        """R-2 must be tight enough to not break the legal cases."""
        router = self._router()
        assert router.get_backend_for_model("light") is router.backends["b1"]
        assert router.get_backend_for_model("model-a") is router.backends["b1"]


# --------------------------------------------------------------------------
# W-3 — /v1/models never enumerates a name the router will 404
# --------------------------------------------------------------------------


class TestListModelsFiltersAmbiguous:
    """W-3: the advertised set is a subset of the routable set.

    Left unfiltered, three vllm backends proxying the same upstream would
    return three identical rows for a name we then 404 on request. That
    is exactly the "advertise an endpoint that structurally 404s"
    failure Bohr called out in W-3.
    """

    @pytest.mark.asyncio
    async def test_ambiguous_name_dropped_from_listing(self) -> None:
        router = _shared_model_router()
        # Simulate three backends proxying the same upstream, each
        # returning an identical row for 'dup-model'.
        for name in ("b1", "b2", "b3"):
            router.backends[name].list_models = AsyncMock(
                return_value={
                    "object": "list",
                    "data": [
                        {
                            "id": "dup-model",
                            "object": "model",
                            "created": 1_700_000_000,
                            "owned_by": "vllm",
                        }
                    ],
                }
            )
        listing = await router.list_all_models()
        ids = [m["id"] for m in listing["data"]]
        assert "dup-model" not in ids, (
            "The router filtered ambiguous 'dup-model' from /v1/models so "
            "the advertised set is a subset of the routable set. Got "
            f"ids: {ids}"
        )
        # Tier aliases must still appear — they route.
        assert {"light", "medium", "heavy"} <= set(ids)

    @pytest.mark.asyncio
    async def test_unambiguous_name_still_listed(self) -> None:
        """The filter is scoped: a name declared by exactly one backend
        keeps its /v1/models row."""
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1", models=[{"name": "solo"}]
                    ),
                },
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        router.backends["b1"].list_models = AsyncMock(
            return_value={
                "object": "list",
                "data": [
                    {
                        "id": "solo",
                        "object": "model",
                        "created": 1,
                        "owned_by": "vllm",
                    }
                ],
            }
        )
        listing = await router.list_all_models()
        ids = {m["id"] for m in listing["data"]}
        assert "solo" in ids

    @pytest.mark.asyncio
    async def test_duplicate_rows_from_shared_upstream_deduplicated(self) -> None:
        """Three backends proxying the same upstream must not produce
        three identical rows for the same non-ambiguous id."""
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1", models=[{"name": "solo"}]
                    ),
                    "b2": BackendSettings(
                        url="http://localhost:1", models=[{"name": "other"}]
                    ),
                },
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        upstream_payload = {
            "object": "list",
            "data": [
                {
                    "id": "shared-extra",
                    "object": "model",
                    "created": 1,
                    "owned_by": "vllm",
                }
            ],
        }
        router.backends["b1"].list_models = AsyncMock(return_value=upstream_payload)
        router.backends["b2"].list_models = AsyncMock(return_value=upstream_payload)
        listing = await router.list_all_models()
        ids = [m["id"] for m in listing["data"]]
        assert ids.count("shared-extra") == 1


# --------------------------------------------------------------------------
# 404 response body shape (OpenAI + Anthropic)
# --------------------------------------------------------------------------


@pytest.fixture
def app_with_shared_model_router(monkeypatch: pytest.MonkeyPatch):
    """FastAPI app whose router 404s 'dup-model' and unknown names.

    Kept as a fixture rather than repeating the boilerplate in each 404-
    shape test: the request-side surface is what these tests measure,
    not the setup.
    """
    from lexora import config as config_module

    def _fake_settings() -> "config_module.Settings":  # type: ignore[name-defined]
        return config_module.Settings(
            routing=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1", models=[{"name": "dup-model"}]
                    ),
                    "b2": BackendSettings(
                        url="http://localhost:2", models=[{"name": "dup-model"}]
                    ),
                },
                tiers={"light": TierSettings(backend="b1", model="dup-model")},
            )
        )

    monkeypatch.setattr(config_module, "create_settings", _fake_settings)
    app = create_app(_fake_settings())
    # The lifespan builds the real router with the fake settings.
    with TestClient(app) as client:
        yield client


class TestNotFoundResponseShape:
    """The API shape for R-1a / R-2 refusals."""

    def test_openai_endpoint_404_shape_for_unknown(
        self, app_with_shared_model_router: TestClient
    ) -> None:
        response = app_with_shared_model_router.post(
            "/v1/chat/completions",
            json={
                "model": "no-such-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 404, response.text
        body = response.json()
        assert body == {
            "error": {
                # The message is human-readable; we assert its structural
                # position, not the exact wording.
                "message": body["error"]["message"],
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        }
        assert "no-such-model" in body["error"]["message"]

    def test_openai_endpoint_404_shape_for_ambiguous(
        self, app_with_shared_model_router: TestClient
    ) -> None:
        response = app_with_shared_model_router.post(
            "/v1/chat/completions",
            json={
                "model": "dup-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 404, response.text
        body = response.json()
        # Ambiguous and unknown share ``code: model_not_found`` on
        # purpose — clients have no useful branch to make between them
        # (msg-082 objection B).
        assert body["error"]["code"] == "model_not_found"
        # The message must tell the operator both offenders.
        assert "b1" in body["error"]["message"]
        assert "b2" in body["error"]["message"]

    def test_anthropic_messages_404_uses_anthropic_shape(
        self, app_with_shared_model_router: TestClient
    ) -> None:
        response = app_with_shared_model_router.post(
            "/v1/messages",
            json={
                "model": "no-such-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 404, response.text
        body = response.json()
        # Anthropic-shaped: the SDK parses this envelope natively; an
        # OpenAI-shaped 404 on /v1/messages would be the one endpoint's
        # response the SDK could not typecheck.
        assert body["type"] == "error"
        assert body["error"]["type"] == "not_found_error"
        assert "no-such-model" in body["error"]["message"]

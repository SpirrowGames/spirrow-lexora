"""Load and validate the shipped ``config/lexora_config.yaml`` (R-4).

Why this file exists (T-silent-routing spec §4 R-4): the routing branch adds
several ways for the config loader to refuse a bad file at boot (strict
YAML loader that rejects duplicate keys, tier/model name collision check,
model-name ambiguity WARNING). Because production runs directly off a
working tree — see ``deploy/`` comments — the moment a bad config lands
under ``spirrow-lexora``'s WorkingDirectory the service enters a restart
loop that shows externally as "not responding", which happens to be the
same symptom as "the naysayer route is down". Catching the fault in CI is
the whole condition on which those boot-time rejections are safe: R-1a /
R-1b / R-5 are gated on R-4 (msg-078 §6, msg-081 §3.3).

This test loads the shipped file as a file (not a fixture), because a
fixture would only measure what we thought to put in it. The failure
modes we care about — a duplicate key added in a rebase, a model
declaration that unexpectedly collides with a tier name, an env override
that would leave a lie behind — all show up textually in
``config/lexora_config.yaml`` and are exactly what R-4 asserts against.

Kept separate from ``test_config.py`` so a diff to the shipped config or
its schema fails a test named after that file, not one named after the
generic config machinery.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from lexora.backends.base import ModelNotFoundError
from lexora.config import create_settings
from lexora.services.router import BackendRouter

SHIPPED_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "lexora_config.yaml"
)


def test_shipped_config_file_exists() -> None:
    """Guard against a rename / move that would leave R-4 silently passing.

    Without this line a rename of the file to somewhere the tests do not
    look would turn ``create_settings(...)`` into a no-op — the loader
    returns ``{}`` for a missing path — and every assertion below would
    pass against the defaults, not the shipped config. The whole point of
    R-4 is that CI reads the same file production will, so a missing file
    is a failure, not an "empty config".
    """
    assert SHIPPED_CONFIG.exists(), (
        f"Shipped config not found at {SHIPPED_CONFIG}. If the file moved, "
        f"update this test to point at the new location — do not delete it."
    )


def test_shipped_config_loads_without_error() -> None:
    """Load-and-validate the shipped config against every gate this branch adds.

    Everything the routing branch enforces at boot is invoked by
    ``create_settings``: the strict YAML loader (R-5), the tier/model
    collision check (R-1b), and the model-name ambiguity WARNING (R-1a,
    non-fatal). A raise here is R-4 doing its job: the shipped file broke
    a rule and would have failed the running service.
    """
    settings = create_settings(SHIPPED_CONFIG)
    assert settings.routing.enabled is True, (
        "Shipped config has routing.enabled: false — that would take the "
        "gateway back into legacy single-backend mode and skip every "
        "guarantee this test measures. Refuse rather than let CI green."
    )


def test_shipped_config_default_model_for_unknown_task_is_routable() -> None:
    """The default model that /generate and /chat fall back to must route.

    ``default_model_for_unknown_task`` is passed through
    ``BackendRouter.get_backend_for_model`` when /generate or /chat run
    without an explicit ``model``. If that name is not registered, or is
    declared by multiple backends (ambiguous), the endpoint returns 404
    for every call that omits ``model`` — a regression that a boot
    validator cannot see because the router is only exercised at request
    time. This test is the boot-time surrogate for that request.
    """
    settings = create_settings(SHIPPED_CONFIG)
    name = settings.routing.default_model_for_unknown_task
    if name is None:
        # An unset default is legal — the /generate / /chat handlers
        # refuse with 400 "no model" in that case. What is not legal is a
        # default that structurally 404s.
        return

    tier_names = set(settings.routing.tiers)
    declared_by: dict[str, list[str]] = {}
    for backend_name, backend_settings in settings.routing.backends.items():
        for model_info in backend_settings.models:
            declared_by.setdefault(model_info.name, []).append(backend_name)

    if name in tier_names:
        # Tier aliases always route unambiguously; nothing else to check.
        return

    declared = declared_by.get(name, [])
    assert declared, (
        f"default_model_for_unknown_task is '{name}' but no tier or "
        f"backend declares it — /generate and /chat without a model would "
        f"404 on every request."
    )
    assert len(declared) == 1, (
        f"default_model_for_unknown_task is '{name}', which is declared "
        f"by multiple backends ({declared}). Requests naming it directly "
        f"are refused with 404 by R-1a; point this field at a tier alias "
        f"or a name only one backend declares."
    )


def _shipped_router() -> BackendRouter:
    """Router built from the shipped config, as production builds it."""
    settings = create_settings(SHIPPED_CONFIG)
    return BackendRouter(
        routing_settings=settings.routing, vllm_settings=settings.vllm
    )


def test_shipped_ambiguous_refusal_still_names_the_three_qwen_tiers() -> None:
    """R-6 regression pin: narrowing the remedy filter must not empty it here.

    All three vLLM backends declare ``Qwen3.8-27B``, so the raw name is
    refused, and the three tiers that reach them (``light`` / ``medium``
    / ``heavy``) all resolve to that same name — every one of them is a
    genuine remedy. Adding the "resolves to the requested model" half of
    the R-6 predicate therefore has to leave this message unchanged. If
    it shrinks, the predicate is stricter than R-6 asked for and the
    shipped gateway stopped telling callers how to reach a model it
    still serves.
    """
    router = _shipped_router()
    # Measured through the request path, so what is pinned is the text a
    # caller actually receives rather than an internal rendering of it.
    with pytest.raises(ModelNotFoundError) as exc_info:
        router.get_backend_for_model("Qwen3.8-27B")
    message = str(exc_info.value)
    assert "declared by multiple backends" in message, (
        "Shipped config no longer collides on 'Qwen3.8-27B', so this pin "
        f"is measuring the wrong refusal. Got: {message}"
    )
    for tier in ("light", "medium", "heavy"):
        assert tier in message, (
            f"Tier '{tier}' resolves to 'Qwen3.8-27B' and reaches a "
            f"declaring backend, so it is a real remedy. Got: {message}"
        )


def _shipped_router_with_stubbed_upstreams() -> BackendRouter:
    """Shipped router whose every upstream serves both Qwen names.

    That is the widest catalogue any backend here reports — the current
    model ID and the pre-rename alias the vLLM servers keep — so it is
    the payload that exercises both listing filters at once.
    """
    router = _shipped_router()
    upstream = {
        "object": "list",
        "data": [
            {"id": "Qwen3.8-27B", "object": "model", "created": 1, "owned_by": "vllm"},
            {"id": "Qwen3-32B", "object": "model", "created": 1, "owned_by": "vllm"},
        ],
    }
    for backend in router.backends.values():
        backend.list_models = AsyncMock(return_value=upstream)
    return router


@pytest.mark.asyncio
async def test_shipped_listing_is_the_tiers_plus_the_declared_concrete_names() -> None:
    """R-7 / T-models-advertise-side regression pin: the advertised set.

    This pin changed deliberately, and the change is the point of
    T-models-advertise-side. It used to read
    ``["light", "medium", "heavy", "naysayer", "frontier"]`` — five tier
    aliases and zero concrete IDs — under the older rule that a concrete
    name is advertised only when some upstream reported it. Under that
    rule the shipped ``claude-code-opus`` / ``claude-code-sonnet`` were
    routable and invisible: ``ClaudeCodeBackend.list_models`` returns a
    hardcoded empty list and no shipped tier maps to that backend, so
    the whole backend was undiscoverable through the HTTP API.

    The advertised set is now the routable set, so every unambiguously
    declared name appears with the declaring backend on its row. What is
    still *absent* is what carries the rest of the meaning:

    * ``Qwen3.8-27B`` — declared by ``heavy`` / ``light`` / ``deep``, so
      it is ambiguous, 404s by name (R-1a), and stays out. Declared is
      not the same as routable.
    * ``Qwen3-32B`` — reported by the stubbed upstreams and declared by
      nobody, so it 404s (R-2) and stays out. W-3 is intact.

    The order is part of the pin: concrete rows first, then tiers, so a
    consumer reading top-down sees the IDs before the aliases that route
    to them.
    """
    router = _shipped_router_with_stubbed_upstreams()

    listing = await router.list_all_models()
    ids = [m["id"] for m in listing["data"]]
    assert ids == [
        "gemini-3.1-pro-preview",
        "claude-code-opus",
        "claude-code-sonnet",
        "claude-sonnet-4-20250514",
        "claude-fable-5",
        "light",
        "medium",
        "heavy",
        "naysayer",
        "frontier",
    ], f"Advertised ids changed. Got: {ids}"
    assert "Qwen3.8-27B" not in ids
    assert "Qwen3-32B" not in ids
    # And the invariant the ids alone do not carry: every advertised
    # name routes, and to the backend the row names.
    for row in listing["data"]:
        assert row["backend"] == router.get_backend_name_for_model(row["id"])


@pytest.mark.asyncio
async def test_shipped_advertised_set_equals_routable_set() -> None:
    """T-models-advertise-side: the invariant, not a list of IDs.

    The pin above goes stale the moment a model is added to the shipped
    config, and updating it is then a bookkeeping edit that cannot fail.
    This states the property instead, so a name added to
    ``config/lexora_config.yaml`` and dropped from ``/v1/models`` fails
    here without anyone having to think of it.

    "Routable" is enumerated from the router's own two lookup tables
    because those are exactly what ``get_backend_for_model`` consults —
    the tier map and the unambiguous by-name index. Rebuilding the set
    from the YAML instead would re-implement the ambiguity rule in the
    test and could agree with a broken router for the wrong reason.
    """
    router = _shipped_router_with_stubbed_upstreams()

    listing = await router.list_all_models()
    advertised = {m["id"] for m in listing["data"]}
    routable = set(router._model_to_backend) | set(router._tier_to_backend)

    assert advertised == routable, (
        f"/v1/models disagrees with the router. Advertised but not "
        f"routable (W-3 violated): {sorted(advertised - routable)}; "
        f"routable but not advertised (hidden): {sorted(routable - advertised)}"
    )
    # Not vacuous: an empty listing would satisfy set equality against an
    # empty routable set, and this config declares plenty.
    assert len(advertised) >= 10


@pytest.mark.asyncio
async def test_shipped_claude_code_models_are_advertised_and_route() -> None:
    """The two names this thread was opened for, by name.

    Kept as a named-instance pin next to the invariant above because the
    invariant would also be satisfied by deleting the ``claude_code``
    backend from the shipped config. These two names are shipped,
    working and were hidden; a change that removes them from the API
    should have to say so here.
    """
    router = _shipped_router_with_stubbed_upstreams()

    listing = await router.list_all_models()
    rows = {m["id"]: m for m in listing["data"]}
    for name in ("claude-code-opus", "claude-code-sonnet"):
        assert name in rows, (
            f"'{name}' is declared by the shipped config and routes, but "
            f"/v1/models does not list it. Got: {sorted(rows)}"
        )
        assert rows[name]["backend"] == "claude_code"
        assert router.get_backend_name_for_model(name) == "claude_code"
    # No upstream reports these — the backend's ``list_models`` is a
    # hardcoded empty list — so both rows exist on this gateway's own
    # authority, which is what ``owned_by`` records.
    assert rows["claude-code-opus"]["owned_by"] == "lexora"

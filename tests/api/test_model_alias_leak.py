"""R-13: `/generate` and `/chat` must send upstream a name upstream declares.

At `develop` (`cd7cdac`) `grep resolve_model src/lexora/api/routes.py` matched
inside four handlers -- `chat_completions`, `completions`, `embeddings`,
`messages` -- and inside neither `generate` nor `chat`. Those two put
`request.model` straight into the dict handed to the backend:

    handler   routes with            puts on the wire
    generate  get_backend_for_model  the requested name, unresolved
    chat      get_backend_for_model  the requested name, unresolved

The requested name is frequently a *tier alias*, which is a name in Lexora's
own config and nowhere else. The shipping `config/lexora_config.yaml` makes
that the default case: `default_model_for_unknown_task: "heavy"`, so a caller
that omits `model` sends the literal string `heavy` to vLLM. Measured
in-process against the shipped config at `cd7cdac`:

    resolve_model("heavy")              -> "Qwen3.8-27B"
    get_backend_for_model("heavy")      -> VLLMBackend "deep"
    get_backend_for_model("Qwen3.8-27B") -> ModelNotFoundError (ambiguous:
                                            declared by heavy, light, deep)

and no backend rewrites the field on the way out for that route:
`["model"] =` appears in `backends/` only at `anthropic.py:141` and
`openai_compatible.py:92`, while `heavy` lands on `VLLMBackend`, which derives
from `Backend` and not from `OpenAICompatibleBackend`, and whose `_post` hands
`data` to `httpx` unchanged.

The two measurements those last three lines rest on are also why the fix has
an order to it, pinned by `TestRoutingUsesTheRequestedName` below: the tier
must reach `get_backend_for_model` *unresolved*. Resolving first hands it
`Qwen3.8-27B`, which three backends declare, and the router refuses ambiguous
request names rather than picking a winner -- so a "tidy" implementation that
resolves once at the top 404s every tier in the shipping config.

What this file does not claim: that production is broken. vLLM's response to
`model: "heavy"` is unmeasured here -- no live server was touched. The defect
asserted is internal and fully measurable in the repo: one decision is
implemented two ways in one file, and the value the odd two send is not a name
any configured backend declares.

Why a real `BackendRouter` instead of a `MagicMock`: the order requirement is
a property of the router's lookup rules (tier index first, ambiguity refusal
on the by-name path). A mock router answers every lookup, so it cannot tell a
correct order from an inverted one. The backends are real objects too; only
their transport methods are replaced.

Every case below is parametrised over both routes, including the routing fence
and the control. An earlier revision of this file drove those two through
`/generate` only, which left `/chat`'s routing unfenced: the isolating
mutation in the last entry, applied to `/chat` alone, passed all 8 cases of
that revision. The lesson generalises past this file -- an isolating mutation
has to be applied *per site*, not once across every site the requirement
covers, or a fence that exists for the requirement can still be absent at one
of the places the requirement holds.

Mutation, so the detectors below are measured rather than asserted. Counts are
this file alone, 12 cases:

- Against `develop` (`cd7cdac`): 4 red / 8 green. Red = both
  `TestTierAliasIsResolvedBeforeItGoesUpstream` cases and both
  `TestDefaultModelIsATierAlias` cases -- each reads the dict the backend was
  handed and finds `dup-model` absent, the alias present. Green, and therefore
  fences rather than detectors, = all six `TestRoutingUsesTheRequestedName`
  cases (`develop` already routes by the requested name; the fix must not lose
  that) and both `test_non_tier_name_is_passed_through_unchanged` cases.
- `/generate` fixed alone (the wire field left unresolved in `chat`): 2 red /
  10 green. `[generate]` is green in both detector classes while `[chat]`
  reds in both -- two independent detectors per class, not one counted twice.
- Order inverted in both handlers (`resolve_model` first, its result passed to
  `get_backend_for_model`): 10 red / 2 green. `TestRoutingUsesTheRequestedName`
  reds on all three tiers on both routes with `ModelNotFoundError` (ambiguous),
  which is the outcome this file exists to forbid. Note what the count says,
  though: the four detectors red too, because a request that 404s never reaches
  a backend and so has no payload to read. **This mutation therefore does not
  isolate `TestRoutingUsesTheRequestedName`** -- it fails everything driven by
  a tier. Only the two control cases survive, because a non-tier name resolves
  to itself and cannot detect a swap.
- Routing forced to `default_backend` while resolution is left correct
  (`backend = backend_router.default_backend`), both handlers: 4 red / 8
  green, and this is the mutation that isolates. Only `[medium-b2-*]` and
  `[heavy-b3-*]` red; `[light-b1-*]` stays green because `b1` *is* the
  default, and all four detectors plus both controls stay green because the
  wire still carries `dup-model`. So the routing fence detects "reached the
  wrong backend" on its own, independently of anything the model-name
  assertions measure.
- The same mutation at one handler at a time: `/generate` alone 2 red / 10
  green (`[medium-b2-generate]`, `[heavy-b3-generate]`), `/chat` alone 2 red /
  10 green (`[medium-b2-chat]`, `[heavy-b3-chat]`). Per-site, so neither
  route's routing is being fenced by the other's. The `/chat` half is the one
  that read 8 green before this file was parametrised.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexora.api.routes import (
    get_backend,
    get_backend_router,
    get_metrics_collector,
    get_model_registry,
    get_rate_limiter,
    get_retry_handler,
    get_stats_collector,
    is_rate_limit_enabled,
    router,
)
from lexora.config import BackendSettings, RoutingSettings, TierSettings, VLLMSettings
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.router import BackendRouter
from lexora.services.stats import StatsCollector

# `SHARED` is declared by three backends and fronted by three tiers, which is
# the shipping config's shape (`Qwen3.8-27B` under `light` / `heavy` / `deep`)
# reduced to the two facts these routes depend on: the concrete name is
# ambiguous as a request field, and each tier still routes. `SOLO` is the
# control -- an unambiguous name with no tier, where resolution is the
# identity and therefore invisible.
SHARED = "dup-model"
SOLO = "solo-model"
TIER_TO_BACKEND = [("light", "b1"), ("medium", "b2"), ("heavy", "b3")]

USAGE = {"prompt_tokens": 7, "completion_tokens": 3}
COMPLETION_RESPONSE: dict[str, Any] = {
    "id": "cmpl-1",
    "choices": [{"index": 0, "text": "Hi", "finish_reason": "stop"}],
    "usage": USAGE,
}
CHAT_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": USAGE,
}


def _generate_body(model: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {"prompt": "Hi"}
    if model is not None:
        body["model"] = model
    return body


def _chat_body(model: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "Hi"}]}
    if model is not None:
        body["model"] = model
    return body


# `(id, body builder, name of the backend coroutine the handler awaits)`. The
# third element is what lets one parametrised body read the dict that actually
# left the handler, for either route.
ROUTES = [
    ("/generate", _generate_body, "completions"),
    ("/chat", _chat_body, "chat_completions"),
]


def _router() -> BackendRouter:
    """A real router; only the backends' transport methods are replaced.

    Every backend answers, so "which one was awaited" is a statement about the
    routing decision rather than about which mock happened to be wired in.
    """
    backend_router = BackendRouter(
        routing_settings=RoutingSettings(
            enabled=True,
            default_backend="b1",
            backends={
                "b1": BackendSettings(url="http://localhost:1", models=[{"name": SHARED}]),
                "b2": BackendSettings(url="http://localhost:2", models=[{"name": SHARED}]),
                "b3": BackendSettings(url="http://localhost:3", models=[{"name": SHARED}]),
                "solo": BackendSettings(url="http://localhost:4", models=[{"name": SOLO}]),
            },
            tiers={
                "light": TierSettings(backend="b1", model=SHARED),
                "medium": TierSettings(backend="b2", model=SHARED),
                "heavy": TierSettings(backend="b3", model=SHARED),
            },
        ),
        vllm_settings=VLLMSettings(),
    )
    for backend in backend_router.backends.values():
        backend.completions = AsyncMock(return_value=COMPLETION_RESPONSE)
        backend.chat_completions = AsyncMock(return_value=CHAT_RESPONSE)
    return backend_router


def _client(backend_router: BackendRouter, default_model: str | None = None) -> TestClient:
    registry = MagicMock()
    registry.get_default_model_for_unknown_task.return_value = default_model

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: backend_router.backends["b1"]
    app.dependency_overrides[get_backend_router] = lambda: backend_router
    app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
    app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
        max_retries=1, base_delay=0.01, max_delay=0.1, jitter=False
    )
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
        default_rate=1000.0, default_burst=1000
    )
    app.dependency_overrides[is_rate_limit_enabled] = lambda: False
    app.dependency_overrides[get_metrics_collector] = lambda: None
    app.dependency_overrides[get_model_registry] = lambda: registry
    return TestClient(app)


def _sent_payload(backend_router: BackendRouter, method: str) -> dict[str, Any]:
    """The single dict some backend was handed, or a failure naming the count.

    Deliberately searches every backend rather than the expected one: a payload
    delivered to the wrong backend must read as a routing failure here, not as
    "the model field was fine".
    """
    calls = [
        (name, getattr(backend, method).call_args[0][0])
        for name, backend in backend_router.backends.items()
        if getattr(backend, method).await_count
    ]
    assert len(calls) == 1, f"expected exactly one backend awaited, got {calls}"
    return calls[0][1]


class TestTierAliasIsResolvedBeforeItGoesUpstream:
    """D-A1 / D-A2: the wire carries the concrete model, not the alias."""

    @pytest.mark.parametrize(
        ("endpoint", "build_body", "method"), ROUTES, ids=["generate", "chat"]
    )
    def test_explicit_tier_is_resolved(
        self, endpoint: str, build_body: Any, method: str
    ) -> None:
        backend_router = _router()
        response = _client(backend_router).post(endpoint, json=build_body("heavy"))

        assert response.status_code == 200
        assert _sent_payload(backend_router, method)["model"] == SHARED


class TestDefaultModelIsATierAlias:
    """D-A1 / D-A2 at the default: omitting `model` is the shipped path.

    `default_model_for_unknown_task` is `"heavy"` in `config/lexora_config.yaml`
    and `get_default_model` returns it verbatim, so this is what a caller that
    names no model produces -- the case with no explicit alias to blame.
    """

    @pytest.mark.parametrize(
        ("endpoint", "build_body", "method"), ROUTES, ids=["generate", "chat"]
    )
    def test_default_tier_is_resolved(
        self, endpoint: str, build_body: Any, method: str
    ) -> None:
        backend_router = _router()
        client = _client(backend_router, default_model="heavy")
        response = client.post(endpoint, json=build_body(None))

        assert response.status_code == 200
        assert _sent_payload(backend_router, method)["model"] == SHARED


class TestRoutingUsesTheRequestedName:
    """D-A3: `get_backend_for_model` keeps receiving the unresolved name.

    Green at `develop` as well -- this is a fence on behaviour the fix must not
    lose, not a detector for the leak. It reds against an implementation that
    resolves first, because `SHARED` is refused as ambiguous, and against one
    that reaches the wrong backend with the name still correct on the wire.
    Only the second of those isolates it; see the module docstring. Both routes
    are driven, because the requirement holds at both and a mutation at one is
    invisible to the other.
    """

    @pytest.mark.parametrize(
        ("endpoint", "build_body", "method"), ROUTES, ids=["generate", "chat"]
    )
    @pytest.mark.parametrize(("tier", "expected"), TIER_TO_BACKEND)
    def test_the_tier_s_own_backend_is_reached(
        self, endpoint: str, build_body: Any, method: str, tier: str, expected: str
    ) -> None:
        backend_router = _router()
        response = _client(backend_router).post(endpoint, json=build_body(tier))

        assert response.status_code == 200
        awaited = [
            name
            for name, backend in backend_router.backends.items()
            if getattr(backend, method).await_count
        ]
        assert awaited == [expected]


@pytest.mark.parametrize(
    ("endpoint", "build_body", "method"), ROUTES, ids=["generate", "chat"]
)
def test_non_tier_name_is_passed_through_unchanged(
    endpoint: str, build_body: Any, method: str
) -> None:
    """F: resolution is the identity for a name no tier fronts.

    The control for the four detectors above. It is why they had to be driven
    with a tier: on a non-tier name a correct implementation and the leaking
    one are indistinguishable, which is the reason
    `tests/api/test_routes_convenience.py` could assert the outgoing `model`
    four times without ever measuring whether resolution happened.
    """
    backend_router = _router()
    response = _client(backend_router).post(endpoint, json=build_body(SOLO))

    assert response.status_code == 200
    assert _sent_payload(backend_router, method)["model"] == SOLO

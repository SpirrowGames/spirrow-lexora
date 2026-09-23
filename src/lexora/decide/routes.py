"""``POST /v1/decide`` FastAPI route.

The route is deliberately thin: pick a provider, call ``evaluate``,
write one decision-log row, return. It exists so the utilisation-side
threads (mindwire / prismind / verimend) have a stable URL to point at;
the interesting behaviour is in the provider layer.

PR 1 scope (msg-246): only ``NullProvider`` was wired. T-decide-jev-provider
adds ``JevProvider`` and the NullProvider fallback on ``ProviderError``. The ``mode`` /
``primary`` / ``fallback`` settings *are* honoured — a config that
selects Jev on PR 1 is refused at startup by
:func:`lexora.decide.config.check_typesafe_api_key`, so the route never
has to reach for a provider that has not been built yet — but the
route resolves everything through :class:`~lexora.decide.config.
DecisionSettings` so the switch to real providers in a later PR is a
one-file change here.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request

from lexora.decide.config import DecisionSettings
from lexora.decide.contract import DecideRequest, DecideResponse
from lexora.decide.log import DecisionLog, build_decision_row
from lexora.decide.providers import (
    DecisionProvider,
    JevProvider,
    NullProvider,
    ProviderError,
)
from lexora.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _get_decision_settings(request: Request) -> DecisionSettings:
    """Fetch the decision settings loaded at app startup."""
    return request.app.state.decision_settings  # type: ignore[no-any-return]


def _get_decision_providers(
    request: Request,
) -> dict[str, DecisionProvider]:
    """Fetch the provider registry loaded at app startup."""
    return request.app.state.decision_providers  # type: ignore[no-any-return]


def _get_decision_log(request: Request) -> DecisionLog:
    """Fetch the decision log writer loaded at app startup."""
    return request.app.state.decision_log  # type: ignore[no-any-return]


def _select_provider(
    settings: DecisionSettings,
    providers: dict[str, DecisionProvider],
) -> DecisionProvider:
    """Return the provider whose answer this request should carry.

    PR 1 behaviour (msg-246):

    * ``mode == "off"`` → NullProvider unconditionally.
    * Otherwise → the provider named by ``primary`` if it is registered,
      else NullProvider. In PR 1 only ``null`` is registered, so a
      ``primary="jev"`` config that got past the startup env check but
      has no provider still gets a NullProvider answer here — a
      followup PR that registers ``jev`` / ``llm`` will change what
      this returns without changing the shape of the response.

    ``shadow`` mode's "return fallback, run primary in the background"
    semantics is a follow-up PR: this PR only implements the caller-
    visible provider, so ``shadow`` collapses to the same code path as
    ``active`` today. The route logs ``provider`` verbatim so a replay
    can filter by whichever provider actually served.
    """
    if settings.mode == "off":
        return providers["null"]
    caller_visible = settings.primary if settings.mode == "active" else settings.fallback
    provider = providers.get(caller_visible)
    if provider is None:
        return providers["null"]
    return provider


@router.post("/v1/decide", response_model=DecideResponse)
async def decide(
    body: DecideRequest,
    settings: DecisionSettings = Depends(_get_decision_settings),
    providers: dict[str, DecisionProvider] = Depends(_get_decision_providers),
    decision_log: DecisionLog = Depends(_get_decision_log),
) -> DecideResponse:
    """Serve one judgment request.

    Contract lives in :mod:`lexora.decide.contract`; nothing in this
    function invents an alternate shape.
    """
    provider = _select_provider(settings, providers)

    decision_id = uuid.uuid4().hex
    start = time.monotonic()
    provider_error: str | None = None
    try:
        answers = await provider.evaluate(state=body.state, questions=body.questions)
    except ProviderError as exc:
        # Safe-default, not fail-loud (Fermi msg-257 §3): the caller
        # (e.g. mindwire's Tier-C gate, D20 fail-open) gets a well-formed
        # NullProvider answer, and the row records who actually answered
        # (``null``) plus why the primary did not (Bohr msg-258 §4).
        # The warning carries the fixed code only — no body, no key.
        provider_error = _format_provider_error(provider.name, exc)
        logger.warning(
            "decide_provider_fallback",
            primary=provider.name,
            code=exc.code,
            discarded=exc.discarded,
            decision_id=decision_id,
        )
        provider = providers["null"]
        answers = await provider.evaluate(state=body.state, questions=body.questions)
    latency_ms = int((time.monotonic() - start) * 1000)

    row = build_decision_row(
        decision_id=decision_id,
        policy=body.policy,
        state=body.state,
        questions={name: q for name, q in body.questions.items()},
        provider=provider.name,
        answers=_answers_for_log(answers),
        latency_ms=latency_ms,
        questions_version=body.questions_version,
        provider_error=provider_error,
    )
    decision_log.write(row)

    # Provider names are constrained to the Literal in DecideResponse,
    # so we assert the type on the way out. Any provider whose name
    # does not match the wire type is a programming error and we would
    # rather fail loudly than silently emit a value the caller cannot
    # parse.
    provider_name = provider.name
    if provider_name not in {"null", "llm", "jev"}:  # pragma: no cover
        raise RuntimeError(
            f"provider.name must be one of 'null'/'llm'/'jev', got {provider_name!r}"
        )

    return DecideResponse(
        answers=answers,
        provider=provider_name,  # type: ignore[arg-type]
        decision_id=decision_id,
        latency_ms=latency_ms,
    )


def _answers_for_log(answers: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly view of ``answers`` for the log.

    ``answers`` already contains plain dicts on the runtime path, but
    keeping the copy explicit makes future providers who return
    :class:`pydantic.BaseModel` instances safe: they get dumped rather
    than stored as opaque object references.
    """
    result: dict[str, Any] = {}
    for name, value in answers.items():
        if hasattr(value, "model_dump"):
            result[name] = value.model_dump(mode="json")
        else:
            result[name] = value
    return result


def _format_provider_error(primary: str, exc: ProviderError) -> str:
    """``"<primary>:<code>"`` plus ``";discarded=<n>"`` when ``n > 0``."""
    text = f"{primary}:{exc.code}"
    if exc.discarded:
        text += f";discarded={exc.discarded}"
    return text


def build_default_providers(
    api_key: str | None = None,
    *,
    timeout_ms: int = 2000,
) -> dict[str, DecisionProvider]:
    """Return the provider registry.

    ``null`` is always registered (it is the fallback). ``jev`` is
    registered only when ``create_app`` passes an API key, which it does
    only when the config references Jev — and in that case the startup
    check has already refused a missing key, so a Jev-configured app
    always gets a real JevProvider. ``llm`` is a follow-up PR.
    """
    providers: dict[str, DecisionProvider] = {"null": NullProvider()}
    if api_key:
        providers["jev"] = JevProvider(api_key, timeout_ms=timeout_ms)
    return providers

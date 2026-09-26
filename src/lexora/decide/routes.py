"""``POST /v1/decide`` FastAPI route.

The route is deliberately thin: pick a provider, call ``evaluate``,
write one decision-log row, return. It exists so the utilisation-side
threads (mindwire / prismind / verimend) have a stable URL to point at;
the interesting behaviour is in the provider layer.

Routing is two-way (Bohr msg-387 v3): ``mode="off"`` answers from
NullProvider, ``mode="active"`` answers from the provider named by
``primary``. On a :class:`~lexora.decide.providers.ProviderError` the
route falls back to NullProvider (Fermi msg-257 §3) and logs the
upstream ``model`` / usage (Bohr msg-342 #2).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request

from lexora.decide.config import DecisionSettings
from lexora.decide.contract import DecideRequest, DecideResponse
from lexora.decide.jev_client import DEFAULT_MODEL
from lexora.decide.log import DecisionLog, build_decision_row
from lexora.decide.providers import (
    DecisionProvider,
    JevProvider,
    NullProvider,
    ProviderError,
    UpstreamMeta,
)
from lexora.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

#: Fallback codes logged at ``error`` rather than ``warning`` (Bohr msg-339
#: #4 / msg-342 #1). Each one points at something a human has to fix: a
#: bad key, a malformed request from Lexora or the caller, a changed
#: response shape, or a Lexora bug.
_ERROR_LEVEL_CODES = frozenset(
    {"auth", "invalid_request", "invalid_response", "internal_error"}
)


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

    * ``mode == "off"`` → NullProvider.
    * ``mode == "active"`` → the provider named by ``primary``.

    The schema only admits ``null`` / ``jev`` for ``primary``, and a
    ``jev`` config cannot start without the key (so ``jev`` is always
    registered). A ``KeyError`` here is therefore a Lexora bug and is
    left to surface rather than being papered over with NullProvider.
    """
    if settings.mode == "off":
        return providers["null"]
    return providers[settings.primary]


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
    upstream: UpstreamMeta | None = None
    failure: dict[str, Any] | None = None
    try:
        result = await provider.evaluate(state=body.state, questions=body.questions)
    except ProviderError as err:
        # Copy out the five plain values and drop ``err`` right away
        # (Bohr msg-344 v7 #2). The fallback and the log write happen
        # below, outside this block.
        failure = {
            "code": err.code,
            "exc_type": err.exc_type,
            "where": err.where,
            "loc": err.loc,
        }
        upstream = err.upstream
        provider_error = f"{provider.name}:{err.code}"
    else:
        upstream = result.upstream
    if failure is not None:
        # Safe default instead of failing loud (Fermi msg-257 §3). The
        # caller, e.g. mindwire's Tier-C gate (D20 fail-open), gets a
        # well-formed NullProvider answer. The row records who answered
        # (``null``) and why the primary did not (Bohr msg-258 §4).
        #
        # Do NOT add ``exc_info=True`` here or anywhere else a
        # ProviderError is logged. The exception chain is cut
        # (msg-344), but ``ProviderError.__traceback__`` still reaches
        # the provider's frame, whose locals hold ``state``.
        log = (
            logger.error if failure["code"] in _ERROR_LEVEL_CODES else logger.warning
        )
        log(
            "decide_provider_fallback",
            primary=provider.name,
            decision_id=decision_id,
            **failure,
        )
        provider = providers["null"]
        result = await provider.evaluate(state=body.state, questions=body.questions)
    answers = result.answers
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
        provider_model=upstream.model if upstream else None,
        provider_input_tokens=upstream.input_tokens if upstream else None,
        provider_output_tokens=upstream.output_tokens if upstream else None,
    )
    # Off the event loop, on DecisionLog's dedicated single-thread
    # executor (msg-464 v3): a slow or lock-contended commit must not
    # freeze the other endpoints, nor occupy the default executor that
    # serves DNS for upstream connections. A write failure still raises
    # here and fails the request.
    await decision_log.awrite(row)

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


def build_default_providers(
    api_key: str | None = None,
    *,
    timeout_ms: int = 2000,
    jev_model: str = DEFAULT_MODEL,
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
        providers["jev"] = JevProvider(api_key, timeout_ms=timeout_ms, model=jev_model)
    return providers

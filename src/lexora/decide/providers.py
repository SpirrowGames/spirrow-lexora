"""Provider abstraction for ``/v1/decide`` and the ``null`` implementation.

The interface is a Protocol rather than an abstract class so the tests
can drop in an ad-hoc fake without inheriting; future ``LlmEmulation``
and ``Jev`` providers will implement it directly. Only the ``null``
implementation shipped in T02 PR 1 (msg-246); :class:`JevProvider` lands
in T-decide-jev-provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol, runtime_checkable

import httpx

from lexora.decide import jev_client
from lexora.decide.contract import Answer, QuestionSpec

ProviderErrorCode = Literal["timeout", "http_status", "network", "auth", "invalid_response"]


class ProviderError(Exception):
    """An upstream provider failed; the route falls back to NullProvider.

    ``code`` is one of a fixed set. ``discarded`` counts sibling upstream
    calls in the same request that had already completed — and so were
    billed — when the failure was observed; their answers are thrown away
    because a request is answered by exactly one provider (Bohr msg-260
    #2). Deliberately holds no HTTP body, header, or API key.
    """

    def __init__(self, code: ProviderErrorCode, *, discarded: int = 0) -> None:
        super().__init__(code if not discarded else f"{code};discarded={discarded}")
        self.code: ProviderErrorCode = code
        self.discarded = discarded


@runtime_checkable
class DecisionProvider(Protocol):
    """Sole entry point provider implementations expose.

    ``name`` MUST match the identifier used in
    :class:`~lexora.decide.config.DecisionSettings` (``"null"`` /
    ``"llm"`` / ``"jev"``); the router uses it verbatim when writing the
    ``provider`` column of the decision log, so a drift here would show
    up as a wrong provider name in every offline evaluation.

    ``evaluate`` returns a dict keyed by the same question names the
    caller supplied. Providers that cannot answer a particular question
    (e.g. NullProvider on a ``choice`` primitive) still emit an entry —
    they never omit the name — because a caller iterating over
    ``answers`` must not have to distinguish "provider did not answer"
    from "question was not asked".
    """

    name: str

    async def evaluate(
        self,
        *,
        state: str,
        questions: dict[str, QuestionSpec],
    ) -> dict[str, Answer]:
        """Answer every question in ``questions`` against ``state``."""


class NullProvider:
    """Deterministic provider that returns the safest possible answer.

    Semantics (msg-237): ``noul`` = 0.5, everything else = the
    caller-visible probabilities are uniform and confidence is 0.0. The
    caller-side interpretation is "no signal" — thresholds on
    ``confidence`` will decline, and thresholds on the raw probability
    will hit their tie-break policy.

    A ``choice`` question that arrived without a ``criteria`` list of
    options still gets a well-formed answer: an empty distribution and
    ``choice = ""``. The alternative — refusing the request — would
    make NullProvider unsafe as a fallback for a caller who happens to
    forget a criteria field on one question, and NullProvider's whole
    point is that the endpoint never fails for lack of upstream.

    A ``score`` question likewise gets ``score = 0.0`` and
    ``legend = []`` when the caller did not supply a legend; the value
    is meaningless but the shape is well-formed, which is what a
    downstream caller iterating over answers needs.
    """

    name = "null"

    async def evaluate(
        self,
        *,
        state: str,
        questions: dict[str, QuestionSpec],
    ) -> dict[str, Answer]:
        # ``state`` is unused deliberately: NullProvider is a shape-
        # correct constant, not a degenerate LLM. The parameter stays
        # in the signature so the Protocol is satisfied and so the two
        # future providers (``llm``, ``jev``) do not have to inherit a
        # different signature.
        del state
        answers: dict[str, Answer] = {}
        for name, question in questions.items():
            answers[name] = _null_answer_for(question)
        return answers


def _null_answer_for(question: QuestionSpec) -> Answer:
    """Return the safe-default answer for one question."""
    if question.type == "noul":
        return {"noul": 0.5}
    if question.type == "choice":
        options = _extract_choice_options(question.criteria)
        if not options:
            return {"choice": "", "probabilities": {}, "confidence": 0.0}
        uniform = 1.0 / len(options)
        return {
            "choice": options[0],
            "probabilities": {opt: uniform for opt in options},
            "confidence": 0.0,
        }
    # score
    legend = _extract_score_legend(question.criteria)
    return {"score": 0.0, "legend": legend, "confidence": 0.0}


def _extract_choice_options(criteria: object) -> list[str]:
    """Best-effort extraction of the option names for a ``choice`` question.

    TypeSafe's ``choice`` criteria may be a list of strings or a list of
    ``{name, description}`` objects (docs at
    https://docs.typesafe.ai/primitives/choice.md — accessible from the
    vendored SKILL.md, not from this codebase at runtime). Lexora
    accepts either form so NullProvider stays useful against real
    question payloads from the utilisation-side threads.
    """
    if not isinstance(criteria, list):
        return []
    options: list[str] = []
    for entry in criteria:
        if isinstance(entry, str):
            options.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str):
                options.append(name)
    return options


def _extract_score_legend(criteria: object) -> list[str]:
    """Best-effort extraction of the ordered level names for a ``score``."""
    if not isinstance(criteria, list):
        return []
    legend: list[str] = []
    for entry in criteria:
        if isinstance(entry, str):
            legend.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str):
                legend.append(name)
    return legend


_JevCall = Callable[[httpx.AsyncClient, str, str, QuestionSpec], Awaitable[Answer]]

_JEV_DISPATCH: dict[str, _JevCall] = {
    "noul": jev_client.call_noul,
    "choice": jev_client.call_choice,
    "score": jev_client.call_score,
}


class JevProvider:
    """Provider backed by the Jev (TypeSafe) judgment API.

    API key (msg-240 §1 / Bohr msg-258 §2): passed in once by
    ``create_app`` from the env and captured in a closure — not stored on
    ``app.state``, on settings, or as a plain attribute, and never put in
    an exception or log line. ``repr(provider)`` does not show it.

    Failure semantics (Bohr msg-260 #2, Einstein msg-259 #2 / msg-261):
    questions are dispatched concurrently under :class:`asyncio.TaskGroup`.
    The first failure cancels the in-flight siblings (fail-fast, to limit
    billed-but-discarded upstream calls) and the whole ``evaluate`` raises
    one :class:`ProviderError`. There is no partial result: a request is
    answered by exactly one provider, so the ``provider`` column never
    mixes Jev answers with NullProvider answers.
    """

    name = "jev"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_ms: int,
        base_url: str = jev_client.DEFAULT_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        def _key() -> str:
            return api_key

        self._key = _key
        self._timeout_s = timeout_ms / 1000.0
        self._base_url = base_url
        self._transport = transport

    def __repr__(self) -> str:
        return f"JevProvider(base_url={self._base_url!r})"

    async def evaluate(
        self,
        *,
        state: str,
        questions: dict[str, QuestionSpec],
    ) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        api_key = self._key()
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:

            async def _one(name: str, question: QuestionSpec) -> None:
                call = _JEV_DISPATCH[question.type]
                answers[name] = await call(client, api_key, state, question)

            try:
                async with asyncio.TaskGroup() as tg:
                    for name, question in questions.items():
                        tg.create_task(_one(name, question))
            except BaseExceptionGroup as eg:
                # Anything that is not a classified Jev failure is a bug;
                # re-raise it unchanged rather than hide it behind a
                # fallback.
                matched, rest = eg.split(jev_client.JevCallError)
                if rest is not None or matched is None:
                    raise
                first = _first_leaf(matched)
                # ``answers`` holds exactly the calls that completed
                # before cancellation: billed upstream, discarded here.
                raise ProviderError(first.code, discarded=len(answers)) from None
        return answers


def _first_leaf(eg: BaseExceptionGroup[jev_client.JevCallError]) -> jev_client.JevCallError:
    exc: BaseException = eg
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    assert isinstance(exc, jev_client.JevCallError)
    return exc

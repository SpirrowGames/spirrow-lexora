"""Provider abstraction for ``/v1/decide``: ``null`` and ``jev``.

The interface is a Protocol rather than an abstract class, so tests can
use an ad-hoc fake without inheriting from anything. ``NullProvider``
shipped in T02 PR 1 (msg-246). :class:`JevProvider` lands in
T-decide-jev-provider, following Bohr's v8 design (msg-339 / 342 / 344 /
346). The planned ``LlmEmulation`` provider will implement the same
Protocol and return the same :class:`ProviderResult`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import httpx

from lexora.decide import jev_client
from lexora.decide.contract import Answer, QuestionSpec

#: Fixed failure codes (Bohr msg-339 #4 table + msg-342 ``internal_error``).
ProviderErrorCode = Literal[
    "timeout",
    "network",
    "auth",
    "invalid_request",
    "rate_limited",
    "overloaded",
    "http_status",
    "invalid_response",
    "internal_error",
]


@dataclass(frozen=True)
class UpstreamMeta:
    """What the upstream call reported about itself (Bohr msg-342 #2).

    ``model`` is the version that actually served the request (e.g.
    ``"jev-1.13.0"``), not the requested alias. Every field is filled
    leniently: one that is missing or has the wrong type is ``None``.
    """

    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ProviderResult:
    """Return type of :meth:`DecisionProvider.evaluate` (Bohr msg-342 #2).

    ``upstream`` is ``None`` for providers that call nothing upstream
    (NullProvider). Metadata travels in the return value, not on
    ``self``, so concurrent requests cannot see each other's values.
    """

    answers: dict[str, Answer]
    upstream: UpstreamMeta | None = None


@dataclass(frozen=True)
class _Failure:
    """Plain-value description of a provider failure (Bohr msg-344/346).

    Holds only strings, a string tuple and an :class:`UpstreamMeta`. It
    never holds the original exception, an httpx ``Request`` /
    ``Response``, or any body.
    """

    code: ProviderErrorCode
    exc_type: str | None = None
    where: str | None = None
    loc: tuple[str, ...] | None = None  # set only for 422
    upstream: UpstreamMeta | None = None

    @staticmethod
    def from_exc(
        code: ProviderErrorCode,
        exc: BaseException,
        upstream: UpstreamMeta | None = None,
    ) -> _Failure:
        """Keep the class name and the innermost ``file:line``.

        The message and traceback are dropped: a ``KeyError`` /
        ``TypeError`` message can contain part of ``state`` (msg-342 #1).
        Frame locals and arguments are never read.
        """
        tb = exc.__traceback__
        while tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
        where = (
            f"{_relpath(tb.tb_frame.f_code.co_filename)}:{tb.tb_lineno}"
            if tb is not None
            else None
        )
        return _Failure(code, type(exc).__name__, where, None, upstream)


_SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src (parent of ``lexora``)


def _relpath(filename: str) -> str:
    """Turn a code filename into a relative, ``/``-separated path.

    Files under Lexora's ``src`` become ``lexora/...``. Library files keep
    only the part after ``site-packages``. Anything else keeps only its
    basename, so an absolute host path never reaches a log line.
    """
    path = Path(filename)
    try:
        return path.resolve().relative_to(_SRC_ROOT).as_posix()
    except (ValueError, OSError):
        pass
    parts = path.parts
    if "site-packages" in parts:
        idx = len(parts) - 1 - parts[::-1].index("site-packages")
        rest = parts[idx + 1 :]
        if rest:
            return "/".join(rest)
    return path.name or filename.replace(os.sep, "/").rsplit("/", 1)[-1]


class ProviderError(Exception):
    """An upstream provider failed; the route falls back to NullProvider.

    Its public surface is exactly five plain values (Bohr msg-346 v8 #1):
    ``code`` / ``exc_type`` / ``where`` / ``loc`` / ``upstream``.
    ``str(err)`` is ``"jev:<code>"``. The exception holds no reference to
    the original exception, an httpx object, a body, or the API key.
    :class:`JevProvider` raises it outside every ``except`` block, so both
    ``__cause__`` and ``__context__`` are ``None`` (msg-344 v7 #1).

    ``upstream`` is set only when a 2xx arrived and the answers were then
    rejected. In that case the call was billed and its usage is still
    logged (msg-342 #2).
    """

    def __init__(self, failure: _Failure, *, provider: str = "jev") -> None:
        super().__init__(f"{provider}:{failure.code}")
        self.code: ProviderErrorCode = failure.code
        self.exc_type = failure.exc_type
        self.where = failure.where
        self.loc = failure.loc
        self.upstream = failure.upstream


class _Classified(Exception):
    """Internal signal: an already-classified failure. Holds a ``_Failure`` only."""

    def __init__(self, failure: _Failure) -> None:
        super().__init__()
        self.failure = failure


@runtime_checkable
class DecisionProvider(Protocol):
    """Sole entry point provider implementations expose.

    ``name`` MUST match the identifier used in
    :class:`~lexora.decide.config.DecisionSettings` (``"null"`` /
    ``"llm"`` / ``"jev"``); the router uses it verbatim when writing the
    ``provider`` column of the decision log, so a drift here would show
    up as a wrong provider name in every offline evaluation.

    ``evaluate`` returns a :class:`ProviderResult` whose ``answers`` is
    keyed by the same question names the caller supplied. Providers that cannot answer a particular question
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
    ) -> ProviderResult:
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
    ) -> ProviderResult:
        # ``state`` is unused deliberately: NullProvider is a shape-
        # correct constant, not a degenerate LLM. The parameter stays
        # in the signature so the Protocol is satisfied and so the two
        # future providers (``llm``, ``jev``) do not have to inherit a
        # different signature.
        del state
        answers: dict[str, Answer] = {}
        for name, question in questions.items():
            answers[name] = _null_answer_for(question)
        return ProviderResult(answers=answers, upstream=None)


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


_Stage = Literal["request", "http", "parse"]


class JevProvider:
    """Provider backed by the Jev (TypeSafe) ``systemone`` API.

    API key (msg-240 §1 / Bohr msg-258 §2): ``create_app`` passes it in
    once from the env and it is captured in a closure. It is not stored on
    ``app.state``, on settings, or as a plain attribute, and it never goes
    into an exception or a log line. ``repr(provider)`` does not show it.

    One request is one upstream call (Bohr msg-339 #1): all questions go
    in a single POST, so either every question is answered or none is.
    One request therefore has exactly one ``provider`` value.

    Failure semantics (Bohr msg-342 #1 / msg-344 / msg-346, Einstein
    msg-340 #1 / msg-343 / msg-345): every exception raised inside
    ``evaluate`` becomes a :class:`ProviderError`. The router catches that
    one type and falls back to NullProvider, so a change in Jev's response
    shape can never turn ``/v1/decide`` into a 500. How exceptions map to
    codes:

    * httpx timeout / ``asyncio.timeout`` → ``timeout``; other
      ``httpx.RequestError`` → ``network``.
    * Status codes → :func:`jev_client.classify_status` (a 422 also carries
      a ``loc``; a 400 is ``invalid_request`` too, but its body is not read).
    * Any exception while reading a 2xx → ``invalid_response``, with the
      leniently read :class:`UpstreamMeta` attached.
    * Any other exception (a Lexora bug while building the request, for
      example) → ``internal_error``.

    The classifying ``try`` is in :meth:`_attempt`, which returns a
    :class:`_Failure` instead of raising. :meth:`evaluate` raises the
    ``ProviderError`` outside any ``except`` block, so the error has no
    ``__cause__`` and no ``__context__``. That matters because the
    original exception can reach the ``state`` text, or through
    ``httpx.Request`` the ``Authorization`` header (msg-344 v7 #1).
    ``_attempt``'s frame, which holds the key and the response, has
    already returned and is not part of the error's traceback.

    Residual risk (msg-344 v7 #2): ``ProviderError.__traceback__`` still
    reaches :meth:`evaluate`'s frame, and that frame holds ``state`` and
    ``self``. Never log it with ``exc_info``.
    """

    name = "jev"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_ms: int,
        model: str = jev_client.DEFAULT_MODEL,
        base_url: str = jev_client.DEFAULT_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        def _key() -> str:
            return api_key

        self._key = _key
        self._timeout_ms = timeout_ms
        self._model = model
        self._base_url = base_url
        self._transport = transport

    def __repr__(self) -> str:
        return f"JevProvider(base_url={self._base_url!r}, model={self._model!r})"

    async def evaluate(
        self,
        *,
        state: str,
        questions: dict[str, QuestionSpec],
    ) -> ProviderResult:
        outcome = await self._attempt(state, questions)
        if isinstance(outcome, _Failure):
            # Outside every ``except``: no implicit __context__ is attached.
            raise ProviderError(outcome, provider=self.name)
        return outcome

    async def _attempt(
        self, state: str, questions: dict[str, QuestionSpec]
    ) -> ProviderResult | _Failure:
        """Run the call. Return a result or a plain :class:`_Failure`; never raise.

        ``asyncio.CancelledError`` is a ``BaseException`` and propagates
        unchanged: the caller is cancelling the request, so this is not a
        Jev failure.
        """
        stage: _Stage = "request"
        meta: UpstreamMeta | None = None
        try:
            body = jev_client.build_body(state, questions, self._model)
            stage = "http"
            resp = await jev_client.call_systemone(
                body,
                api_key=self._key(),
                timeout_ms=self._timeout_ms,
                base_url=self._base_url,
                transport=self._transport,
            )
            status_code = jev_client.classify_status(resp.status_code)
            if status_code is not None:
                # Only a 422 body is read; a 400 is invalid_request too but
                # its body is never touched (msg-368 v9).
                loc = (
                    jev_client.extract_422_loc(resp)
                    if resp.status_code == 422
                    else None
                )
                raise _Classified(_Failure(status_code, loc=loc))
            stage = "parse"
            payload = resp.json()
            model, tokens_in, tokens_out = jev_client.extract_meta(payload)
            meta = UpstreamMeta(model, tokens_in, tokens_out)
            answers = jev_client.parse_answers(payload, questions)
        except (httpx.TimeoutException, TimeoutError) as exc:
            return _Failure.from_exc("timeout", exc)
        except httpx.RequestError as exc:
            return _Failure.from_exc("network", exc)
        except _Classified as signal:
            return signal.failure
        except jev_client.InvalidResponse:
            # Shape check failed: the class name is enough, and there is
            # no "where" worth keeping for a deliberate raise.
            return _Failure("invalid_response", "InvalidResponse", None, None, meta)
        except Exception as exc:  # noqa: BLE001 — msg-342 #1: the boundary is here
            code: ProviderErrorCode = (
                "invalid_response" if stage == "parse" else "internal_error"
            )
            return _Failure.from_exc(code, exc, upstream=meta)
        return ProviderResult(answers=answers, upstream=meta)

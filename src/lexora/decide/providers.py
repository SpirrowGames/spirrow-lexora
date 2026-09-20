"""Provider abstraction for ``/v1/decide`` and the ``null`` implementation.

The interface is a Protocol rather than an abstract class so the tests
can drop in an ad-hoc fake without inheriting; future ``LlmEmulation``
and ``Jev`` providers will implement it directly. Only the ``null``
implementation ships in this PR (msg-246 confirmed T02 PR 1 scope).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lexora.decide.contract import Answer, QuestionSpec


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

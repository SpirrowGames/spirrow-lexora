"""Wire types for the ``/v1/decide`` endpoint.

The three answer variants map 1:1 to TypeSafe's judgment primitives, but
Lexora deliberately does NOT re-export TypeSafe's names on this layer —
the callers must be able to swap providers without changing their code
(msg-237). Provider-neutral field names in, provider-neutral field names
out; the provider bridge is in :mod:`lexora.decide.providers`.

``questions_hash`` (see :func:`compute_questions_hash`) is what pins the
same question set across replays and provider swaps, and it is computed
by Lexora regardless of whether the caller sent ``questions_version``
(Einstein msg-243 objection #1, Bohr disposition in msg-244). The
callers cannot arrange for the hash to be missing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

#: Length of the hex digest kept in the decision log. Full sha256 is 64
#: hex chars; 16 hex chars = 64 bits, more than enough to distinguish the
#: question sets a single deployment ever sees. Fixed here so the schema
#: and the writer cannot drift.
QUESTIONS_HASH_HEX_LENGTH = 16


class QuestionSpec(BaseModel):
    """A single judgment the caller asks Lexora to make.

    ``type`` follows TypeSafe's primitives (``noul``, ``choice``,
    ``score``) but the schema is intentionally permissive on
    ``instructions`` / ``criteria`` — Lexora forwards whatever the
    caller sent, so a caller that upgrades to a new TypeSafe field does
    not have to wait for a Lexora release.
    """

    type: Literal["noul", "choice", "score"] = Field(
        description="Judgment primitive.",
    )
    instructions: str = Field(description="What the judgment is about.")
    criteria: Any | None = Field(
        default=None,
        description="Criteria object; shape depends on the primitive.",
    )

    model_config = {"extra": "allow"}


class DecideRequest(BaseModel):
    """The full ``POST /v1/decide`` request body.

    ``questions_version`` is opaque to Lexora — it is recorded verbatim
    if present (msg-244 Bohr disposition #1). It is optional so that a
    caller that has not adopted question versioning yet still gets a
    complete decision-log row (``questions_hash`` is server-computed).
    """

    state: str = Field(description="The named JSON / free text the questions read from.")
    questions: dict[str, QuestionSpec] = Field(
        description="Named questions to evaluate over ``state``.",
    )
    policy: str = Field(
        description=(
            "Caller-supplied tag identifying which policy / call site "
            "sent the request. Recorded in the decision log so replays "
            "can filter by call site."
        ),
    )
    questions_version: str | None = Field(
        default=None,
        description=(
            "Opaque string the caller may send to label a specific "
            "question-set revision. Lexora does not interpret it; the "
            "decision log stores it verbatim (may be null)."
        ),
    )

    model_config = {"extra": "allow"}


class NoulAnswer(BaseModel):
    """A ``noul`` judgment result: probability of "yes"."""

    noul: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of the condition holding (0..1).",
    )


class ChoiceAnswer(BaseModel):
    """A ``choice`` judgment result: selected option + distribution."""

    choice: str = Field(description="Selected option.")
    probabilities: dict[str, float] = Field(
        description="Probability per option; sums to (approximately) 1.0.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Distribution concentration (not correctness).",
    )


class ScoreAnswer(BaseModel):
    """A ``score`` judgment result: ordered level + level legend."""

    score: float = Field(description="Probability-weighted position on the legend.")
    legend: list[str] = Field(description="Ordered level names.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Distribution concentration (not correctness).",
    )


#: Union of the three answer shapes. Kept as a dict rather than a
#: discriminated Pydantic union so unknown provider-specific keys still
#: reach the caller without a schema change (Bohr msg-246 stance: schema
#: is a contract for the primitives, not a filter for the primitives).
Answer = dict[str, Any]


class DecideResponse(BaseModel):
    """The ``POST /v1/decide`` response body."""

    answers: dict[str, Answer] = Field(description="Answer per question name.")
    provider: Literal["null", "llm", "jev"] = Field(
        description="Which provider actually served the answer.",
    )
    decision_id: str = Field(description="Server-assigned decision id.")
    latency_ms: int = Field(ge=0, description="Server-observed latency.")


def _canonical_questions_bytes(questions: dict[str, Any]) -> bytes:
    """Serialise a questions object into a byte string that is stable.

    Canonical form (msg-244 disposition #1):

    * Keys are emitted in sorted order at every level (``sort_keys=True``
      in :func:`json.dumps`).
    * ``ensure_ascii=False`` so a non-ASCII character in
      ``instructions`` produces the same bytes on every host — the
      escaped-ASCII form would differ character-for-character between
      Python versions.
    * No extra whitespace (``separators=(",", ":")``).

    ``QuestionSpec`` objects are normalised through
    :meth:`~pydantic.BaseModel.model_dump` first so ``criteria=None``
    lands as ``null`` rather than the Python ``None`` sentinel — a
    Pydantic default that differs across dump modes would otherwise
    silently change the hash.
    """
    normalised: dict[str, Any] = {}
    for name, question in questions.items():
        if isinstance(question, QuestionSpec):
            normalised[name] = question.model_dump(mode="json", exclude_none=False)
        else:
            normalised[name] = question
    return json.dumps(
        normalised,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_questions_hash(questions: dict[str, Any]) -> str:
    """Return the 16-hex-char sha256 prefix used in the decision log.

    Server-computed on every request; never derived from
    ``questions_version`` (Einstein msg-243 objection #1). The
    ``state`` field is deliberately excluded — the hash names the
    question set, and different states asked with the same question set
    must group together for offline evaluation (msg-237 "116 判断点
    リプレイ").
    """
    payload = _canonical_questions_bytes(questions)
    digest = hashlib.sha256(payload).hexdigest()
    return digest[:QUESTIONS_HASH_HEX_LENGTH]


def compute_state_hash(state: str) -> str:
    """Return the 16-hex-char sha256 prefix of the state string.

    Same width as :func:`compute_questions_hash`. Recorded on the
    decision-log row so replays can distinguish "the same question set
    over different states" from "the same state under different question
    versions" without the log carrying the state text itself.
    """
    digest = hashlib.sha256(state.encode("utf-8")).hexdigest()
    return digest[:QUESTIONS_HASH_HEX_LENGTH]

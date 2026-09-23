"""Thin httpx layer for the Jev (TypeSafe) ``systemone`` endpoint.

Internal to :mod:`lexora.decide`. Only :class:`lexora.decide.providers.
JevProvider` calls into this module; nothing else in Lexora (and nothing
on the utilisation side — the 9/18 approval forbids a direct TypeSafe
dependency there) is meant to reach Jev except through ``/v1/decide``.

WIRE FORMAT — source: docs.typesafe.ai/api.md as quoted verbatim by Fermi
in T-decide-jev-provider msg-336 (retrieved 2026-09-23). Design: Bohr
msg-339 (v5) / msg-342 (v6) / msg-344 (v7) / msg-346 (v8).

* One endpoint, ``POST https://api.typesafe.ai/v1/systemone``; every
  question of a request travels in ONE call (so there is no partial
  success and nothing to fan out).
* Headers: ``Authorization: Bearer <API_KEY>``, ``Content-Type:
  application/json``.
* Request: ``{"state", "model", "questions": {qid: {"type",
  "instructions", "criteria"}}}`` — ``state`` / ``model`` / ``questions``
  are required.
* Response: ``{"model": "jev-1.13.0", "answers": {qid: {...}}, "usage":
  {"input_tokens", "output_tokens"}}``. Per answer: noul ``{"noul"}``;
  choice ``{"choice", "probabilities", "confidence"}``; score ``{"score",
  "legend": {"0": ..., "1": ...}, "probabilities", "confidence"}``.
* Errors: 401 bad key / 422 validation (body names the field) / 429 rate
  limited / 529 overloaded.

One part is NOT in the quoted spec: the 422 body layout. Fermi's quote
says only that the body names the offending field. :func:`extract_422_loc`
assumes the FastAPI-style ``{"detail": [{"loc": [...], ...}]}`` layout
that Bohr's v8 test case uses. If the assumption is wrong, the 422 is
still classified as ``invalid_request``; only the optional ``loc``
detail is lost (``None``).

Everything here either returns plain data or raises. It never logs, and
it never copies an upstream body or header into an exception. Turning a
failure into a :class:`~lexora.decide.providers.ProviderError` (with the
exception chain severed) is the provider's job.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Literal

import httpx

from lexora.decide.contract import Answer, QuestionSpec

#: TypeSafe API origin (msg-336).
DEFAULT_BASE_URL = "https://api.typesafe.ai"

#: The single judgment endpoint (msg-336).
SYSTEMONE_PATH = "/v1/systemone"

#: Default ``model`` request field (msg-339 #2): ``jev-latest`` until the
#: logged ``provider_model`` values show which version to pin.
DEFAULT_MODEL = "jev-latest"

StatusCode = Literal[
    "auth", "invalid_request", "rate_limited", "overloaded", "http_status"
]

#: Bounds on the 422 ``loc`` detail (msg-346 v8 #2).
LOC_MAX_ELEMENTS = 16
LOC_MAX_ELEMENT_CHARS = 64


class InvalidResponse(Exception):
    """A 2xx body that does not have the documented shape.

    Carries no message on purpose. The provider turns it into
    ``invalid_response`` and drops it.
    """


def build_body(
    state: str, questions: dict[str, QuestionSpec], model: str
) -> dict[str, Any]:
    """Build the one request body for all questions (msg-339 #1).

    ``criteria=None`` is left out. Extra ``QuestionSpec`` fields from the
    caller go through unchanged (``extra="allow"``: Lexora forwards what
    the caller sent).
    """
    return {
        "state": state,
        "model": model,
        "questions": {
            qid: spec.model_dump(mode="json", exclude_none=True)
            for qid, spec in questions.items()
        },
    }


async def call_systemone(
    body: dict[str, Any],
    *,
    api_key: str,
    timeout_ms: int,
    base_url: str = DEFAULT_BASE_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.Response:
    """POST ``body`` once and return the fully read response.

    ``timeout_ms`` bounds the whole call (msg-339 #1): httpx's own timeout
    covers each phase separately, so an ``asyncio.timeout`` wraps the
    call on top of it. Raises ``httpx.TimeoutException`` /
    ``TimeoutError`` / ``httpx.RequestError`` unchanged; the provider
    classifies them. There is no retry, 429/529 included (msg-339 #4).
    """
    timeout_s = timeout_ms / 1000.0
    async with httpx.AsyncClient(
        base_url=base_url, timeout=timeout_s, transport=transport
    ) as client:
        async with asyncio.timeout(timeout_s):
            return await client.post(
                SYSTEMONE_PATH,
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )


def classify_status(status: int) -> StatusCode | None:
    """Map a status code to its fixed code (msg-339 #4 table). 2xx → None."""
    if 200 <= status < 300:
        return None
    if status == 401:
        return "auth"
    if status == 422:
        return "invalid_request"
    if status == 429:
        return "rate_limited"
    if status == 529:
        return "overloaded"
    return "http_status"


def extract_422_loc(resp: httpx.Response) -> tuple[str, ...] | None:
    """Return the structural path from a 422 body, or ``None``.

    Tolerant (msg-346 v8 #2): a 422 status already decides the code, and
    reading the body only adds optional detail. Only ``loc`` is read.
    ``msg`` / ``input`` / ``ctx`` may echo caller input, so they are
    never touched. Elements go through ``str()``, each is cut to
    :data:`LOC_MAX_ELEMENT_CHARS`, and at most :data:`LOC_MAX_ELEMENTS`
    are kept. The first ``detail`` entry that has a readable ``loc`` wins.

    Only the exceptions a body with the wrong shape can raise are caught
    (Einstein advisory on v8). A ``NameError`` / ``AttributeError`` from a
    bug in this function still propagates. The provider then records it
    as ``internal_error``, so the fallback still happens, but the bug
    shows up in the tests instead of reading as "no loc".
    """
    try:
        payload = resp.json()  # JSONDecodeError / UnicodeDecodeError are ValueErrors
        detail = payload["detail"] if isinstance(payload, dict) else None
        if not isinstance(detail, list):
            return None
        for entry in detail:
            if not isinstance(entry, dict):
                continue
            loc = entry.get("loc")
            if not isinstance(loc, list) or not loc:
                continue
            return tuple(
                str(part)[:LOC_MAX_ELEMENT_CHARS] for part in loc[:LOC_MAX_ELEMENTS]
            )
        return None
    except (KeyError, TypeError, ValueError):
        return None


def _opt_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def extract_meta(payload: object) -> tuple[str | None, int | None, int | None]:
    """Read ``(model, input_tokens, output_tokens)`` leniently (msg-342 #2).

    A field of the wrong type becomes ``None`` and never raises. This runs
    BEFORE the strict answer check, so a billed response whose answers
    are then thrown away still leaves its usage in the log.
    """
    if not isinstance(payload, dict):
        return None, None, None
    model = payload.get("model")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return (
        model if isinstance(model, str) else None,
        _opt_int(usage.get("input_tokens")),
        _opt_int(usage.get("output_tokens")),
    )


def _is_prob(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def normalise_legend(legend: object) -> list[str]:
    """``{"0": "Calm", "1": ...}`` → ``["Calm", ...]`` (msg-339 #3).

    Keys must be exactly the decimal strings ``"0"`` .. ``"n-1"``. Any gap,
    duplicate, non-numeric key, or non-string value raises
    :class:`InvalidResponse`. The Lexora contract (``ScoreAnswer.legend:
    list[str]``) stays unchanged.
    """
    if not isinstance(legend, dict) or not legend:
        raise InvalidResponse
    expected = {str(i) for i in range(len(legend))}
    if set(legend) != expected:
        raise InvalidResponse
    out = [legend[str(i)] for i in range(len(legend))]
    if not all(isinstance(v, str) for v in out):
        raise InvalidResponse
    return out


def _parse_one(qtype: str, raw: object) -> Answer:
    if not isinstance(raw, dict):
        raise InvalidResponse
    answer: Answer = dict(raw)  # unknown keys pass through (contract.Answer)
    if qtype == "noul":
        if not _is_prob(raw.get("noul")):
            raise InvalidResponse
        return answer
    if qtype == "choice":
        probs = raw.get("probabilities")
        if (
            not isinstance(raw.get("choice"), str)
            or not isinstance(probs, dict)
            or not all(isinstance(k, str) and _is_prob(v) for k, v in probs.items())
            or not _is_prob(raw.get("confidence"))
        ):
            raise InvalidResponse
        return answer
    # score
    if not _is_number(raw.get("score")) or not _is_prob(raw.get("confidence")):
        raise InvalidResponse
    answer["legend"] = normalise_legend(raw.get("legend"))
    # ``probabilities`` keeps Jev's index keys ("0", "1", ...); after
    # normalisation ``legend[i]`` lines up with key ``str(i)``.
    return answer


def parse_answers(
    payload: object, questions: dict[str, QuestionSpec]
) -> dict[str, Answer]:
    """Strictly validate ``answers`` for every requested qid (msg-339 #3/#4).

    A missing qid, a non-dict ``answers``, or a malformed answer raises
    :class:`InvalidResponse`. Silently filling in a default would put a
    made-up value in the log under ``provider="jev"`` (msg-260). Answers
    for qids that were not requested are dropped.
    """
    if not isinstance(payload, dict):
        raise InvalidResponse
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise InvalidResponse
    out: dict[str, Answer] = {}
    for qid, spec in questions.items():
        if qid not in answers:
            raise InvalidResponse
        out[qid] = _parse_one(spec.type, answers[qid])
    return out

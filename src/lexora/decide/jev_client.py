"""Thin httpx layer for the Jev (TypeSafe) judgment primitives.

Internal to :mod:`lexora.decide`. Only :class:`lexora.decide.providers.
JevProvider` calls into this module; nothing else in Lexora (and nothing
on the utilisation side — the 9/18 approval forbids a direct TypeSafe
dependency there) is meant to reach Jev except through ``/v1/decide``.

Everything HTTP-shaped — base URL, path, headers, status handling,
response parsing — is closed inside this module so the provider layer
only ever sees a Lexora :data:`~lexora.decide.contract.Answer` or a
:class:`JevCallError` carrying a fixed code.

WIRE-FORMAT ASSUMPTIONS (UNVERIFIED — T-decide-jev-provider):
    The implementer of this module could not read TypeSafe's API
    reference (the ``typesafe-ai`` skill is not in this checkout and
    docs.typesafe.ai was unreachable from the build host). The four
    constants/functions below encode the assumed wire shape and are the
    ONLY place it is encoded:

    * :data:`DEFAULT_BASE_URL` and :data:`_PATH_TEMPLATE` — endpoint.
    * :func:`_auth_headers` — ``Authorization: Bearer <key>``.
    * :func:`_request_body` — ``{"state", "instructions", "criteria",
      ...extra QuestionSpec fields}``.
    * :func:`_parse_answer` — the response carries the Lexora answer keys
      (``noul`` / ``choice``+``probabilities``+``confidence`` /
      ``score``+``legend``+``confidence``) at the top level.

    If any of these is wrong, every Jev call fails with ``http_status``
    or ``invalid_response`` and the route falls back to NullProvider —
    i.e. the failure is safe, but Jev never actually answers. Verify with
    ``pytest -m smoke`` (tests/decide/test_jev_smoke.py) before setting
    ``[decision] primary = "jev"`` anywhere that matters.

No upstream response body or header is ever copied into an exception or
a log line: :class:`JevCallError` carries a code, nothing else.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx

from lexora.decide.contract import Answer, QuestionSpec

#: Assumed TypeSafe API base URL (see module docstring — UNVERIFIED).
DEFAULT_BASE_URL = "https://api.typesafe.ai"

#: Assumed per-primitive path (see module docstring — UNVERIFIED).
_PATH_TEMPLATE = "/v1/{primitive}"

JevErrorCode = Literal["timeout", "http_status", "network", "auth", "invalid_response"]


class JevCallError(Exception):
    """A single Jev call failed. Carries a fixed code only.

    Deliberately holds no response body, header, or URL — the "no raw
    body" rule of the decision log (msg-244 #3) applies to exceptions
    too, because exception text ends up in logs.
    """

    def __init__(self, code: JevErrorCode) -> None:
        super().__init__(code)
        self.code: JevErrorCode = code


def _auth_headers(api_key: str) -> dict[str, str]:
    """Assumed auth header shape (UNVERIFIED)."""
    return {"Authorization": f"Bearer {api_key}"}


def _request_body(state: str, question: QuestionSpec) -> dict[str, Any]:
    """Assumed request body (UNVERIFIED).

    ``criteria`` and any extra QuestionSpec fields are forwarded verbatim
    (contract.py ``extra="allow"``: a caller adopting a new TypeSafe field
    should not have to wait for a Lexora release). ``type`` selects the
    path and is not repeated in the body.
    """
    body = question.model_dump(mode="json", exclude_none=True)
    body.pop("type", None)
    body["state"] = state
    return body


def _is_prob(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _parse_answer(primitive: str, payload: object) -> Answer:
    """Map an (assumed) Jev response onto the Lexora answer shape.

    Strict: a missing or ill-typed field is ``invalid_response``, never a
    silently-defaulted value — a defaulted value would be logged under
    ``provider="jev"`` and pollute the calibration population (msg-260).
    """
    if not isinstance(payload, dict):
        raise JevCallError("invalid_response")
    if primitive == "noul":
        p = payload.get("noul")
        if not _is_prob(p):
            raise JevCallError("invalid_response")
        return {"noul": float(p)}  # type: ignore[arg-type]
    if primitive == "choice":
        choice = payload.get("choice")
        probs = payload.get("probabilities")
        conf = payload.get("confidence")
        if (
            not isinstance(choice, str)
            or not isinstance(probs, dict)
            or not all(isinstance(k, str) and _is_prob(v) for k, v in probs.items())
            or not _is_prob(conf)
        ):
            raise JevCallError("invalid_response")
        return {
            "choice": choice,
            "probabilities": {k: float(v) for k, v in probs.items()},
            "confidence": float(conf),  # type: ignore[arg-type]
        }
    # score
    score = payload.get("score")
    legend = payload.get("legend")
    conf = payload.get("confidence")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not isinstance(legend, list)
        or not all(isinstance(x, str) for x in legend)
        or not _is_prob(conf)
    ):
        raise JevCallError("invalid_response")
    return {"score": float(score), "legend": list(legend), "confidence": float(conf)}  # type: ignore[arg-type]


async def _call(
    client: httpx.AsyncClient,
    api_key: str,
    primitive: Literal["noul", "choice", "score"],
    state: str,
    question: QuestionSpec,
) -> Answer:
    """POST one primitive and return the parsed answer.

    Error classification (Bohr msg-258 §9): timeout → ``timeout``;
    401/403 → ``auth``; any other non-2xx (incl. 429 — no retry in this
    PR, msg-258 §8) → ``http_status``; other transport failure →
    ``network``; non-JSON or wrong shape → ``invalid_response``.
    ``from None`` drops the httpx exception chain so its message (which
    can contain the URL) does not ride along into a traceback.
    """
    try:
        resp = await client.post(
            _PATH_TEMPLATE.format(primitive=primitive),
            json=_request_body(state, question),
            headers=_auth_headers(api_key),
        )
    except httpx.TimeoutException:
        raise JevCallError("timeout") from None
    except httpx.TransportError:
        raise JevCallError("network") from None
    if resp.status_code in (401, 403):
        raise JevCallError("auth")
    if not 200 <= resp.status_code < 300:
        raise JevCallError("http_status")
    try:
        payload = resp.json()
    except ValueError:
        raise JevCallError("invalid_response") from None
    return _parse_answer(primitive, payload)


async def call_noul(
    client: httpx.AsyncClient, api_key: str, state: str, question: QuestionSpec
) -> Answer:
    return await _call(client, api_key, "noul", state, question)


async def call_choice(
    client: httpx.AsyncClient, api_key: str, state: str, question: QuestionSpec
) -> Answer:
    return await _call(client, api_key, "choice", state, question)


async def call_score(
    client: httpx.AsyncClient, api_key: str, state: str, question: QuestionSpec
) -> Answer:
    return await _call(client, api_key, "score", state, question)

"""Unit tests for :class:`lexora.decide.providers.JevProvider` (v8 design).

HTTP is faked with :class:`httpx.MockTransport`, so nothing here reaches
the network. The fake speaks the wire format Fermi quoted from
docs.typesafe.ai/api.md (T-decide-jev-provider msg-336). What these tests
pin down is Lexora's side of that format; they cannot prove TypeSafe
serves it. ``tests/decide/test_jev_smoke.py`` checks that against the
live API.
"""

from __future__ import annotations

import gc
import json
import re
import types
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from lexora.decide import jev_client
from lexora.decide.contract import QuestionSpec
from lexora.decide.providers import (
    DecisionProvider,
    JevProvider,
    ProviderError,
    ProviderResult,
    UpstreamMeta,
)

API_KEY = "sk-test-SECRET-value-0123456789"
STATE = "STATE-SENTINEL-confidential-4c1d"
INPUT_SENTINEL = "INPUT-SENTINEL-echoed-by-422-9b2e"

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def _provider(handler: Handler, timeout_ms: int = 2000, model: str = "jev-latest") -> JevProvider:
    return JevProvider(
        API_KEY,
        timeout_ms=timeout_ms,
        model=model,
        base_url="https://jev.test",
        transport=httpx.MockTransport(handler),
    )


def _respond(status: int, payload: object = None, *, text: str | None = None) -> Handler:
    async def handler(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=payload)

    return handler


def _q(type_: str, instructions: str = "i", criteria: object = None) -> QuestionSpec:
    return QuestionSpec(type=type_, instructions=instructions, criteria=criteria)  # type: ignore[arg-type]


def _ok(answers: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 4},
        **extra,
    }


async def _raises(provider: JevProvider, **questions: QuestionSpec) -> ProviderError:
    with pytest.raises(ProviderError) as excinfo:
        await provider.evaluate(state=STATE, questions=questions or {"q": _q("noul")})
    return excinfo.value


class TestShape:
    def test_name_and_protocol(self) -> None:
        p = _provider(_respond(200, {}))
        assert p.name == "jev"
        assert isinstance(p, DecisionProvider)


class TestSingleCall:
    """Bohr msg-339 #1: one request == one POST to /v1/systemone."""

    async def test_all_questions_in_one_post(self) -> None:
        seen: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200,
                json=_ok(
                    {
                        "a": {"type": "noul", "noul": 0.95},
                        "b": {
                            "type": "choice",
                            "choice": "billing",
                            "probabilities": {"billing": 0.9, "other": 0.1},
                            "confidence": 0.81,
                        },
                    }
                ),
            )

        questions = {
            "a": _q("noul", "Escalate?"),
            "b": _q("choice", "Which?", {"billing": "money", "other": "else"}),
        }
        result = await _provider(handler, model="jev-9.9.9").evaluate(
            state="the state", questions=questions
        )
        assert len(seen) == 1
        req = seen[0]
        assert req.method == "POST"
        assert req.url.path == "/v1/systemone"
        assert req.headers["authorization"] == f"Bearer {API_KEY}"
        assert req.headers["content-type"] == "application/json"
        body = json.loads(req.content)
        assert body["state"] == "the state"
        assert body["model"] == "jev-9.9.9"
        assert set(body["questions"]) == {"a", "b"}
        # criteria=None is omitted, not sent as null.
        assert body["questions"]["a"] == {"type": "noul", "instructions": "Escalate?"}
        assert body["questions"]["b"]["criteria"] == {"billing": "money", "other": "else"}
        assert isinstance(result, ProviderResult)
        assert result.answers["a"]["noul"] == 0.95
        assert result.answers["b"]["choice"] == "billing"
        assert result.upstream == UpstreamMeta("jev-1.13.0", 100, 4)

    async def test_extra_question_fields_forwarded(self) -> None:
        seen: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_ok({"q": {"noul": 0.1}}))

        spec = QuestionSpec.model_validate(
            {"type": "noul", "instructions": "i", "future_field": {"x": 1}}
        )
        await _provider(handler).evaluate(state="s", questions={"q": spec})
        assert json.loads(seen[0].content)["questions"]["q"]["future_field"] == {"x": 1}

    async def test_default_model_is_jev_latest(self) -> None:
        seen: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_ok({"q": {"noul": 0.1}}))

        await JevProvider(
            API_KEY, timeout_ms=1000, base_url="https://jev.test",
            transport=httpx.MockTransport(handler),
        ).evaluate(state="s", questions={"q": _q("noul")})
        assert json.loads(seen[0].content)["model"] == "jev-latest"


class TestScoreLegend:
    """Bohr msg-339 #3: dict legend → list; probabilities keep index keys."""

    async def test_legend_normalised(self) -> None:
        raw = {
            "type": "score",
            "score": 1.05,
            "legend": {"2": "Very angry", "0": "Calm", "1": "Frustrated"},
            "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
            "confidence": 0.92,
        }
        result = await _provider(_respond(200, _ok({"s": raw}))).evaluate(
            state="s", questions={"s": _q("score", criteria=["Calm", "Frustrated", "Very angry"])}
        )
        ans = result.answers["s"]
        assert ans["legend"] == ["Calm", "Frustrated", "Very angry"]
        assert ans["probabilities"] == {"0": 0.0, "1": 0.95, "2": 0.05}
        assert ans["score"] == 1.05
        assert ans["type"] == "score"  # unknown/extra keys pass through

    @pytest.mark.parametrize(
        "legend",
        [
            {"0": "a", "2": "c"},  # gap
            {"1": "a", "2": "b"},  # does not start at 0
            {"0": "a", "x": "b"},  # non-numeric
            {"0": "a", "1": 3},  # non-string value
            {},
            ["a", "b"],
            None,
        ],
    )
    async def test_bad_legend_is_invalid_response(self, legend: object) -> None:
        raw = {"score": 1.0, "legend": legend, "confidence": 0.5}
        err = await _raises(_provider(_respond(200, _ok({"s": raw}))), s=_q("score"))
        assert err.code == "invalid_response"


class TestErrorClassification:
    """Bohr msg-339 #4 table."""

    async def test_timeout(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        err = await _raises(_provider(handler))
        assert err.code == "timeout"
        assert err.exc_type == "ReadTimeout"
        assert err.upstream is None

    async def test_overall_deadline(self) -> None:
        import asyncio

        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(5)
            return httpx.Response(200, json=_ok({"q": {"noul": 0.1}}))

        err = await _raises(_provider(handler, timeout_ms=50))
        assert err.code == "timeout"

    async def test_network(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        err = await _raises(_provider(handler))
        assert err.code == "network"
        assert err.exc_type == "ConnectError"

    @pytest.mark.parametrize(
        ("status", "code"),
        [
            (400, "invalid_request"),  # msg-368 v9 (measured in msg-367)
            (401, "auth"),
            (403, "auth"),  # msg-368 v9 (measured in msg-367)
            (422, "invalid_request"),
            (429, "rate_limited"),
            (529, "overloaded"),
            (500, "http_status"),
            (503, "http_status"),
            (404, "http_status"),
        ],
    )
    async def test_status(self, status: int, code: str) -> None:
        err = await _raises(_provider(_respond(status, {"error": "no"})))
        assert err.code == code
        assert err.upstream is None
        assert err.exc_type is None

    async def test_classify_status_v9_table(self) -> None:
        assert jev_client.classify_status(400) == "invalid_request"
        assert jev_client.classify_status(403) == "auth"

    async def test_400_body_is_not_read(self) -> None:
        """msg-368 v9: a 400 is invalid_request with loc None, even if its
        body happens to carry a FastAPI-style loc; the body is never read."""
        body = {
            "detail": {
                "error_type": "api_usage_error",
                "message": f"Invalid request. {STATE}",
                "loc": ["body", "questions"],
            }
        }
        err = await _raises(_provider(_respond(400, body)))
        assert err.code == "invalid_request"
        assert err.loc is None
        assert STATE not in repr(vars(err))

        listy = {"detail": [{"loc": ["body", "questions"], "input": STATE}]}
        err = await _raises(_provider(_respond(400, listy)))
        assert err.code == "invalid_request"
        assert err.loc is None

    async def test_no_retry_on_429(self) -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(429)

        await _raises(_provider(handler))
        assert calls == 1


class TestInvalidResponse:
    """Bohr msg-342: every 2xx parse failure → invalid_response, never raised raw."""

    @pytest.mark.parametrize(
        "payload",
        [
            {"model": "jev-1", "answers": []},
            {"model": "jev-1", "answers": {"q": {"noul": "high"}}},
            {"model": "jev-1", "answers": {"q": {"noul": 1.5}}},
            {"model": "jev-1", "answers": {"q": [0.5]}},
            {"model": "jev-1", "answers": {}},  # requested qid missing
            {"model": "jev-1"},
            [1, 2, 3],
        ],
    )
    async def test_bad_shape(self, payload: object) -> None:
        err = await _raises(_provider(_respond(200, payload)))
        assert err.code == "invalid_response"

    async def test_not_json(self) -> None:
        err = await _raises(_provider(_respond(200, text="<html>not json")))
        assert err.code == "invalid_response"
        assert err.exc_type == "JSONDecodeError"
        assert err.upstream is None

    async def test_choice_missing_confidence(self) -> None:
        payload = _ok({"c": {"choice": "a", "probabilities": {"a": 1.0}}})
        err = await _raises(_provider(_respond(200, payload)), c=_q("choice"))
        assert err.code == "invalid_response"

    async def test_usage_kept_on_invalid_response(self) -> None:
        """msg-342 #2: billed but discarded → usage/model still carried."""
        payload = _ok({"q": {"noul": "nope"}})
        err = await _raises(_provider(_respond(200, payload)))
        assert err.code == "invalid_response"
        assert err.upstream == UpstreamMeta("jev-1.13.0", 100, 4)

    async def test_broken_usage_with_good_answers_is_success(self) -> None:
        payload = {"model": 7, "answers": {"q": {"noul": 0.2}}, "usage": "lots"}
        result = await _provider(_respond(200, payload)).evaluate(
            state="s", questions={"q": _q("noul")}
        )
        assert result.answers["q"]["noul"] == 0.2
        assert result.upstream == UpstreamMeta(None, None, None)

    async def test_unexpected_parse_exception_is_wrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exception type the parser did not plan for still becomes
        invalid_response (msg-342 #1) with exc_type/where only."""

        def boom(payload: object, questions: object) -> object:
            raise KeyError(STATE)

        monkeypatch.setattr(jev_client, "parse_answers", boom)
        err = await _raises(_provider(_respond(200, _ok({"q": {"noul": 0.1}}))))
        assert err.code == "invalid_response"
        assert err.exc_type == "KeyError"
        assert err.upstream == UpstreamMeta("jev-1.13.0", 100, 4)

    async def test_internal_error_on_request_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200)

        def boom(state: str, questions: object, model: str) -> object:
            raise TypeError(f"cannot serialise {state}")

        monkeypatch.setattr(jev_client, "build_body", boom)
        err = await _raises(_provider(handler))
        assert err.code == "internal_error"
        assert err.exc_type == "TypeError"
        assert calls == 0


class TestExtract422Loc:
    """Bohr msg-346 v8 #2 + Einstein advisory (narrow catch)."""

    async def test_loc_extracted(self) -> None:
        body = {
            "detail": [
                {
                    "loc": ["body", "questions", "q1", "criteria"],
                    "msg": f"bad {INPUT_SENTINEL}",
                    "input": INPUT_SENTINEL,
                    "ctx": {"x": INPUT_SENTINEL},
                }
            ]
        }
        err = await _raises(_provider(_respond(422, body)), q1=_q("noul"))
        assert err.code == "invalid_request"
        assert err.loc == ("body", "questions", "q1", "criteria")
        assert err.exc_type is None
        assert INPUT_SENTINEL not in repr(vars(err))

    async def test_int_elements_and_first_readable_entry(self) -> None:
        body = {"detail": ["junk", {"msg": "no loc"}, {"loc": ["body", 3]}, {"loc": ["x"]}]}
        err = await _raises(_provider(_respond(422, body)))
        assert err.loc == ("body", "3")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"text": "<html>oops"},
            {"payload": {"detail": {"loc": ["body"]}}},
            {"payload": {"detail": [{"msg": "no loc"}]}},
            {"payload": {"error": "x"}},
            {"payload": ["detail"]},
            {"payload": {"detail": []}},
        ],
    )
    async def test_broken_body_keeps_code(self, kwargs: dict[str, Any]) -> None:
        err = await _raises(_provider(_respond(422, **kwargs)))
        assert err.code == "invalid_request"  # NOT internal_error
        assert err.loc is None

    async def test_truncation(self) -> None:
        loc = ["x" * 200] + [f"p{i}" for i in range(40)]
        err = await _raises(_provider(_respond(422, {"detail": [{"loc": loc}]})))
        assert err.loc is not None
        assert len(err.loc) == jev_client.LOC_MAX_ELEMENTS
        assert err.loc[0] == "x" * jev_client.LOC_MAX_ELEMENT_CHARS

    def test_bug_in_extractor_is_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Einstein advisory: only shape-mismatch exceptions are caught."""

        class Resp:
            def json(self) -> object:
                raise AttributeError("typo in extractor")

        with pytest.raises(AttributeError):
            jev_client.extract_422_loc(Resp())  # type: ignore[arg-type]

    async def test_bug_in_extractor_still_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(resp: object) -> object:
            raise NameError("typo")

        monkeypatch.setattr(jev_client, "extract_422_loc", boom)
        err = await _raises(_provider(_respond(422, {"detail": []})))
        assert err.code == "internal_error"
        assert err.exc_type == "NameError"


# ---------------------------------------------------------------------------
# Exception-chain severance (Bohr msg-344 v7, Einstein msg-343)
# ---------------------------------------------------------------------------

_WHERE = re.compile(r"^[^/\\:][^:]*:\d+$")


async def _ok_noul_but_bad(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_ok({"q": {"noul": "x"}}))


async def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


async def _read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _status(n: int) -> Handler:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            n, json={"detail": [{"loc": ["body"], "input": STATE, "msg": API_KEY}]}
        )

    return handler


_ALL_FAILURES: list[Any] = [
    pytest.param(_read_timeout, "timeout", id="timeout"),
    pytest.param(_connect_error, "network", id="network"),
    pytest.param(_status(400), "invalid_request", id="400"),
    pytest.param(_status(401), "auth", id="401"),
    pytest.param(_status(403), "auth", id="403"),
    pytest.param(_status(422), "invalid_request", id="422"),
    pytest.param(_status(429), "rate_limited", id="429"),
    pytest.param(_status(529), "overloaded", id="529"),
    pytest.param(_status(502), "http_status", id="5xx"),
    pytest.param(_ok_noul_but_bad, "invalid_response", id="invalid_response"),
    pytest.param(_respond(200, text=f"{STATE} not json"), "invalid_response", id="not-json"),
    pytest.param(_respond(200, {"answers": []}), "invalid_response", id="answers-list"),
    pytest.param(
        _respond(200, _ok({"q": {"score": 1, "legend": None, "confidence": 0.1}})),
        "invalid_response",
        id="legend-null",
    ),
]

#: Objects the reachability walk does not enter. Traceback/frame: the
#: residual risk accepted in msg-344 v7 #2 (``__traceback__`` reaches
#: ``evaluate``'s frame, which holds ``state`` and ``self``). Types,
#: modules and code/function objects lead to global interpreter state
#: (``sys.modules`` → everything), which says nothing about what the
#: exception itself holds.
_OPAQUE = (
    types.TracebackType,
    types.FrameType,
    type,
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.CodeType,
    types.MethodType,
)


def _reachable_strings(root: object) -> list[str]:
    seen: set[int] = set()
    out: list[str] = []
    stack: list[object] = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, (str, bytes)):
            out.append(obj if isinstance(obj, str) else obj.decode("latin-1"))
            continue
        if obj is not root and isinstance(obj, _OPAQUE):
            continue
        stack.extend(gc.get_referents(obj))
        # Frozen dataclasses with __slots__ are not always traversed; add
        # instance dicts explicitly.
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict) and not isinstance(obj, _OPAQUE):
            stack.append(d)
    return out


class TestChainSevered:
    @pytest.mark.parametrize(("handler", "code"), _ALL_FAILURES)
    async def test_no_chain_and_no_sentinels(self, handler: Handler, code: str) -> None:
        err = await _raises(_provider(handler))
        assert err.code == code
        assert err.__cause__ is None
        assert err.__context__ is None
        for text in (str(err), repr(err), repr(vars(err))):
            assert STATE not in text
            assert API_KEY not in text
        assert str(err) == f"jev:{code}"
        assert set(vars(err)) == {"code", "exc_type", "where", "loc", "upstream"}
        if err.where is not None:
            assert _WHERE.match(err.where), err.where
        if err.exc_type is not None:
            assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", err.exc_type)

    @pytest.mark.parametrize(("handler", "code"), _ALL_FAILURES)
    async def test_nothing_reachable_holds_key_or_state(
        self, handler: Handler, code: str
    ) -> None:
        err = await _raises(_provider(handler))
        for s in _reachable_strings(err):
            assert API_KEY not in s
            assert STATE not in s

    async def test_internal_error_chain_severed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(state: str, questions: object, model: str) -> object:
            raise TypeError(f"cannot serialise {state}")

        monkeypatch.setattr(jev_client, "build_body", boom)
        err = await _raises(_provider(_respond(200, {})))
        assert err.code == "internal_error"
        assert err.__cause__ is None and err.__context__ is None
        assert STATE not in repr(vars(err))
        assert err.where is not None and _WHERE.match(err.where)
        for s in _reachable_strings(err):
            assert STATE not in s

    def test_walk_would_find_a_leak(self) -> None:
        """The reachability walk is not vacuous: a chained exception is found."""
        try:
            raise KeyError(STATE)
        except KeyError as inner:
            try:
                raise ProviderError.__new__(ProviderError) from inner
            except ProviderError as leaky:
                caught = leaky
        assert any(STATE in s for s in _reachable_strings(caught))


class TestApiKeyHygiene:
    async def test_key_absent_from_repr(self) -> None:
        provider = _provider(_respond(200, {}))
        assert API_KEY not in repr(provider)
        assert API_KEY not in repr(vars(provider))

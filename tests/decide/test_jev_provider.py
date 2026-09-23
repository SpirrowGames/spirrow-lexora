"""Unit tests for :class:`lexora.decide.providers.JevProvider`.

HTTP is faked with :class:`httpx.MockTransport`; nothing here reaches the
network. The wire shape the fake speaks is the one assumed in
:mod:`lexora.decide.jev_client` (marked UNVERIFIED there) — these tests pin
Lexora's behaviour around that shape (mapping, error classification,
fail-fast, key hygiene), not TypeSafe's actual API. That is what
``tests/decide/test_jev_smoke.py`` is for.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable

import httpx
import pytest

from lexora.decide.contract import QuestionSpec
from lexora.decide.providers import DecisionProvider, JevProvider, ProviderError

API_KEY = "sk-test-SECRET-value-0123456789"

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def _provider(handler: Handler, timeout_ms: int = 2000) -> JevProvider:
    return JevProvider(
        API_KEY,
        timeout_ms=timeout_ms,
        base_url="https://jev.test",
        transport=httpx.MockTransport(handler),
    )


def _ok(payload: object) -> Handler:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return handler


def _q(type_: str, instructions: str = "i", criteria: object = None) -> QuestionSpec:
    return QuestionSpec(type=type_, instructions=instructions, criteria=criteria)  # type: ignore[arg-type]


async def _raises(provider: JevProvider, **questions: QuestionSpec) -> ProviderError:
    with pytest.raises(ProviderError) as excinfo:
        await provider.evaluate(state="s", questions=questions)
    return excinfo.value


class TestShape:
    def test_name_and_protocol(self) -> None:
        p = _provider(_ok({}))
        assert p.name == "jev"
        assert isinstance(p, DecisionProvider)


class TestHappyPath:
    async def test_noul(self) -> None:
        seen: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"noul": 0.83})

        answers = await _provider(handler).evaluate(
            state="the state", questions={"esc": _q("noul", "Escalate?")}
        )
        assert answers == {"esc": {"noul": 0.83}}
        req = seen[0]
        assert req.url.path == "/v1/noul"
        body = json.loads(req.content)
        assert body == {"state": "the state", "instructions": "Escalate?"}

    async def test_choice_forwards_criteria(self) -> None:
        seen: list[httpx.Request] = []
        payload = {"choice": "b", "probabilities": {"a": 0.2, "b": 0.8}, "confidence": 0.6}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=payload)

        answers = await _provider(handler).evaluate(
            state="s", questions={"pick": _q("choice", criteria=["a", "b"])}
        )
        assert answers == {"pick": payload}
        assert seen[0].url.path == "/v1/choice"
        assert json.loads(seen[0].content)["criteria"] == ["a", "b"]

    async def test_score(self) -> None:
        payload = {"score": 1.4, "legend": ["low", "mid", "high"], "confidence": 0.3}
        answers = await _provider(_ok(payload)).evaluate(
            state="s", questions={"sev": _q("score", criteria=["low", "mid", "high"])}
        )
        assert answers == {"sev": payload}

    async def test_multiple_questions_all_answered(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"noul": 0.1})

        answers = await _provider(handler).evaluate(
            state="s", questions={"a": _q("noul"), "b": _q("noul")}
        )
        assert set(answers) == {"a", "b"}


class TestErrorClassification:
    async def test_timeout(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        err = await _raises(_provider(handler), q=_q("noul"))
        assert err.code == "timeout"
        assert err.discarded == 0

    @pytest.mark.parametrize("status", [401, 403])
    async def test_auth(self, status: int) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": "bad key"})

        assert (await _raises(_provider(handler), q=_q("noul"))).code == "auth"

    @pytest.mark.parametrize("status", [429, 500, 503])
    async def test_http_status(self, status: int) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, text="upstream says no")

        assert (await _raises(_provider(handler), q=_q("noul"))).code == "http_status"

    async def test_network(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        assert (await _raises(_provider(handler), q=_q("noul"))).code == "network"

    async def test_invalid_json(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json")

        assert (await _raises(_provider(handler), q=_q("noul"))).code == "invalid_response"

    @pytest.mark.parametrize(
        ("type_", "payload"),
        [
            ("noul", {}),
            ("noul", {"noul": 1.5}),
            ("noul", {"noul": "high"}),
            ("noul", [0.5]),
            ("choice", {"choice": "a", "probabilities": {"a": 1.0}}),
            ("score", {"score": 1.0, "legend": "low", "confidence": 0.1}),
        ],
    )
    async def test_invalid_shape(self, type_: str, payload: object) -> None:
        err = await _raises(_provider(_ok(payload)), q=_q(type_))
        assert err.code == "invalid_response"


class TestFailFast:
    """Einstein msg-259 #2 / Bohr msg-260 #2: TaskGroup cancels siblings."""

    async def test_one_failure_fails_whole_request_and_cancels_siblings(self) -> None:
        slow_finished = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            instr = json.loads(request.content)["instructions"]
            if instr == "fast-ok":
                return httpx.Response(200, json={"noul": 0.9})
            if instr == "fails":
                await asyncio.sleep(0.05)
                raise httpx.ReadTimeout("timed out", request=request)
            # "slow": would complete long after the failure.
            await asyncio.sleep(5)
            slow_finished.set()
            return httpx.Response(200, json={"noul": 0.1})

        start = time.monotonic()
        err = await _raises(
            _provider(handler),
            a=_q("noul", "fast-ok"),
            b=_q("noul", "fails"),
            c=_q("noul", "slow"),
        )
        elapsed = time.monotonic() - start

        assert err.code == "timeout"
        # "fast-ok" completed (billed upstream) before the failure and was
        # thrown away; "slow" was cancelled, not counted.
        assert err.discarded == 1
        assert "discarded=1" in str(err)
        assert elapsed < 2.0, "sibling was not cancelled"
        assert not slow_finished.is_set()

    async def test_non_jev_exception_is_not_masked(self) -> None:
        """A bug (not a classified upstream failure) must not become a
        silent NullProvider fallback."""

        async def handler(request: httpx.Request) -> httpx.Response:
            raise RuntimeError("bug")

        with pytest.raises(BaseExceptionGroup):
            await _provider(handler).evaluate(state="s", questions={"q": _q("noul")})


class TestApiKeyHygiene:
    async def test_authorization_header_carries_key(self) -> None:
        seen: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"noul": 0.5})

        await _provider(handler).evaluate(state="s", questions={"q": _q("noul")})
        assert seen[0].headers["authorization"] == f"Bearer {API_KEY}"

    async def test_key_absent_from_errors_and_repr(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text=f"bad key {API_KEY}")

        provider = _provider(handler)
        err = await _raises(provider, q=_q("noul"))
        assert API_KEY not in str(err)
        assert API_KEY not in repr(err)
        assert err.__cause__ is None
        assert API_KEY not in repr(provider)
        assert API_KEY not in repr(vars(provider))

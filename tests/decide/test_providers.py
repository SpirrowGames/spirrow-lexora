"""Tests for :mod:`lexora.decide.providers`."""

from __future__ import annotations

import pytest

from lexora.decide.contract import QuestionSpec
from lexora.decide.providers import DecisionProvider, NullProvider, ProviderResult


class TestNullProviderName:
    def test_name_matches_wire_literal(self) -> None:
        """NullProvider.name is exactly ``"null"``.

        The router writes ``provider.name`` verbatim into the decision
        log; a mismatch here would produce rows an offline evaluation
        cannot key on.
        """
        assert NullProvider().name == "null"


class TestNullProviderNoul:
    @pytest.mark.asyncio
    async def test_noul_returns_half(self) -> None:
        """noul answer is 0.5 — the "no signal" value per msg-237."""
        result = (await NullProvider().evaluate(
            state="anything",
            questions={"q": QuestionSpec(type="noul", instructions="i")},
        )).answers
        assert result == {"q": {"noul": 0.5}}


class TestNullProviderChoice:
    @pytest.mark.asyncio
    async def test_choice_with_string_options(self) -> None:
        """A choice question with a list of option names gets uniform mass."""
        result = (await NullProvider().evaluate(
            state="s",
            questions={
                "q": QuestionSpec(type="choice", instructions="i", criteria=["a", "b"])
            },
        )).answers
        answer = result["q"]
        assert answer["choice"] == "a"
        assert set(answer["probabilities"]) == {"a", "b"}
        assert answer["probabilities"]["a"] == pytest.approx(0.5)
        assert answer["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_choice_with_object_options(self) -> None:
        """TypeSafe's ``[{name, description}, ...]`` shape also works."""
        result = (await NullProvider().evaluate(
            state="s",
            questions={
                "q": QuestionSpec(
                    type="choice",
                    instructions="i",
                    criteria=[
                        {"name": "left", "description": "..."},
                        {"name": "right", "description": "..."},
                    ],
                )
            },
        )).answers
        assert set(result["q"]["probabilities"]) == {"left", "right"}

    @pytest.mark.asyncio
    async def test_choice_without_criteria_still_shape_correct(self) -> None:
        """A malformed choice question still gets a shape-correct answer.

        Rationale: NullProvider is the safety-net for the caller, so
        it must never raise. Downstream callers iterating over
        ``answers`` see the same key set they asked for.
        """
        result = (await NullProvider().evaluate(
            state="s",
            questions={"q": QuestionSpec(type="choice", instructions="i")},
        )).answers
        assert result["q"] == {"choice": "", "probabilities": {}, "confidence": 0.0}


class TestNullProviderScore:
    @pytest.mark.asyncio
    async def test_score_returns_zero_and_confidence_zero(self) -> None:
        result = (await NullProvider().evaluate(
            state="s",
            questions={
                "q": QuestionSpec(
                    type="score",
                    instructions="i",
                    criteria=["low", "mid", "high"],
                )
            },
        )).answers
        assert result["q"] == {
            "score": 0.0,
            "legend": ["low", "mid", "high"],
            "confidence": 0.0,
        }


class TestProtocolConformance:
    def test_null_provider_is_a_decision_provider(self) -> None:
        """NullProvider satisfies the runtime-checkable Protocol.

        Follow-up PRs that add ``llm`` / ``jev`` providers will inherit
        the same contract; this test pins the invariant so a future
        edit to the Protocol that quietly drops a member fails here
        instead of at deployment.
        """
        assert isinstance(NullProvider(), DecisionProvider)

    async def test_null_provider_reports_no_upstream(self) -> None:
        """Bohr msg-342 #2: NullProvider returns a ProviderResult with
        ``upstream=None``, which puts NULL in all ``provider_*`` columns."""
        result = await NullProvider().evaluate(
            state="s", questions={"q": QuestionSpec(type="noul", instructions="i")}
        )
        assert isinstance(result, ProviderResult)
        assert result.upstream is None

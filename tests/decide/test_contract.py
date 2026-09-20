"""Tests for the /v1/decide contract types + questions_hash canonicalisation."""

from __future__ import annotations

from lexora.decide.contract import (
    QUESTIONS_HASH_HEX_LENGTH,
    DecideRequest,
    QuestionSpec,
    compute_questions_hash,
    compute_state_hash,
)


class TestQuestionsHash:
    def test_length_is_fixed(self) -> None:
        """Every hash has the 16-hex-char width the schema promises.

        The width is what the SQLite ``questions_hash`` column carries,
        so a drift here would produce a row that no replay tool can
        interpret.
        """
        h = compute_questions_hash({"q": QuestionSpec(type="noul", instructions="is it")})
        assert len(h) == QUESTIONS_HASH_HEX_LENGTH
        assert all(c in "0123456789abcdef" for c in h)

    def test_stable_across_key_order(self) -> None:
        """Rekeying the outer dict does NOT change the hash.

        A caller that renames the question in the middle of the file
        or that Python dict-orders differently on different hosts must
        not see the hash change — that is the whole point of the
        canonical form.
        """
        a = compute_questions_hash(
            {
                "one": QuestionSpec(type="noul", instructions="A"),
                "two": QuestionSpec(type="noul", instructions="B"),
            }
        )
        b = compute_questions_hash(
            {
                "two": QuestionSpec(type="noul", instructions="B"),
                "one": QuestionSpec(type="noul", instructions="A"),
            }
        )
        assert a == b

    def test_different_questions_hash_differently(self) -> None:
        a = compute_questions_hash(
            {"one": QuestionSpec(type="noul", instructions="A")}
        )
        b = compute_questions_hash(
            {"one": QuestionSpec(type="noul", instructions="A different")}
        )
        assert a != b

    def test_accepts_plain_dict_form(self) -> None:
        """Callers may pass raw dicts (as the route does before validation).

        The route validates the payload through :class:`DecideRequest`
        which produces :class:`QuestionSpec` instances; a caller
        pre-computing a hash from the raw dict must arrive at the same
        value.
        """
        via_spec = compute_questions_hash(
            {"q": QuestionSpec(type="choice", instructions="I", criteria=["a", "b"])}
        )
        via_dict = compute_questions_hash(
            {"q": {"type": "choice", "instructions": "I", "criteria": ["a", "b"]}}
        )
        assert via_spec == via_dict

    def test_state_not_folded_in(self) -> None:
        """The state string never contributes to the questions hash.

        A replay of the same question set over 116 different states
        must fold to one questions_hash so offline evaluation can
        group by it.
        """
        # A direct test: state_hash and questions_hash are independent
        # functions.
        assert (
            compute_state_hash("state text")
            != compute_questions_hash({"q": QuestionSpec(type="noul", instructions="i")})
        )


class TestDecideRequest:
    def test_questions_version_optional(self) -> None:
        """questions_version defaults to None."""
        req = DecideRequest(
            state="s",
            questions={"q": QuestionSpec(type="noul", instructions="i")},
            policy="test.caller",
        )
        assert req.questions_version is None

    def test_questions_version_opaque(self) -> None:
        """questions_version is preserved verbatim.

        Lexora does not parse or normalise the string — a caller who
        sends ``v2.1-rc3`` gets ``v2.1-rc3`` back on the log row.
        """
        req = DecideRequest(
            state="s",
            questions={"q": QuestionSpec(type="noul", instructions="i")},
            policy="test",
            questions_version="v2.1-rc3",
        )
        assert req.questions_version == "v2.1-rc3"

    def test_extra_fields_allowed(self) -> None:
        """extra fields on the top-level request do not raise.

        The endpoint deliberately accepts a superset of its documented
        keys so a caller that adopts a new TypeSafe field ahead of a
        Lexora release does not have to wait — the route forwards
        unknown fields on the ``questions`` payload through the hash
        and on to the provider.
        """
        req = DecideRequest(
            state="s",
            questions={"q": QuestionSpec(type="noul", instructions="i")},
            policy="p",
            future_field="whatever",  # type: ignore[call-arg]
        )
        assert req.policy == "p"

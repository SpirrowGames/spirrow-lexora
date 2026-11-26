"""Ingress redaction on ``/v1/decide``.

T-decide-endpoint-bypasses-ingress-redaction (Bohr msg-1154 §6, msg-1156
v2, msg-1158 v3; cleared by Einstein). The rules are the tier path's
(D-1 / D-2): these tests check that the decide path reaches them, what it
sends, what it records, and that a key hit or a scanner failure sends
nothing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lexora.config import Settings
from lexora.decide.config import DecisionSettings
from lexora.decide.contract import QuestionSpec, compute_questions_hash, compute_state_hash
from lexora.ingress import gate
from lexora.ingress.gate import KeyRedacted, check_keys, redact_decide_payload, redact_request
from lexora.main import create_app

PREFIXED = "sk-ant-api03-1234567890abcdefghij"
PLAIN = "correcthorsebattery"
_JEV_KEY = "sk-route-SECRET-0123456789"


def _q(**kw: Any) -> QuestionSpec:
    kw.setdefault("type", "noul")
    kw.setdefault("instructions", "Escalate?")
    return QuestionSpec(**kw)


# ---------------------------------------------------------------------------
# redact_decide_payload
# ---------------------------------------------------------------------------


class TestPayload:
    def test_text_state_matches_the_tier_path(self) -> None:
        text = f"deploy notes\npassword: {PLAIN}\nkey {PREFIXED}\n"
        tier = {"messages": [{"role": "user", "content": text}]}
        redact_request(tier)
        out = redact_decide_payload(text, {"q": _q()})
        assert out.state == tier["messages"][0]["content"]
        assert PLAIN not in out.state and PREFIXED not in out.state
        assert out.state.count("\n") == text.count("\n")

    def test_compact_json_state_is_not_wiped_by_a_key_name(self) -> None:
        state = json.dumps(
            {"max_tokens": 1024, "thread": "T-x", "body": f"see {PREFIXED} please"},
            separators=(",", ":"),
        )
        out = redact_decide_payload(state, {"q": _q()})
        parsed = json.loads(out.state)  # still valid JSON
        assert parsed["thread"] == "T-x"
        assert parsed["body"].startswith("see [REDACTED:")
        assert PREFIXED not in out.state
        assert out.redactions is not None

    def test_nothing_found_keeps_exact_bytes(self) -> None:
        state = '{"a":  1,   "b": "plain words"}'  # odd spacing
        questions = {"q": _q(criteria={"true": "yes", "false": "no"})}
        out = redact_decide_payload(state, questions)
        assert out.state is state
        assert out.redactions is None
        assert out.questions["q"].model_dump() == questions["q"].model_dump()

    def test_questions_every_field_but_type(self) -> None:
        questions = {
            "pick": QuestionSpec(
                type="choice",
                instructions=f"use {PREFIXED}",
                criteria={"a": f"token: {PLAIN}", "b": "fine"},
                hint={"deep": [f"x {PREFIXED}"]},
            )
        }
        out = redact_decide_payload("s", questions)
        dumped = json.dumps(out.questions["pick"].model_dump())
        assert PREFIXED not in dumped and PLAIN not in dumped
        spec = out.questions["pick"]
        assert isinstance(spec, QuestionSpec)
        assert spec.type == "choice"
        assert set(spec.criteria) == {"a", "b"}  # option names untouched
        assert spec.criteria["b"] == "fine"

    def test_one_numbering_across_state_and_questions(self) -> None:
        out = redact_decide_payload(f"k {PREFIXED}", {"q": _q(instructions=f"k {PREFIXED}")})
        label = out.state.split()[-1]
        assert label.startswith("[REDACTED:")
        assert out.questions["q"].instructions.split()[-1] == label

    def test_ordinary_question_sets_are_untouched(self) -> None:
        """Fixed question texts of the noul / choice / score shapes the
        callers use must not trip a rule (a hit would change every call)."""
        questions = {
            "tierc": _q(
                instructions="Does this step need a human decision (cost, dispute, goal)?",
                criteria={"true": "a human must decide", "false": "agents may proceed"},
            ),
            "route": QuestionSpec(
                type="choice",
                instructions="Which participant acts next?",
                criteria={"proposer": "design", "implementer": "code", "naysayer": "review"},
            ),
            "level": QuestionSpec(
                type="score",
                instructions="How engaged is the learner?",
                criteria=["low", "medium", "high"],
            ),
        }
        out = redact_decide_payload("s", questions)
        assert out.redactions is None
        for name, spec in questions.items():
            assert out.questions[name].model_dump() == spec.model_dump()


class TestKeys:
    @pytest.mark.parametrize(
        "questions",
        [
            {f"token: {PLAIN}": _q()},
            {"q": _q(hint={f"x {PREFIXED}": "ordinary"})},
            {"q": _q(metadata={"inner": {f"x {PREFIXED}": "ordinary"}})},
            {"q": QuestionSpec(type="choice", instructions="i", criteria=[{f"x {PREFIXED}": "d"}])},
        ],
        ids=["question-name", "extra-key", "nested-extra-key", "criteria-list-dict"],
    )
    def test_key_hit_in_questions_raises(self, questions: dict[str, QuestionSpec]) -> None:
        with pytest.raises(KeyRedacted) as info:
            redact_decide_payload("s", questions)
        assert PREFIXED not in str(info.value) and PLAIN not in str(info.value)

    def test_key_hit_three_levels_down_in_state_raises(self) -> None:
        state = json.dumps({"a": {"b": [{f"x {PREFIXED}": 1}]}})
        with pytest.raises(KeyRedacted):
            redact_decide_payload(state, {"q": _q()})

    def test_key_names_alone_do_not_hit(self) -> None:
        check_keys({"max_tokens": 1, "api_key": {"password": [{"token": 2}]}, "roster": {}})


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


class _RecordingJev:
    name = "jev"

    def __init__(self, error: Exception | None = None) -> None:
        self.sent: list[tuple[str, dict[str, QuestionSpec]]] = []
        self._error = error

    async def evaluate(self, *, state, questions):  # type: ignore[no-untyped-def]
        from lexora.decide.providers import ProviderResult

        self.sent.append((state, questions))
        if self._error is not None:
            raise self._error
        return ProviderResult(answers={n: {"noul": 0.9} for n in questions})


def _app(monkeypatch: pytest.MonkeyPatch, fake: _RecordingJev, mode: str = "active"):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TYPESAFE_API_KEY", _JEV_KEY)
    app = create_app(
        settings=Settings(decision=DecisionSettings(primary="jev", mode=mode, log_path=":memory:"))  # type: ignore[arg-type]
    )
    app.state.decision_providers["jev"] = fake
    return app


def _timeout() -> Exception:
    from lexora.decide.providers import ProviderError, _Failure

    return ProviderError(_Failure("timeout"))


_STATE = f"note\ntoken: {PLAIN}\n"
_QUESTIONS = {
    "esc": {"type": "noul", "instructions": "Escalate?"},
    "pick": {"type": "choice", "instructions": "Which?", "criteria": {"a": "x", "b": "y"}},
}
_BODY = {"state": _STATE, "questions": _QUESTIONS, "policy": "mindwire.conductor.tierc"}


class TestRoute:
    def test_active_sends_redacted_and_records_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _RecordingJev()
        app = _app(monkeypatch, fake)
        resp = TestClient(app).post("/v1/decide", json=_BODY)
        assert resp.status_code == 200, resp.text
        assert resp.json()["provider"] == "jev"
        ((sent_state, _),) = fake.sent
        assert PLAIN not in sent_state
        (row,) = app.state.decision_log.fetch_all()
        assert row["redactions"] == "assignment:1"
        # Hashes are of what the caller sent, not of what was sent on.
        assert row["state_hash"] == compute_state_hash(_STATE)
        assert row["questions_hash"] == compute_questions_hash(
            {n: QuestionSpec(**q) for n, q in _QUESTIONS.items()}
        )
        assert PLAIN not in json.dumps(row)

    def test_nothing_found_leaves_redactions_null(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _RecordingJev()
        app = _app(monkeypatch, fake)
        TestClient(app).post("/v1/decide", json={**_BODY, "state": "plain"})
        assert fake.sent[0][0] == "plain"
        (row,) = app.state.decision_log.fetch_all()
        assert row["redactions"] is None

    def test_shadow_sends_redacted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _RecordingJev()
        app = _app(monkeypatch, fake, mode="shadow")
        resp = TestClient(app).post("/v1/decide", json=_BODY)
        assert resp.json()["provider"] == "null"
        ((sent_state, _),) = fake.sent
        assert PLAIN not in sent_state
        rows = app.state.decision_log.fetch_all()
        assert len(rows) == 2
        assert {r["redactions"] for r in rows} == {"assignment:1"}

    @staticmethod
    def _fallback_body(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """The answer a Jev timeout gives today (msg-257 §3)."""
        app = _app(monkeypatch, _RecordingJev(error=_timeout()))
        resp = TestClient(app).post("/v1/decide", json=_BODY)
        assert resp.status_code == 200
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider_error"] == "jev:timeout"
        return resp.json()

    @pytest.mark.parametrize("mode", ["active", "shadow"])
    def test_scanner_failure_sends_nothing(self, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
        expected = self._fallback_body(monkeypatch)

        def boom(self, text):  # type: ignore[no-untyped-def]
            raise RuntimeError("scanner bug")

        monkeypatch.setattr(gate.Redactor, "redact", boom)
        fake = _RecordingJev()
        app = _app(monkeypatch, fake, mode=mode)
        resp = TestClient(app).post("/v1/decide", json=_BODY)
        assert resp.status_code == 200, resp.text
        assert fake.sent == []  # neither answering nor shadow
        body = resp.json()
        for key in ("decision_id", "latency_ms"):
            body.pop(key)
            expected.pop(key, None)
        assert body == expected
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider"] == "null"
        assert row["provider_error"] == "ingress:redaction_failed"
        assert row["redactions"] is None

    def test_key_hit_sends_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        expected = self._fallback_body(monkeypatch)
        fake = _RecordingJev()
        app = _app(monkeypatch, fake)
        questions = {**_QUESTIONS, "esc": {**_QUESTIONS["esc"], "meta": {"m": {f"x {PREFIXED}": 1}}}}
        resp = TestClient(app).post("/v1/decide", json={**_BODY, "questions": questions})
        assert resp.status_code == 200, resp.text
        assert fake.sent == []
        body = resp.json()
        for key in ("decision_id", "latency_ms"):
            body.pop(key)
            expected.pop(key, None)
        assert body == expected
        (row,) = app.state.decision_log.fetch_all()
        assert row["provider"] == "null"
        assert row["provider_error"] == "ingress:key_redacted"
        assert PREFIXED not in json.dumps(row)

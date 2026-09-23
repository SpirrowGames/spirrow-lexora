"""PR-2b: the ``type: fallback`` backend (T-naysayer-codex-backend msg-448
B-1 / B-3 / B-5, msg-450 B-7). Codex is the fake CLI from ``test_codex``;
Gemini is an in-process fake."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from lexora.backends.answer_route import current_answer_route
from lexora.backends.base import Backend, UsageSink
from lexora.backends.codex import CodexAuthError, CodexBackend, CodexQuotaError
from lexora.backends.fallback import (
    FALLBACK_ERRORS,
    NOTICE_MAX_CHARS,
    FallbackBackend,
    extract_verdict,
)
from lexora.services.cost_tracker import CostTracker
from lexora.tools import shadow_report
from tests.backends.test_codex import VERSION, make_backend, record_pass

GEMINI_MODEL = "gemini-3.1-pro-preview"
REQUEST = {"model": GEMINI_MODEL, "messages": [{"role": "user", "content": "review this"}]}
GEMINI_TEXT = "Looks fine.\n\nVERDICT: APPROVE"


class FakeGemini(Backend):
    fills_usage_sink = True

    def __init__(self, text: str = GEMINI_TEXT, delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    def _response(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "g",
            "object": "chat.completion",
            "created": 0,
            "model": request.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": self.text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        }

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        await asyncio.sleep(self.delay)
        return self._response(request)

    async def chat_completions_stream(
        self, request: dict[str, Any], usage_sink: UsageSink | None = None
    ) -> AsyncIterator[bytes]:
        self.calls.append(request)
        await asyncio.sleep(self.delay)
        if usage_sink is not None:
            usage_sink.prompt_tokens, usage_sink.completion_tokens = 100, 10
        for delta in ({"role": "assistant", "content": ""}, {"content": self.text}):
            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': delta}]})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def completions(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._response(request)

    async def completions_stream(self, request: dict[str, Any], usage_sink: UsageSink | None = None) -> AsyncIterator[bytes]:
        yield b"data: [DONE]\n\n"

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def list_models(self) -> dict[str, Any]:
        return {"data": []}

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class Clock:
    def __init__(self) -> None:
        self.now = datetime.now(timezone.utc) - timedelta(seconds=1)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
async def wrappers() -> AsyncIterator[list[FallbackBackend]]:
    made: list[FallbackBackend] = []
    yield made
    for w in made:
        await w.close()


def _make(
    wrappers: list[FallbackBackend],
    tmp_path: Path,
    scenario: str = "ok",
    *,
    verified: bool = True,
    mode: str = "fallback",
    gemini: FakeGemini | None = None,
    webhook: str | None = "https://example.invalid/hook",
    ledger: bool = True,
    clock: Clock | None = None,
    timeout: float = 30.0,
) -> FallbackBackend:
    codex = make_backend(tmp_path, scenario, timeout=timeout)
    if verified:
        record_pass(codex)
    w = FallbackBackend(
        name="naysayer-fb",
        primary=codex,
        fallback=gemini or FakeGemini(),
        fallback_name="gemini",
        mode=mode,
        webhook_url=webhook,
        clock=clock or Clock(),
    )
    w.tier_label = "naysayer"
    if ledger:
        w.attach_ledger(CostTracker(tmp_path / "costs.db"))
    wrappers.append(w)
    return w


def _capture_posts(w: FallbackBackend, ok: bool = True) -> list[str]:
    posted: list[str] = []

    async def post(text: str) -> bool:
        posted.append(text)
        return ok

    w._post = post  # type: ignore[method-assign]
    return posted


async def _sends(w: FallbackBackend) -> None:
    while w._send_tasks:
        await asyncio.gather(*list(w._send_tasks))
        await asyncio.sleep(0)  # let the done-callbacks empty the set


async def _drain(agen: Any) -> list[bytes]:
    return [c async for c in agen]


def _text(resp: dict[str, Any]) -> str:
    return resp["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# B-1: what falls back, and what does not
# --------------------------------------------------------------------------


class TestFallbackDecision:
    async def test_codex_answers_when_open(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path)
        resp = await w.chat_completions(REQUEST)
        assert _text(resp).startswith("REVIEW: ")
        assert resp["model"] == "gpt-5-codex"  # the tier's Gemini model never reaches codex
        assert w.fallback.calls == []  # type: ignore[attr-defined]
        route = current_answer_route()
        assert route is not None and (route.backend, route.answered_by, route.model) == ("codex", "codex", "gpt-5-codex")

    @pytest.mark.parametrize(
        ("scenario", "verified", "reason"),
        [
            pytest.param("ok", False, "verification_missing", id="gate-closed"),
            pytest.param("quota", True, "quota", id="quota"),
            pytest.param("tool_use", True, "tool_use_violation", id="latch"),
            pytest.param("exit0_no_terminal", True, "unverifiable_run", id="unverifiable-latch"),
        ],
    )
    async def test_falls_back_to_gemini(
        self, tmp_path: Path, wrappers: list, scenario: str, verified: bool, reason: str
    ) -> None:
        w = _make(wrappers, tmp_path, scenario, verified=verified)
        resp = await w.chat_completions(REQUEST)
        assert _text(resp) == GEMINI_TEXT  # a latched run's text is never returned
        assert w.fallback.calls[0]["model"] == GEMINI_MODEL  # type: ignore[attr-defined]
        route = current_answer_route()
        assert route is not None and (route.backend, route.answered_by) == ("gemini", "gemini-fallback")
        assert w._last_reason == reason

    async def test_timeout_falls_back(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, "sleep", timeout=1.0)
        assert _text(await w.chat_completions(REQUEST)) == GEMINI_TEXT
        assert w._last_reason == "timeout"

    async def test_launch_failure_falls_back(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path)
        w.primary._wrap = lambda inner, workdir: [str(tmp_path / "no-such-bwrap")]  # type: ignore[method-assign]
        assert _text(await w.chat_completions(REQUEST)) == GEMINI_TEXT
        assert w._last_reason == "launch_failed"

    async def test_auth_failure_is_not_in_the_b1_list(self, tmp_path: Path, wrappers: list) -> None:
        """B-1 lists quota / launch / not-verified / timeout / latch only."""
        w = _make(wrappers, tmp_path)

        async def auth_fails(*_a: Any, **_k: Any) -> Any:
            # A classified auth failure (terminal event present). The fake
            # CLI's "auth" scenario has no terminal event, so it is a D-1d
            # latch -- which does fall back.
            raise CodexAuthError("not logged in")

        w.primary.chat_completions = auth_fails  # type: ignore[method-assign]
        with pytest.raises(CodexAuthError):
            await w.chat_completions(REQUEST)
        assert w.fallback.calls == []  # type: ignore[attr-defined]
        assert not issubclass(CodexAuthError, FALLBACK_ERRORS)

    async def test_quota_hold_skips_codex_on_the_next_request(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, "quota")
        await w.chat_completions(REQUEST)
        started_before = len(w.primary.state_store.runs())
        await w.chat_completions(REQUEST)
        assert len(w.primary.state_store.runs()) == started_before  # no codex run during the hold
        assert w._last_reason == "quota_hold"

    async def test_one_version_check_per_request(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path)
        calls = [0]

        async def version() -> str:
            calls[0] += 1
            return VERSION

        w.primary._codex_version = version  # type: ignore[method-assign]
        await w.chat_completions(REQUEST)
        await _drain(w.chat_completions_stream(REQUEST, UsageSink()))
        assert calls[0] == 2


class TestNoSwitchAfterTheFirstByte:
    """B-1: Gemini only while nothing has been sent to the client."""

    async def test_stream_codex_answers(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path)
        sink = UsageSink()
        body = b"".join(await _drain(w.chat_completions_stream(REQUEST, sink)))
        assert b"REVIEW: " in body and w.fallback.calls == []  # type: ignore[attr-defined]
        assert (sink.prompt_tokens, sink.completion_tokens) == (120, 7)

    async def test_stream_failure_before_first_byte_falls_back(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, "quota")
        body = b"".join(await _drain(w.chat_completions_stream(REQUEST, UsageSink())))
        assert b"VERDICT: APPROVE" in body and b"REVIEW" not in body

    async def test_error_after_first_byte_propagates(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path)

        async def one_then_fail(*_a: Any, **_k: Any) -> AsyncIterator[bytes]:
            yield b"data: first\n\n"
            raise CodexQuotaError("late failure")

        w.primary.chat_completions_stream = one_then_fail  # type: ignore[method-assign]
        received: list[bytes] = []
        with pytest.raises(CodexQuotaError):
            async for chunk in w.chat_completions_stream(REQUEST, UsageSink()):
                received.append(chunk)
        assert received == [b"data: first\n\n"]
        assert w.fallback.calls == []  # type: ignore[attr-defined]

    async def test_codex_emits_nothing_before_its_run_finishes(self, tmp_path: Path, wrappers: list) -> None:
        """The premise of the rule: codex's stream yields only after
        ``_run_gated`` returned, so every codex failure precedes byte 1."""
        codex = make_backend(tmp_path, "quota")
        record_pass(codex)
        stream = codex.chat_completions_stream(REQUEST, UsageSink())
        with pytest.raises(CodexQuotaError):
            await stream.__anext__()


# --------------------------------------------------------------------------
# B-3: notifications
# --------------------------------------------------------------------------


class TestNotifications:
    async def test_start_once_then_end_once(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, "quota")
        posted = _capture_posts(w)
        for _ in range(3):
            await w.chat_completions(REQUEST)
        await _sends(w)
        assert [p.split(":")[0] for p in posted] == ["[Lexora naysayer] fallback STARTED"]
        # codex comes back
        w.primary._quota_hold_until = None
        w.primary._scenario = "ok"  # type: ignore[attr-defined]
        await w.chat_completions(REQUEST)
        await w.chat_completions(REQUEST)
        await _sends(w)
        assert [p.split(":")[0] for p in posted] == [
            "[Lexora naysayer] fallback STARTED",
            "[Lexora naysayer] fallback ENDED",
        ]
        assert w._fallback_since is None

    async def test_reminder_every_six_hours_with_ledger_counts(self, tmp_path: Path, wrappers: list) -> None:
        clock = Clock()
        w = _make(wrappers, tmp_path, "quota", clock=clock)
        posted = _capture_posts(w)
        await w.chat_completions(REQUEST)
        await _sends(w)
        # two fallback rows in the ledger, written as a route would
        for _ in range(2):
            w.ledger.record(model=GEMINI_MODEL, endpoint="/v1/chat/completions", tokens_input=1000, tokens_output=100, backend="naysayer-fb")
        clock.now += timedelta(hours=5, minutes=59)
        await w.notice_tick()
        assert len(posted) == 1
        clock.now += timedelta(minutes=1)
        await w.notice_tick()
        assert len(posted) == 2 and "fallback CONTINUING" in posted[1]
        assert "gemini_calls=2" in posted[1]
        assert "quota_hold_until=" in posted[1] and "quota_hold_until=unknown" not in posted[1]
        await w.notice_tick()
        assert len(posted) == 2  # not again until another 6 h

    async def test_failed_post_is_retried_next_cycle(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, "quota")
        posted = _capture_posts(w, ok=False)
        resp = await w.chat_completions(REQUEST)  # the request is not blocked
        assert _text(resp) == GEMINI_TEXT
        await _sends(w)
        assert w._pending_notice is not None
        assert posted  # one failed attempt
        retried = _capture_posts(w)  # the webhook recovers
        await w.notice_tick()
        assert retried and "fallback STARTED" in retried[0]
        assert w._pending_notice is None

    async def test_no_webhook_sends_nothing(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, "quota", webhook=None)
        posted = _capture_posts(w)
        await w.chat_completions(REQUEST)
        await _sends(w)
        assert posted == [] and w._fallback_since is not None

    async def test_notice_is_short_and_carries_only_the_b3_fields(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, "quota")
        w._last_reason = "x" * 2000
        w._fallback_since = datetime.now(timezone.utc)
        text = w._notice_text("CONTINUING")
        assert len(text) <= NOTICE_MAX_CHARS
        w._last_reason = "quota"
        keys = [part.split("=")[0] for part in w._notice_text("STARTED").split(": ", 1)[1].split()]
        assert keys == ["reason", "fallback_since", "gemini_calls", "gemini_cost_usd", "quota_hold_until"]

    async def test_real_post_failure_does_not_raise(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, webhook="http://127.0.0.1:9/unreachable")
        assert await w._post("x") is False


# --------------------------------------------------------------------------
# B-2: ledger rows (the route-level half is tests/api/test_fallback_routes.py)
# --------------------------------------------------------------------------


class TestLedger:
    def _rows(self, db: Path) -> list[dict[str, Any]]:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM request_costs ORDER BY id")]

    async def test_codex_row_is_free_and_priced(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path)
        await w.chat_completions(REQUEST)
        w.ledger.record(model=GEMINI_MODEL, endpoint="/v1/chat/completions", tokens_input=120, tokens_output=7, backend="naysayer-fb", tier="naysayer")
        row = self._rows(tmp_path / "costs.db")[-1]
        assert (row["backend"], row["answered_by"], row["model"]) == ("codex", "codex", "gpt-5-codex")
        assert (row["cost_usd"], row["pricing_known"]) == (0.0, 1)

    async def test_fallback_row_is_priced_as_gemini(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, verified=False)
        await w.chat_completions(REQUEST)
        w.ledger.record(model=GEMINI_MODEL, endpoint="/v1/chat/completions", tokens_input=1000, tokens_output=100, backend="naysayer-fb")
        row = self._rows(tmp_path / "costs.db")[-1]
        assert (row["backend"], row["answered_by"], row["model"]) == ("gemini", "gemini-fallback", GEMINI_MODEL)
        assert row["cost_usd"] > 0 and row["pricing_known"] == 1
        assert w.ledger.fallback_totals(w._fallback_since) == (1, row["cost_usd"])

    def test_rows_without_a_route_keep_null(self, tmp_path: Path) -> None:
        ledger = CostTracker(tmp_path / "c.db")
        ledger.record(model=GEMINI_MODEL, endpoint="/x", tokens_input=1, tokens_output=1, backend="gemini")
        row = self._rows(tmp_path / "c.db")[-1]
        assert (row["backend"], row["answered_by"]) == ("gemini", None)

    def test_migration_adds_the_column_to_an_old_db(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE request_costs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
                "model TEXT NOT NULL, backend TEXT, endpoint TEXT NOT NULL, user_id TEXT, tokens_input INTEGER NOT NULL "
                "DEFAULT 0, tokens_output INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0.0, "
                "duration_seconds REAL, success INTEGER NOT NULL DEFAULT 1)"
            )
            conn.execute("INSERT INTO request_costs (timestamp, model, endpoint) VALUES ('t', 'm', '/e')")
        CostTracker(db)
        CostTracker(db)  # idempotent
        rows = self._rows(db)
        assert rows[0]["answered_by"] is None


# --------------------------------------------------------------------------
# B-5: shadow mode
# --------------------------------------------------------------------------


async def _shadow_idle(w: FallbackBackend) -> None:
    if w._shadow_task is not None:
        await w._shadow_task


class TestShadow:
    async def test_gemini_answers_codex_runs_and_is_compared(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, mode="shadow")
        resp = await w.chat_completions(REQUEST)
        assert _text(resp) == GEMINI_TEXT
        route = current_answer_route()
        assert route is not None and (route.backend, route.answered_by) == ("gemini", None)
        await _shadow_idle(w)
        [row] = w.ledger.shadow_comparisons()
        assert (row["tier"], row["gemini_verdict"], row["codex_verdict"], row["codex_reason"]) == (
            "naysayer", "APPROVE", "unparsed", None,
        )
        assert row["gemini_seconds"] is not None and row["codex_seconds"] is not None
        assert "REVIEW" not in json.dumps(row) and "VERDICT" not in json.dumps(row)  # no text stored
        with sqlite3.connect(tmp_path / "costs.db") as conn:
            shadow_rows = conn.execute(
                "SELECT backend, answered_by, cost_usd, pricing_known, endpoint FROM request_costs"
            ).fetchall()
        assert shadow_rows == [("codex", "codex-shadow", 0.0, 1, "shadow")]

    async def test_stream_shadow(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, mode="shadow")
        body = b"".join(await _drain(w.chat_completions_stream(REQUEST, UsageSink())))
        assert b"VERDICT: APPROVE" in body
        await _shadow_idle(w)
        assert w.ledger.shadow_comparisons()[0]["gemini_verdict"] == "APPROVE"

    async def test_one_shadow_at_a_time(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, "sleep", mode="shadow", timeout=2.0)
        await w.chat_completions(REQUEST)
        await w.chat_completions(REQUEST)
        await w.chat_completions(REQUEST)
        assert w.shadow_skipped == 2
        assert w.primary.inflight_runs() == 1  # the shadow run counts (B-5)
        await _shadow_idle(w)
        [row] = w.ledger.shadow_comparisons()
        assert (row["codex_verdict"], row["codex_reason"]) == ("error", "timeout")

    async def test_no_shadow_run_during_a_hold(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, mode="shadow")
        w.primary._quota_hold_until = datetime.now(timezone.utc) + timedelta(hours=1)
        await w.chat_completions(REQUEST)
        await _shadow_idle(w)
        assert w.ledger.shadow_comparisons() == []
        assert w.primary.state_store.runs() == []

    async def test_gemini_failure_is_recorded_as_error(self, tmp_path: Path, wrappers: list) -> None:
        class Broken(FakeGemini):
            async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
                raise RuntimeError("gemini down")

        w = _make(wrappers, tmp_path, mode="shadow", gemini=Broken())
        with pytest.raises(RuntimeError):
            await w.chat_completions(REQUEST)
        await _shadow_idle(w)
        assert w.ledger.shadow_comparisons()[0]["gemini_verdict"] == "error"


class TestVerdicts:
    @pytest.mark.parametrize(
        ("text", "verdict"),
        [
            ("...\nVERDICT: REQUEST_CHANGES (ci=success)\n...\nVERDICT: REQUEST_CHANGES", "REQUEST_CHANGES"),
            ("VERDICT: APPROVE", "APPROVE"),
            ("**Objections**\n- **class: correctness** (blocks: true)", "blocking"),
            ("**Objections**\n- class: structure (blocks: false)", "non_blocking"),
            ("**Endorsements**\n- fine", "non_blocking"),
            ("just prose", "unparsed"),
            ("", "unparsed"),
        ],
    )
    def test_extract(self, text: str, verdict: str) -> None:
        assert extract_verdict(text) == verdict


class TestShadowReport:
    def test_summary(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        ledger = CostTracker(tmp_path / "costs.db")
        for g, c, reason in [
            ("APPROVE", "APPROVE", None),
            ("APPROVE", "REQUEST_CHANGES", None),
            ("blocking", "blocking", None),
            ("unparsed", "APPROVE", None),
            ("APPROVE", "error", "quota"),
        ]:
            ledger.record_shadow_comparison(tier="naysayer", gemini_verdict=g, codex_verdict=c, codex_reason=reason, gemini_seconds=2.0, codex_seconds=4.0)
        assert shadow_report.main(["--db", str(tmp_path / "costs.db")]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["compared"] == 3 and out["disagreements"] == 1
        assert out["agreement_rate"] == round(2 / 3, 4)
        assert out["unparsed"] == {"gemini": 1, "codex": 0}
        assert out["codex_failures"] == {"quota": 1}
        assert out["median_seconds"] == {"gemini": 2.0, "codex": 4.0}


class TestStatusFields:
    async def test_mode_follows_availability(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, verified=False)
        availability = await w.primary.codex_availability()
        fields = w.status_fields(availability)
        assert fields["mode"] == "fallback" and fields["fallback_since"] is None  # before any request
        record_pass(w.primary)
        assert w.status_fields(await w.primary.codex_availability())["mode"] == "codex"

    async def test_close_cancels_background_work(self, tmp_path: Path) -> None:
        w = _make([], tmp_path, "sleep", mode="shadow", timeout=30.0)
        await w.chat_completions(REQUEST)
        await asyncio.sleep(0.3)
        await w.close()
        assert w._shadow_task is not None and w._shadow_task.done()

"""``mode: codex_only`` of the fallback wrapper (2026-11-10, Takahito: the
naysayer path calls no Gemini). Every case where ``fallback`` mode would call
Gemini is refused with ``NaysayerCodexUnavailable``, the notices still go out
(worded as "codex unavailable"), and no other backend exists to be called."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lexora.backends.answer_route import current_answer_route
from lexora.backends.base import BackendError
from lexora.backends.codex import CodexUnsupportedInputError
from lexora.backends.fallback import FallbackBackend, NaysayerCodexUnavailable
from lexora.config import FallbackSettings
from lexora.services.cost_tracker import CostTracker
from tests.backends.test_codex import make_backend, record_pass
from tests.backends.test_fallback import (
    REQUEST,
    Clock,
    FakeGemini,
    _capture_posts,
    _drain,
    _sends,
    _text,
)


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
    clock: Clock | None = None,
    timeout: float = 30.0,
) -> FallbackBackend:
    codex = make_backend(tmp_path, scenario, timeout=timeout)
    if verified:
        record_pass(codex)
    w = FallbackBackend(
        name="naysayer-codex",
        primary=codex,
        fallback=None,
        fallback_name=None,
        mode="codex_only",
        webhook_url="https://example.invalid/hook",
        clock=clock or Clock(),
        codex_timeout_s=timeout,
        slot_wait_s=600.0,
    )
    w.tier_label = "naysayer"
    w.attach_ledger(CostTracker(tmp_path / "costs.db"))
    wrappers.append(w)
    return w


class TestConstruction:
    def test_codex_only_refuses_a_fallback_backend(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="takes no fallback"):
            FallbackBackend(
                name="x",
                primary=make_backend(tmp_path, "ok"),
                fallback=FakeGemini(),
                fallback_name="gemini",
                mode="codex_only",
            )

    @pytest.mark.parametrize("mode", ["fallback", "shadow"])
    def test_other_modes_still_need_one(self, tmp_path: Path, mode: str) -> None:
        with pytest.raises(ValueError, match="needs a fallback backend"):
            FallbackBackend(name="x", primary=make_backend(tmp_path, "ok"), fallback=None, fallback_name=None, mode=mode)

    def test_budget_has_no_gemini_floor(self, tmp_path: Path) -> None:
        # 600 + 270 = 870 <= 900 holds only because codex_only reserves no floor.
        w = FallbackBackend(
            name="x",
            primary=make_backend(tmp_path, "ok"),
            fallback=None,
            fallback_name=None,
            mode="codex_only",
            slot_wait_s=600.0,
            codex_timeout_s=270.0,
        )
        assert w.fallback_floor_s == 0.0


class TestSettings:
    def test_codex_only_loads_without_fallback_or_floor(self) -> None:
        s = FallbackSettings(primary="codex", mode="codex_only", slot_wait_s=600, codex_timeout_s=270)
        assert s.fallback is None and s.fallback_floor_s is None and s.floor_s == 0.0

    def test_codex_only_refuses_a_named_fallback(self) -> None:
        with pytest.raises(ValidationError, match="'fallback' must not be set"):
            FallbackSettings(primary="codex", fallback="gemini", mode="codex_only")

    def test_codex_only_refuses_a_floor(self) -> None:
        with pytest.raises(ValidationError, match="'fallback_floor_s' must not be set"):
            FallbackSettings(primary="codex", mode="codex_only", fallback_floor_s=600)

    def test_codex_only_budget_is_codex_alone(self) -> None:
        with pytest.raises(ValidationError, match="fallback budget does not hold"):
            FallbackSettings(primary="codex", mode="codex_only", slot_wait_s=700, codex_timeout_s=270)

    @pytest.mark.parametrize("mode", ["fallback", "shadow"])
    def test_other_modes_require_fallback_and_default_the_floor(self, mode: str) -> None:
        with pytest.raises(ValidationError, match="requires 'fallback'"):
            FallbackSettings(primary="codex", mode=mode)
        s = FallbackSettings(primary="codex", fallback="gemini", mode=mode)
        assert s.fallback_floor_s == 600.0 and s.floor_s == 600.0


class TestRefusal:
    async def test_codex_answers_when_open(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path)
        resp = await w.chat_completions(REQUEST)
        assert _text(resp).startswith("REVIEW: ")
        route = current_answer_route()
        assert route is not None and (route.backend, route.answered_by) == ("codex", "codex")

    @pytest.mark.parametrize(
        ("scenario", "verified", "reason"),
        [
            pytest.param("ok", False, "verification_missing", id="gate-closed"),
            pytest.param("quota", True, "quota", id="quota"),
            pytest.param("tool_use", True, "tool_use_violation", id="latch"),
            pytest.param("exit0_no_terminal", True, "unverifiable_run", id="unverifiable-latch"),
            pytest.param("failed", True, "codex_failed", id="unclassified"),
        ],
    )
    async def test_refused_where_fallback_mode_calls_gemini(
        self, tmp_path: Path, wrappers: list, scenario: str, verified: bool, reason: str
    ) -> None:
        w = _make(wrappers, tmp_path, scenario, verified=verified)
        with pytest.raises(NaysayerCodexUnavailable) as info:
            await w.chat_completions(REQUEST)
        assert info.value.reason == reason
        assert isinstance(info.value, BackendError)  # the route answers 502
        assert f"reason={reason}" in str(info.value)
        assert w.fallback is None

    async def test_stream_is_refused_before_any_byte(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, verified=False)
        with pytest.raises(NaysayerCodexUnavailable):
            await _drain(w.chat_completions_stream(REQUEST))

    async def test_timeout_is_refused(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, "sleep", timeout=1.0)
        with pytest.raises(NaysayerCodexUnavailable, match="reason=timeout"):
            await w.chat_completions(REQUEST)

    async def test_caller_error_stays_the_callers(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path)

        async def fails(*_a: Any, **_k: Any) -> Any:
            raise CodexUnsupportedInputError("bad input")

        w.primary.chat_completions = fails  # type: ignore[method-assign]
        with pytest.raises(CodexUnsupportedInputError):
            await w.chat_completions(REQUEST)

    @pytest.mark.parametrize("call", ["completions", "embeddings"])
    async def test_what_codex_does_not_serve_is_refused(self, tmp_path: Path, wrappers: list, call: str) -> None:
        w = _make(wrappers, tmp_path)
        with pytest.raises(NaysayerCodexUnavailable, match="reason=unsupported"):
            await getattr(w, call)({"model": "x", "prompt": "p", "input": "i"})

    async def test_health_is_codex_alone(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, verified=False)
        assert await w.health_check() is False
        record_pass(w.primary)
        assert await w.health_check() is True

    async def test_nothing_is_billed_to_gemini(self, tmp_path: Path, wrappers: list) -> None:
        w = _make(wrappers, tmp_path, verified=False)
        with pytest.raises(NaysayerCodexUnavailable):
            await w.chat_completions(REQUEST)
        with sqlite3.connect(tmp_path / "costs.db") as conn:
            rows = conn.execute("SELECT backend FROM request_costs WHERE backend LIKE '%gemini%'").fetchall()
        assert rows == []


class TestNotices:
    async def test_started_counts_refusals_and_ended_follows_codex(self, tmp_path: Path, wrappers: list) -> None:
        clock = Clock()
        w = _make(wrappers, tmp_path, verified=False, clock=clock)
        posted = _capture_posts(w)
        for _ in range(3):
            with pytest.raises(NaysayerCodexUnavailable):
                await w.chat_completions(REQUEST)
        await _sends(w)
        assert len(posted) == 1
        assert posted[0].startswith(
            "[Lexora naysayer] codex unavailable STARTED (no fallback: requests are refused): "
            "reason=verification_missing "
        )
        assert "refused_requests=1" in posted[0]
        assert "gemini" not in posted[0].lower()
        assert w._refused_since == 3
        fields = w.status_fields(await w.primary.codex_availability())
        assert fields["mode"] == "unavailable" and fields["configured_mode"] == "codex_only"
        assert fields["refused_since"] == 3 and fields["fallback_calls"] is None

        record_pass(w.primary)
        await w.chat_completions(REQUEST)
        clock.now += timedelta(minutes=16)
        await w.notice_tick()
        await _sends(w)
        assert posted[-1].startswith("[Lexora naysayer] codex unavailable ENDED")
        assert "refused_requests=3" in posted[-1]
        assert w.status_fields(await w.primary.codex_availability())["mode"] == "codex"

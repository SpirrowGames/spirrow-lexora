"""``GET /v1/naysayer/status`` (T-naysayer-codex-backend msg-421 S-1,
msg-424 S-1', msg-429 S-1'')."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexora.api.routes import get_backend_router, router
from lexora.backends.codex import CodexLaunchError
from lexora.backends.gemini import GeminiBackend
from tests.backends.test_codex import VERSION, make_backend, record_pass


def _client(backend: Any, *, tier: bool = True, name: str = "codex") -> TestClient:
    backend_router = MagicMock()
    backend_router.is_tier.side_effect = lambda t: tier and t == "naysayer"
    backend_router.get_backend_name_for_model.return_value = name
    backend_router.get_backend_by_name.side_effect = lambda n: backend if n == name else None
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend_router] = lambda: backend_router
    return TestClient(app)


def _status(backend: Any) -> dict[str, Any]:
    response = _client(backend).get("/v1/naysayer/status")
    assert response.status_code == 200
    return response.json()


def test_open_codex(tmp_path: Path) -> None:
    backend = make_backend(tmp_path)
    record_pass(backend)
    backend._in_flight.update({7, 8})
    assert _status(backend) == {
        "tier": "naysayer",
        "backend": "codex",
        "primary": "codex",
        "codex": {"codex_disabled_reason": None, "quota_hold_until": None, "inflight_runs": 2},
    }


def test_never_verified(tmp_path: Path) -> None:
    body = _status(make_backend(tmp_path))
    assert body["codex"]["codex_disabled_reason"] == "verification_missing"


def test_state_unreadable_is_200(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_backend(tmp_path)
    record_pass(backend)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(backend.state_store, "uncleared_violations", boom)
    assert _status(backend)["codex"]["codex_disabled_reason"] == "state_unreadable"


def test_launch_failure_is_200(tmp_path: Path) -> None:
    backend = make_backend(tmp_path)
    record_pass(backend)

    async def broken() -> str:
        raise CodexLaunchError("codex --version failed")

    backend._codex_version = broken  # type: ignore[method-assign]
    assert _status(backend)["codex"]["codex_disabled_reason"] == "launch_failed"


def test_hold_and_a_latch_are_both_visible(tmp_path: Path) -> None:
    backend = make_backend(tmp_path)
    record_pass(backend)
    until = datetime.now(timezone.utc) + timedelta(hours=1)
    backend._quota_hold_until = until
    assert _status(backend)["codex"] == {
        "codex_disabled_reason": "quota_hold",
        "quota_hold_until": until.isoformat(),
        "inflight_runs": 0,
    }
    backend.state_store.record_run_started("codex", "dead-instance")
    codex = _status(backend)["codex"]
    assert codex["codex_disabled_reason"] == "run_unfinished"
    assert codex["quota_hold_until"] == until.isoformat()


def test_body_carries_no_content(tmp_path: Path) -> None:
    backend = make_backend(tmp_path)
    record_pass(backend)
    text = _client(backend).get("/v1/naysayer/status").text
    assert VERSION not in text
    assert str(tmp_path) not in text.replace("\\\\", "\\")


def test_gemini_tier_has_no_codex_block() -> None:
    gemini = MagicMock(spec=GeminiBackend)
    body = _client(gemini, name="gemini").get("/v1/naysayer/status").json()
    assert body == {"tier": "naysayer", "backend": "gemini", "primary": "gemini", "codex": None}


def test_no_naysayer_tier_is_404(tmp_path: Path) -> None:
    response = _client(make_backend(tmp_path), tier=False).get("/v1/naysayer/status")
    assert response.status_code == 404

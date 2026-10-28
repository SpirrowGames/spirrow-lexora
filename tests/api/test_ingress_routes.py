"""The ingress gate through the real route handlers (ADR-14 §7.3,
spirrow-mindwire#482; T-naysayer-codex-backend msg-800 §4-§7 / msg-802 §5).

Every route that sends a tier request is driven through FastAPI and read back
out of a real SQLite ledger: which backend was called, what text it received,
and what the row says (``backend``, ``answered_by``, ``redactions``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexora.backends.fallback import FallbackBackend
from lexora.ingress import gate
from lexora.ingress.gate import N3_ROUTE_HEADER, TierIngress
from lexora.ingress.redact import Redactor
from lexora.services.cost_tracker import CostTracker
from tests.api.test_trace_id import ALL_ROUTES, _app, _backend
from tests.backends.test_codex import make_backend, record_pass
from tests.backends.test_fallback import GEMINI_MODEL, FakeGemini

SECRET = "sk-proj-abcdefghij1234567890KLMNOP"
CODEX_TIER = TierIngress(redact=True, codex_chain=True, unrouted="gemini")
REDACT_ONLY = TierIngress(redact=True, codex_chain=False, unrouted=None)
PLAIN = TierIngress(redact=False, codex_chain=False, unrouted=None)


def _wire(app: FastAPI, ingress: TierIngress | None, unrouted: Any = None) -> None:
    from lexora.api.routes import get_backend_router

    router = app.dependency_overrides[get_backend_router]()
    router.tier_ingress = MagicMock(return_value=ingress)
    router.get_backend_by_name = MagicMock(side_effect=lambda name: unrouted if name == "gemini" else None)


def _rows(db: Path) -> list[tuple[str, str | None, str | None]]:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT backend, answered_by, redactions FROM request_costs ORDER BY id").fetchall()


def _called(backend: MagicMock) -> bool:
    return any(
        getattr(backend, m).called
        for m in ("chat_completions", "completions", "embeddings", "chat_completions_stream", "completions_stream")
    )


def _sent(backend: MagicMock) -> dict[str, Any]:
    for m in ("chat_completions", "completions", "embeddings", "chat_completions_stream", "completions_stream"):
        mock = getattr(backend, m)
        if mock.called:
            return mock.call_args.args[0]
    raise AssertionError("backend not called")


class TestN3Route:
    @pytest.mark.parametrize(("path", "body"), ALL_ROUTES)
    def test_v1_reaches_the_tier_backend(self, tmp_path: Path, path: str, body: dict[str, Any]) -> None:
        primary, unrouted = _backend(), _backend()
        app = _app(primary, CostTracker(tmp_path / "c.db"), backend_name="naysayer-codex")
        _wire(app, CODEX_TIER, unrouted)
        response = TestClient(app).post(path, json=body, headers={N3_ROUTE_HEADER: "v1"})
        assert response.status_code == 200, response.text
        assert _called(primary) and not _called(unrouted)
        assert _rows(tmp_path / "c.db") == [("naysayer-codex", None, None)]

    @pytest.mark.parametrize(("path", "body"), ALL_ROUTES)
    @pytest.mark.parametrize("value", [None, "", "v2", "V1", "v1.0", "v1; tier=naysayer", "x" * 80])
    def test_anything_but_v1_is_answered_by_unrouted(
        self, tmp_path: Path, path: str, body: dict[str, Any], value: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = MagicMock()
        monkeypatch.setattr(gate, "logger", log)
        primary, unrouted = _backend(), _backend()
        app = _app(primary, CostTracker(tmp_path / "c.db"), backend_name="naysayer-codex")
        _wire(app, CODEX_TIER, unrouted)
        headers = {} if value is None else {N3_ROUTE_HEADER: value}
        response = TestClient(app).post(path, json=body, headers=headers)
        assert response.status_code == 200, response.text
        assert _called(unrouted) and not _called(primary)
        assert _rows(tmp_path / "c.db") == [("gemini", "gemini", None)]
        expected: dict[str, Any] = {"n3_route": "absent", "tier": body["model"], "unrouted": "gemini"}
        if value is not None:
            expected["n3_route_value"] = value[:32]
        log.info.assert_any_call("n3_route_unrouted", **expected)

    @pytest.mark.parametrize(("path", "body"), ALL_ROUTES)
    @pytest.mark.parametrize("ingress", [REDACT_ONLY, PLAIN, None], ids=["redact-only", "plain", "not-a-tier"])
    def test_header_is_ignored_without_codex(
        self, tmp_path: Path, path: str, body: dict[str, Any], ingress: TierIngress | None
    ) -> None:
        primary, unrouted = _backend(), _backend()
        app = _app(primary, CostTracker(tmp_path / "c.db"))
        _wire(app, ingress, unrouted)
        response = TestClient(app).post(path, json=body)
        assert response.status_code == 200, response.text
        assert _called(primary) and not _called(unrouted)


class TestRedaction:
    @pytest.mark.parametrize(("path", "body"), ALL_ROUTES)
    def test_every_route_sends_redacted_text_and_records_counts(
        self, tmp_path: Path, path: str, body: dict[str, Any]
    ) -> None:
        primary = _backend()
        app = _app(primary, CostTracker(tmp_path / "c.db"))
        _wire(app, REDACT_ONLY)
        body = _with_secret(body)
        response = TestClient(app).post(path, json=body)
        assert response.status_code == 200, response.text
        sent = repr(_sent(primary))
        assert SECRET not in sent
        assert "[REDACTED:key#1]" in sent
        rows = _rows(tmp_path / "c.db")
        assert len(rows) == 1 and rows[0][2] is not None and rows[0][2].startswith("key:")
        assert SECRET not in rows[0][2]

    def test_messages_system_and_blocks_are_redacted(self, tmp_path: Path) -> None:
        primary = _backend()
        app = _app(primary, CostTracker(tmp_path / "c.db"))
        _wire(app, REDACT_ONLY)
        body = {
            "model": "naysayer-gemini",
            "max_tokens": 16,
            "system": [{"type": "text", "text": f"sys {SECRET}"}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": f"a {SECRET}"}, {"type": "text", "text": "b"}]},
                {"role": "assistant", "content": f"c {SECRET}"},
            ],
        }
        assert TestClient(app).post("/v1/messages", json=body).status_code == 200
        sent = _sent(primary)
        assert SECRET not in repr(sent)
        assert sent["messages"][0] == {"role": "system", "content": "sys [REDACTED:key#1]"}
        assert _rows(tmp_path / "c.db")[0][2] == "key:3"

    @pytest.mark.parametrize(("path", "body"), ALL_ROUTES)
    def test_redact_false_sends_text_unchanged(self, tmp_path: Path, path: str, body: dict[str, Any]) -> None:
        primary = _backend()
        app = _app(primary, CostTracker(tmp_path / "c.db"))
        _wire(app, PLAIN)
        response = TestClient(app).post(path, json=_with_secret(body))
        assert response.status_code == 200, response.text
        assert SECRET in repr(_sent(primary))
        assert _rows(tmp_path / "c.db")[0][2] is None

    @pytest.mark.parametrize(("path", "body"), ALL_ROUTES)
    def test_scanner_failure_calls_no_backend(
        self, tmp_path: Path, path: str, body: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(self: Redactor, text: str) -> str:
            raise ValueError("scanner bug")

        monkeypatch.setattr(Redactor, "redact", boom)
        primary = _backend()
        app = _app(primary, CostTracker(tmp_path / "c.db"))
        _wire(app, REDACT_ONLY)
        response = TestClient(app, raise_server_exceptions=False).post(path, json=_with_secret(body))
        assert response.status_code == 500
        assert "scanner bug" not in response.text and SECRET not in response.text
        assert not _called(primary)
        assert _rows(tmp_path / "c.db") == []

    def test_a_later_unredacted_request_does_not_inherit_the_count(self, tmp_path: Path) -> None:
        primary = _backend()
        ledger = CostTracker(tmp_path / "c.db")
        app = _app(primary, ledger)
        _wire(app, REDACT_ONLY)
        body = {"model": "naysayer-gemini", "messages": [{"role": "user", "content": SECRET}]}
        with TestClient(app) as client:
            client.post("/v1/chat/completions", json=body)
            _wire(app, PLAIN)
            client.post("/v1/chat/completions", json=body)
        assert [r[2] for r in _rows(tmp_path / "c.db")] == ["key:1", None]

    def test_row_records_the_extent_next_to_the_counts(self, tmp_path: Path) -> None:
        """msg-898 P-4: ``redaction_extent`` sits beside ``redactions`` (which
        keeps its format), holds sizes only, and is NULL where it is."""
        primary = _backend()
        app = _app(primary, CostTracker(tmp_path / "c.db"))
        _wire(app, REDACT_ONLY)
        body = {"model": "naysayer-gemini", "messages": [{"role": "user", "content": f"a {SECRET}\nb {SECRET}"}]}
        with TestClient(app) as client:
            client.post("/v1/chat/completions", json=body)
            _wire(app, PLAIN)
            client.post("/v1/chat/completions", json=body)
        with sqlite3.connect(tmp_path / "c.db") as conn:
            rows = conn.execute("SELECT redactions, redaction_extent FROM request_costs ORDER BY id").fetchall()
        assert rows == [("key:2", f"key:2/{2 * len(SECRET)}"), (None, None)]


class TestShadowRow:
    def test_shadow_row_carries_the_redactions(self, tmp_path: Path) -> None:
        """The shadow run is a task spawned by the wrapper; it inherits the
        context the ingress wrapper stamped, so its row says what codex got."""
        codex = make_backend(tmp_path, "ok")
        record_pass(codex)
        w = FallbackBackend(
            name="naysayer-fb", primary=codex, fallback=FakeGemini(), fallback_name="gemini", mode="shadow", webhook_url=None
        )
        w.tier_label = "naysayer"
        ledger = CostTracker(tmp_path / "c.db")
        w.attach_ledger(ledger)
        app = _app(w, ledger, backend_name="naysayer-fb", resolved=GEMINI_MODEL)
        _wire(app, CODEX_TIER, FakeGemini())
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "naysayer", "messages": [{"role": "user", "content": SECRET}]},
                headers={N3_ROUTE_HEADER: "v1"},
            )
            assert response.status_code == 200, response.text
            if w._shadow_task is not None:
                client.portal.call(_await, w._shadow_task)
        with sqlite3.connect(tmp_path / "c.db") as conn:
            rows = conn.execute(
                "SELECT endpoint, redactions, redaction_extent FROM request_costs ORDER BY endpoint"
            ).fetchall()
        extent = f"key:1/{len(SECRET)}"
        assert sorted(rows) == [("/v1/chat/completions", "key:1", extent), ("shadow", "key:1", extent)]


async def _await(task: Any) -> None:
    await task


def _with_secret(body: dict[str, Any]) -> dict[str, Any]:
    body = dict(body)
    if "messages" in body:
        body["messages"] = [{"role": "user", "content": f"review this: {SECRET}"}]
    if "prompt" in body:
        body["prompt"] = f"review this: {SECRET}"
    if "input" in body:
        body["input"] = f"embed: {SECRET}"
    return body

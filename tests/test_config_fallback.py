"""``type: fallback`` configuration and router wiring (T-naysayer-codex-backend
msg-448 B-1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lexora.backends.fallback import FallbackBackend
from lexora.config import BackendSettings, RoutingSettings, VLLMSettings
from lexora.services.router import BackendRouter


def _backends(tmp_path: Path, **fallback: Any) -> dict[str, Any]:
    return {
        "gemini": {"type": "gemini", "url": "https://example.invalid", "models": ["gemini-3.1-pro-preview"]},
        "codex": {
            "type": "codex",
            "models": ["gpt-5-codex"],
            "codex": {"codex_home": str(tmp_path / "home"), "state_db_path": str(tmp_path / "codex.db")},
        },
        "naysayer-fb": {"type": "fallback", "fallback": {"primary": "codex", "fallback": "gemini", **fallback}},
    }


def _routing(tmp_path: Path, **fallback: Any) -> RoutingSettings:
    return RoutingSettings(
        enabled=True,
        default_backend="gemini",
        backends=_backends(tmp_path, **fallback),
        tiers={"naysayer": {"backend": "naysayer-fb", "model": "gemini-3.1-pro-preview"}},
    )


class TestSettings:
    def test_section_required_on_type_fallback(self) -> None:
        with pytest.raises(ValidationError, match="requires a 'fallback:' section"):
            BackendSettings(type="fallback")

    def test_section_refused_elsewhere(self) -> None:
        with pytest.raises(ValidationError, match="only read by backend type 'fallback'"):
            BackendSettings(type="gemini", fallback={"primary": "a", "fallback": "b"})

    def test_mode_is_fallback_or_shadow(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            _routing(tmp_path, mode="both")
        assert _routing(tmp_path, mode="shadow").backends["naysayer-fb"].fallback.mode == "shadow"  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        ("patch", "message"),
        [
            ({"primary": "nope"}, "primary 'nope' is not a configured backend"),
            ({"primary": "gemini"}, "primary 'gemini' is type 'gemini', expected 'codex'"),
            ({"fallback": "codex"}, "fallback 'codex' is type 'codex', expected 'gemini'"),
        ],
    )
    def test_references_are_checked(self, tmp_path: Path, patch: dict[str, str], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            _routing(tmp_path, **patch)


class TestRouter:
    def test_router_builds_the_wrapper_over_the_named_backends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LEXORA_FALLBACK_WEBHOOK_URL", raising=False)
        router = BackendRouter(_routing(tmp_path, mode="shadow"), VLLMSettings())
        wrapper = router.get_backend_for_model("naysayer")
        assert isinstance(wrapper, FallbackBackend)
        assert wrapper.primary is router.get_backend_by_name("codex")
        assert wrapper.fallback is router.get_backend_by_name("gemini")
        assert (wrapper.fallback_name, wrapper.mode, wrapper.tier_label) == ("gemini", "shadow", "naysayer")
        assert wrapper.webhook_url is None
        assert router.resolve_model("naysayer") == "gemini-3.1-pro-preview"

    def test_webhook_comes_from_the_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXORA_FALLBACK_WEBHOOK_URL", "https://example.invalid/hook")
        router = BackendRouter(_routing(tmp_path), VLLMSettings())
        wrapper = router.get_backend_by_name("naysayer-fb")
        assert isinstance(wrapper, FallbackBackend) and wrapper.webhook_url == "https://example.invalid/hook"

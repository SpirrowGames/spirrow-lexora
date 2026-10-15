"""Tier ingress settings and startup checks S-1..S-4 (ADR-14 §7.3,
spirrow-mindwire#482; T-naysayer-codex-backend msg-800 §1-§3 revised in
msg-802 §2-§3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lexora.config import RoutingSettings, TierSettings, create_settings

SHIPPED = Path(__file__).resolve().parent.parent / "config" / "lexora_config.yaml"


def _backends(tmp_path: Path) -> dict[str, Any]:
    return {
        "gemini": {"type": "gemini", "url": "https://example.invalid", "models": ["gemini-3.1-pro-preview"]},
        "gemini2": {"type": "gemini", "url": "https://example.invalid", "models": ["gemini-3.1-pro-preview"]},
        "codex": {
            "type": "codex",
            "models": ["gpt-6.1-sol"],
            "codex": {"codex_home": str(tmp_path / "home"), "state_db_path": str(tmp_path / "codex.db")},
        },
        "wrapped": {"type": "fallback", "fallback": {"primary": "codex", "fallback": "gemini"}},
    }


def _routing(tmp_path: Path, **tiers: dict[str, Any]) -> RoutingSettings:
    return RoutingSettings(enabled=True, default_backend="gemini", backends=_backends(tmp_path), tiers=tiers)


class TestRedactIsRequired:
    def test_missing_redact_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="redact"):
            TierSettings(backend="gemini")  # type: ignore[call-arg]

    def test_explicit_false_is_accepted_without_codex(self, tmp_path: Path) -> None:
        r = _routing(tmp_path, plain={"backend": "gemini", "redact": False})
        assert r.tiers["plain"].redact is False


class TestStartupChecks:
    @pytest.mark.parametrize("backend", ["codex", "wrapped"], ids=["direct-codex", "behind-wrapper"])
    def test_s1_codex_chain_with_redact_false(self, tmp_path: Path, backend: str) -> None:
        with pytest.raises(ValidationError, match=r"tier 't'.*'redact' must be true \(S-1\)"):
            _routing(tmp_path, t={"backend": backend, "redact": False, "unrouted": "gemini"})

    @pytest.mark.parametrize("backend", ["codex", "wrapped"], ids=["direct-codex", "behind-wrapper"])
    def test_s2_codex_chain_without_unrouted(self, tmp_path: Path, backend: str) -> None:
        with pytest.raises(ValidationError, match=r"tier 't'.*'unrouted' must name.*\(S-2\)"):
            _routing(tmp_path, t={"backend": backend, "redact": True})

    def test_s3_unrouted_not_configured(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match=r"'unrouted' names 'nope', which is not a configured backend \(S-3\)"):
            _routing(tmp_path, t={"backend": "wrapped", "redact": True, "unrouted": "nope"})

    @pytest.mark.parametrize("unrouted", ["codex", "wrapped"])
    def test_s3_unrouted_chain_has_codex(self, tmp_path: Path, unrouted: str) -> None:
        with pytest.raises(ValidationError, match=r"whose backend chain contains codex \(S-3\)"):
            _routing(tmp_path, t={"backend": "wrapped", "redact": True, "unrouted": unrouted})

    def test_s4_unrouted_on_a_tier_without_codex(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match=r"has no codex, so it would never be used \(S-4\)"):
            _routing(tmp_path, t={"backend": "gemini", "redact": True, "unrouted": "gemini2"})

    @pytest.mark.parametrize("backend", ["codex", "wrapped"], ids=["direct-codex", "behind-wrapper"])
    def test_codex_in_the_chain_is_not_itself_refused(self, tmp_path: Path, backend: str) -> None:
        r = _routing(tmp_path, t={"backend": backend, "redact": True, "unrouted": "gemini"})
        assert r.chain_has_codex(backend)
        assert not r.chain_has_codex("gemini")

    def test_stray_unrouted_on_the_wrapper_is_refused(self, tmp_path: Path) -> None:
        backends = _backends(tmp_path)
        backends["wrapped"]["fallback"]["unrouted"] = "gemini"
        with pytest.raises(ValidationError, match="unrouted"):
            RoutingSettings(enabled=True, default_backend="gemini", backends=backends, tiers={})


class TestShippedConfig:
    def test_passes_every_startup_check_with_the_specified_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXORA_FRONTIER_MODEL", raising=False)
        monkeypatch.delenv("LEXORA_CODEX_RO_BINDS", raising=False)
        tiers = create_settings(SHIPPED).routing.tiers
        assert {name: t.redact for name, t in tiers.items()} == {
            "light": False,
            "medium": False,
            "heavy": False,
            "naysayer": True,
            "naysayer-gemini": True,
            "frontier": False,
        }
        assert {name: t.unrouted for name, t in tiers.items() if t.unrouted} == {"naysayer": "gemini"}

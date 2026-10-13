"""The codex backend's tool-disabling setting (measured 2026-10-13).

Two halves: the feature / tool flags in ``TOOL_DISABLE_FEATURE_OVERRIDES``
and a model catalog written by ``lexora.tools.codex_catalog`` and named by
the ``model_catalog_json`` override. Covered here: the catalog content is
part of the config hash, the override is read the way the CLI reads it, the
catalog edits, and the catalog tool. Whether the real CLI then declares no
tool is what ``verify_codex`` (V-2/3) measures on the host.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from lexora.backends import codex as codex_mod
from lexora.backends.codex import (
    CATALOG_KEYS_REMOVED,
    CATALOG_KEYS_SET,
    CODEX_MODEL_PIN,
    MODEL_CATALOG_OVERRIDE_KEY,
    TOOL_DISABLE_FEATURE_OVERRIDES,
    TOOL_DISABLE_OVERRIDE_KEYS,
    CodexBackend,
    apply_catalog_tool_edits,
    model_catalog_digest,
    model_catalog_path,
    override_key,
    override_value,
)
from lexora.backends.codex_verification import CodexStateStore
from lexora.tools import codex_catalog as cc


def _backend(tmp_path: Path, overrides: list[str], home: str = "/srv/codex") -> CodexBackend:
    return CodexBackend(
        codex_home=home, state_store=CodexStateStore(tmp_path / "s.db"), models=["m"], cli_overrides=overrides
    )


def _bundled(**changes: Any) -> dict[str, Any]:
    entry = {
        "slug": CODEX_MODEL_PIN.model,
        "context_window": CODEX_MODEL_PIN.context_window,
        "max_context_window": CODEX_MODEL_PIN.max_context_window,
        "tool_mode": "code_mode_only",
        "multi_agent_version": "v2",
        "apply_patch_tool_type": "freeform",
        "experimental_supported_tools": ["send_user_message_async", "clock"],
        "base_instructions": "You are Codex.",
        "shell_type": "unified_exec",
    }
    entry.update(changes)
    return {"models": [{"slug": "gpt-5.5", "context_window": 1}, entry]}


# --------------------------------------------------------------------------
# The registered keys
# --------------------------------------------------------------------------


def test_disable_keys_are_the_catalog_override_and_every_flag() -> None:
    assert TOOL_DISABLE_OVERRIDE_KEYS == {MODEL_CATALOG_OVERRIDE_KEY} | {override_key(o) for o in TOOL_DISABLE_FEATURE_OVERRIDES}
    assert all(override_value(o) == "false" for o in TOOL_DISABLE_FEATURE_OVERRIDES)


def test_control_clone_drops_the_catalog_and_the_flags(tmp_path: Path) -> None:
    overrides = ['model_catalog_json="/srv/codex/cat.json"', *TOOL_DISABLE_FEATURE_OVERRIDES, "features.other=true"]
    clone = _backend(tmp_path, overrides).control_clone("/tmp/dummy")
    assert clone.cli_overrides == ["features.other=true"]


# --------------------------------------------------------------------------
# Reading the override and hashing the catalog
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ('model_catalog_json="/srv/codex/cat.json"', "/srv/codex/cat.json"),
        ("model_catalog_json=/srv/codex/cat.json", "/srv/codex/cat.json"),
        ("model_catalog_json = '/srv/x y.json'", "/srv/x y.json"),
        ("features.goals=false", "false"),
    ],
)
def test_override_value_reads_like_the_cli(override: str, value: str) -> None:
    assert override_value(override) == value


def test_catalog_path_last_override_wins() -> None:
    assert model_catalog_path(["a=b"]) is None
    assert model_catalog_path(['model_catalog_json="/a"', 'MODEL_CATALOG_JSON="/b"']) == "/b"


def test_catalog_content_is_part_of_the_config_hash(tmp_path: Path) -> None:
    catalog = tmp_path / "cat.json"
    catalog.write_text('{"models": []}')
    # as_posix: the override value is parsed like a TOML string, so a Windows
    # path's backslashes would be read as escapes on a dev host.
    backend = _backend(tmp_path, [f'model_catalog_json="{catalog.as_posix()}"'])
    first = backend.config_hash()
    assert backend.config_hash() == first
    catalog.write_text('{"models": [{"slug": "x"}]}')
    assert backend.config_hash() != first
    catalog.unlink()
    assert model_catalog_digest(backend.cli_overrides) == "unreadable:FileNotFoundError"
    assert backend.config_hash() != first


def test_hash_without_a_catalog_override_is_unchanged(tmp_path: Path) -> None:
    """No ``model_catalog_json``: the payload has no catalog field, so hashes
    recorded before this field existed still match."""
    backend = _backend(tmp_path, ["a=b"])
    expected = codex_mod.hashlib.sha256(
        json.dumps(
            {
                "template": codex_mod.COMMAND_TEMPLATE_VERSION,
                "codex_bin": "codex",
                "codex_home": "/srv/codex",
                "bwrap_bin": "bwrap",
                "ro_binds": [],
                "cli_overrides": ["a=b"],
                "model_mapping": {},
                "models": ["m"],
                "env_allowlist": list(codex_mod.LOCALE_ENV_ALLOWLIST),
                "sandbox_path": codex_mod.SANDBOX_PATH,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert backend.config_hash() == expected


# --------------------------------------------------------------------------
# Catalog edits and the catalog tool
# --------------------------------------------------------------------------


def test_catalog_edits_remove_and_overwrite_only_the_tool_keys() -> None:
    entry = _bundled()["models"][1]
    edited = apply_catalog_tool_edits(entry)
    assert not set(CATALOG_KEYS_REMOVED) & set(edited)
    assert {k: edited[k] for k in CATALOG_KEYS_SET} == dict(CATALOG_KEYS_SET)
    untouched = set(entry) - set(CATALOG_KEYS_REMOVED) - set(CATALOG_KEYS_SET)
    assert {k: edited[k] for k in untouched} == {k: entry[k] for k in untouched}
    assert entry["tool_mode"] == "code_mode_only"  # the input is not mutated


def test_build_catalog_keeps_only_the_pinned_model() -> None:
    catalog = cc.build_catalog(_bundled())
    assert [m["slug"] for m in catalog["models"]] == [CODEX_MODEL_PIN.model]
    assert "tool_mode" not in catalog["models"][0]


@pytest.mark.parametrize(
    "bundled",
    [
        {"models": []},
        _bundled(context_window=1),
        _bundled(max_context_window=1),
        {"models": [_bundled()["models"][1], _bundled()["models"][1]]},
    ],
)
def test_build_catalog_refuses_a_catalog_that_does_not_match_the_pin(bundled: dict[str, Any]) -> None:
    with pytest.raises(cc.CatalogError):
        cc.build_catalog(bundled)


def test_render_is_deterministic() -> None:
    assert cc.render(cc.build_catalog(_bundled())) == cc.render(cc.build_catalog(_bundled()))


@pytest.mark.parametrize("path", ["/elsewhere/cat.json", "/srv/codex/../cat.json", "relative.json", "/srv/codex"])
def test_target_must_lie_inside_codex_home(tmp_path: Path, path: str) -> None:
    with pytest.raises(cc.CatalogError):
        cc.target_path(_backend(tmp_path, [f'model_catalog_json="{path}"']))


def test_target_requires_the_override(tmp_path: Path) -> None:
    with pytest.raises(cc.CatalogError):
        cc.target_path(_backend(tmp_path, list(TOOL_DISABLE_FEATURE_OVERRIDES)))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="target_path validates POSIX paths: the catalog is only ever written on the Linux (bwrap) host",
)
class TestMain:
    @pytest.fixture
    def backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CodexBackend:
        home = tmp_path / "home"
        home.mkdir()
        backend = _backend(tmp_path, [f'model_catalog_json="{home}/cat.json"'], home=str(home))
        monkeypatch.setattr("lexora.tools.verify_codex._load_backend", lambda config, name: backend)
        monkeypatch.setattr(cc, "bundled_catalog", lambda codex_bin: _bundled())
        return backend

    def test_writes_600_then_reports_up_to_date(self, backend: CodexBackend, capsys: pytest.CaptureFixture[str]) -> None:
        path = Path(backend.codex_home) / "cat.json"
        assert cc.main(["--check"]) == 1
        assert cc.main([]) == 0
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert path.read_bytes() == cc.render(cc.build_catalog(_bundled()))
        assert cc.main(["--check"]) == 0
        assert cc.main([]) == 0
        assert "already up to date" in capsys.readouterr().out

    def test_check_reports_a_changed_file(self, backend: CodexBackend) -> None:
        cc.main([])
        (Path(backend.codex_home) / "cat.json").write_text("{}")
        assert cc.main(["--check"]) == 1

    def test_a_pin_mismatch_writes_nothing(self, backend: CodexBackend, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cc, "bundled_catalog", lambda codex_bin: _bundled(context_window=1))
        assert cc.main([]) == 2
        assert not (Path(backend.codex_home) / "cat.json").exists()

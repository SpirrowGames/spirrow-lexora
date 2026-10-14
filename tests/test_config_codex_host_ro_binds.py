"""``LEXORA_CODEX_RO_BINDS``: host-specific codex sandbox binds live in the
per-host deployment step, not in the shared config.

T-naysayer-codex-backend, human decision after msg-783 ("move the
host-specific path out of the shared config and into the per-host
deployment step"). The systemd-resolved stub path is a fact about one
host; the shared YAML ships no such bind, and the host adds it through
the unit's ``EnvironmentFile``. These tests pin:

* the shipped config carries no host-specific ``ro_binds``;
* the variable appends to every codex backend's YAML ``ro_binds``;
* its entries go through the same sandbox-widening validator as YAML
  entries (so the env surface cannot widen the sandbox);
* the entries reach the config hash (so changing the variable closes the
  verification gate);
* empty elements, or a config with no codex backend, refuse to start.
"""

from pathlib import Path

import pytest

from lexora.config import CODEX_HOST_RO_BINDS_ENV, create_settings

SHIPPED_CONFIG = Path(__file__).resolve().parent.parent / "config" / "lexora_config.yaml"
STUB = "/run/systemd/resolve/stub-resolv.conf"


def _codex_backends(settings):
    return {
        name: backend
        for name, backend in settings.routing.backends.items()
        if backend.type == "codex"
    }


def _write_config(tmp_path: Path, *, codex: bool) -> Path:
    backend = (
        '    c:\n      type: "codex"\n      models:\n        - name: "gpt-6.1-sol"\n'
        '      codex:\n        codex_home: "/srv/lexora/codex-home"\n'
        '        ro_binds: ["/opt/codex"]\n'
        if codex
        else '    g:\n      type: "openai_compatible"\n      url: "http://x"\n'
        '      models:\n        - name: "m"\n'
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(f"routing:\n  enabled: true\n  backends:\n{backend}", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _no_host_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CODEX_HOST_RO_BINDS_ENV, raising=False)


def test_shipped_config_has_no_host_specific_ro_binds() -> None:
    text = SHIPPED_CONFIG.read_text(encoding="utf-8")
    assert STUB not in text
    for backend in _codex_backends(create_settings(SHIPPED_CONFIG)).values():
        assert backend.codex is not None
        assert backend.codex.ro_binds == []


def test_env_appends_to_every_codex_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CODEX_HOST_RO_BINDS_ENV, STUB)
    backends = _codex_backends(create_settings(SHIPPED_CONFIG))
    assert backends
    for backend in backends.values():
        assert backend.codex is not None
        assert backend.codex.ro_binds == [STUB]


def test_env_entries_follow_yaml_entries_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CODEX_HOST_RO_BINDS_ENV, f"{STUB}:/opt/codex")
    (backend,) = _codex_backends(create_settings(_write_config(tmp_path, codex=True))).values()
    assert backend.codex is not None
    assert backend.codex.ro_binds == ["/opt/codex", STUB]


@pytest.mark.parametrize("value", ["/", "/home/lexora", "/srv/lexora", "relative/path", "/a/../b"])
def test_env_entries_cannot_widen_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(CODEX_HOST_RO_BINDS_ENV, value)
    with pytest.raises(ValueError, match="ro_binds"):
        create_settings(_write_config(tmp_path, codex=True))


@pytest.mark.parametrize("value", [":" + STUB, STUB + ":", f"{STUB}::/opt/codex", " "])
def test_empty_elements_refuse_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(CODEX_HOST_RO_BINDS_ENV, value)
    with pytest.raises(ValueError, match="empty element"):
        create_settings(_write_config(tmp_path, codex=True))


def test_set_without_a_codex_backend_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CODEX_HOST_RO_BINDS_ENV, STUB)
    with pytest.raises(ValueError, match="no routing backend is of type 'codex'"):
        create_settings(_write_config(tmp_path, codex=False))


def test_empty_variable_is_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CODEX_HOST_RO_BINDS_ENV, "")
    (backend,) = _codex_backends(create_settings(_write_config(tmp_path, codex=True))).values()
    assert backend.codex is not None
    assert backend.codex.ro_binds == ["/opt/codex"]


def test_env_entries_reach_the_config_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lexora.backends.factory import create_backend

    path = _write_config(tmp_path, codex=True)
    ((name, cfg),) = _codex_backends(create_settings(path)).items()
    monkeypatch.setenv(CODEX_HOST_RO_BINDS_ENV, STUB)
    ((_, cfg_host),) = _codex_backends(create_settings(path)).items()
    assert create_backend(name, cfg).config_hash() != create_backend(name, cfg_host).config_hash()


@pytest.mark.parametrize(
    "value",
    [
        f'{STUB}   # ":" separated',  # the assignment PR #91's gate flagged
        f"{STUB} #comment",
        f"{STUB}#x",
        f"{STUB} :/opt/codex",
        f"{STUB}\t",
    ],
)
def test_inline_comment_or_whitespace_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """systemd EnvironmentFile keeps a trailing comment in the value, a sourcing
    shell drops it; refusing both ways keeps service and verify_codex in step."""
    monkeypatch.setenv(CODEX_HOST_RO_BINDS_ENV, value)
    with pytest.raises(ValueError, match="whitespace or '#'"):
        create_settings(_write_config(tmp_path, codex=True))


DOCS = [
    Path(__file__).resolve().parent.parent / "deploy" / "RUNBOOK.md",
    Path(__file__).resolve().parent.parent / "deploy" / "environment.example",
]


def _documented_assignments() -> list[tuple[str, str]]:
    found = []
    for doc in DOCS:
        for line in doc.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text.startswith("#"):
                text = text.lstrip("#").strip()  # environment.example ships it commented out
            if text.startswith(f"{CODEX_HOST_RO_BINDS_ENV}="):
                found.append((doc.name, text))
    return found


def test_docs_document_the_assignment() -> None:
    names = {name for name, _ in _documented_assignments()}
    assert names == {"RUNBOOK.md", "environment.example"}


@pytest.mark.parametrize("doc,line", _documented_assignments())
def test_documented_assignment_loads_as_systemd_reads_it(
    monkeypatch: pytest.MonkeyPatch, doc: str, line: str
) -> None:
    """Copy-paste the documented line into an EnvironmentFile: systemd takes
    everything after '=' to the end of the line as the value."""
    _, _, value = line.partition("=")
    monkeypatch.setenv(CODEX_HOST_RO_BINDS_ENV, value)
    for backend in _codex_backends(create_settings(SHIPPED_CONFIG)).values():
        assert backend.codex is not None
        assert backend.codex.ro_binds == [STUB]

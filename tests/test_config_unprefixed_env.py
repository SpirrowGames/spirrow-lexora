"""Config sections never read unprefixed environment variables.

T-config-unprefixed-env (msg-602, step 1 as revised by msg-612).

``create_settings`` builds each YAML section as ``X(**yaml[section])``. When
``X`` was a ``BaseSettings`` without ``env_prefix``, every field the YAML left
out was silently filled from an environment variable of the same name
(``API_KEY`` onto every backend, ``PORT`` onto the server, ``ENABLED=false``
switching off rate limiting, ...). The child classes are now ``BaseModel``
with ``extra="forbid"``; only the root ``Settings`` stays ``BaseSettings``
(``LEXORA_`` prefix).

Two layers of test:

* ``TestStructuralGuard`` walks every class reachable from ``Settings`` and
  asserts none is a ``BaseSettings`` -- so a child class added later cannot
  reintroduce the leak without turning CI red, whatever its name. It also
  keeps ``protected_namespaces=()`` scoped to ``BackendSettings`` and fails
  any field name that shadows a ``BaseModel`` attribute.
* ``TestBehaviour`` sets one unprefixed env var per field name of every
  reachable class and checks ``create_settings()`` is unchanged, on both
  paths that fall back to the environment: (a) section present, field
  omitted; (b) section absent (no-arg / ``default_factory`` construction).

On develop 73e0afb (before the fix) the structural guard and the behaviour
tests failed.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings

from lexora.backends.factory import resolve_api_key
from lexora.config import (
    BackendSettings,
    ClassifierSettings,
    ModelInfo,
    Settings,
    TierSettings,
    create_settings,
)


def _reachable_models(root: type[BaseModel]) -> list[type[BaseModel]]:
    """Every pydantic model class reachable from ``root``'s field types.

    Unwraps generic containers (``dict[str, X]``, ``list[X]``,
    ``X | None``, ``Annotated[...]``) so nested classes are found however
    they are declared.
    """
    seen: dict[type[BaseModel], None] = {}

    def walk(tp: object) -> None:
        if typing.get_origin(tp) is not None:
            for arg in typing.get_args(tp):
                walk(arg)
            return
        if isinstance(tp, type) and issubclass(tp, BaseModel) and tp not in seen:
            seen[tp] = None
            for field in tp.model_fields.values():
                walk(field.annotation)

    walk(root)
    return list(seen)


def _child_classes() -> list[type[BaseModel]]:
    """Every class reachable from ``Settings``, minus the root itself.

    The recursive walk is the single source: no hand-kept list of the classes
    ``create_settings`` constructs (msg-617 advisory, accepted in msg-622).
    """
    return [cls for cls in _reachable_models(Settings) if cls is not Settings]


#: Child classes allowed to disable pydantic's protected-namespace check.
#: ``BackendSettings.model_mapping`` starts with ``model_``; pydantic < 2.10
#: (permitted by ``pydantic>=2.5.0``) warns on it. Suppression is an
#: exception: adding a class here must be a reviewed decision
#: (T-config-unprefixed-env msg-617 / msg-620 / msg-624).
_PROTECTED_NAMESPACES_SUPPRESSED: frozenset[type[BaseModel]] = frozenset(
    {BackendSettings}
)


#: One unprefixed env value per field name of every child class. Values
#: differ from the defaults and from the YAML below, so a leak shows up as a
#: changed value (or a load error) rather than being masked.
UNPREFIXED_ENV: dict[str, str] = {
    # BackendSettings
    "TYPE": "openai_compatible",
    "URL": "http://evil.invalid:9",
    "TIMEOUT": "7.0",
    "CONNECT_TIMEOUT": "1.5",
    "MODELS": '["envmodel"]',
    "API_KEY": "LEAK",
    "API_KEY_ENV": "SOME_OTHER_VAR",
    "MODEL_MAPPING": '{"a": "b"}',
    "THINKING_MODE": "think",
    "REASONING_EFFORT": "low",
    "DEFAULT_MAX_TOKENS": "17",
    "PAID_KEY_ACKNOWLEDGED": "true",
    "GOVERNANCE_GATE_ENABLED": "true",
    "HEALTH_CHECK": "false",
    "ERROR_PASSTHROUGH": "true",
    "CODEX": "{}",
    "FALLBACK": "envfallback",
    # ModelInfo
    "NAME": "envname",
    "CAPABILITIES": '["envcap"]',
    "DESCRIPTION": "envdesc",
    # TierSettings / ClassifierSettings
    "BACKEND": "envbackend",
    "MODEL": "envmodel",
    # RoutingSettings / ClassifierSettings / RateLimitSettings
    "ENABLED": "true",
    "DEFAULT_BACKEND": "envbackend",
    "BACKENDS": "{}",
    "DEFAULT_MODEL_FOR_UNKNOWN_TASK": "envmodel",
    "TIERS": "{}",
    "CLASSIFIER": "{}",
    # ServerSettings
    "HOST": "9.9.9.9",
    "PORT": "4242",
    # QueueSettings
    "MAX_SIZE": "3",
    "DEFAULT_TIMEOUT": "2.5",
    # RateLimitSettings
    "DEFAULT_RATE": "0.5",
    "DEFAULT_BURST": "2",
    # RetrySettings
    "MAX_RETRIES": "9",
    "BASE_DELAY": "0.25",
    "MAX_DELAY": "3.0",
    "EXPONENTIAL_BASE": "5.0",
    "RESPECT_RETRY_AFTER": "false",
    "MAX_RETRY_AFTER": "4.0",
    # LoggingSettings
    "LEVEL": "DEBUG",
    "FORMAT": "json",
    # CodexSettings / FallbackSettings / DecisionSettings were already
    # BaseModel before this change; covered so the map is complete and a
    # regression there shows too.
    "CODEX_HOME": "/env/codex",
    "CODEX_BIN": "/env/codex-bin",
    "BWRAP_BIN": "/env/bwrap",
    "RO_BINDS": '["/env"]',
    "CLI_OVERRIDES": '["x=y"]',
    "MAX_CONCURRENCY": "77",
    "STATE_DB_PATH": "/env/state.db",
    "DATA_CONTROLS_FILE": "/env/dc.json",
    "PRIMARY": "jev",
    "MODE": "active",
    "TIMEOUT_MS": "1234",
    "JEV_MODEL": "hijack",
    "LOG_PATH": "/env/decisions.db",
}

#: Path (a): every section present, each with most fields left out. Backend
#: ``b`` gives only ``url`` and ``models`` (one as str, one as dict); tier
#: ``t`` only ``backend``; the classifier only ``backend``.
_PARTIAL_YAML = """\
vllm:
  connect_timeout: 5.0
server:
  host: 127.0.0.1
queue:
  default_timeout: 60.0
rate_limit:
  default_rate: 10.0
retry:
  base_delay: 1.0
logging:
  level: INFO
routing:
  enabled: false
  default_backend: b
  backends:
    b:
      url: http://localhost:8000
      models:
        - m1
        - name: m2
  tiers:
    t:
      backend: b
  classifier:
    backend: b
decision:
  timeout_ms: 3000
"""


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in UNPREFIXED_ENV.items():
        monkeypatch.setenv(name, value)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in UNPREFIXED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LEXORA_FRONTIER_MODEL", raising=False)


class TestStructuralGuard:
    def test_walker_finds_nested_children(self) -> None:
        """Sanity: the walker sees dict / list / optional members."""
        found = set(_reachable_models(Settings))
        for cls in (BackendSettings, ModelInfo, TierSettings, ClassifierSettings):
            assert cls in found

    @pytest.mark.parametrize("cls", _child_classes(), ids=lambda c: c.__name__)
    def test_child_is_not_base_settings(self, cls: type[BaseModel]) -> None:
        assert not issubclass(cls, BaseSettings), (
            f"{cls.__name__} is a BaseSettings without the root's LEXORA_ "
            f"prefix: create_settings would fill its YAML-omitted fields from "
            f"unprefixed env vars. Make it a BaseModel with "
            f"model_config = ConfigDict(extra='forbid')."
        )

    @pytest.mark.parametrize("cls", _child_classes(), ids=lambda c: c.__name__)
    def test_child_forbids_extra(self, cls: type[BaseModel]) -> None:
        """``BaseModel`` defaults to ``extra='ignore'``: a YAML typo would
        pass silently unless ``forbid`` is explicit (msg-602 step 3)."""
        assert cls.model_config.get("extra") == "forbid"

    def test_backend_settings_suppresses_protected_namespaces(self) -> None:
        """``model_mapping`` would warn on pydantic < 2.10 otherwise."""
        assert BackendSettings.model_config.get("protected_namespaces") == ()

    def test_protected_namespace_suppression_is_scoped(self) -> None:
        """Only the allow-listed classes may set ``protected_namespaces=()``:
        elsewhere pydantic's own check must stay in force."""
        suppressed = {
            cls
            for cls in _child_classes()
            if cls.model_config.get("protected_namespaces") == ()
        }
        assert suppressed == set(_PROTECTED_NAMESPACES_SUPPRESSED)

    @pytest.mark.parametrize("cls", _child_classes(), ids=lambda c: c.__name__)
    def test_no_field_shadows_base_model_attribute(self, cls: type[BaseModel]) -> None:
        """No field name may equal a ``BaseModel`` attribute.

        pydantic >= 2.10 hard-errors only for the ``model_validate`` /
        ``model_dump`` prefixes; every other shadow (``model_copy``,
        ``model_fields``, legacy ``copy`` / ``json`` / ``dict`` ...) is just a
        ``UserWarning``, and with ``protected_namespaces=()`` even
        ``model_dump`` is. This raises all of them to a CI failure, on every
        child class (msg-624 / msg-626).
        """
        shadowing = sorted(name for name in cls.model_fields if hasattr(BaseModel, name))
        assert not shadowing, (
            f"{cls.__name__} fields shadow BaseModel attributes: {shadowing}"
        )

    def test_root_keeps_prefixed_base_settings(self) -> None:
        assert issubclass(Settings, BaseSettings)
        assert Settings.model_config.get("env_prefix") == "LEXORA_"

    def test_env_map_covers_every_child_field(self) -> None:
        names = {name.upper() for cls in _child_classes() for name in cls.model_fields}
        missing = sorted(names - set(UNPREFIXED_ENV))
        assert not missing, f"add these field names to UNPREFIXED_ENV: {missing}"


class TestBehaviour:
    @pytest.mark.parametrize(
        "yaml_text",
        [_PARTIAL_YAML, ""],
        ids=["a-section-present-field-omitted", "b-section-absent"],
    )
    def test_unprefixed_env_does_not_change_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, yaml_text: str
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_text, encoding="utf-8")
        _clear_env(monkeypatch)
        baseline = create_settings(config_file).model_dump()
        _set_env(monkeypatch)
        assert create_settings(config_file).model_dump() == baseline

    def test_api_key_not_attached_to_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The heaviest case in msg-602: an unrelated ``API_KEY`` sent to
        every backend whose YAML has no ``api_key``. (``resolve_api_key``
        prefers ``api_key`` over ``api_key_env``, so the leak also overrode
        a correctly named ``api_key_env``.)"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(_PARTIAL_YAML, encoding="utf-8")
        _clear_env(monkeypatch)
        monkeypatch.setenv("API_KEY", "LEAK")
        backend = create_settings(config_file).routing.backends["b"]
        assert backend.api_key is None
        assert resolve_api_key(backend) is None

    def test_api_key_env_named_lookup_still_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``api_key_env`` is the intended, named env read (msg-602 step 5)
        and is untouched by the fix."""
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("MY_BACKEND_KEY", "named-secret")
        backend = BackendSettings(api_key_env="MY_BACKEND_KEY")
        assert resolve_api_key(backend) == "named-secret"

    def test_yaml_typo_still_rejected(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("server:\n  prot: 8002\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            create_settings(config_file)

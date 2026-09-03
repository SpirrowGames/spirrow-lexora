"""Load and validate the shipped ``config/lexora_config.yaml`` (R-4).

Why this file exists (T-silent-routing spec §4 R-4): the routing branch adds
several ways for the config loader to refuse a bad file at boot (strict
YAML loader that rejects duplicate keys, tier/model name collision check,
model-name ambiguity WARNING). Because production runs directly off a
working tree — see ``deploy/`` comments — the moment a bad config lands
under ``spirrow-lexora``'s WorkingDirectory the service enters a restart
loop that shows externally as "not responding", which happens to be the
same symptom as "the naysayer route is down". Catching the fault in CI is
the whole condition on which those boot-time rejections are safe: R-1a /
R-1b / R-5 are gated on R-4 (msg-078 §6, msg-081 §3.3).

This test loads the shipped file as a file (not a fixture), because a
fixture would only measure what we thought to put in it. The failure
modes we care about — a duplicate key added in a rebase, a model
declaration that unexpectedly collides with a tier name, an env override
that would leave a lie behind — all show up textually in
``config/lexora_config.yaml`` and are exactly what R-4 asserts against.

Kept separate from ``test_config.py`` so a diff to the shipped config or
its schema fails a test named after that file, not one named after the
generic config machinery.
"""

from pathlib import Path

import pytest

from lexora.config import create_settings

SHIPPED_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "lexora_config.yaml"
)


def test_shipped_config_file_exists() -> None:
    """Guard against a rename / move that would leave R-4 silently passing.

    Without this line a rename of the file to somewhere the tests do not
    look would turn ``create_settings(...)`` into a no-op — the loader
    returns ``{}`` for a missing path — and every assertion below would
    pass against the defaults, not the shipped config. The whole point of
    R-4 is that CI reads the same file production will, so a missing file
    is a failure, not an "empty config".
    """
    assert SHIPPED_CONFIG.exists(), (
        f"Shipped config not found at {SHIPPED_CONFIG}. If the file moved, "
        f"update this test to point at the new location — do not delete it."
    )


def test_shipped_config_loads_without_error() -> None:
    """Load-and-validate the shipped config against every gate this branch adds.

    Everything the routing branch enforces at boot is invoked by
    ``create_settings``: the strict YAML loader (R-5), the tier/model
    collision check (R-1b), and the model-name ambiguity WARNING (R-1a,
    non-fatal). A raise here is R-4 doing its job: the shipped file broke
    a rule and would have failed the running service.
    """
    settings = create_settings(SHIPPED_CONFIG)
    assert settings.routing.enabled is True, (
        "Shipped config has routing.enabled: false — that would take the "
        "gateway back into legacy single-backend mode and skip every "
        "guarantee this test measures. Refuse rather than let CI green."
    )


def test_shipped_config_default_model_for_unknown_task_is_routable() -> None:
    """The default model that /generate and /chat fall back to must route.

    ``default_model_for_unknown_task`` is passed through
    ``BackendRouter.get_backend_for_model`` when /generate or /chat run
    without an explicit ``model``. If that name is not registered, or is
    declared by multiple backends (ambiguous), the endpoint returns 404
    for every call that omits ``model`` — a regression that a boot
    validator cannot see because the router is only exercised at request
    time. This test is the boot-time surrogate for that request.
    """
    settings = create_settings(SHIPPED_CONFIG)
    name = settings.routing.default_model_for_unknown_task
    if name is None:
        # An unset default is legal — the /generate / /chat handlers
        # refuse with 400 "no model" in that case. What is not legal is a
        # default that structurally 404s.
        return

    tier_names = set(settings.routing.tiers)
    declared_by: dict[str, list[str]] = {}
    for backend_name, backend_settings in settings.routing.backends.items():
        for model_info in backend_settings.models:
            declared_by.setdefault(model_info.name, []).append(backend_name)

    if name in tier_names:
        # Tier aliases always route unambiguously; nothing else to check.
        return

    declared = declared_by.get(name, [])
    assert declared, (
        f"default_model_for_unknown_task is '{name}' but no tier or "
        f"backend declares it — /generate and /chat without a model would "
        f"404 on every request."
    )
    assert len(declared) == 1, (
        f"default_model_for_unknown_task is '{name}', which is declared "
        f"by multiple backends ({declared}). Requests naming it directly "
        f"are refused with 404 by R-1a; point this field at a tier alias "
        f"or a name only one backend declares."
    )

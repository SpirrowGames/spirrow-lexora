"""Decision endpoint settings and startup env checks.

Contract (msg-237 / msg-244 / msg-246, narrowed by Bohr msg-387 v3):

* ``primary``: which provider answers under ``active`` (``null`` | ``jev``
  — Fermi msg-257 scope 2).
* ``mode``: ``off`` answers from :class:`NullProvider` unconditionally;
  ``active`` answers from ``primary`` and falls back to NullProvider on
  error / timeout / 429 (Fermi msg-257 §3). There is no other mode.
* ``timeout_ms``: per-call upstream deadline in milliseconds.
* ``jev_model``: the ``model`` value sent to Jev (default ``jev-latest``).
  The version that actually served each request is logged in
  ``provider_model``.

The schema accepts only what the code implements (msg-387 v3). ``shadow``
(answer from one provider, run another in the background for logging),
``llm`` (``LlmEmulationProvider``) and a ``fallback`` provider selector are
not implemented, so they are not in the schema: a config naming any of
them fails validation at load instead of being accepted and silently
doing nothing. ``DecisionSettings`` forbids unknown keys (the
pydantic-settings default, pinned by a test), so a leftover ``fallback``
key in ``[decision]`` stops start-up too. The PR that implements shadow /
LlmEmulation adds the value back together with the behaviour. This
withdraws the earlier "validate the mode up front so a later PR does not
have to revisit the schema" policy.

The defaults (``primary="null"``, ``mode="off"``) never reach Jev, so no
metered call happens until an operator opts in.

Startup env check (msg-239 / msg-240):

* If ``primary == "jev"`` and the ``TYPESAFE_API_KEY`` environment
  variable is unset (or empty), start-up is **refused** with a fixed
  error message that does not leak the value, its length, or a prefix.
  ``mode == "off"`` does not soften this: the operator has still declared
  intent to use Jev, and the value they set ``mode`` to next is not
  something startup can observe.

Nothing else in this module talks to Jev or reads the env variable. The
provider implementation only accesses the value through an injected HTTP
client (see :class:`lexora.decide.providers.DecisionProvider`), which is
why unit tests can exercise the provider without the env being set — the
env check is a **startup** guarantee, not a per-call one.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

#: The single env variable Lexora reads for the TypeSafe API key. Fixed by
#: msg-239: chosen to match TypeSafe's official SDK default so a Lexora
#: operator does not have to remember two names. The value is only read at
#: startup; it is never logged, never included in a decision-log row, and
#: never returned in an error message. See ``check_typesafe_api_key``.
TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"

#: Fixed message emitted when the startup check refuses the config. Does
#: not carry the caller-supplied value, its length, or a prefix (msg-240
#: §1): the operator already knows what env variable is missing; adding
#: derivative facts about the value only widens what a stack trace or log
#: capture exposes.
_MISSING_KEY_MESSAGE = (
    f'{TYPESAFE_API_KEY_ENV} is required when [decision].primary is "jev"'
)


DecisionProviderName = Literal["null", "jev"]
DecisionMode = Literal["off", "active"]


class DecisionSettings(BaseSettings):
    """Configuration for the ``/v1/decide`` endpoint.

    Defaults are deliberately safe: ``primary="null"`` + ``mode="off"``
    means an operator who ships this branch without touching
    ``[decision]`` gets the NullProvider on every call and never talks to
    an external service. A config that names ``primary="jev"`` must
    ship the env variable too (see :func:`check_typesafe_api_key`).
    """

    primary: DecisionProviderName = Field(
        default="null",
        description=(
            "Provider that answers under mode=active (null | jev). "
            "Ignored under mode=off."
        ),
    )
    mode: DecisionMode = Field(
        default="off",
        description=(
            "off = always answer from NullProvider (safest); active = "
            "answer from primary, fall back to NullProvider on "
            "error/timeout/429."
        ),
    )
    timeout_ms: int = Field(
        default=2000,
        ge=1,
        description="Per-call upstream deadline in milliseconds.",
    )
    jev_model: str = Field(
        default="jev-latest",
        min_length=1,
        description=(
            "Value sent as the required ``model`` field to Jev's systemone "
            "endpoint (Bohr msg-339 #2). ``jev-latest`` until the logged "
            "``provider_model`` values show which version to pin. Env: "
            "``LEXORA_DECISION__JEV_MODEL``."
        ),
    )
    log_path: str = Field(
        default="data/decisions.db",
        description=(
            "Filesystem path for the SQLite decision log. Parent directory "
            "is created if missing (see :class:`lexora.decide.log."
            "DecisionLog`). Must be a durable path — msg-237 requires the "
            "log for calibration-curve fitting and for replaying "
            "mindwire's 116 decision points through the same code path, "
            "which is only possible if rows survive a process restart. "
            "The one exception is tests, which pass ``:memory:`` "
            "explicitly through the settings override. The regular "
            "``data/decisions.db`` default matches the sibling convention "
            "in ``services/cost_tracker.py`` (``data/costs.db``) so an "
            "operator does not have to learn two directory layouts."
        ),
    )


def references_jev(settings: DecisionSettings) -> bool:
    """Whether the current config would ever call the Jev provider.

    A ``mode=off`` config that still names ``primary="jev"`` is treated
    as referencing Jev on purpose (see the startup-check rationale in the
    module docstring).
    """
    return settings.primary == "jev"


def check_typesafe_api_key(
    settings: DecisionSettings,
    *,
    environ: dict[str, str] | None = None,
) -> None:
    """Refuse to continue startup when ``jev`` is named but the key is missing.

    Called from :func:`lexora.main.create_app`'s lifespan startup, after
    ``create_settings`` has run and before any provider is instantiated
    (msg-240 §1). Tests inject ``environ`` so they can assert both
    branches without mutating ``os.environ`` from other processes.

    The check has one shape: ``references_jev`` and the value is missing
    or empty. It does not examine the value's length, prefix, or
    character set — a syntactically well-formed key that TypeSafe rejects
    is a runtime error, not a startup one, and the point of the startup
    check is to keep the two failure modes distinguishable.

    Args:
        settings: The loaded :class:`DecisionSettings`.
        environ: Env mapping. Defaults to ``os.environ``.

    Raises:
        RuntimeError: With the fixed message. The exception does not
            carry the caller-supplied value or any derivative of it.
    """
    if not references_jev(settings):
        return
    env = environ if environ is not None else os.environ
    value = env.get(TYPESAFE_API_KEY_ENV)
    if value is None or value == "":
        raise RuntimeError(_MISSING_KEY_MESSAGE)

"""Decision endpoint settings and startup env checks.

Contract (msg-237 / msg-244 / msg-246):

* ``primary``: which provider Lexora asks first (``null`` | ``llm`` | ``jev``).
* ``mode``: ``off`` returns from :class:`NullProvider` unconditionally;
  ``shadow`` returns ``fallback``'s answer to the caller while also running
  ``primary`` in the background for logging; ``active`` returns
  ``primary``'s answer and only falls back on error / timeout / 429. The
  mode selects behaviour, it does NOT re-select provider identity — a
  ``mode=off`` config that names ``primary=jev`` is still a config
  intending to use Jev, and it still fails closed if the key is missing
  when the operator later flips to shadow / active. (PR 1 only implements
  ``NullProvider``; the mode field is validated up front so a future PR
  that adds ``LlmEmulationProvider`` / ``JevProvider`` does not have to
  revisit the schema.)
* ``fallback``: which provider serves the answer when ``primary`` fails
  under ``active``, and which serves the caller under ``shadow``.
* ``timeout_ms``: per-call upstream deadline in milliseconds.

Startup env check (msg-239 / msg-240):

* If ``primary == "jev"`` **or** ``fallback == "jev"`` and the
  ``TYPESAFE_API_KEY`` environment variable is unset (or empty), start-up
  is **refused** with a fixed error message that does not leak the value,
  its length, or a prefix. ``mode == "off"`` does not soften this: the
  operator has still declared intent to use Jev, and the value they set
  ``mode`` to next is not something startup can observe.

  Bohr endorsed by Einstein in msg-243 / msg-245: examining both
  ``primary`` and ``fallback`` is not overzealous — a config that names
  ``fallback=jev`` without an API key is a time bomb that goes off only
  when the primary is already failing.

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
    f"{TYPESAFE_API_KEY_ENV} is required when [decision].primary or "
    f'[decision].fallback is "jev"'
)


DecisionProviderName = Literal["null", "llm", "jev"]
DecisionMode = Literal["off", "shadow", "active"]


class DecisionSettings(BaseSettings):
    """Configuration for the ``/v1/decide`` endpoint.

    Defaults are deliberately safe: ``primary="null"`` + ``mode="off"``
    means an operator who ships this branch without touching
    ``[decision]`` gets the NullProvider on every call and never talks to
    an external service. A config that names ``jev`` for either slot must
    ship the env variable too (see :func:`check_typesafe_api_key`).
    """

    primary: DecisionProviderName = Field(
        default="null",
        description=(
            "Which provider Lexora asks first. In shadow mode this is the "
            "one whose answer is logged but NOT returned to the caller; in "
            "active mode this is the one whose answer is returned."
        ),
    )
    mode: DecisionMode = Field(
        default="off",
        description=(
            "off = always answer from NullProvider (safest); shadow = "
            "return fallback's answer, log primary in the background; "
            "active = return primary's answer, fall back on error/timeout/429."
        ),
    )
    fallback: DecisionProviderName = Field(
        default="llm",
        description=(
            "Provider used when primary fails under active, and used as "
            "the caller-facing provider under shadow."
        ),
    )
    timeout_ms: int = Field(
        default=2000,
        ge=1,
        description="Per-call upstream deadline in milliseconds.",
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

    A ``mode=off`` config that still names ``jev`` for ``primary`` or
    ``fallback`` is treated as referencing Jev on purpose (see the
    startup-check rationale in the module docstring). The one and only
    exception is a config where neither slot is ``jev``.
    """
    return settings.primary == "jev" or settings.fallback == "jev"


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

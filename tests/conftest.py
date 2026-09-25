"""Session-wide test environment.

The shipped ``config/lexora_config.yaml`` sets ``[decision].primary = "jev"``
(production runs Jev), and ``lexora.main`` builds ``app = create_app()`` at
import time. ``create_app`` refuses to start when a Jev config has no
``TYPESAFE_API_KEY`` (see ``lexora.decide.config``), so without a key every
test module that imports ``lexora.main`` fails at collection — in CI and on
any machine that does not hold the real key.

A placeholder is set here, before any test module is imported. It never
reaches TypeSafe: the only tests that call the real API are the ``smoke``
ones, which are deselected by default, and ``setdefault`` leaves a real key
in the environment untouched so ``pytest -m smoke`` still uses it. Tests
that exercise the missing-key path remove the variable with ``monkeypatch``
or pass ``environ`` explicitly, so they are unaffected.
"""

from __future__ import annotations

import os

os.environ.setdefault("TYPESAFE_API_KEY", "test-placeholder-not-a-real-key")

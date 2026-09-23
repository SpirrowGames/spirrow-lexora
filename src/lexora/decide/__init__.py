"""Judgment-only decision endpoint (``/v1/decide``).

T-decide-endpoint (T02) PR 1 lands the contract + NullProvider + decision
log + startup env checks. LlmEmulationProvider and JevProvider are
follow-up PRs; the interface in :mod:`lexora.decide.providers` is what
they will implement against.

Callers do not import from submodules directly — the public surface is
this package and :mod:`lexora.decide.routes` (which mounts the route).
"""

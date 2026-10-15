"""Lexora's ingress: what happens to a tier request before a backend sees it.

ADR-2026-05-31-14 §7.3 (spirrow-mindwire#482): credential redaction for tiers
with ``redact: true`` (``redact.py``) and the N-3 route declaration
``X-Mindwire-N3-Route: v1`` for tiers whose backend chain contains codex
(``gate.py``).
"""

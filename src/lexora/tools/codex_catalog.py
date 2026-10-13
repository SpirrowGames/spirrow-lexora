"""``python -m lexora.tools.codex_catalog`` -- write the tool-less model catalog.

The codex backend's tool-disabling setting has two halves (measured
2026-10-13 on codex-cli 0.160.0, see ``TOOL_DISABLE_FEATURE_OVERRIDES`` in
``lexora.backends.codex``):

* feature / tool flags passed as ``-c`` overrides, and
* a model catalog, named by the ``model_catalog_json`` override, in which the
  pinned model's entry no longer asks for the tools no flag can remove
  (code mode's ``exec``, ``apply_patch``, the ``collaboration.*`` tools,
  ``clock.*``).

This tool writes that catalog. It reads the catalog bundled in the
installed CLI (``codex debug models``, run with an empty ``CODEX_HOME`` so
the login is never touched), keeps only ``CODEX_MODEL_PIN.model``, checks
that the entry still carries the pinned context windows, applies
``CATALOG_KEYS_REMOVED`` / ``CATALOG_KEYS_SET``, and writes the result
(mode 600) to the path the override names. That path must lie inside
``codex_home``: it is the only host directory the sandboxed CLI can read.

The file's sha256 is part of the backend's config hash, so writing a
different catalog closes the gate (``verification_stale``) until
``verify_codex`` passes again. Order: this tool, then ``verify_codex``.
Re-run both after every CLI upgrade.

``--check`` writes nothing; it exits 1 when the file is missing or differs
from what would be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from lexora.backends.codex import (
    CODEX_MODEL_PIN,
    MODEL_CATALOG_OVERRIDE_KEY,
    CodexBackend,
    CodexModelPin,
    apply_catalog_tool_edits,
    model_catalog_path,
)


class CatalogError(Exception):
    """The bundled catalog cannot produce a catalog for the pinned model."""


def build_catalog(bundled: Mapping[str, Any], pin: CodexModelPin = CODEX_MODEL_PIN) -> dict[str, Any]:
    """A one-model catalog: the pinned entry with its tool keys edited.

    Refuses (``CatalogError``) when the bundled catalog has no single entry
    for ``pin.model`` or that entry's context windows differ from the pin:
    the pin was read from a different catalog, and has to be read again
    before anything is written.
    """
    models = bundled.get("models") if isinstance(bundled, Mapping) else None
    entries = [m for m in models or [] if isinstance(m, Mapping) and m.get("slug") == pin.model]
    if len(entries) != 1:
        raise CatalogError(f"the bundled catalog has {len(entries)} entries for {pin.model!r}, expected 1")
    entry = entries[0]
    for key, expected in (("context_window", pin.context_window), ("max_context_window", pin.max_context_window)):
        if entry.get(key) != expected:
            raise CatalogError(
                f"{pin.model!r} {key} is {entry.get(key)!r} in the bundled catalog but {expected!r} in "
                f"CODEX_MODEL_PIN; read the pin again for this CLI version before writing a catalog"
            )
    return {"models": [apply_catalog_tool_edits(entry)]}


def render(catalog: Mapping[str, Any]) -> bytes:
    """Deterministic bytes, so the same CLI version always yields the same hash."""
    return (json.dumps(catalog, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def bundled_catalog(codex_bin: str) -> dict[str, Any]:
    """``codex debug models`` with an empty, throwaway ``CODEX_HOME``."""
    with tempfile.TemporaryDirectory(prefix="lexora-codex-catalog-") as home:
        env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "CODEX_HOME": home, "HOME": home}
        try:
            proc = subprocess.run(
                [codex_bin, "debug", "models"], env=env, capture_output=True, text=True, timeout=60, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CatalogError(f"`{codex_bin} debug models` could not run: {exc}") from exc
    if proc.returncode != 0:
        raise CatalogError(f"`{codex_bin} debug models` exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"`{codex_bin} debug models` did not print JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CatalogError("`codex debug models` printed JSON that is not an object")
    return parsed


def target_path(backend: CodexBackend) -> Path:
    """The catalog path from the backend's overrides; must sit in ``codex_home``."""
    raw = model_catalog_path(backend.cli_overrides)
    if raw is None:
        raise CatalogError(f"backend '{backend.name}' has no {MODEL_CATALOG_OVERRIDE_KEY} override in cli_overrides")
    path, home = PurePosixPath(raw), PurePosixPath(backend.codex_home)
    if not path.is_absolute() or ".." in path.parts or home not in path.parents:
        raise CatalogError(
            f"{MODEL_CATALOG_OVERRIDE_KEY} {raw!r} must be an absolute path inside codex_home "
            f"{backend.codex_home!r}; the sandboxed CLI cannot read anything else"
        )
    return Path(raw)


def write_atomically(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".lexora-catalog-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lexora.tools.codex_catalog", description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="Lexora YAML config (default: Lexora's usual lookup)")
    parser.add_argument("--backend", help="name of the codex backend (required if several)")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if the file is missing or differs")
    args = parser.parse_args(argv)

    from lexora.tools.verify_codex import _load_backend

    backend = _load_backend(args.config, args.backend)
    try:
        path = target_path(backend)
        data = render(build_catalog(bundled_catalog(backend.codex_bin)))
    except CatalogError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    digest = hashlib.sha256(data).hexdigest()
    current = path.read_bytes() if path.exists() else None
    if args.check:
        state = "missing" if current is None else ("up to date" if current == data else "differs")
        print(f"{path}: {state} (expected sha256 {digest})")
        return 0 if current == data else 1
    if current == data:
        print(f"{path}: already up to date (sha256 {digest})")
        return 0
    write_atomically(path, data)
    print(f"wrote {path} (sha256 {digest}); run verify_codex next -- the config hash changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

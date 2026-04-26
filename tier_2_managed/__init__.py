"""Importable shim for the on-disk ``tier-2-managed/`` package.

Mirrors the ``tier_1_naive`` shim from Phase 128 (and the ``tier_3_graph``
shim from Plan 129-03). Hyphenated directory names are not valid Python
identifiers; this shim loads each sibling module from ``tier-2-managed/``
via ``importlib`` and registers them under ``tier_2_managed.<name>``.

Usage from external code (tests, main.py, Plan 04 CLI)::

    from tier_2_managed.store import GeminiFileSearchStore  # noqa
    from tier_2_managed.store import (
        get_or_create_store, upload_pdf, upload_with_retry,
        list_existing_documents, delete_store,
    )

The shim is intentionally tiny — it owns no logic, only path resolution.
Each underlying module (``store.py`` here; ``query.py`` and ``main.py``
arrive in Plan 129-04) remains a clean library file inside
``tier-2-managed/`` and is the source of truth.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TIER2_DIR = Path(__file__).resolve().parent.parent / "tier-2-managed"

# Make the hyphenated directory itself importable as a sys.path entry so
# that intra-tier flat imports (e.g. when main.py is executed as a script)
# continue to work, AND so that any module loaded via importlib below can
# resolve sibling imports like ``from store import upload_pdf``.
_tier2_str = str(_TIER2_DIR)
if _tier2_str not in sys.path:
    sys.path.insert(0, _tier2_str)


def _load(submodule: str):
    """Load ``tier-2-managed/<submodule>.py`` as ``tier_2_managed.<submodule>``."""
    full_name = f"{__name__}.{submodule}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    src = _TIER2_DIR / f"{submodule}.py"
    if not src.exists():
        raise ImportError(f"tier_2_managed shim: missing source file {src}")
    spec = importlib.util.spec_from_file_location(full_name, src)
    if spec is None or spec.loader is None:
        raise ImportError(f"tier_2_managed shim: could not load spec for {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Eagerly register all known tier-2-managed submodules so that
# ``from tier_2_managed.X import Y`` works without the caller knowing about
# the shim mechanics. ``query`` and ``main`` are added in Plan 129-04 and
# may be missing during partial Plan 02 execution; the try/except keeps
# the shim usable until then.
for _name in ("store", "query", "main"):
    try:
        _load(_name)
    except ImportError:
        pass

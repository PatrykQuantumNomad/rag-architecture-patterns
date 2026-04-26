"""Importable shim for the on-disk ``tier-1-naive/`` package.

The on-disk package directory is ``tier-1-naive`` (hyphen) because tier
directories are organized for human readability and the docs/blog. Hyphens
are not valid Python identifiers, so ``tier-1-naive`` cannot be imported
directly. This shim package provides an importable name (``tier_1_naive``)
that loads each sibling module from ``tier-1-naive/`` via ``importlib`` and
registers it under the dotted path callers expect:

    from tier_1_naive.prompt import build_prompt
    from tier_1_naive.main import main, DEFAULT_QUERY
    ...

The shim is intentionally tiny — it owns no logic, only path resolution.
Each underlying module (`ingest.py`, `store.py`, etc.) remains a clean
library file inside ``tier-1-naive/`` and is the source of truth.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TIER1_DIR = Path(__file__).resolve().parent.parent / "tier-1-naive"

# Make the hyphenated directory itself importable as a sys.path entry so
# that intra-tier flat imports (e.g. when main.py is executed as a script)
# continue to work, AND so that any module loaded via importlib below can
# resolve sibling imports like ``from prompt import build_prompt``.
_tier1_str = str(_TIER1_DIR)
if _tier1_str not in sys.path:
    sys.path.insert(0, _tier1_str)


def _load(submodule: str):
    """Load ``tier-1-naive/<submodule>.py`` and register as ``tier_1_naive.<submodule>``."""
    full_name = f"{__name__}.{submodule}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    src = _TIER1_DIR / f"{submodule}.py"
    if not src.exists():
        raise ImportError(f"tier_1_naive shim: missing source file {src}")
    spec = importlib.util.spec_from_file_location(full_name, src)
    if spec is None or spec.loader is None:
        raise ImportError(f"tier_1_naive shim: could not load spec for {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Eagerly register all known tier-1-naive submodules so that
# ``from tier_1_naive.X import Y`` works without the caller knowing about
# the shim mechanics.
for _name in ("ingest", "embed_openai", "chat", "store", "retrieve", "prompt", "main"):
    try:
        _load(_name)
    except ImportError:
        # ``main`` may not exist yet during partial Plan 04 execution; skip
        # gracefully so the shim loads even when only a subset of modules
        # has been written.
        pass

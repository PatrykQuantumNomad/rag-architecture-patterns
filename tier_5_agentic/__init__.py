"""Importable shim for the on-disk ``tier-5-agentic/`` package.

The on-disk package directory is ``tier-5-agentic`` (hyphen) because tier
directories are organized for human readability and the docs/blog. Hyphens
are not valid Python identifiers, so ``tier-5-agentic`` cannot be imported
directly. This shim package provides an importable name (``tier_5_agentic``)
that loads each sibling module from ``tier-5-agentic/`` via ``importlib`` and
registers it under the dotted path callers expect:

    from tier_5_agentic.tools import search_text_chunks, lookup_paper_metadata
    from tier_5_agentic.agent import build_agent, DEFAULT_MODEL, INSTRUCTIONS
    from tier_5_agentic.main import main, DEFAULT_QUERY, MAX_TURNS

The shim is intentionally tiny — it owns no logic, only path resolution.
Each underlying module (``tools.py``, ``agent.py``, ``main.py``) remains a
clean library file inside ``tier-5-agentic/`` and is the source of truth.

Mirrors the ``tier_1_naive`` (Phase 128) and ``tier_3_graph`` (Phase 129)
shims verbatim. Same hyphen-dir → underscore-shim convention.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TIER5_DIR = Path(__file__).resolve().parent.parent / "tier-5-agentic"

# Make the hyphenated directory itself importable as a sys.path entry so
# that intra-tier flat imports (e.g. when main.py is executed as a script)
# continue to work, AND so that any module loaded via importlib below can
# resolve sibling imports like ``from tools import search_text_chunks``.
_tier5_str = str(_TIER5_DIR)
if _tier5_str not in sys.path:
    sys.path.insert(0, _tier5_str)


def _load(submodule: str):
    """Load ``tier-5-agentic/<submodule>.py`` and register as ``tier_5_agentic.<submodule>``."""
    full_name = f"{__name__}.{submodule}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    src = _TIER5_DIR / f"{submodule}.py"
    if not src.exists():
        raise ImportError(f"tier_5_agentic shim: missing source file {src}")
    spec = importlib.util.spec_from_file_location(full_name, src)
    if spec is None or spec.loader is None:
        raise ImportError(f"tier_5_agentic shim: could not load spec for {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Eagerly register all known tier-5-agentic submodules so that
# ``from tier_5_agentic.X import Y`` works without the caller knowing about
# the shim mechanics. Submodules may not yet exist during partial Plan
# execution; the try/except handles graceful skip.
for _name in ("tools", "agent", "main"):
    try:
        _load(_name)
    except ImportError:
        # Submodule not yet authored; skip gracefully so the shim still
        # loads when only a subset of modules has been written.
        pass

"""Importable shim for the on-disk ``tier-3-graph/`` package.

The on-disk package directory is ``tier-3-graph`` (hyphen) because tier
directories are organized for human readability and the docs/blog. Hyphens
are not valid Python identifiers, so ``tier-3-graph`` cannot be imported
directly. This shim package provides an importable name (``tier_3_graph``)
that loads each sibling module from ``tier-3-graph/`` via ``importlib`` and
registers it under the dotted path callers expect:

    from tier_3_graph.rag import build_rag, EMBED_DIMS
    from tier_3_graph.cost_adapter import CostAdapter
    from tier_3_graph.main import main, DEFAULT_QUERY  # added by Plan 05

The shim is intentionally tiny — it owns no logic, only path resolution.
Each underlying module (``rag.py``, ``cost_adapter.py``, etc.) remains a
clean library file inside ``tier-3-graph/`` and is the source of truth.

Mirrors the ``tier_1_naive`` shim from Phase 128 Plan 04 verbatim.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TIER3_DIR = Path(__file__).resolve().parent.parent / "tier-3-graph"

# Make the hyphenated directory itself importable as a sys.path entry so
# that intra-tier flat imports (e.g. when main.py is executed as a script)
# continue to work, AND so that any module loaded via importlib below can
# resolve sibling imports like ``from cost_adapter import CostAdapter``.
_tier3_str = str(_TIER3_DIR)
if _tier3_str not in sys.path:
    sys.path.insert(0, _tier3_str)


def _load(submodule: str):
    """Load ``tier-3-graph/<submodule>.py`` and register as ``tier_3_graph.<submodule>``."""
    full_name = f"{__name__}.{submodule}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    src = _TIER3_DIR / f"{submodule}.py"
    if not src.exists():
        raise ImportError(f"tier_3_graph shim: missing source file {src}")
    spec = importlib.util.spec_from_file_location(full_name, src)
    if spec is None or spec.loader is None:
        raise ImportError(f"tier_3_graph shim: could not load spec for {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Eagerly register all known tier-3-graph submodules so that
# ``from tier_3_graph.X import Y`` works without the caller knowing about
# the shim mechanics. ``ingest``, ``query``, and ``main`` may not yet exist
# during partial Plan execution (Plan 05 adds them); the try/except handles
# graceful skip.
for _name in ("rag", "cost_adapter", "ingest", "query", "main"):
    try:
        _load(_name)
    except ImportError:
        # Submodule not yet authored (Plan 05 territory); skip gracefully so
        # the shim still loads when only a subset of modules has been
        # written.
        pass

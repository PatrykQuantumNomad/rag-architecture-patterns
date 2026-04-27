"""Importable shim for the on-disk ``tier-4-multimodal/`` package.

The on-disk package directory is ``tier-4-multimodal`` (hyphen) for human
readability and the docs/blog. Hyphens are not valid Python identifiers, so
``tier-4-multimodal`` cannot be imported directly. This shim package
provides an importable name (``tier_4_multimodal``) that loads each sibling
module from ``tier-4-multimodal/`` via ``importlib`` and registers it under
the dotted path callers expect:

    from tier_4_multimodal.rag import build_rag, EMBED_DIMS
    from tier_4_multimodal.cost_adapter import CostAdapter
    from tier_4_multimodal.ingest_pdfs import ingest_pdfs
    from tier_4_multimodal.ingest_images import ingest_standalone_images
    from tier_4_multimodal.query import query  # added by Plan 03
    from tier_4_multimodal.main import main, DEFAULT_QUERY  # added by Plan 03

Mirrors the ``tier_1_naive`` (Phase 128 Plan 04) and ``tier_3_graph``
(Phase 129 Plan 03) shims verbatim — same path-resolution + sys.path
injection + graceful-skip-on-missing-source pattern. ``query`` and
``main`` are NOT yet authored (Plan 03 lands them); the loop below
gracefully skips any submodule whose source file does not yet exist.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TIER4_DIR = Path(__file__).resolve().parent.parent / "tier-4-multimodal"

# Make the hyphenated directory itself importable as a sys.path entry so
# that intra-tier flat imports (e.g. when main.py is executed as a script)
# continue to work, AND so that any module loaded via importlib below can
# resolve sibling imports like ``from cost_adapter import CostAdapter``.
_tier4_str = str(_TIER4_DIR)
if _tier4_str not in sys.path:
    sys.path.insert(0, _tier4_str)


def _load(submodule: str):
    """Load ``tier-4-multimodal/<submodule>.py`` and register as ``tier_4_multimodal.<submodule>``."""
    full_name = f"{__name__}.{submodule}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    src = _TIER4_DIR / f"{submodule}.py"
    if not src.exists():
        raise ImportError(f"tier_4_multimodal shim: missing source file {src}")
    spec = importlib.util.spec_from_file_location(full_name, src)
    if spec is None or spec.loader is None:
        raise ImportError(f"tier_4_multimodal shim: could not load spec for {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Eagerly register all known tier-4-multimodal submodules so that
# ``from tier_4_multimodal.X import Y`` works without the caller knowing
# about the shim mechanics. ``query`` and ``main`` may not yet exist
# during partial Plan execution (Plan 03 adds them); the try/except handles
# graceful skip.
for _name in ("rag", "cost_adapter", "ingest_pdfs", "ingest_images", "query", "main"):
    try:
        _load(_name)
    except ImportError:
        # Submodule not yet authored (Plan 03 territory); skip gracefully so
        # the shim still loads when only a subset of modules has been
        # written.
        pass

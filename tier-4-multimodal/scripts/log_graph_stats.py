"""Tier 4 — provenance writer for graphml + kv_store + library versions.

Purpose
-------
Reads ``rag_anything_storage/tier-4-multimodal/`` (graphml + kv_store JSONs),
captures pinned library versions via ``importlib.metadata``, and writes a
single Pydantic-typed JSON manifest to
``evaluation/results/diagnostics/tier-4-graph-stats-{TS}.json``.

The manifest is the immutable record of "this rebuild" — Pitfall 6 of
.planning/phases/02-tier-4-graphml-regeneration/02-RESEARCH.md notes that
LLM variance produces ±10% differences across re-runs of
``ingest_from_mineru.py``. Phase 9's frozen blog cites the manifest's
graphml_size_bytes + node/edge counts as the ground truth for the captured
eval; subsequent re-ingests do NOT invalidate the manifest, only produce
additional ones.

Read-only operation
-------------------
The script is read-only over ``rag_anything_storage/tier-4-multimodal/`` +
``importlib.metadata`` + ``git rev-parse``. It does NOT call
``insert_content_list``, does NOT spawn subprocesses other than ``git``,
and does NOT depend on ``OPENROUTER_API_KEY``.

Pydantic shape
--------------
Mirrors ``evaluation/harness/diagnostics.py`` (Phase 1 ``FallbackLog``):
``BaseModel`` subclass + ``model_dump_json(indent=2)`` for writes, single
ISO 8601 ``Z`` timestamp at construction. The typed shape lets Phase 9's
frozen-doc generator load the manifest with ``GraphStats(**json.load(f))``
and reference fields by name rather than by JSON-key string.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

import networkx as nx
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console  # noqa: E402

RAG_STORAGE = _REPO_ROOT / "rag_anything_storage" / "tier-4-multimodal"
DIAGNOSTICS_DIR = _REPO_ROOT / "evaluation" / "results" / "diagnostics"


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class GraphStats(BaseModel):
    """Frozen-doc-grade snapshot of a single Tier 4 graph rebuild.

    All fields are required so that loading a manifest via
    ``GraphStats(**json.load(...))`` raises pydantic ValidationError on
    schema drift rather than silently returning ``None`` for missing keys.
    """

    timestamp: str
    git_sha: str
    working_dir: str
    graphml_path: str
    graphml_node_count: int
    graphml_edge_count: int
    graphml_size_bytes: int
    graphml_mtime: str
    kv_full_docs_count: int
    kv_text_chunks_count: int
    kv_full_entities_count: int
    kv_full_relations_count: int
    raganything_version: str
    lightrag_version: str
    mineru_version: str


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the short git sha of HEAD, or ``"unknown"`` on any error.

    Mirrors ``tier-4-multimodal/scripts/eval_capture.py:_git_sha`` (project
    convention — keep the SHA-resolution path identical across diagnostics
    writers so the resulting JSON files are interchangeable).
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _iso_z_now() -> str:
    """ISO 8601 UTC with trailing Z (e.g. ``"2026-05-04T19:30:00Z"``).

    Mirrors ``CostTracker._iso_timestamp`` and ``eval_capture.py:92`` —
    no microseconds, no offset, single 'Z' suffix.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_pkg_version(name: str) -> str:
    """Return the installed version of ``name`` or ``"not-installed"``.

    Mirrors ``evaluation.harness.diagnostics._captured_versions`` — never
    raises so the manifest is always well-formed.
    """
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "not-installed"


def _kv_count(working_dir: Path, filename: str) -> int:
    """Return ``len(json.load(working_dir/filename))`` or 0 if missing.

    Defensive default: a partial-ingest may leave one of the kv_store
    files absent. The provenance writer should NOT crash in that case;
    it should record ``count=0`` and let the caller (or Phase 9's
    verifier) flag the anomaly.
    """
    p = working_dir / filename
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001 — malformed JSON also defaults to 0
        return 0
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    return 0


def collect_graph_stats(working_dir: Path) -> GraphStats:
    """Read the LightRAG storage tree under ``working_dir`` and snapshot.

    Raises ``FileNotFoundError`` if the graphml is absent — the caller
    must run ``ingest_from_mineru.py`` first to produce it. kv_store JSON
    files are loaded defensively (missing files default to count=0).
    """
    working_dir = Path(working_dir).resolve()
    graphml = working_dir / "graph_chunk_entity_relation.graphml"
    if not graphml.exists():
        raise FileNotFoundError(
            f"graphml missing at {graphml} — run ingest_from_mineru.py first"
        )
    g = nx.read_graphml(graphml)
    mtime = datetime.fromtimestamp(
        graphml.stat().st_mtime, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return GraphStats(
        timestamp=_iso_z_now(),
        git_sha=_git_sha(),
        working_dir=str(working_dir),
        graphml_path=str(graphml),
        graphml_node_count=g.number_of_nodes(),
        graphml_edge_count=g.number_of_edges(),
        graphml_size_bytes=graphml.stat().st_size,
        graphml_mtime=mtime,
        kv_full_docs_count=_kv_count(working_dir, "kv_store_full_docs.json"),
        kv_text_chunks_count=_kv_count(working_dir, "kv_store_text_chunks.json"),
        kv_full_entities_count=_kv_count(working_dir, "kv_store_full_entities.json"),
        kv_full_relations_count=_kv_count(working_dir, "kv_store_full_relations.json"),
        raganything_version=_safe_pkg_version("raganything"),
        lightrag_version=_safe_pkg_version("lightrag-hku"),
        mineru_version=_safe_pkg_version("mineru"),
    )


def write_stats(stats: GraphStats, diagnostics_dir: Path) -> Path:
    """Write the stats as indent-2 JSON to a timestamped file.

    Filename: ``tier-4-graph-stats-{ts_safe}.json`` where ``ts_safe`` is
    the timestamp with ``:`` replaced by ``_`` (Windows-safe). The
    directory is created on first use. Returns the resolved path.
    """
    diagnostics_dir = Path(diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    ts_safe = stats.timestamp.replace(":", "_")
    out = diagnostics_dir / f"tier-4-graph-stats-{ts_safe}.json"
    out.write_text(stats.model_dump_json(indent=2))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="log_graph_stats",
        description=(
            "Tier 4 graph-stats provenance writer. Reads "
            "rag_anything_storage/tier-4-multimodal/, captures library "
            "versions, writes a Pydantic-typed JSON manifest to "
            "evaluation/results/diagnostics/."
        ),
    )
    p.add_argument(
        "--working-dir",
        type=str,
        default=str(RAG_STORAGE),
        help=f"LightRAG storage dir to inspect (default {RAG_STORAGE}).",
    )
    p.add_argument(
        "--diagnostics-dir",
        type=str,
        default=str(DIAGNOSTICS_DIR),
        help=f"Output directory for the manifest (default {DIAGNOSTICS_DIR}).",
    )
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Print the JSON to stdout instead of writing a file.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        stats = collect_graph_stats(Path(args.working_dir))
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    if args.print_only:
        print(stats.model_dump_json(indent=2))
        return 0
    out = write_stats(stats, Path(args.diagnostics_dir))
    console.print(f"[green]Wrote {out}[/green]")
    console.print(
        f"[dim]nodes={stats.graphml_node_count} "
        f"edges={stats.graphml_edge_count} "
        f"docs={stats.kv_full_docs_count} "
        f"raganything={stats.raganything_version} "
        f"lightrag={stats.lightrag_version} "
        f"mineru={stats.mineru_version}[/dim]"
    )
    return 0


__all__ = [
    "GraphStats",
    "_git_sha",
    "_iso_z_now",
    "_safe_pkg_version",
    "_kv_count",
    "collect_graph_stats",
    "write_stats",
    "build_parser",
    "main",
    "RAG_STORAGE",
    "DIAGNOSTICS_DIR",
]


if __name__ == "__main__":
    raise SystemExit(main())

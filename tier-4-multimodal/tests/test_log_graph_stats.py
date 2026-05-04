"""Unit tests for tier-4-multimodal/scripts/log_graph_stats.py.

ZERO live operations. All tests use ``tmp_path`` to build a fake
LightRAG-shaped storage tree (graphml + kv_store JSONs) and exercise
the helpers + Pydantic model offline. The real
``rag_anything_storage/tier-4-multimodal/`` is NOT touched.

The script under test lives in a non-package directory
(``tier-4-multimodal/scripts/``); we load it via
``importlib.util.spec_from_file_location`` so tests don't depend on the
``tier_4_multimodal`` shim adding a ``scripts`` submodule.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "tier-4-multimodal" / "scripts" / "log_graph_stats.py"


def _load_module():
    """Load log_graph_stats.py without going through the shim."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "log_graph_stats_under_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


# Minimal graphml that networkx 3.x can read back to a 2-node, 1-edge graph.
# Indentation MUST start at column 0 — networkx is strict about XML headers.
_FAKE_GRAPHML_XML = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph id="G" edgedefault="directed">
    <node id="A"/>
    <node id="B"/>
    <edge source="A" target="B"/>
  </graph>
</graphml>
"""


def _build_fake_storage(tmp_path: Path) -> Path:
    """Build a minimal LightRAG-shaped storage tree at ``tmp_path/storage``."""
    wd = tmp_path / "storage"
    wd.mkdir()
    (wd / "graph_chunk_entity_relation.graphml").write_text(_FAKE_GRAPHML_XML)
    (wd / "kv_store_full_docs.json").write_text(json.dumps({"p1": {}, "p2": {}}))
    (wd / "kv_store_text_chunks.json").write_text(
        json.dumps({"c1": {}, "c2": {}, "c3": {}})
    )
    (wd / "kv_store_full_entities.json").write_text(json.dumps({"e1": {}}))
    (wd / "kv_store_full_relations.json").write_text(json.dumps({}))
    return wd


# ---------------------------------------------------------------------------
# collect_graph_stats
# ---------------------------------------------------------------------------


def test_collect_graph_stats_happy_path(tmp_path: Path) -> None:
    wd = _build_fake_storage(tmp_path)
    stats = mod.collect_graph_stats(wd)
    assert stats.graphml_node_count == 2
    assert stats.graphml_edge_count == 1
    assert stats.kv_full_docs_count == 2
    assert stats.kv_text_chunks_count == 3
    assert stats.kv_full_entities_count == 1
    assert stats.kv_full_relations_count == 0
    assert stats.graphml_size_bytes > 0
    # Library versions MUST come from the live importlib.metadata install,
    # not from a string hard-coded in the model. This guards against the
    # model accidentally returning placeholder strings.
    assert stats.raganything_version == "1.2.10"
    assert stats.lightrag_version == "1.4.15"
    assert stats.mineru_version == "3.1.4"


def test_collect_graph_stats_missing_graphml(tmp_path: Path) -> None:
    """The helper MUST refuse to write a manifest if the graphml is absent."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as exc_info:
        mod.collect_graph_stats(empty)
    assert "graphml" in str(exc_info.value).lower()


def test_collect_graph_stats_missing_kv_files(tmp_path: Path) -> None:
    """Missing kv_store JSONs default to count=0 (defensive — partial-ingest)."""
    wd = _build_fake_storage(tmp_path)
    # Delete one of the kv_store files BEFORE calling collect_graph_stats.
    (wd / "kv_store_full_relations.json").unlink()
    stats = mod.collect_graph_stats(wd)
    assert stats.kv_full_relations_count == 0
    # Other counts unaffected.
    assert stats.kv_full_docs_count == 2


# ---------------------------------------------------------------------------
# write_stats
# ---------------------------------------------------------------------------


def test_write_stats_creates_file(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    # Build a stats object by hand — set every required field.
    stats = mod.GraphStats(
        timestamp="2026-05-04T19:30:00Z",
        git_sha="abc1234",
        working_dir=str(tmp_path / "storage"),
        graphml_path=str(tmp_path / "storage" / "graph_chunk_entity_relation.graphml"),
        graphml_node_count=42,
        graphml_edge_count=99,
        graphml_size_bytes=12345,
        graphml_mtime="2026-05-04T19:29:00Z",
        kv_full_docs_count=75,
        kv_text_chunks_count=500,
        kv_full_entities_count=1000,
        kv_full_relations_count=2500,
        raganything_version="1.2.10",
        lightrag_version="1.4.15",
        mineru_version="3.1.4",
    )
    out = mod.write_stats(stats, diagnostics)
    assert out.exists()
    assert out.parent == diagnostics
    assert out.name.startswith("tier-4-graph-stats-")
    assert out.name.endswith(".json")
    # Roundtrip: load JSON and reconstruct GraphStats.
    loaded = json.loads(out.read_text())
    rehydrated = mod.GraphStats(**loaded)
    assert rehydrated.graphml_node_count == 42
    assert rehydrated.kv_full_docs_count == 75
    assert rehydrated.raganything_version == "1.2.10"


def test_write_stats_filename_replaces_colons(tmp_path: Path) -> None:
    """Colons in ISO timestamps break Windows filenames; we replace with _."""
    stats = mod.GraphStats(
        timestamp="2026-05-04T19:30:00Z",
        git_sha="abc1234",
        working_dir="/tmp/x",
        graphml_path="/tmp/x/graph.graphml",
        graphml_node_count=1,
        graphml_edge_count=0,
        graphml_size_bytes=10,
        graphml_mtime="2026-05-04T19:30:00Z",
        kv_full_docs_count=0,
        kv_text_chunks_count=0,
        kv_full_entities_count=0,
        kv_full_relations_count=0,
        raganything_version="1.2.10",
        lightrag_version="1.4.15",
        mineru_version="3.1.4",
    )
    out = mod.write_stats(stats, tmp_path / "diagnostics")
    assert ":" not in out.name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_iso_z_now_format() -> None:
    """ISO 8601 with trailing Z, no microseconds, no offset."""
    ts = mod._iso_z_now()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts), ts


def test_git_sha_returns_string() -> None:
    """_git_sha returns a non-empty string (real sha or 'unknown')."""
    sha = mod._git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0


# ---------------------------------------------------------------------------
# CLI surface (just smoke that the parser builds + main accepts argv)
# ---------------------------------------------------------------------------


def test_main_print_only_succeeds(tmp_path: Path, capsys) -> None:
    """`--print-only` skips the file write but exercises collect_graph_stats."""
    wd = _build_fake_storage(tmp_path)
    rc = mod.main(
        [
            "--working-dir",
            str(wd),
            "--diagnostics-dir",
            str(tmp_path / "diagnostics_unused"),
            "--print-only",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    # The JSON dump should appear on stdout.
    assert "graphml_node_count" in captured.out
    # And the diagnostics dir was NOT created.
    assert not (tmp_path / "diagnostics_unused").exists()


def test_main_writes_file(tmp_path: Path) -> None:
    wd = _build_fake_storage(tmp_path)
    diagnostics = tmp_path / "diagnostics"
    rc = mod.main(
        [
            "--working-dir",
            str(wd),
            "--diagnostics-dir",
            str(diagnostics),
        ]
    )
    assert rc == 0
    files = list(diagnostics.glob("tier-4-graph-stats-*.json"))
    assert len(files) == 1


def test_main_returns_2_on_missing_graphml(tmp_path: Path) -> None:
    """If the working_dir has no graphml, main returns 2 (not raise)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = mod.main(
        [
            "--working-dir",
            str(empty),
            "--diagnostics-dir",
            str(tmp_path / "diagnostics"),
        ]
    )
    assert rc == 2

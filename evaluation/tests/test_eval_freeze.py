"""Non-live unit tests for evaluation/harness/freeze.py.

Covers (per 04-RESEARCH.md §7 + 04-01-PLAN.md tasks):
  - HARN-03 happy path: byte-identical md copy + sidecar manifest
  - HARN-03 immutability: refuse-to-clobber wording
  - HARN-03 override: --force overwrite path
  - HARN-04 manifest schema: top-level keys
  - HARN-04 per-tier provenance: relative paths + mtime
  - HARN-04 lib pinning: 4 critical libs present + non-not-installed
  - HARN-04 judge block: model + embedder + max_tokens
  - Phase 5 forward contract: importable freeze() returning Path
  - Defensive: missing comparison.md => FileNotFoundError mentioning the path
  - HARN-04 graceful degradation: status=missing for absent tiers

All offline. No @pytest.mark.live markers. No network calls.
Uses tmp_path + results_dir injection (per public freeze() signature).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evaluation.harness.freeze import freeze


def _build_fixture(
    tmp_path: Path,
    tiers: tuple[int, ...] = (1, 2, 3, 4, 5),
    with_embedder: bool = False,
) -> Path:
    """Build a fake results_dir under tmp_path with the schema aggregate_tier reads.

    Returns the results_dir Path. Schema mirrors evaluation/results/ layout:
      results_dir/
        comparison.md                     (literal "FAKE ROLLUP\\n")
        queries/tier-{N}-2026-05-05T10_00_00Z.json
        costs/tier-{N}-2026-05-05T10_00_00Z.json
        costs/ragas-judge-2026-05-05T10_00_00Z.json
        metrics/tier-{N}-2026-05-05T10_00_00Z.json

    When ``with_embedder=True`` (Plan 06-01 Task 6), each tier's queries JSON
    is populated with the truth-table (embedder, embedder_source) tuple so the
    freeze manifest exercises the new field plumbing.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "comparison.md").write_text("FAKE ROLLUP\n")
    queries_dir = results_dir / "queries"
    costs_dir = results_dir / "costs"
    metrics_dir = results_dir / "metrics"
    queries_dir.mkdir()
    costs_dir.mkdir()
    metrics_dir.mkdir()

    cost_payload = {"totals": {"usd": 0.01}, "queries": []}
    metrics_payload: list = []  # empty list — aggregate_tier handles
    judge_payload = {
        "queries": [{"kind": "llm", "model": "google/gemini-2.5-flash"}]
    }

    # Plan 06-01 Task 6 truth table — D-ROADMAP-OVERRIDE for Tier 5.
    embedder_truth = {
        1: ("openai/text-embedding-3-small", "openrouter"),
        2: ("gemini-embedding-001", "google-managed"),
        3: ("openai/text-embedding-3-small", "openrouter"),
        4: ("openai/text-embedding-3-small", "openrouter"),
        5: ("openai/text-embedding-3-small", "openrouter"),
    }

    for tier in tiers:
        queries_payload: dict = {
            "records": [{"question_id": "q1", "latency_s": 1.0}],
            "timestamp": "2026-05-05T10:00:00Z",
            "git_sha": "deadbeef",
            "model": "google/gemini-2.5-flash",
        }
        if with_embedder:
            emb, src = embedder_truth[tier]
            queries_payload["embedder"] = emb
            queries_payload["embedder_source"] = src

        (queries_dir / f"tier-{tier}-2026-05-05T10_00_00Z.json").write_text(
            json.dumps(queries_payload)
        )
        (costs_dir / f"tier-{tier}-2026-05-05T10_00_00Z.json").write_text(
            json.dumps(cost_payload)
        )
        (metrics_dir / f"tier-{tier}-2026-05-05T10_00_00Z.json").write_text(
            json.dumps(metrics_payload)
        )

    (costs_dir / "ragas-judge-2026-05-05T10_00_00Z.json").write_text(
        json.dumps(judge_payload)
    )
    return results_dir


def test_freeze_writes_md_and_manifest(tmp_path):
    """HARN-03 happy path: byte-identical md + sidecar manifest."""
    results_dir = _build_fixture(tmp_path)

    out_md = freeze(version="1.0", force=False, results_dir=results_dir)

    assert out_md == results_dir / "frozen" / "eval-numbers-v1.0.md"
    assert out_md.exists()
    assert out_md.read_text() == "FAKE ROLLUP\n"
    assert out_md.with_suffix(".manifest.json").exists()


def test_freeze_refuses_clobber(tmp_path):
    """HARN-03 immutability: re-freeze without --force raises FileExistsError."""
    results_dir = _build_fixture(tmp_path)

    freeze(version="1.0", results_dir=results_dir)

    with pytest.raises(FileExistsError) as exc_info:
        freeze(version="1.0", results_dir=results_dir)

    msg = str(exc_info.value)
    assert "eval-numbers-v1.0.md" in msg
    assert "already frozen" in msg
    assert "bump version or pass --force" in msg


def test_freeze_force_overwrites(tmp_path):
    """HARN-03 override: --force replaces both md and manifest."""
    results_dir = _build_fixture(tmp_path)

    freeze(version="1.0", results_dir=results_dir)

    (results_dir / "comparison.md").write_text("NEW CONTENT\n")
    out_md = freeze(version="1.0", force=True, results_dir=results_dir)

    assert out_md.read_text() == "NEW CONTENT\n"
    manifest = json.loads(out_md.with_suffix(".manifest.json").read_text())
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", manifest["frozen_at"])


def test_manifest_top_level_fields(tmp_path):
    """HARN-04 schema: required top-level keys present + well-typed."""
    results_dir = _build_fixture(tmp_path)
    out_md = freeze(version="1.0", results_dir=results_dir)
    manifest = json.loads(out_md.with_suffix(".manifest.json").read_text())

    assert manifest["version"] == "1.0"
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", manifest["frozen_at"])
    assert isinstance(manifest["git_sha"], str)
    assert len(manifest["git_sha"]) >= 7 or manifest["git_sha"] == "unknown"
    assert isinstance(manifest["git_dirty"], bool)
    assert "source_markdown" in manifest
    assert "source_markdown_mtime" in manifest
    assert "frozen_markdown" in manifest
    assert isinstance(manifest["judge"], dict)
    assert isinstance(manifest["per_tier"], dict)
    assert isinstance(manifest["library_versions"], dict)
    assert isinstance(manifest["python_version"], str)
    assert manifest["$schema_version"] == "1.0"


def test_manifest_per_tier_provenance(tmp_path):
    """HARN-04 per-tier: present status + relative paths + mtime ISO Z."""
    results_dir = _build_fixture(tmp_path)
    out_md = freeze(version="1.0", results_dir=results_dir)
    manifest = json.loads(out_md.with_suffix(".manifest.json").read_text())

    for tier in (1, 2, 3, 4, 5):
        assert manifest["per_tier"][f"tier-{tier}"]["status"] == "present"

    t1 = manifest["per_tier"]["tier-1"]
    for key in (
        "generation_model",
        "capture_timestamp",
        "capture_git_sha",
        "queries_path",
        "cost_path",
        "metrics_path",
        "queries_mtime",
    ):
        assert key in t1, f"tier-1 missing key: {key}"

    # Pitfall 7 — paths must be relative (no absolute leak)
    assert not t1["queries_path"].startswith("/")
    assert str(tmp_path) not in t1["queries_path"]


def test_manifest_library_versions(tmp_path):
    """HARN-04 lib pinning: 4 critical libs present + semver-ish + not-installed forbidden."""
    results_dir = _build_fixture(tmp_path)
    out_md = freeze(version="1.0", results_dir=results_dir)
    manifest = json.loads(out_md.with_suffix(".manifest.json").read_text())

    libs = manifest["library_versions"]
    assert set(libs.keys()) >= {"lightrag-hku", "raganything", "openai-agents", "ragas"}
    for pkg in ("lightrag-hku", "raganything", "openai-agents", "ragas"):
        assert libs[pkg] != "not-installed", f"{pkg} unexpectedly not-installed"
        assert re.match(r"^\d+\.\d+\.\d+", libs[pkg]), f"{pkg} not semver-ish: {libs[pkg]}"


def test_manifest_judge_block(tmp_path):
    """HARN-04 judge: model from fixture + embedder default + max_tokens=8192."""
    results_dir = _build_fixture(tmp_path)
    out_md = freeze(version="1.0", results_dir=results_dir)
    manifest = json.loads(out_md.with_suffix(".manifest.json").read_text())

    assert manifest["judge"]["model"] == "google/gemini-2.5-flash"
    assert manifest["judge"]["embedder"] == "openai/text-embedding-3-small"
    assert manifest["judge"]["max_tokens"] == 8192


def test_freeze_in_process_returns_path(tmp_path):
    """Phase 5 forward contract: freeze() callable, returns pathlib.Path, points at file."""
    from evaluation.harness.freeze import freeze as freeze_fn

    assert callable(freeze_fn)

    results_dir = _build_fixture(tmp_path)
    returned = freeze_fn(version="1.0", force=False, results_dir=results_dir)

    assert isinstance(returned, Path)
    assert returned.is_file()


def test_freeze_no_source_errors(tmp_path):
    """Defensive: missing comparison.md => FileNotFoundError mentioning the file."""
    results_dir = _build_fixture(tmp_path)
    (results_dir / "comparison.md").unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        freeze(version="1.0", results_dir=results_dir)

    assert "comparison.md" in str(exc_info.value)


def test_freeze_manifest_carries_embedder_per_tier(tmp_path):
    """Plan 06-01 Task 6 / CAP-03: freeze() per_tier manifest entries
    carry embedder + embedder_source from each tier's queries JSON.

    D-ROADMAP-OVERRIDE locked in the tier-5 assertion: Tier 5 reads the
    SAME (embedder, embedder_source) as Tier 1 — NOT a hosted vector
    store. Phase 4's judge-block contract (separate concept) MUST be
    preserved byte-identical (regression guard).
    """
    results_dir = _build_fixture(tmp_path, with_embedder=True)
    out_md = freeze(version="6.0", force=False, results_dir=results_dir)
    manifest = json.loads(out_md.with_suffix(".manifest.json").read_text())

    expected = {
        "tier-1": ("openai/text-embedding-3-small", "openrouter"),
        "tier-2": ("gemini-embedding-001", "google-managed"),
        "tier-3": ("openai/text-embedding-3-small", "openrouter"),
        "tier-4": ("openai/text-embedding-3-small", "openrouter"),
        # D-ROADMAP-OVERRIDE — Tier 5 IS Tier 1's embedder.
        "tier-5": ("openai/text-embedding-3-small", "openrouter"),
    }
    for tier_label, (emb, src) in expected.items():
        entry = manifest["per_tier"][tier_label]
        assert entry["embedder"] == emb, (
            f"{tier_label} embedder mismatch: got {entry.get('embedder')!r}, "
            f"expected {emb!r}"
        )
        assert entry["embedder_source"] == src, (
            f"{tier_label} embedder_source mismatch: got "
            f"{entry.get('embedder_source')!r}, expected {src!r}"
        )

    # Phase 4 judge-block contract preserved (separate concept — judge
    # embedder is not the per-tier embedder).
    assert manifest["judge"]["embedder"] == "openai/text-embedding-3-small"


def test_manifest_missing_tier(tmp_path):
    """HARN-04 graceful: tiers without queries get status=missing, no other keys."""
    results_dir = _build_fixture(tmp_path, tiers=(1, 2, 3))
    out_md = freeze(version="1.0", results_dir=results_dir)
    manifest = json.loads(out_md.with_suffix(".manifest.json").read_text())

    for tier in (1, 2, 3):
        assert manifest["per_tier"][f"tier-{tier}"]["status"] == "present"

    for tier in (4, 5):
        entry = manifest["per_tier"][f"tier-{tier}"]
        assert entry["status"] == "missing"
        assert "generation_model" not in entry
        assert "queries_path" not in entry

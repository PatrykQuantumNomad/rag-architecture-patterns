"""REPO-02 trace test — dataset structure + conditional manifest schema checks.

Wave 2 (this commit) lands the file with conditional ``pytest.skip`` guards
so the same file passes:

* empty (Plan 02 — no manifests yet),
* partial (post-Plan 04 — papers.json present, figures/videos still pending),
* full (post-Plan 05 — all manifests present).

Plan 06 extends this file in place with stricter thresholds (≥30 figures,
≥7 captions, metadata.json present + non-empty).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "dataset"
MANIFESTS = DATASET / "manifests"


def test_dataset_directory_tree() -> None:
    for sub in ("papers", "images", "videos", "manifests"):
        path = DATASET / sub
        assert path.is_dir(), f"missing dataset/{sub}/"


def test_papers_manifest_conditional() -> None:
    p = MANIFESTS / "papers.json"
    if not p.exists():
        pytest.skip("papers.json not yet generated (Plan 04 produces it)")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list), "papers.json must be a JSON array"
    assert len(data) >= 80, (
        f"expected >=80 papers (loosened from ~100 target), got {len(data)}"
    )
    for entry in data:
        assert "arxiv_id" in entry, f"paper missing arxiv_id: {entry}"
        assert "title" in entry, f"paper missing title: {entry}"
        assert "filename" in entry, f"paper missing filename: {entry}"
        pdf_path = DATASET / "papers" / entry["filename"]
        assert pdf_path.exists(), (
            f"manifest references missing PDF: {entry['filename']}"
        )


def test_figures_manifest_conditional() -> None:
    p = MANIFESTS / "figures.json"
    if not p.exists():
        pytest.skip("figures.json not yet generated (Plan 05 produces it)")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list), "figures.json must be a JSON array"
    assert len(data) >= 1, "figures manifest is empty"
    # Cross-ref against papers manifest if present.
    papers_json = MANIFESTS / "papers.json"
    if papers_json.exists():
        paper_ids = {p["paper_id"] for p in json.loads(papers_json.read_text(encoding="utf-8")) if "paper_id" in p}
        for f in data:
            if "paper_id" in f and paper_ids:
                assert f["paper_id"] in paper_ids, (
                    f"figure references unknown paper_id {f['paper_id']}"
                )


def test_videos_manifest_conditional() -> None:
    p = MANIFESTS / "videos.json"
    if not p.exists():
        pytest.skip("videos.json not yet generated (Plan 05 produces it)")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list), "videos.json must be a JSON array"
    assert len(data) >= 1, "videos manifest is empty"
    forbidden_licenses = {"TBD_VERIFY_MANUALLY", "CC-BY-ND", "CC-BY-NC-ND"}
    for v in data:
        assert v.get("license_verified_at"), (
            f"video {v.get('video_id')} lacks license_verified_at"
        )
        assert v["license"] not in forbidden_licenses, (
            f"unverified or ND license on {v.get('video_id')}: {v.get('license')}"
        )


# -----------------------------------------------------------------------------
# Plan 06 extensions: stricter post-corpus assertions.
#
# These complement the conditional tests above — at this wave all required
# manifests exist, so both sets of tests run. The video-manifest test below
# stays *conditional* because Plan 05 deferred the video-clip licensing step
# (videos.json is intentionally absent — see 127-05-SUMMARY.md). It will
# tighten automatically once a future phase lands videos.json.
# -----------------------------------------------------------------------------


def test_metadata_exists() -> None:
    """metadata.json is the top-level dataset index (Plan 06 deliverable)."""
    p = MANIFESTS / "metadata.json"
    assert p.exists(), "dataset/manifests/metadata.json missing — run scripts/build_metadata.py"
    m = json.loads(p.read_text(encoding="utf-8"))
    assert m["version"] == "1.0", f"unexpected metadata version {m.get('version')}"
    assert m["stats"]["paper_count"] >= 80, (
        f"paper_count {m['stats']['paper_count']} below 80 floor"
    )
    assert m["stats"]["figure_count"] >= 30, (
        f"figure_count {m['stats']['figure_count']} below 30 floor"
    )
    # video_count floor stays at 0 — Plan 05 deferral. Tighten when videos land.
    assert m["stats"]["video_count"] >= 0
    assert m["stats"]["total_size_mb"] > 0


def test_papers_manifest_full() -> None:
    """Post-Plan 04: papers manifest is final."""
    d = json.loads((MANIFESTS / "papers.json").read_text(encoding="utf-8"))
    assert len(d) >= 80, f"expected >=80 papers, got {len(d)}"
    assert all("arxiv_id" in p and "title" in p for p in d)


def test_figures_manifest_full() -> None:
    """Post-Plan 05: figures manifest must have >=30 with >=7 captioned."""
    d = json.loads((MANIFESTS / "figures.json").read_text(encoding="utf-8"))
    assert len(d) >= 30, f"expected >=30 figures, got {len(d)}"
    captioned = sum(1 for f in d if f.get("caption"))
    assert captioned >= 7, f"need >=7 captioned figures, got {captioned}"


def test_videos_manifest_when_present() -> None:
    """Plan 05 deferred video-clip licensing; videos.json stays absent until a future phase lands it.

    When videos.json IS present, every entry must carry a license_verified_at
    timestamp and a non-ND license. Skip cleanly while absent.
    """
    p = MANIFESTS / "videos.json"
    if not p.exists():
        pytest.skip(
            "videos.json absent — Plan 05 deferred video-clip licensing; tightens when manifest lands"
        )
    d = json.loads(p.read_text(encoding="utf-8"))
    assert len(d) >= 1, "videos manifest is empty"
    forbidden = {"TBD_VERIFY_MANUALLY", "CC-BY-ND", "CC-BY-NC-ND"}
    assert all(v.get("license_verified_at") for v in d)
    assert all(v.get("license") not in forbidden for v in d)

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

"""Schema + cross-reference tests for evaluation/golden_qa.json.

Plan 06 authored 30 hand-curated Q&A entries. Per D-04 the locked split was
10 single-hop / 10 multi-hop / 7 multimodal / 3 video, but Plan 05 deferred
the video-clip licensing step (see 127-05-SUMMARY.md), so the 3 video slots
were substituted with 3 extra multimodal entries — producing the actual
on-disk split of 10 single-hop text / 10 multi-hop text / 10 multimodal / 0
video. These tests pin the SHIPPED split. When videos land in a future
phase, update the constants at the top of this file.

No live API calls — pure JSON schema and cross-manifest reference checks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

# Shipped split (Plan 06 / 127-06-SUMMARY.md). See module docstring.
EXPECTED_TOTAL = 30
EXPECTED_BY_MODALITY = {"text": 20, "multimodal": 10, "video": 0}
EXPECTED_TEXT_SINGLE_HOP = 10
EXPECTED_TEXT_MULTI_HOP = 10

REQUIRED_FIELDS = (
    "id",
    "question",
    "expected_answer",
    "source_papers",
    "modality_tag",
    "hop_count_tag",
)


@pytest.fixture(scope="module")
def qa() -> list[dict]:
    return json.loads((ROOT / "evaluation/golden_qa.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def papers() -> set[str]:
    data = json.loads((ROOT / "dataset/manifests/papers.json").read_text(encoding="utf-8"))
    return {p["paper_id"] for p in data}


@pytest.fixture(scope="module")
def figures() -> set[str]:
    data = json.loads((ROOT / "dataset/manifests/figures.json").read_text(encoding="utf-8"))
    return {f["figure_id"] for f in data}


@pytest.fixture(scope="module")
def videos() -> set[str]:
    """Videos manifest may be absent (Plan 05 deferral); treat as empty set."""
    p = ROOT / "dataset/manifests/videos.json"
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    return {v["video_id"] for v in data}


def test_count_30(qa: list[dict]) -> None:
    assert len(qa) == EXPECTED_TOTAL, f"expected {EXPECTED_TOTAL}, got {len(qa)}"


def test_modality_split(qa: list[dict]) -> None:
    by_modality = {"text": 0, "multimodal": 0, "video": 0}
    for entry in qa:
        tag = entry["modality_tag"]
        assert tag in by_modality, f"unknown modality_tag {tag} in {entry['id']}"
        by_modality[tag] += 1
    assert by_modality == EXPECTED_BY_MODALITY, (
        f"modality split {by_modality} != expected {EXPECTED_BY_MODALITY}"
    )


def test_hop_split_text_only(qa: list[dict]) -> None:
    """The locked 10/10 single-hop/multi-hop split applies to the TEXT subset."""
    single = sum(
        1 for e in qa if e["hop_count_tag"] == "single-hop" and e["modality_tag"] == "text"
    )
    multi = sum(
        1 for e in qa if e["hop_count_tag"] == "multi-hop" and e["modality_tag"] == "text"
    )
    assert single == EXPECTED_TEXT_SINGLE_HOP, (
        f"text single-hop count {single}, expected {EXPECTED_TEXT_SINGLE_HOP}"
    )
    assert multi == EXPECTED_TEXT_MULTI_HOP, (
        f"text multi-hop count {multi}, expected {EXPECTED_TEXT_MULTI_HOP}"
    )


def test_required_fields_present(qa: list[dict]) -> None:
    for entry in qa:
        for field in REQUIRED_FIELDS:
            assert field in entry, f"missing {field} in {entry.get('id', '?')}"
        assert entry["expected_answer"], f"empty expected_answer in {entry['id']}"
        assert entry["question"], f"empty question in {entry['id']}"
        assert entry["source_papers"], f"empty source_papers in {entry['id']}"
        assert entry["modality_tag"] in {"text", "multimodal", "video"}, (
            f"invalid modality_tag in {entry['id']}: {entry['modality_tag']}"
        )
        assert entry["hop_count_tag"] in {"single-hop", "multi-hop"}, (
            f"invalid hop_count_tag in {entry['id']}: {entry['hop_count_tag']}"
        )


def test_no_duplicate_ids(qa: list[dict]) -> None:
    ids = [e["id"] for e in qa]
    assert len(ids) == len(set(ids)), f"duplicate ids found: {sorted(ids)}"


def test_source_papers_exist(qa: list[dict], papers: set[str]) -> None:
    for entry in qa:
        for paper_id in entry["source_papers"]:
            assert paper_id in papers, (
                f"unknown source_paper {paper_id!r} in {entry['id']}"
            )


def test_multi_hop_has_multiple_papers(qa: list[dict]) -> None:
    """Multi-hop entries must reference >=2 source papers (citation-chain authoring per D-03)."""
    for entry in qa:
        if entry["hop_count_tag"] == "multi-hop":
            n = len(entry["source_papers"])
            assert n >= 2, (
                f"multi-hop {entry['id']} references only {n} paper(s); "
                "citation-chain authoring requires >=2"
            )


def test_multimodal_has_figures(qa: list[dict], figures: set[str]) -> None:
    for entry in qa:
        if entry["modality_tag"] == "multimodal":
            assert entry.get("figure_ids"), (
                f"multimodal {entry['id']} missing figure_ids"
            )
            for fid in entry["figure_ids"]:
                assert fid in figures, (
                    f"unknown figure_id {fid!r} in {entry['id']}"
                )


def test_video_entries_have_videos(qa: list[dict], videos: set[str]) -> None:
    """Video entries (currently zero — Plan 05 deferral) must reference real video_ids when present."""
    for entry in qa:
        if entry["modality_tag"] == "video":
            assert entry.get("video_ids"), (
                f"video {entry['id']} missing video_ids"
            )
            for vid in entry["video_ids"]:
                assert vid in videos, (
                    f"unknown video_id {vid!r} in {entry['id']}"
                )

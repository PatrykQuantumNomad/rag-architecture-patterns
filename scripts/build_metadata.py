"""Aggregate papers/figures/videos manifests into dataset/manifests/metadata.json.

Idempotent. Re-run after corpus changes. NOT a curation tool — it never touches
the source manifests; it only indexes them.

Plan 06 ripple: dataset/manifests/videos.json is intentionally absent (Plan 05
deferred video-clip licensing verification — see 127-05-SUMMARY.md). The
aggregator treats absent videos.json as an empty list (video_count=0) so the
metadata index stays buildable without dropping the manifests.videos field.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path


def _load_manifest(path: Path) -> list[dict]:
    """Load a manifest JSON list, returning [] if the file is absent.

    Absent files are treated as empty rather than errors so build_metadata
    runs cleanly across wave boundaries (e.g., pre-videos.json wave) and
    reflects the true on-disk state.
    """
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def build(repo_root: Path = Path(".")) -> dict:
    ds = repo_root / "dataset"
    papers = _load_manifest(ds / "manifests/papers.json")
    figures = _load_manifest(ds / "manifests/figures.json")
    videos = _load_manifest(ds / "manifests/videos.json")
    total_bytes = sum(f.stat().st_size for f in ds.rglob("*") if f.is_file())
    return {
        "version": "1.0",
        "phase": "127-repository-skeleton-enterprise-dataset",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "manifests": {
            "papers": "manifests/papers.json",
            "figures": "manifests/figures.json",
            "videos": "manifests/videos.json",
        },
        "evaluation": {"golden_qa": "../evaluation/golden_qa.json"},
        "stats": {
            "paper_count": len(papers),
            "figure_count": len(figures),
            "video_count": len(videos),
            "total_size_mb": round(total_bytes / 1024 / 1024, 1),
        },
    }


def main() -> None:
    md = build()
    out = Path("dataset/manifests/metadata.json")
    out.write_text(json.dumps(md, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(md, indent=2))


if __name__ == "__main__":
    main()

"""Unit tests for ``shared.loader.DatasetLoader``.

Validates that loader is robust to the staged-dataset reality: ``manifests/*``
files materialize across Plans 04 (papers), 05 (figures, videos), and 06
(metadata). Tests therefore exercise BOTH the empty / pre-Plan-04 state and
the populated state via tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path

from shared.loader import DatasetLoader


def _build_dataset(tmp_path: Path) -> Path:
    """Create a fake dataset/manifests/ tree under tmp_path."""
    root = tmp_path / "dataset"
    (root / "manifests").mkdir(parents=True)
    return root


def test_papers_returns_empty_list_when_manifest_missing(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    loader = DatasetLoader(dataset_root=root)
    assert loader.papers() == []


def test_figures_returns_empty_list_when_manifest_missing(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    loader = DatasetLoader(dataset_root=root)
    assert loader.figures() == []


def test_videos_returns_empty_list_when_manifest_missing(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    loader = DatasetLoader(dataset_root=root)
    assert loader.videos() == []


def test_metadata_returns_empty_dict_when_manifest_missing(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    loader = DatasetLoader(dataset_root=root)
    assert loader.metadata() == {}


def test_papers_reads_manifest_when_present(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    payload = [{"paper_id": "x1", "title": "Demo", "filename": "x1.pdf"}]
    (root / "manifests" / "papers.json").write_text(json.dumps(payload))
    loader = DatasetLoader(dataset_root=root)
    assert loader.papers() == payload


def test_figures_reads_manifest_when_present(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    payload = [{"figure_id": "f1", "paper_id": "x1", "filename": "f1.png"}]
    (root / "manifests" / "figures.json").write_text(json.dumps(payload))
    loader = DatasetLoader(dataset_root=root)
    assert loader.figures() == payload


def test_videos_reads_manifest_when_present(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    payload = [
        {
            "video_id": "v1",
            "title": "Talk",
            "license": "CC-BY",
            "license_verified_at": "2026-04-25",
        }
    ]
    (root / "manifests" / "videos.json").write_text(json.dumps(payload))
    loader = DatasetLoader(dataset_root=root)
    assert loader.videos() == payload


def test_metadata_reads_manifest_when_present(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    payload = {"generated_at": "2026-04-25", "paper_count": 1}
    (root / "manifests" / "metadata.json").write_text(json.dumps(payload))
    loader = DatasetLoader(dataset_root=root)
    assert loader.metadata() == payload


def test_default_dataset_root_does_not_call_settings() -> None:
    """Loader must work without GEMINI_API_KEY (no get_settings() at construction)."""
    # If this constructor instantiated Settings(), Pydantic would raise here
    # under a fresh checkout without .env. We expect it to succeed silently.
    loader = DatasetLoader()
    # papers() may return [] or a real list depending on dataset state.
    result = loader.papers()
    assert isinstance(result, list)

"""Tier 4 — standalone-image ingest via ``insert_content_list`` (Pattern 3).

Reads ``dataset/manifests/figures.json`` (Phase 127's 581-image set) and
hands RAG-Anything a content list of ``{type: image, img_path, ...}`` entries
for the multimodal vision pass.

CRITICAL — Pitfall 4
--------------------
``img_path`` MUST be absolute. Relative paths are silently accepted by
``insert_content_list`` but produce empty image entities in
``vdb_entities.json`` (RAG-Anything resolves the path against an internal
working directory rather than the caller's CWD). Both code paths below
call ``Path(...).resolve()`` BEFORE composing the entry.

The pure helper ``build_image_content_list`` is exposed alongside the
``async`` ingest function so non-live tests can assert the content_list
shape without touching the real ``rag`` (no LLM calls, no model fetch).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


async def ingest_standalone_images(rag, dataset_root: Path) -> int:
    """Ingest the standalone-figures bundle into the RAG-Anything store.

    Returns the number of figures actually queued for ingest (manifest-
    present and on-disk-present). Returns 0 if the manifest is missing or
    no figures resolve on disk (callers can no-op the ingest cleanly).
    """
    figures_path = dataset_root / "manifests" / "figures.json"
    if not figures_path.exists():
        return 0
    figures = json.loads(figures_path.read_text())
    images_dir = (dataset_root / "images").resolve()  # absolute (Pitfall 4)
    content_list = build_image_content_list(figures, images_dir)
    if not content_list:
        return 0
    await rag.insert_content_list(
        content_list=content_list,
        file_path="dataset_figures_bundle",  # synthetic source path for KG node
        doc_id="figures-bundle",
    )
    return len(content_list)


def build_image_content_list(
    figures: Iterable[dict], images_dir: Path
) -> list[dict]:
    """Pure builder — exposed for non-live testing (no rag side effects).

    Same logic the async path uses, factored out so tests can fabricate a
    synthetic ``figures`` list + ``tmp_path`` images_dir and assert the
    content-list shape without invoking the live RAG-Anything store.

    Filters out manifest entries whose ``filename`` is missing on disk
    (matches the live path's behavior — the live ingest never sends a
    nonexistent ``img_path`` to RAG-Anything).
    """
    images_dir = Path(images_dir).resolve()  # absolute (Pitfall 4)
    out: list[dict] = []
    for i, f in enumerate(figures):
        img_path = images_dir / f["filename"]
        if not img_path.exists():
            continue
        captions = [f["caption"]] if f.get("caption") else []
        out.append(
            {
                "type": "image",
                "img_path": str(img_path),
                "image_caption": captions,
                "page_idx": i,
            }
        )
    return out

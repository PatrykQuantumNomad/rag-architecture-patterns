"""Non-live unit tests for tier-4-multimodal/ingest_images.py content_list shape.

Exercises the pure ``build_image_content_list`` helper (no rag side effects,
no LLM calls, no MineRU model fetch). Asserts the Pitfall 4 invariant
(absolute ``img_path``) and the empty-input edge case.

Pitfall 4 (re-stated)
---------------------
RAG-Anything's ``insert_content_list`` silently accepts relative ``img_path``
strings but produces empty image entities in ``vdb_entities.json`` because it
resolves the path against an internal working directory rather than the
caller's CWD. The pure helper calls ``Path(...).resolve()`` BEFORE composing
each entry; this test asserts the resulting paths are absolute.
"""
from __future__ import annotations

from pathlib import Path

from tier_4_multimodal.ingest_images import build_image_content_list


def test_content_list_uses_absolute_img_path(tmp_path):
    """Manifest entries that resolve on disk produce absolute-path content entries.

    Synthetic 3-figure manifest:
      * a.png — exists, has caption → keep, caption=[caption]
      * b.png — exists, no caption  → keep, caption=[]
      * missing.png — absent on disk → drop entirely
    """
    images = tmp_path / "images"
    images.mkdir()
    # Existence is what matters; the helper does not validate the bytes.
    (images / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (images / "b.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    figures = [
        {"filename": "a.png", "caption": "fig a"},
        {"filename": "b.png"},  # no caption
        {"filename": "missing.png"},  # absent — should be filtered
    ]
    out = build_image_content_list(figures, images)

    # Filtered shape: 2 of 3 figures land in the content list.
    assert len(out) == 2

    for entry in out:
        assert entry["type"] == "image"
        assert isinstance(entry["image_caption"], list)
        # Pitfall 4 — img_path MUST be absolute.
        assert Path(entry["img_path"]).is_absolute(), (
            "Pitfall 4 — img_path must be absolute or RAG-Anything silently "
            "produces empty image entities in vdb_entities.json"
        )
        # page_idx is preserved as the manifest position (informational).
        assert "page_idx" in entry

    # Caption shape: a.png keeps caption; b.png has no caption (empty list).
    captions = {Path(e["img_path"]).name: e["image_caption"] for e in out}
    assert captions["a.png"] == ["fig a"]
    assert captions["b.png"] == []


def test_content_list_empty_when_no_figures(tmp_path):
    """Empty manifest produces empty content list (no exceptions, no entries)."""
    out = build_image_content_list([], tmp_path)
    assert out == []

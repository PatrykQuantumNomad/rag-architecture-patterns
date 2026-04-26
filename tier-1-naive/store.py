"""Tier 1: ChromaDB persistence layer.

Owns the on-disk path convention (chroma_db/tier-1-naive/) — Tier 5 reads
from this same path but writes its own indices to chroma_db/tier-5-agentic/
(128-RESEARCH.md Pitfall 8 — collection collisions across tiers).

Cosine HNSW is configured at FIRST creation only — re-passing the metadata
to get_or_create_collection is silently ignored in ChromaDB 1.x
(Pitfall 4). To change indexing parameters, use reset=True.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

# Tier 1 owns this path exclusively. Plan 04 of Phase 130 (Tier 5) reads
# from it as a shared baseline but writes to its own subdirectory.
CHROMA_PATH: Path = Path("chroma_db") / "tier-1-naive"
COLLECTION_NAME: str = "enterprise_kb_naive"


def open_collection(reset: bool = False, path: Path | None = None):
    """Open or create the Tier 1 ChromaDB collection.

    Parameters
    ----------
    reset
        If True, delete the collection before re-creating. Use this
        whenever you change indexing parameters (HNSW config) — Pitfall 4.
    path
        Override the on-disk path (tests pass a tmp_path). Defaults to
        ``CHROMA_PATH``.

    Returns
    -------
    chromadb.api.models.Collection.Collection
    """
    chroma_path = Path(path) if path is not None else CHROMA_PATH
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            # Collection didn't exist — that's fine; reset is idempotent.
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

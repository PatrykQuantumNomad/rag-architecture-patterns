"""Non-live unit tests for tier-1 store + retrieve layers.

Uses a tmp_path PersistentClient + synthetic 1536-dim vectors — no
OpenAI calls, no real PDFs.

Import strategy: the on-disk package directory is ``tier-1-naive`` (hyphen),
which Python cannot import as ``tier_1_naive`` directly (hyphens are not
valid identifiers). We load ``store.py`` and ``retrieve.py`` via
``importlib.util`` and pin the modules under synthetic names for the
test session — same pattern as ``test_chunker.py`` from Plan 02.
"""
from __future__ import annotations

import importlib.util
import pathlib
import random
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_STORE_PATH = _REPO_ROOT / "tier-1-naive" / "store.py"
_RETRIEVE_PATH = _REPO_ROOT / "tier-1-naive" / "retrieve.py"


def _import_store_and_retrieve():
    if "_tier1_store" in sys.modules:
        store_mod = sys.modules["_tier1_store"]
    else:
        spec = importlib.util.spec_from_file_location("_tier1_store", _STORE_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load store module from {_STORE_PATH}")
        store_mod = importlib.util.module_from_spec(spec)
        sys.modules["_tier1_store"] = store_mod
        spec.loader.exec_module(store_mod)

    if "_tier1_retrieve" in sys.modules:
        retrieve_mod = sys.modules["_tier1_retrieve"]
    else:
        spec = importlib.util.spec_from_file_location(
            "_tier1_retrieve", _RETRIEVE_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load retrieve module from {_RETRIEVE_PATH}"
            )
        retrieve_mod = importlib.util.module_from_spec(spec)
        sys.modules["_tier1_retrieve"] = retrieve_mod
        spec.loader.exec_module(retrieve_mod)

    return (
        store_mod.open_collection,
        retrieve_mod.retrieve_top_k,
        store_mod.CHROMA_PATH,
        store_mod.COLLECTION_NAME,
    )


def _synthetic_vec(seed: int, dim: int = 1536) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def test_constants_match_research_decision():
    _, _, chroma_path, coll_name = _import_store_and_retrieve()
    assert chroma_path.parts[-2:] == ("chroma_db", "tier-1-naive"), (
        "Persistence path locked by 128-RESEARCH.md Pitfall 8"
    )
    assert coll_name == "enterprise_kb_naive"


def test_open_collection_creates_and_persists(tmp_path):
    open_collection, *_ = _import_store_and_retrieve()
    coll = open_collection(reset=False, path=tmp_path / "chroma")
    assert coll.count() == 0
    coll.add(
        ids=["a", "b"],
        embeddings=[_synthetic_vec(1), _synthetic_vec(2)],
        documents=["doc a", "doc b"],
        metadatas=[
            {"paper_id": "x", "page": 1, "chunk_idx": 0},
            {"paper_id": "y", "page": 2, "chunk_idx": 0},
        ],
    )
    assert coll.count() == 2

    # Reopen in a new client to verify on-disk persistence.
    coll2 = open_collection(reset=False, path=tmp_path / "chroma")
    assert coll2.count() == 2


def test_open_collection_reset_wipes(tmp_path):
    open_collection, *_ = _import_store_and_retrieve()
    coll = open_collection(reset=False, path=tmp_path / "chroma")
    coll.add(
        ids=["a"],
        embeddings=[_synthetic_vec(1)],
        documents=["doc a"],
        metadatas=[{"paper_id": "x", "page": 1, "chunk_idx": 0}],
    )
    assert coll.count() == 1
    coll_reset = open_collection(reset=True, path=tmp_path / "chroma")
    assert coll_reset.count() == 0


def test_retrieve_top_k_returns_normalized_shape(tmp_path):
    open_collection, retrieve_top_k, *_ = _import_store_and_retrieve()
    coll = open_collection(reset=False, path=tmp_path / "chroma")
    vecs = [_synthetic_vec(s) for s in range(5)]
    coll.add(
        ids=[f"c{i}" for i in range(5)],
        embeddings=vecs,
        documents=[f"doc {i}" for i in range(5)],
        metadatas=[
            {"paper_id": f"p{i}", "page": 1, "chunk_idx": 0} for i in range(5)
        ],
    )
    res = retrieve_top_k(coll, query_vec=vecs[0], k=3)
    assert set(res.keys()) == {"documents", "metadatas", "distances", "similarities"}
    assert len(res["documents"]) == 3
    assert len(res["metadatas"]) == 3
    assert len(res["distances"]) == 3
    assert len(res["similarities"]) == 3
    # similarities = 1 - distances (Pitfall 2)
    for d, s in zip(res["distances"], res["similarities"]):
        assert abs((1.0 - d) - s) < 1e-9
    # Query vec is itself in the collection — best match should be very close.
    assert res["distances"][0] <= 1e-3
    assert res["similarities"][0] >= 1.0 - 1e-3


def test_open_collection_idempotent_reopen(tmp_path):
    """Reopening the same collection without reset returns the same state."""
    open_collection, *_ = _import_store_and_retrieve()
    coll1 = open_collection(reset=False, path=tmp_path / "chroma")
    coll1.add(
        ids=["only"],
        embeddings=[_synthetic_vec(42)],
        documents=["payload"],
        metadatas=[{"paper_id": "p", "page": 1, "chunk_idx": 0}],
    )
    coll2 = open_collection(reset=False, path=tmp_path / "chroma")
    assert coll2.count() == 1

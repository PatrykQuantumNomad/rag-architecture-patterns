"""Tier 1: top-k retrieval with cosine-similarity normalization.

ChromaDB returns cosine *distance* in result["distances"]. This module
converts distance -> similarity (= 1 - distance) so callers (and the
shared/display.render_query_result table) see intuitive higher-is-better
scores in [0, 1] for unit-normalized OpenAI embeddings (Pitfall 2).
"""
from __future__ import annotations

from typing import Any


def retrieve_top_k(
    collection, query_vec: list[float], k: int
) -> dict[str, list[Any]]:
    """Query the collection and return a flat, normalized result dict.

    Parameters
    ----------
    collection
        A chromadb collection (e.g., from store.open_collection).
    query_vec
        1536-dim OpenAI embedding of the user's question.
    k
        Number of chunks to return.

    Returns
    -------
    dict
        Keys:
          documents     — list[str], length min(k, collection.count())
          metadatas     — list[dict], same length
          distances     — list[float], cosine distances [0, 2]
          similarities  — list[float], 1 - distance, [-1, 1]
    """
    res = collection.query(query_embeddings=[query_vec], n_results=k)
    # ChromaDB returns batched results — index [0] for the single query.
    docs = list(res["documents"][0])
    metas = list(res["metadatas"][0])
    dists = [float(d) for d in res["distances"][0]]
    sims = [1.0 - d for d in dists]
    return {
        "documents": docs,
        "metadatas": metas,
        "distances": dists,
        "similarities": sims,
    }

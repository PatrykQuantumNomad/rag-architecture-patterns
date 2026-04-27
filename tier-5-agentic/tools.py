"""Tier 5 — agent function tools.

Two `@function_tool`-decorated callables exposed to the OpenAI Agents SDK:

* ``search_text_chunks(query, k=5)`` — semantic search over Tier 1's ChromaDB
  collection (``chroma_db/tier-1-naive/``). Wraps Tier 1's existing
  ``embed_batch`` + ``open_collection`` + ``retrieve_top_k`` so Tier 5 reuses
  Phase 128's index without re-embedding the corpus.
* ``lookup_paper_metadata(paper_id)`` — resolves an arXiv id to its title,
  authors, year, and abstract via :class:`shared.loader.DatasetLoader`.

Pitfall 9 (130-RESEARCH.md) is a HARD INVARIANT here: ``open_collection`` is
ALWAYS called with ``reset=False``. Tier 5 reads from Tier 1's index and
NEVER mutates it. A ``reset=True`` would wipe Phase 128's empirically
verified collection — see ROADMAP TIER-05 success criterion 2.

Pitfall 6 (Pattern 6): ``@function_tool`` derives the JSON schema the LLM
sees from Python type hints + ``pydantic.Field`` metadata; docstrings become
the ``description`` field the planner reads. Both tools use
``Annotated[T, Field(...)]`` so the SDK's pydantic model generator picks up
bounds (``ge=1, le=20``) and per-arg descriptions cleanly.

Module-level singletons (``_collection``, ``_loader``, ``_oai_client``) are
lazy-initialized on first tool call. This keeps ``import tier_5_agentic.tools``
side-effect-free for the non-live unit tests in ``tests/test_tools.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from pydantic import Field

from agents import function_tool

# Repo-root sys.path bootstrap mirrors Tier 1's pattern — guarantees
# ``shared.*`` and ``tier_1_naive.*`` resolve regardless of CWD when this
# module is imported via ``tier_5_agentic.tools`` (which the agent does).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.loader import DatasetLoader  # noqa: E402
from tier_1_naive.store import open_collection  # noqa: E402
from tier_1_naive.retrieve import retrieve_top_k  # noqa: E402
from tier_1_naive.embed_openai import (  # noqa: E402
    build_openai_client,
    EMBED_MODEL,
)

# --- Module-level singletons (lazy-init) ---------------------------------

_collection = None
_loader: DatasetLoader | None = None
_oai_client = None


def _get_collection():
    """Open Tier 1's collection in READ-ONLY mode. Pitfall 9 invariant."""
    global _collection
    if _collection is None:
        # ``reset=False`` ALWAYS — Tier 5 NEVER wipes Tier 1's index. This
        # is the load-bearing invariant of TIER-05 success criterion 2.
        _collection = open_collection(reset=False)
    return _collection


def _get_loader() -> DatasetLoader:
    global _loader
    if _loader is None:
        _loader = DatasetLoader()
    return _loader


def _get_oai_client():
    """Lazy OpenAI/OpenRouter client for query embeddings.

    Uses Tier 1's ``build_openai_client`` factory which fast-fails with a
    friendly SystemExit if ``OPENROUTER_API_KEY`` is unset (Pitfall 10
    inheritance from Phase 128). Lazy because the non-live agent
    construction tests must not require a key.
    """
    global _oai_client
    if _oai_client is None:
        _oai_client = build_openai_client()
    return _oai_client


def _embed_query(text: str) -> list[float]:
    """Embed a single query string into a 1536-dim vector via OpenRouter.

    Bypasses :class:`shared.cost_tracker.CostTracker` because Tier 5's primary
    cost surface is the agent's LLM iterations (captured in ``main.py`` via
    ``result.context_wrapper.usage``). Embedding cost during tool calls is
    a small fraction of total run cost; if Phase 133's evaluation tightens
    auditability, this is the seam to wire a per-tool tracker through.
    """
    client = _get_oai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return list(resp.data[0].embedding)


# --- Tools ---------------------------------------------------------------


@function_tool
async def search_text_chunks(
    query: Annotated[
        str,
        Field(description="The user's natural-language search query"),
    ],
    k: Annotated[
        int,
        Field(
            default=5,
            ge=1,
            le=20,
            description="How many chunks to return (1-20)",
        ),
    ] = 5,
) -> list[dict]:
    """Search the 100-paper enterprise knowledge base by semantic similarity over text chunks.

    Returns a list of {paper_id, page, snippet, similarity}. Use this FIRST when the user
    asks a question — it returns evidence chunks. Then call `lookup_paper_metadata` if you
    need authors/year for any cited paper.
    """
    coll = _get_collection()
    qvec = _embed_query(query)
    res = retrieve_top_k(coll, qvec, k=k)
    out: list[dict] = []
    for d, m, s in zip(
        res.get("documents", []),
        res.get("metadatas", []),
        res.get("similarities", []),
    ):
        out.append(
            {
                "paper_id": (m or {}).get("paper_id"),
                "page": (m or {}).get("page"),
                "snippet": (d or "")[:400],
                "similarity": float(s) if s is not None else 0.0,
            }
        )
    return out


@function_tool
def lookup_paper_metadata(
    paper_id: Annotated[
        str,
        Field(description="arXiv ID, e.g. '1706.03762'"),
    ],
) -> dict:
    """Look up paper metadata (title, authors, year, abstract) by arXiv ID.

    Use this AFTER `search_text_chunks` to disambiguate authors / year / publication
    of any paper you intend to cite in your final answer.
    """
    loader = _get_loader()
    for p in loader.papers():
        if p.get("paper_id") == paper_id:
            return {
                "paper_id": p["paper_id"],
                "title": p.get("title", ""),
                "authors": p.get("authors", []),
                "year": p.get("year"),
                "abstract": (p.get("abstract") or "")[:500],
            }
    return {"error": f"paper_id {paper_id} not found in dataset/manifests/papers.json"}

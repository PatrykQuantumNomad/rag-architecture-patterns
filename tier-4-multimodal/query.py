"""Tier 4 — async query wrapper with multimodal-content branching (Pattern 4).

If the caller hands in a ``multimodal_content`` list (per the RAG-Anything
README — a list of ``{type: image|text|table|equation, ...}`` dicts), the
query routes through ``rag.aquery_with_multimodal``. Otherwise it falls
through to plain ``rag.aquery``: RAG-Anything still surfaces image/table
chunks because everything is co-indexed in the same multimodal KG.

Pitfall 7 (chunks shape) — RAG-Anything 1.2.10 returns a string answer; it
does NOT expose a structured ``chunks`` field publicly. The companion
helper ``to_display_chunks`` returns ``[]`` for string responses so
``shared.display.render_query_result`` can render the standard Tier panel
without faking citations. If a future raganything version surfaces a
structured response, the dict-branch below populates from
``chunks``/``context``/``retrieved_chunks`` keys.
"""
from __future__ import annotations

from typing import Any, Optional


async def run_query(
    rag,
    question: str,
    mode: str = "hybrid",
    multimodal_content: Optional[list] = None,
) -> str:
    """Run a Tier 4 query.

    Parameters
    ----------
    rag
        A ``RAGAnything`` instance (post ``build_rag()`` from ``rag.py``).
    question
        Natural-language query string.
    mode
        One of ``naive`` / ``local`` / ``global`` / ``hybrid`` / ``mix`` —
        passed straight through to LightRAG's QueryParam under the hood.
    multimodal_content
        Optional list of multimodal content dicts. When supplied (truthy),
        routes via ``aquery_with_multimodal``. When ``None`` or empty, uses
        plain ``aquery`` (the multimodal KG still surfaces image/table
        chunks because they were co-indexed at ingest time).

    Returns
    -------
    str
        The model's answer text. (RAG-Anything 1.2.10 returns a string;
        Pitfall 7 — no structured chunks list.)
    """
    if multimodal_content:
        return await rag.aquery_with_multimodal(
            question,
            multimodal_content=multimodal_content,
            mode=mode,
        )
    return await rag.aquery(question, mode=mode)


def to_display_chunks(response: Any) -> list[dict]:
    """Defensive adapter: RAG-Anything response → ``shared.display`` chunks shape.

    RAG-Anything 1.2.10 returns a plain ``str`` — there is no structured
    ``chunks`` field exposed publicly (Pitfall 7). Returning ``[]`` lets
    ``render_query_result`` print "No chunks retrieved." which is the
    honest representation; the agent self-cites in the answer text.

    Forward-compat: if a future raganything release returns a dict with a
    chunks/context/retrieved_chunks key, populate from there.
    """
    if isinstance(response, dict):
        for key in ("chunks", "context", "retrieved_chunks"):
            value = response.get(key)
            if isinstance(value, list):
                return [
                    {
                        "doc_id": (c or {}).get("doc_id")
                        or (c or {}).get("source")
                        or "",
                        "score": float((c or {}).get("score") or 0.0),
                        "snippet": str(
                            (c or {}).get("text") or (c or {}).get("snippet") or ""
                        )[:400],
                    }
                    for c in value
                    if c
                ]
    return []


__all__ = ["run_query", "to_display_chunks"]

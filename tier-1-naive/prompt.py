"""Tier 1: prompt builder for retrieval-augmented chat completion.

Context-stuffed pattern — concatenates retrieved chunks with provenance
markers ([paper_id p.<page>]) so the LLM is steered toward inline
citations. The 'naive' framing matters: no reranking, no query rewriting,
no chain-of-thought scaffolding. Just stuff the top-k chunks and ask.
"""
from __future__ import annotations


def build_prompt(question: str, docs: list[str], metas: list[dict]) -> str:
    """Compose the RAG prompt from retrieved chunks.

    Parameters
    ----------
    question
        The user's natural-language question.
    docs
        The k chunk texts (in retrieval order).
    metas
        The k chunk metadata dicts (each must have ``paper_id`` and ``page``).

    Returns
    -------
    str
        A single prompt string with --- CONTEXT ---, --- QUESTION ---, and
        Answer: sentinels. Each context block is numbered and tagged with
        paper_id + page so the LLM can cite inline.
    """
    ctx_blocks: list[str] = []
    for i, (doc, m) in enumerate(zip(docs, metas), start=1):
        ctx_blocks.append(
            f"[{i}] paper_id={m['paper_id']} page={m['page']}\n{doc.strip()}"
        )
    ctx = "\n\n".join(ctx_blocks)
    return (
        "You are answering a research question using ONLY the provided context excerpts. "
        "If the answer is not in the excerpts, say so explicitly. "
        "Cite sources inline as [paper_id p.<page>].\n\n"
        f"--- CONTEXT ---\n{ctx}\n\n"
        f"--- QUESTION ---\n{question}\n\n"
        "Answer:"
    )

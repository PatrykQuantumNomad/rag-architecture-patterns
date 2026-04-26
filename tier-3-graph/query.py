"""Tier 3 query path — wraps ``rag.aquery`` with per-call token tracking.

LightRAG's per-query LLM calls are captured by the ``CostAdapter`` wired
through ``build_rag(llm_token_tracker=...)`` (Plan 03). That captures usage
into the shared ``CostTracker`` automatically. This module's job is the
small additional accounting needed by the renderer:

The repo-wide ``shared.display.render_query_result`` signature wants
per-query ``input_tokens`` / ``output_tokens`` — but ``CostTracker`` records
*cumulative* tokens across ingest + query. So we snapshot the LLM totals
BEFORE and AFTER ``rag.aquery(...)`` and return the delta. That delta is the
honest per-query token count (entity-extraction during ingest is excluded).

Probe-validated path (Plan 03 SUMMARY — Outcome A):

    Plan 03 confirmed lightrag-hku==1.4.15 supports ``token_tracker`` on
    both ``openai_complete_if_cache`` AND ``openai_embed``. Therefore the
    pre/post-delta strategy works WITHOUT a tiktoken fallback.

    The Outcome-C fallback (tiktoken estimation) is documented in this
    module's docstring as a contingency for future LightRAG releases that
    might remove ``token_tracker`` support — but it is NOT active because
    Plan 03 verified Outcome A.

Edge cases:

* If LightRAG cache-hits on the query (no LLM calls), pre==post and we
  return ``(answer, 0, 0)`` — render_query_result will print "0 tokens"
  which is the honest answer for a cached response.
* If LightRAG made multiple internal LLM calls (e.g., gleaning during
  hybrid mode), the delta sums them all — also honest.
* ``max(0, ...)`` guards against the impossible "post < pre" case (would
  only happen if another async task popped LLM entries from the tracker
  mid-query; we do not do that, but the guard prevents negative tokens
  surfacing in the UI on any future regression).
"""
from __future__ import annotations

from lightrag import QueryParam


async def run_query(
    rag,
    question: str,
    mode: str,
    model: str,  # noqa: ARG001 — kept in signature for the Outcome-C fallback path
    tracker,
) -> tuple[str, int, int]:
    """Run a graph query and return (answer, input_tokens, output_tokens).

    The ``model`` parameter is unused on the active Outcome-A path because
    the model is already baked into the LightRAG instance via ``build_rag``.
    It is retained in the public signature so the Outcome-C fallback (which
    needs to know the model slug to call ``tracker.record_llm(model, ...)``
    after a tiktoken estimate) is a one-line swap, not a contract change.

    Parameters
    ----------
    rag
        Initialized ``LightRAG`` instance (storages opened).
    question
        The user's query string.
    mode
        One of ``naive`` / ``local`` / ``global`` / ``hybrid`` / ``mix``.
        ``hybrid`` is the safe default (always works without a reranker);
        ``mix`` may require a separately-configured reranker.
    model
        OpenRouter LLM slug (kept for the Outcome-C fallback; unused
        otherwise — see module docstring).
    tracker
        ``shared.cost_tracker.CostTracker`` whose pre/post LLM-token totals
        we snapshot to compute per-query deltas.

    Returns
    -------
    tuple[str, int, int]
        ``(answer_text, per_query_input_tokens, per_query_output_tokens)``.
    """
    pre_in = sum(
        q["input_tokens"] for q in tracker.queries if q.get("kind") == "llm"
    )
    pre_out = sum(
        q["output_tokens"] for q in tracker.queries if q.get("kind") == "llm"
    )

    answer = await rag.aquery(question, param=QueryParam(mode=mode))

    post_in = sum(
        q["input_tokens"] for q in tracker.queries if q.get("kind") == "llm"
    )
    post_out = sum(
        q["output_tokens"] for q in tracker.queries if q.get("kind") == "llm"
    )

    return answer, max(0, post_in - pre_in), max(0, post_out - pre_out)


__all__ = ["run_query"]

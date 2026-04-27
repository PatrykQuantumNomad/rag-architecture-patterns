"""Tier 3 (LightRAG Graph) — eval adapter.

Library-first per Pattern 3. Pitfall 7 mitigation: LightRAG returns a STRING,
not structured chunks. We make TWO ``aquery`` calls per question:

1. ``only_need_context=True``  -> assembled context string (no LLM cost on this path)
2. ``only_need_context=False`` -> final answer (LLM cost via ``CostAdapter``)

The context string is split on LightRAG's documented ``-----`` delimiter into a
``list[str]``. Per A3 in 131-RESEARCH the separator is empirically verified;
if a future LightRAG release changes it, the only damage is that ``contexts``
becomes one big string (RAGAS still scores; ``context_precision`` degrades).

Decisions referenced
--------------------
- Pattern 3 (per-tier adapter contract).
- 131-RESEARCH Example B: two-phase aquery (context probe + answer).
- Pitfall 7 (LightRAG returns string): split on ``-----``.
- Pitfall 11 (CostTracker collision): optional ``tracker`` injection;
  optional ``rag`` injection so Plan 04 can construct ONE LightRAG instance
  per harness run (storage init is ~30s; calling 30 times would dominate
  latency).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lightrag import QueryParam  # noqa: E402

from shared.cost_tracker import CostTracker  # noqa: E402

from tier_3_graph.rag import build_rag, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL  # noqa: E402
from tier_3_graph.cost_adapter import CostAdapter  # noqa: E402
from tier_3_graph.query import run_query as tier3_run_query  # noqa: E402

from evaluation.harness.records import EvalRecord  # noqa: E402


def _split_context(context_str: str) -> list[str]:
    """Split LightRAG's concatenated context string on ``-----`` delimiter.

    LightRAG 1.4.15 separates retrieved chunks/entities/relations with lines of
    five hyphens. Conservatively split, strip, drop empties.

    Edge cases:
    * Empty / falsy input -> ``[]``.
    * No delimiter present -> single-element list with the trimmed input.
    """
    if not context_str:
        return []
    parts = [p.strip() for p in context_str.split("-----")]
    return [p for p in parts if p]


async def run_tier3(
    question_id: str,
    question: str,
    mode: str = "hybrid",
    rag=None,
    tracker: Optional[CostTracker] = None,
) -> EvalRecord:
    """Run a Tier 3 query; return an EvalRecord.

    Parameters
    ----------
    question_id : str
        Golden Q&A id.
    question : str
        Natural-language query.
    mode : str
        LightRAG retrieval mode — one of ``naive`` / ``local`` / ``global`` /
        ``hybrid`` / ``mix``. ``hybrid`` is the safe default (always works
        without a reranker).
    rag : LightRAG | None
        When provided, reuse the caller's already-initialized LightRAG instance
        (Plan 04's run.py constructs ONE per harness invocation — avoids the
        ~30s storage init per question). When ``None``, this adapter builds and
        initializes a fresh one (smoke test path).
    tracker : Optional[CostTracker]
        When provided, the adapter records into it. When ``None``, a fresh
        ``CostTracker("tier-3-eval")`` is constructed. Note: when ``rag`` is
        also caller-supplied, the caller is responsible for wiring the
        ``CostAdapter`` -> ``tracker`` linkage. This adapter only wires the
        ``CostAdapter`` on the own-build path.
    """
    if tracker is None:
        tracker = CostTracker("tier-3-eval")

    if rag is None:
        adapter = CostAdapter(tracker, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL)
        rag = build_rag(llm_token_tracker=adapter)
        await rag.initialize_storages()

    t0 = time.monotonic()
    # Step 1 — context probe (no LLM call when only_need_context=True; Pitfall 7).
    context_str = await rag.aquery(
        question, param=QueryParam(mode=mode, only_need_context=True)
    )
    contexts = _split_context(str(context_str) if context_str else "")

    # Step 2 — full answer; CostAdapter records LLM tokens into tracker.
    answer, _in_tok, _out_tok = await tier3_run_query(
        rag, question, mode, DEFAULT_LLM_MODEL, tracker
    )
    latency = time.monotonic() - t0

    return EvalRecord(
        question_id=question_id,
        question=question,
        answer=answer or "",
        retrieved_contexts=contexts,
        latency_s=latency,
        cost_usd_at_capture=tracker.total_usd(),
    )


__all__ = ["run_tier3", "_split_context"]

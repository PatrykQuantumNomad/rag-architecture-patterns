"""Tier 1 (Naive ChromaDB + OpenAI) — eval adapter.

Library-first per Pattern 3 in 131-RESEARCH. Reuses tier_1_naive's exact
embed -> retrieve -> chat path. The adapter is ``async def`` (Pitfall 5) but
the underlying tier-1 surface is sync; we wrap heavy calls in
``asyncio.to_thread`` to keep the harness loop non-blocking.

Pitfall 9 from 130-RESEARCH (READ-ONLY ChromaDB) inherited: ``open_collection``
is called with ``reset=False`` ALWAYS; this adapter NEVER mutates Tier 1's index.

Decisions referenced
--------------------
- Pattern 3 (per-tier adapter contract): uniform
  ``async run_tierN(question_id, question, **kwargs) -> EvalRecord``.
- Pitfall 5 (async/sync boundaries): sync tier-1 surface wrapped in
  ``asyncio.to_thread`` so Plan 04's harness loop stays non-blocking.
- Pitfall 9 (Tier 1 read-only): ``open_collection(reset=False)`` is the
  invariant — this adapter never mutates the index.
- Pitfall 11 (CostTracker collision): optional ``tracker`` argument so
  Plan 04 can supply ONE tracker per tier per harness invocation.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

# Repo-root sys.path bootstrap so this module imports cleanly regardless of
# pytest collection cwd (Plan 04's run.py is invoked from repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.cost_tracker import CostTracker  # noqa: E402
from shared.pricing import PRICES  # noqa: E402,F401  (kept for parity with sister adapters)

from tier_1_naive.store import open_collection  # noqa: E402
from tier_1_naive.embed_openai import build_openai_client, embed_batch  # noqa: E402
from tier_1_naive.retrieve import retrieve_top_k  # noqa: E402
from tier_1_naive.prompt import build_prompt  # noqa: E402
from tier_1_naive.chat import build_chat_client, complete as chat_complete  # noqa: E402

from evaluation.harness.records import EvalRecord  # noqa: E402


# OpenRouter slug WITHOUT the "openrouter/" prefix; tier-1 chat.complete takes
# the bare slug per Phase 128 Plan 06's OpenRouter migration.
DEFAULT_MODEL = "google/gemini-2.5-flash"


async def run_tier1(
    question_id: str,
    question: str,
    k: int = 5,
    model: str = DEFAULT_MODEL,
    tracker: Optional[CostTracker] = None,
) -> EvalRecord:
    """Run a Tier 1 query; return an EvalRecord with answer + contexts + latency + cost.

    Parameters
    ----------
    question_id : str
        Golden Q&A id (e.g. ``"single-hop-001"``). Passed through to the record.
    question : str
        Natural-language query.
    k : int
        Top-K chunks to retrieve.
    model : str
        OpenRouter slug WITHOUT the ``openrouter/`` prefix (Tier 1's
        ``chat.complete`` passes it straight through to the OpenAI SDK; the
        ``shared.pricing.PRICES`` table keys match this form).
    tracker : Optional[CostTracker]
        When provided, the adapter records into it (Plan 04 path — one tracker
        per tier per run). When ``None``, the adapter constructs a fresh one.

    Returns
    -------
    EvalRecord
        Populated record. ``error="tier1_chroma_empty"`` when the index has no
        documents (skip-with-error so the harness doesn't crash mid-loop).
    """
    if tracker is None:
        tracker = CostTracker("tier-1-eval")

    coll = open_collection(reset=False)  # Pitfall 9 — READ-ONLY invariant
    if coll.count() == 0:
        return EvalRecord(
            question_id=question_id,
            question=question,
            answer="",
            latency_s=0.0,
            cost_usd_at_capture=tracker.total_usd(),
            error="tier1_chroma_empty",
        )

    t0 = time.monotonic()
    embed_client = build_openai_client()
    chat_client = build_chat_client()

    # The heavy bits are sync — wrap to keep the harness loop yielding (Pitfall 5).
    [qv] = await asyncio.to_thread(embed_batch, embed_client, [question], tracker)
    res = await asyncio.to_thread(retrieve_top_k, coll, qv, k)
    prompt = build_prompt(question, res["documents"], res["metadatas"])
    chat_resp = await asyncio.to_thread(chat_complete, chat_client, prompt, model, tracker)

    latency = time.monotonic() - t0

    return EvalRecord(
        question_id=question_id,
        question=question,
        answer=chat_resp.text,
        retrieved_contexts=list(res.get("documents") or []),
        latency_s=latency,
        cost_usd_at_capture=tracker.total_usd(),
    )


__all__ = ["run_tier1", "DEFAULT_MODEL"]

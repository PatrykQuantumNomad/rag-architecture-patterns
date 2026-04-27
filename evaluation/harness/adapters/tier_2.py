"""Tier 2 (Gemini File Search) — eval adapter.

Library-first per Pattern 3. Calls ``tier_2_managed.query.query`` directly with
the caller-supplied ``store_name``. Plan 04's run.py reads the store_id from
the live ``.store_id`` file (gitignored — Phase 129 Plan 02).

Pitfall 6 from 129-RESEARCH (grounding_metadata may be None) inherited:
``to_display_chunks`` already defends every level. We map the chunk dicts to a
plain ``list[str]`` of ``ctx.text`` values for RAGAS.

Decisions referenced
--------------------
- Pattern 3 (per-tier adapter contract): uniform
  ``async run_tierN(question_id, question, **kwargs) -> EvalRecord``.
- Pattern 7 PRICES key for Tier 2: ``google/gemini-2.5-flash`` — NO
  ``openrouter/`` prefix because Tier 2 uses the native Gemini SDK directly.
- Pitfall 5 (async/sync): sync ``query`` call wrapped in
  ``asyncio.to_thread``.
- Pitfall 6 (grounding may be None): defensive ``getattr`` fallbacks on
  ``resp.usage_metadata``; zero-token path on missing usage.
- Pitfall 11 (CostTracker collision): optional ``tracker`` injection.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from google import genai  # noqa: E402

from shared.config import get_settings  # noqa: E402
from shared.cost_tracker import CostTracker  # noqa: E402
from shared.pricing import PRICES  # noqa: E402

from tier_2_managed.query import (  # noqa: E402
    query as tier2_query,
    to_display_chunks,
    DEFAULT_MODEL,
)

from evaluation.harness.records import EvalRecord  # noqa: E402


async def run_tier2(
    question_id: str,
    question: str,
    store_name: str,
    model: str = DEFAULT_MODEL,
    tracker: Optional[CostTracker] = None,
) -> EvalRecord:
    """Run a Tier 2 query; return an EvalRecord.

    Parameters
    ----------
    question_id : str
        Golden Q&A id (e.g. ``"single-hop-001"``).
    question : str
        Natural-language query.
    store_name : str
        FileSearch store identifier (``fileSearchStores/<id>``). Plan 04's run.py
        reads this from ``tier-2-managed/.store_id``.
    model : str
        Native Gemini model slug (NOT the openrouter form). Matches a key in
        ``shared.pricing.PRICES`` (``gemini-2.5-flash``).
    tracker : Optional[CostTracker]
        When provided, the adapter records into it. When ``None``, a fresh
        ``CostTracker("tier-2-eval")`` is constructed.
    """
    if tracker is None:
        tracker = CostTracker("tier-2-eval")

    settings = get_settings()
    if not settings.gemini_api_key:
        return EvalRecord(
            question_id=question_id,
            question=question,
            answer="",
            latency_s=0.0,
            cost_usd_at_capture=tracker.total_usd(),
            error="missing_gemini_api_key",
        )

    client = genai.Client(api_key=settings.gemini_api_key)

    t0 = time.monotonic()
    # Sync underlying call — wrap to keep harness loop yielding (Pitfall 5).
    resp = await asyncio.to_thread(tier2_query, client, store_name, question, model)
    latency = time.monotonic() - t0

    # Defensive answer extraction — try the canonical path then fall back to
    # ``resp.text`` (the native Gemini SDK exposes both).
    answer = ""
    try:
        answer = resp.candidates[0].content.parts[0].text or ""
    except (AttributeError, IndexError, TypeError):
        answer = getattr(resp, "text", "") or ""

    # Extract contexts via the existing chunk adapter, then map to list[str].
    chunks = to_display_chunks(resp)
    contexts: list[str] = []
    for ch in chunks:
        # to_display_chunks returns dicts with 'text' / 'snippet' / 'title' fields.
        if isinstance(ch, dict):
            txt = ch.get("text") or ch.get("snippet") or ch.get("title") or ""
            if txt:
                contexts.append(txt)

    # Cost — resp.usage_metadata may be None per Pitfall 6; defensive getattr.
    usage = getattr(resp, "usage_metadata", None)
    in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
    out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
    if model in PRICES:
        try:
            tracker.record_llm(model, in_tok, out_tok)
        except KeyError:
            # Belt-and-braces fail-soft — Pattern 11 in 131-RESEARCH.
            pass

    return EvalRecord(
        question_id=question_id,
        question=question,
        answer=answer,
        retrieved_contexts=contexts,
        latency_s=latency,
        cost_usd_at_capture=tracker.total_usd(),
    )


__all__ = ["run_tier2"]

"""Tier 5 (Agentic) — eval adapter.

Library-first per Pattern 3. The agent self-cites in answer text;
``retrieved_contexts`` is left empty — Pitfall 9 from 131-RESEARCH (Tier 5 may
legitimately decline retrieval). RAGAS will return NaN for
``faithfulness`` / ``context_precision`` when contexts are empty; Plan 05
handles that via ``nan_reason='empty_contexts'``.

Pitfall 8 from 131-RESEARCH: ``MaxTurnsExceeded`` sets
``error='max_turns_exceeded'``. Plan 05 will see ``error != None`` and tag
``nan_reason='agent_truncated'`` BEFORE calling RAGAS (skip the bad sample,
don't pollute the mean).

Decisions referenced
--------------------
- Pattern 3 (per-tier adapter contract).
- Pitfall 8: ``MaxTurnsExceeded`` -> ``error='max_turns_exceeded'`` and
  answer prefixed ``[truncated ...]``.
- Pitfall 9: ``retrieved_contexts=[]`` honest empty (agent self-cites).
- Pitfall 11: optional ``tracker`` and ``agent`` injection.
- Pattern 12 in 131-RESEARCH (PRICES key strip): the tier-5 agent uses the
  ``openrouter/<provider>/<model>`` slug in the SDK call but
  ``shared.pricing.PRICES`` keys are provider-only — strip the prefix.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents import Runner  # noqa: E402
from agents.exceptions import MaxTurnsExceeded  # noqa: E402

from shared.cost_tracker import CostTracker  # noqa: E402
from shared.pricing import PRICES  # noqa: E402

from tier_5_agentic.agent import build_agent, DEFAULT_MODEL  # noqa: E402

from evaluation.harness.records import EvalRecord  # noqa: E402


def _strip_openrouter_prefix(model: str) -> str:
    """Mirror tier-5-agentic/main.py — PRICES uses provider-only slugs.

    ``openrouter/google/gemini-2.5-flash`` -> ``google/gemini-2.5-flash``.
    Models without the prefix are returned unchanged.
    """
    return model.split("/", 1)[1] if model.startswith("openrouter/") else model


async def run_tier5(
    question_id: str,
    question: str,
    max_turns: int = 10,
    model: str = DEFAULT_MODEL,
    agent=None,
    tracker: Optional[CostTracker] = None,
) -> EvalRecord:
    """Run a Tier 5 agent query; return an EvalRecord.

    Parameters
    ----------
    question_id : str
        Golden Q&A id.
    question : str
        Natural-language query.
    max_turns : int
        Hard cap on agent loop iterations (ROADMAP-locked at 10).
        ``MaxTurnsExceeded`` is caught (Pitfall 8) and surfaced as
        ``error='max_turns_exceeded'`` with the answer prefixed ``[truncated]``.
    model : str
        Agent model slug — full ``openrouter/<provider>/<model>`` form for the
        SDK; the PRICES key is derived via ``_strip_openrouter_prefix``.
    agent : Agent | None
        When provided, reuse the caller's agent (Plan 04 may construct ONE per
        harness invocation). When ``None``, ``build_agent(model=model)`` is
        invoked.
    tracker : Optional[CostTracker]
        When provided, the adapter records into it. When ``None``, a fresh
        ``CostTracker("tier-5-eval")`` is constructed.
    """
    if tracker is None:
        tracker = CostTracker("tier-5-eval")

    if agent is None:
        agent = build_agent(model=model)

    error: Optional[str] = None
    answer = ""
    usage = None

    t0 = time.monotonic()
    try:
        result = await Runner.run(agent, question, max_turns=max_turns)
        answer = result.final_output or ""
        usage = result.context_wrapper.usage
    except MaxTurnsExceeded as exc:  # Pitfall 8
        error = "max_turns_exceeded"
        answer = f"[truncated — agent exceeded max_turns={max_turns}] {exc}"
        # Some agents-SDK versions don't expose .usage on the exception
        # (Pitfall 6 in 130-RESEARCH); defensive getattr.
        usage = getattr(exc, "usage", None)
    latency = time.monotonic() - t0

    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)

    pricing_key = _strip_openrouter_prefix(model)
    if pricing_key in PRICES:
        try:
            tracker.record_llm(pricing_key, in_tok, out_tok)
        except KeyError:
            # Pattern 11 — fail-soft on unknown model.
            pass

    return EvalRecord(
        question_id=question_id,
        question=question,
        answer=answer,
        retrieved_contexts=[],  # Pitfall 9 honest empty — agent self-cites
        latency_s=latency,
        cost_usd_at_capture=tracker.total_usd(),
        error=error,
    )


__all__ = ["run_tier5", "_strip_openrouter_prefix"]

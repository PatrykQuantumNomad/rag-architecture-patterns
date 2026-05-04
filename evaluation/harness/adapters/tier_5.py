"""Tier 5 (Agentic) — eval adapter.

Library-first per Pattern 3. The adapter walks ``RunResult.new_items``
filtered by ``ToolCallOutputItem`` and projects each ``item.output`` into
provenance-prefixed strings (``[paper_id=<id>] <text>``) for RAGAS
``retrieved_contexts``. When the agent legitimately declines retrieval
(zero tool calls), the walk yields an empty list — Pitfall 9 of
130-RESEARCH preserved (no synthesis from ``final_output``).

Pitfall 8 from 130-RESEARCH: ``MaxTurnsExceeded`` sets
``error='max_turns_exceeded'``. Plan 05 will see ``error != None`` and tag
``nan_reason='agent_truncated'`` BEFORE calling RAGAS (skip the bad sample,
don't pollute the mean). The walk is never reached on this path.

Decisions referenced
--------------------
- Pattern 3 (per-tier adapter contract).
- Pitfall 8: ``MaxTurnsExceeded`` -> ``error='max_turns_exceeded'`` and
  answer prefixed ``[truncated ...]``.
- Pitfall 9: zero tool calls → ``retrieved_contexts=[]`` honest empty.
- Pitfall 11: optional ``tracker`` and ``agent`` injection.
- Pattern 12 in 131-RESEARCH (PRICES key strip): the tier-5 agent uses the
  ``openrouter/<provider>/<model>`` slug in the SDK call but
  ``shared.pricing.PRICES`` keys are provider-only — strip the prefix.
- Pitfall 1 of 132-RESEARCH: read ``item.output`` (raw tool return value),
  NEVER ``item.raw_item`` (stringified Responses-API payload).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents import Runner, ToolCallOutputItem  # noqa: E402
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


def _extract_contexts_from_run_items(new_items) -> list[str]:
    """Project ``ToolCallOutputItem.output`` values into provenance-prefixed strings.

    Walks ``new_items``, keeps only ``ToolCallOutputItem`` instances, and
    surfaces each tool's payload as one or more ``[paper_id=<id>] <text>``
    entries in RAGAS ``retrieved_contexts``. Repeated chunks (same
    ``paper_id`` + ``page``) and repeated metadata (same ``paper_id``) are
    deduped first-occurrence-wins — small bounded contexts produce cleaner
    RAGAS judgments.

    Pitfall 1 of 132-RESEARCH (HARD): we read ``item.output`` (the raw
    return value of the ``@function_tool`` callable), never
    ``item.raw_item`` (a stringified Responses-API payload). See
    ``agents/items.py:382`` for the contract.

    Tool error payloads (``{"error": "..."}`` returned by
    ``lookup_paper_metadata`` on miss, see ``tier-5-agentic/tools.py:171``)
    are skipped silently — they are NOT context.
    """
    contexts: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for item in new_items:
        # Pitfall 2 of 132-RESEARCH: a future SDK rename of ``ToolCallOutputItem``
        # would silently break this isolated isinstance check; openai-agents
        # is pinned at 0.14.6 in pyproject.toml so the shape is stable.
        if not isinstance(item, ToolCallOutputItem):
            continue
        out = item.output  # raw tool return; NOT item.raw_item (Pitfall 1 of 132-RESEARCH)
        if isinstance(out, list):
            # search_text_chunks: list[dict] keyed paper_id, page, snippet, similarity
            for hit in out:
                if not isinstance(hit, dict):
                    continue
                if hit.get("error"):
                    continue
                pid = hit.get("paper_id")
                page = hit.get("page")
                snippet = hit.get("snippet")
                if not snippet:
                    continue
                key = ("chunk", str(pid), str(page))
                if key in seen:
                    continue
                seen.add(key)
                contexts.append(f"[paper_id={pid}] {snippet}")
        elif isinstance(out, dict):
            # lookup_paper_metadata: success {paper_id, title, authors, year, abstract}
            # or error {"error": "..."} — skip the error path silently.
            if "error" in out:
                continue
            pid = out.get("paper_id")
            abstract = out.get("abstract")
            if not abstract:
                continue
            key = ("meta", str(pid))
            if key in seen:
                continue
            seen.add(key)
            contexts.append(f"[paper_id={pid}] {abstract}")
        # Any other shape (str, int, None) is silently skipped — Tier 5 has
        # only two tools and both return list[dict] or dict.
    return contexts


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
    contexts: list[str] = []

    t0 = time.monotonic()
    try:
        result = await Runner.run(agent, question, max_turns=max_turns)
        answer = result.final_output or ""
        usage = result.context_wrapper.usage
        # Walk tool outputs to populate retrieved_contexts. Pitfall 9 of
        # 130-RESEARCH preserved: a zero-tool-call run yields contexts=[]
        # honestly via this walk (no synthesis from final_output).
        contexts = _extract_contexts_from_run_items(getattr(result, "new_items", []) or [])
    except MaxTurnsExceeded as exc:  # Pitfall 8
        error = "max_turns_exceeded"
        answer = f"[truncated — agent exceeded max_turns={max_turns}] {exc}"
        # Some agents-SDK versions don't expose .usage on the exception
        # (Pitfall 6 in 130-RESEARCH); defensive getattr.
        usage = getattr(exc, "usage", None)
        # Pitfall 8 preserved: truncation surfaces error and skips the walk;
        # contexts stays [] so Plan 05 can tag nan_reason='agent_truncated'.
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
        retrieved_contexts=contexts,
        latency_s=latency,
        cost_usd_at_capture=tracker.total_usd(),
        error=error,
    )


__all__ = ["run_tier5", "_strip_openrouter_prefix", "_extract_contexts_from_run_items"]

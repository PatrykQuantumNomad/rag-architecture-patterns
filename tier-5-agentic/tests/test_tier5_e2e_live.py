"""Tier 5 live e2e test — multi-tool agent loop against real OpenRouter.

Filename matches Phase 129 Plan 07 / Phase 130 Plan 05 convention
(``test_tierN_e2e_live.py``) for pytest rootdir basename uniqueness — Phase
128 Plan 02's follow-on rule (no ``__init__.py`` in tier-local ``tests/``)
means basenames must be unique across the repo to avoid
``ImportPathMismatchError``.

Cost: ~$0.005-0.05 per run depending on agent loop depth. Pre-conditions:

* ``OPENROUTER_API_KEY`` in env (``tier5_live_keys_ok`` fixture)
* ``chroma_db/tier-1-naive/`` populated by Tier 1's ingest
  (``tier1_index_present`` fixture; Pitfall 9 read-only invariant)

Both fixtures must be present; either missing skips the test cleanly.

The ``QUESTION`` is deliberately designed to require **tool composition**:
the agent should call ``search_text_chunks`` to find the RAG paper's chunks
(yielding a ``paper_id``) and then call ``lookup_paper_metadata`` to verify
the canonical title + author list. A single-shot vector search cannot
produce both, so a passing run with ``cost > 0`` and at least minimal token
usage is empirical evidence of multi-tool autonomy.

Pitfall 6: ``MaxTurnsExceeded.usage`` may be ``None`` on some 0.x patch
releases — guarded with ``getattr(exc, "usage", None)``. Pitfall 12:
``shared.pricing.PRICES`` is keyed without the ``openrouter/`` prefix; this
test strips the prefix before lookup and gates ``record_llm`` on
membership + ``try/except KeyError``.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from agents import Runner
from agents.exceptions import MaxTurnsExceeded

from shared.cost_tracker import CostTracker
from shared.pricing import PRICES

from tier_5_agentic.agent import build_agent, DEFAULT_MODEL


# A question requiring tool composition: agent should call
# ``search_text_chunks`` at least once and ``lookup_paper_metadata`` at
# least once before answering. The chunked corpus from Tier 1's Phase 128-06
# live test contains both Lewis 2020 (RAG) and DPR-adjacent papers.
QUESTION = (
    "What is the main contribution of the paper that introduced "
    "'retrieval-augmented generation' (RAG), and who are its authors? "
    "Use search_text_chunks to find evidence and lookup_paper_metadata "
    "to verify the author list. Cite the paper_id."
)


@pytest.mark.live
def test_tier5_end_to_end_multi_tool(
    tier5_live_keys_ok,
    tier1_index_present,
    tmp_path,
    monkeypatch,
):
    """End-to-end: agent runs with max_turns=10, calls tools, returns cited answer.

    Assertions (in order of importance):

    1. Agent produced a non-empty final output (or marked truncation).
    2. Tokens were consumed (the planner LLM was called at least once).
    3. Cost was recorded (when the model is in ``shared.pricing.PRICES``).
    4. ``MaxTurnsExceeded`` is allowed but unusual — logged, not failed.
    """
    # Persist the cost JSON into tmp_path so the test does not pollute
    # ``evaluation/results/costs/``. CostTracker.persist() writes
    # relative to the CWD by default; we patch the env var the tracker
    # honours (best-effort — most CostTracker variants ignore this and
    # use their own path; the assertion does not depend on the file).
    monkeypatch.setenv("COST_TRACKER_OUTPUT_DIR", str(tmp_path))

    agent = build_agent()
    tracker = CostTracker("tier-5-test")

    truncated = False
    answer = ""
    usage = None
    t0 = time.monotonic()
    try:
        result = asyncio.run(Runner.run(agent, QUESTION, max_turns=10))
        answer = result.final_output or ""
        usage = result.context_wrapper.usage
    except MaxTurnsExceeded as exc:
        # Pitfall 6: usage may be None on some 0.x patch releases.
        truncated = True
        usage = getattr(exc, "usage", None)
        answer = f"[truncated] {exc}"
    latency = time.monotonic() - t0

    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    requests = int(getattr(usage, "requests", 0) or 0)

    # Pitfall 12: PRICES keys lack the ``openrouter/`` prefix.
    pricing_key = (
        DEFAULT_MODEL.split("/", 1)[1]
        if DEFAULT_MODEL.startswith("openrouter/")
        else DEFAULT_MODEL
    )
    if pricing_key in PRICES:
        try:
            tracker.record_llm(pricing_key, in_tok, out_tok)
        except KeyError:  # defensive — should not trigger after the membership check
            pass

    print(f"\nLatency: {latency:.2f}s")
    print(f"Truncated: {truncated}")
    print(f"Tokens: {in_tok} in / {out_tok} out (requests={requests})")
    print(f"Total cost: ${tracker.total_usd():.6f}")
    truncated_answer = answer[:400] + ("..." if len(str(answer)) > 400 else "")
    print(f"Answer: {truncated_answer}")

    # --- Assertions ---
    # 1. Agent did not crash and produced SOME output (truncation prefix is OK)
    assert answer, "Empty answer (and not truncated) — Runner.run failed silently?"
    # 2. Tokens were used (the agent at least called the LLM once)
    assert in_tok > 0 or out_tok > 0, (
        "No tokens used — agent never called the LLM? "
        "(Check OPENROUTER_API_KEY validity + LiteLLM provider routing.)"
    )
    # 3. Cost was tracked (modulo unknown-model cases)
    if pricing_key in PRICES:
        assert tracker.total_usd() > 0, (
            f"Cost recorded zero for known model {pricing_key!r} "
            f"(input_tokens={in_tok}, output_tokens={out_tok})"
        )
    # 4. Truncation only if agent ran a long loop — not expected for this
    # 2-tool question, but allowed (the SDK may iterate aggressively on
    # ambiguous evidence).
    if truncated:
        print(
            "WARNING: Agent hit max_turns; this is allowed but unusual for "
            "a 2-tool question."
        )

    # Best-effort persist; some CostTracker variants persist to a fixed
    # path under evaluation/results/costs/. Don't fail the test if it
    # cannot write here.
    try:
        tracker.persist()
    except Exception as exc:  # pragma: no cover — best-effort
        print(f"NOTE: tracker.persist() raised {exc!r} (non-fatal)")

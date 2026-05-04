# Phase 1 Tier 5 Fallback Runbook

**Status:** Operator checklist — invoked when the Plan 01-02 smoke gate returns FAIL or INCONCLUSIVE.
**Authority:** This runbook encodes the D-06 fallback ordering verbatim. **Instrument first, mutate second.** No silent partial-ship.

---

## When to read this runbook

Per `.planning/phases/01-tier-5-adapter-fix/01-CONTEXT.md` D-06 exit policy:

> Phase 1 ships when the smoke gate passes (≥4/5 populated AND non-NaN RAGAS on the measurable subset). If after instrumentation + the 3 STACK.md mutations the gate still fails, mark Phase 1 incomplete, write a fallback-log doc, and escalate to the user before Phase 7.

If the smoke gate from Plan 01-02 returned **PASS**, this runbook is dormant scaffolding — Phase 1 ships clean. If it returned **FAIL** or **INCONCLUSIVE**, work the steps below in order.

---

## Step 0 — Open the fallback log

Open a Python REPL at the repo root and stamp a fresh log:

```python
from evaluation.harness.diagnostics import (
    FallbackAttempt,
    FallbackLog,
    SmokeQuestionResult,
    SpanObservation,
    open_fallback_log,
    write_fallback_log,
)

log = open_fallback_log()
# log.git_sha, log.opened_at, log.captured_versions are populated automatically.
```

The log is the **provenance artifact** for Phase 9's frozen doc — if the user later authorizes a degraded ship, the frozen-doc "Tier 5: partial-fix caveat" line cites this JSON. Every attempt below appends a `FallbackAttempt(...)` to `log.attempts`.

---

## Step 1 — Instrument first

Install the diagnostic stack on demand (the `[debug-tier5]` extra is opt-in; default install is byte-identical without it):

```bash
uv sync --extra debug-tier5
export RAG_DEBUG_TIER5_TRACING=1
```

Start a local Phoenix collector:

```bash
phoenix serve            # default OTLP port 4317; UI at http://localhost:6006
```

Re-run the smoke (same command from the Plan 01-02 checkpoint):

```bash
python -m evaluation.harness.run --tiers 5 --smoke-question-ids <ids> --yes
```

Observe the spans in the Phoenix UI to answer THREE questions:

1. **Is the agent calling tools?** Look for `ToolCallItem` spans — if absent, the agent is self-citing from training (Pitfall 9 of 132-RESEARCH); diagnose the agent prompt before any STACK.md mutation.
2. **Are tools returning data?** Look for `ToolCallOutputItem` raw_item payloads — if the chunks are empty, the underlying retrieval (Tier 1's ChromaDB) is the issue, not the adapter.
3. **Is the adapter walking new_items?** The walk added in Plan 01-01 Task 2 is in-process. If spans show tool outputs but `retrieved_contexts` remains empty in the captured `EvalRecord`, log:

   ```python
   obs = SpanObservation(
       agent_called_tools=True,
       tools_returned_data=True,
       adapter_saw_tool_call_output_items=False,
       notes="adapter walk did not surface tool outputs — proceed to Step 2",
   )
   ```

   …and proceed to Step 2. Otherwise diagnose why the agent is not calling tools first.

Record the attempt:

```python
from evaluation.harness.run import _ts
log.attempts.append(FallbackAttempt(
    kind="instrument",
    description="install debug-tier5, RAG_DEBUG_TIER5_TRACING=1, observed Phoenix spans",
    smoke_outcome="FAIL",            # or "INCONCLUSIVE" / "PASS"
    smoke_results=[...],             # SmokeQuestionResult per question
    span_observation=obs,
    cost_usd=0.04,
    timestamp=_ts(),
))
```

---

## Step 2 — STACK.md mutations (only after Step 1 spans observed)

Each mutation is followed by **one** smoke run. Log the attempt; on PASS, jump to Step 3.

### 2a — Simplify schema

`tier-5-agentic/tools.py` schema fields with `Annotated[T, Field(...)]` re-typed to bare types. Mitigates LiteLLM #16651 (silent failure on `additionalProperties` schema variants — see `01-RESEARCH.md` Sources). Re-run smoke:

```python
log.attempts.append(FallbackAttempt(
    kind="simplify_schema",
    description="bare types in tier-5-agentic/tools.py — LiteLLM #16651 mitigation",
    smoke_outcome="PASS",            # or FAIL / INCONCLUSIVE
    cost_usd=0.04,
    timestamp=_ts(),
))
```

If PASS — close the log and ship (Step 3, `final_disposition="RESOLVED"`).

### 2b — Switch model slug

`tier-5-agentic/agent.py` `DEFAULT_MODEL` toggle between `openrouter/google/gemini-2.5-flash` and `openrouter/google/gemini-2.5-flash-001`. Re-run smoke. Log with `kind="switch_model_slug"`. If PASS, ship.

### 2c — Bump openai-agents (LAST RESORT)

**Requires explicit user authorization via a `checkpoint:decision` in the executor flow.** Bumping the SDK pin invalidates Phase 130 pricing assumptions and cascades into the Phase 9 frozen-doc cost claims. **Do NOT do this autonomously.** The `AttemptKind = "bump_openai_agents"` Literal in `evaluation/harness/diagnostics.py` is the documented marker.

If user authorizes and you bump the pin, log with `kind="bump_openai_agents"` and re-run smoke.

---

## Step 3 — Close the log

Set the disposition and persist:

```python
log.closed_at = _ts()
log.final_disposition = "RESOLVED"     # smoke gate now passes
# OR "DEGRADED_SHIP"  — user authorized partial-fix per D-06
# OR "ESCALATED"      — all 3 mutations exhausted, smoke still failing

path = write_fallback_log(log)         # -> evaluation/results/diagnostics/tier-5-fallback-{TS}.json
print(f"Fallback log written: {path}")
```

Record the returned path in the Plan 01-03 SUMMARY.md **Notes** section so Phase 7's full rerun knows whether the runbook was exercised.

---

## Cost budget

Per Pitfall 7 of 132-RESEARCH:

- **Per attempt:** ≤ 1 smoke run (5 questions) ≈ ≤ $0.05.
- **Total fallback budget:** ≤ $0.50 (instrument + 3 mutations + 1 buffer + ≤ 5 re-runs).
- **If exceeded:** halt and escalate to the user; do NOT keep iterating on the user's bill.

The `cost_usd` field on each `FallbackAttempt` makes the running total auditable.

---

## Anti-patterns (DO NOT)

1. **DO NOT** skip Step 1 and jump straight to mutations. Without spans, you are mutating blind — every "fix" is indistinguishable from a no-op.
2. **DO NOT** modify the agent prompt or tool schemas beyond the simplify-schema step (2a). Pitfall 9 of 130-RESEARCH (agent self-citation) is load-bearing for the answer-text invariant; do not "fix" it by re-prompting.
3. **DO NOT** install `[debug-tier5]` into the default environment. Always use `uv sync --extra debug-tier5` so the diagnostic stack lives in an opt-in install (D-09 + CLEANUP-02).
4. **DO NOT** bump `openai-agents` without explicit user authorization. The pin invalidates Phase 130 pricing — escalate via `checkpoint:decision` first.

---

## References

- `.planning/phases/01-tier-5-adapter-fix/01-CONTEXT.md` — D-06 fallback ordering, exit policy.
- `.planning/phases/01-tier-5-adapter-fix/01-RESEARCH.md` — STACK.md decision tree sources.
- `evaluation/harness/diagnostics.py` — FallbackLog / write_fallback_log writer.
- `pyproject.toml` `[debug-tier5]` extra — diagnostic dependencies (opt-in only).
- `.planning/REQUIREMENTS.md` v1.1 — CLEANUP-02 (debug-tier5 extras tracking).

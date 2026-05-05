# Phase 3: NaN Reason Instrumentation - Research

**Researched:** 2026-05-05
**Domain:** RAGAS 0.4.3 metric scoring + per-row failure-mode attribution
**Confidence:** HIGH

## Summary

The phase-3 ask is **incremental, not greenfield**. The codebase already has a working `nan_reason` field on `ScoreRecord`, a working `_short_circuit_nan()` in `evaluation/harness/score.py`, a 4-reason taxonomy (`empty_contexts`, `agent_truncated`, `tier4_unavailable`, `cached_miss`), and a working per-reason rollup in `evaluation/harness/compare.py` (the `nan_breakdown` dict on aggregate_tier and the per-tier breakdown lines in `comparison.md`). The phase-3 deltas are precisely two: (1) add `empty_statements` and `json_parse_failure` reasons for failures that happen **inside** the RAGAS `evaluate()` call (not pre-call short-circuits), and (2) ensure those reasons reach the same `nan_breakdown` aggregator that already works for the pre-call reasons.

The non-trivial engineering bit is that RAGAS 0.4.3's `Executor.wrap_callable_with_index` swallows per-row exceptions and returns `np.nan` to the dataframe (`/Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/executor.py:71-84`). By the time `evaluate()` returns, the exception type — and therefore the reason — is gone. Three viable approaches exist (see Architecture Patterns); the recommended approach is a **custom `langchain_core` callback handler** that subclasses `BaseCallbackHandler` and records `(row_index, metric_name, exception_type)` via `on_chain_error`, then post-evaluate maps results-with-NaN back to that record. This is a ~50-LOC addition with no new dependencies.

**Primary recommendation:** Add a `NaNReasonTracer(BaseCallbackHandler)` to `score.py` that captures per-row exception types via `on_chain_error`, pass it into `evaluate(callbacks=[...])` alongside the existing `RagasTracer` (RAGAS appends its own automatically), and after `evaluate()` returns map each NaN cell to its captured reason (`RagasOutputParserException` → `json_parse_failure`, `LLMDidNotFinishException` → `llm_did_not_finish`, NaN with no captured exception → `empty_statements` for faithfulness). Extend the `nan_reason` taxonomy in `_short_circuit_nan()` and the post-evaluate mapper to emit these new strings. The existing `compare.py::aggregate_tier` `nan_breakdown` dict requires no changes — it already buckets by whatever string is in the `nan_reason` field.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Pre-evaluate failure detection (empty contexts, tier4 unavailable, agent truncation) | `score.py::_short_circuit_nan` | — | Already in place; no change needed for these three |
| Mid-evaluate failure capture (RAGAS exceptions) | `score.py::NaNReasonTracer` (NEW) | `langchain_core.callbacks.BaseCallbackHandler` | Only place we can intercept exceptions before RAGAS Executor swallows them |
| Post-evaluate NaN classification (map results dataframe → nan_reason) | `score.py::score_query_log` (extend existing) | — | Already does NaN→None mapping; extend to also assign reason |
| Per-row reason persistence | `records.py::ScoreRecord.nan_reason` | — | Already exists; no schema change |
| Per-tier reason rollup | `compare.py::aggregate_tier::nan_breakdown` | — | Already exists; no aggregator change |
| Per-tier reason rendering | `compare.py::emit_markdown` "NaN breakdown per tier" section | — | Already renders per-reason counts; no template change |

**Key insight:** Of the 6 capabilities listed, **5 already exist and need zero touch**. The only NEW capability is "mid-evaluate failure capture" — everything else is reused. The plan should be heavily test-driven on the new tracer; the existing aggregator tests already prove the rollup works.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HARN-05 | User can distinguish RAGAS NaN reasons (`empty_contexts` vs. `empty_statements` vs. `json_parse_failure`) in per-row metrics output | `empty_contexts` already implemented (`score.py:163`); `empty_statements` requires post-evaluate detection via faithfulness `_ascore` returning NaN with no captured exception (faithfulness.py:210-211); `json_parse_failure` requires `RagasOutputParserException` capture via callback (`pydantic_prompt.py:557`) |

## Standard Stack

### Core (already installed, no change)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ragas | 0.4.3 [VERIFIED: `.venv/lib/python3.13/site-packages/ragas/_version.py`] | RAGAS metric scoring | Already locked by score.py + tests; v0.4.x is current stable line |
| langchain-core | (transitive via ragas) [VERIFIED: ragas imports `langchain_core.callbacks.BaseCallbackHandler`] | Callback handler base class | RAGAS uses langchain callback API natively; subclassing is the documented extension hook |
| pydantic | 2.10–3.x [VERIFIED: pyproject.toml dependency pin] | `ScoreRecord` model | Already in records.py; nan_reason is `Optional[str]` so adding new string values is a pure-data change |

### Supporting (already installed, no change)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | (transitive via ragas) | `np.nan` detection in dataframe rows | Already used in `_to_float_or_none` (`score.py:171-181`) — the NaN→None mapper |
| pytest | (project test dep) | Test framework for new tracer + classifier tests | All existing `test_eval_score.py` tests use plain pytest — extend, don't introduce alternatives |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom `BaseCallbackHandler` | Subclass `Faithfulness` and override `_ascore` to catch `RagasOutputParserException`/`LLMDidNotFinishException` | Faithfulness-only; would not cover `json_parse_failure` happening in `answer_relevancy` or `context_precision`. Callback approach is one mechanism for all three metrics. |
| Custom `BaseCallbackHandler` | Re-run NaN rows post-hoc with `raise_exceptions=True` | Doubles judge cost on the failing rows; non-deterministic if LLM responses differ on retry. Callback approach captures the original failure mode in-flight. |
| Per-row `try/except` wrapping `evaluate()` | Call `evaluate()` once per row | Loses RAGAS's batching (`batch_size=10`), inflates wall time and cost ledger fragmentation. Reject. |
| New PyPI dep | (none needed) | The langchain callback API is already a transitive dep. No `pip install` step required. |

**Installation:** None required. All deps already pinned in pyproject.toml `[tier-1]`/`[tier-3]`/`[tier-4]` and the `.venv` has them installed.

**Version verification:**
```bash
python3 -c "import ragas; print(ragas.__version__)"  # → 0.4.3 [VERIFIED 2026-05-05]
```

## Architecture Patterns

### System Architecture Diagram

```
                        evaluation/harness/score.py::amain
                                     │
                                     ▼
                        ┌────────────────────────────┐
                        │  per-tier QueryLog read    │
                        │  golden_qa.json index      │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────┐
                        │  score_query_log(log,...)  │
                        └────────────┬───────────────┘
                                     │
                ┌────────────────────┼─────────────────────┐
                │                    │                     │
                ▼                    ▼                     ▼
   ┌───────────────────┐  ┌────────────────────┐  ┌────────────────────┐
   │ _short_circuit_   │  │ EvaluationDataset  │  │ NaNReasonTracer    │
   │ nan(rec)          │  │  + RAGAS evaluate()│◄─│  (NEW callback)    │
   │ pre-call branch   │  │                    │  │ on_chain_error     │
   │ (4 reasons today) │  └─────────┬──────────┘  │ records exception  │
   └─────────┬─────────┘            │             │ type per (row,     │
             │                      │             │  metric)           │
             │                      │             └─────────┬──────────┘
             │                      ▼                       │
             │           ┌────────────────────┐             │
             │           │ result.to_pandas() │             │
             │           │ NaN cells exist    │             │
             │           └─────────┬──────────┘             │
             │                     │                        │
             │                     ▼                        │
             │           ┌────────────────────┐             │
             │           │ _classify_nan(     │◄────────────┘
             │           │   row, tracer)     │
             │           │ NEW post-call mapper│
             │           │ • exception present│
             │           │   → json_parse_    │
             │           │     failure /      │
             │           │     llm_did_not_   │
             │           │     finish         │
             │           │ • no exception +   │
             │           │   faithfulness=NaN │
             │           │   → empty_         │
             │           │     statements     │
             │           └─────────┬──────────┘             
             │                     │
             ▼                     ▼
     ┌─────────────────────────────────────┐
     │ ScoreRecord(question_id, ..., nan_reason)│
     │ persisted to metrics/tier-N-*.json  │
     └────────────────┬────────────────────┘
                      │
                      ▼
        ┌────────────────────────────────────┐
        │ compare.py::aggregate_tier         │
        │   nan_breakdown[reason] += 1       │
        │   (already implemented; lines      │
        │   110-114 of compare.py — NO CHANGE)│
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │ comparison.md "NaN breakdown per   │
        │ tier" footer (compare.py:268-279   │
        │ — NO CHANGE)                       │
        └────────────────────────────────────┘
```

### Recommended Code Structure

The phase touches exactly TWO files (both existing — no new files, no new modules):

```
evaluation/harness/
├── score.py        # ADD NaNReasonTracer class + _classify_post_evaluate_nan() helper
│                   # MODIFY score_query_log() to wire tracer into evaluate(callbacks=)
│                   # MODIFY result→ScoreRecord mapping to call _classify_post_evaluate_nan()
│                   # NO CHANGE to _short_circuit_nan() body (the 4 existing reasons stay)
└── (compare.py — UNCHANGED; it already buckets by whatever nan_reason string appears)

evaluation/tests/
└── test_eval_score.py  # ADD tests for NaNReasonTracer (capture exception per-row)
                        # ADD tests for _classify_post_evaluate_nan() (3 reason mappings)
                        # ADD integration test feeding stub evaluate() output through full path
```

**Optional but recommended:** Add ONE new test file or extend `test_eval_compare.py` with a regression assertion that the new reason strings (`empty_statements`, `json_parse_failure`, `llm_did_not_finish`) flow through `aggregate_tier::nan_breakdown` correctly. This is a single test (~15 LOC) using the existing `_seed_results` helper.

### Pattern 1: Capture per-row exceptions via langchain callback

**What:** RAGAS uses `langchain_core` callbacks throughout `evaluate()`. Subclassing `BaseCallbackHandler` and overriding `on_chain_error` lets us record exceptions BEFORE the executor swallows them.

**When to use:** Any time you need to know WHY a RAGAS row produced NaN (faithfulness / answer_relevancy / context_precision all run through this callback chain).

**Example:**
```python
# Source: derived from langchain_core.callbacks.BaseCallbackHandler API
# verified against /Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/callbacks.py
# (RagasTracer uses the same base class) [VERIFIED]
from langchain_core.callbacks import BaseCallbackHandler
from ragas.callbacks import ChainType
from typing import Any
import uuid

class NaNReasonTracer(BaseCallbackHandler):
    """Capture per-row exception types so post-evaluate NaN classification has structured signals."""
    def __init__(self) -> None:
        # (run_id) -> {"row_index": int, "metric_name": str, "type": ChainType}
        self._chains: dict[str, dict] = {}
        # (row_index, metric_name) -> exception type name (e.g. "RagasOutputParserException")
        self.errors: dict[tuple[int, str], str] = {}

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                       tags=None, metadata=None, **kwargs):
        chain_type = (metadata or {}).get("type")
        entry = {"type": chain_type, "name": serialized.get("name", "")}
        # ROW chain carries row_index in metadata
        if chain_type == ChainType.ROW:
            entry["row_index"] = (metadata or {}).get("row_index")
        # METRIC chain inherits row_index from parent ROW chain
        elif chain_type == ChainType.METRIC and parent_run_id:
            parent = self._chains.get(str(parent_run_id), {})
            entry["row_index"] = parent.get("row_index")
            entry["metric_name"] = serialized.get("name", "").rsplit("-", 1)[0]
        # PROMPT chain inherits both from parent METRIC chain
        elif chain_type == ChainType.RAGAS_PROMPT and parent_run_id:
            parent = self._chains.get(str(parent_run_id), {})
            entry["row_index"] = parent.get("row_index")
            entry["metric_name"] = parent.get("metric_name")
        self._chains[str(run_id)] = entry

    def on_chain_error(self, error: BaseException, *, run_id, **kwargs):
        chain = self._chains.get(str(run_id), {})
        row_idx = chain.get("row_index")
        metric = chain.get("metric_name")
        if row_idx is not None and metric is not None:
            # Record only the leaf-most exception (don't overwrite with parent re-raises)
            key = (row_idx, metric)
            if key not in self.errors:
                self.errors[key] = type(error).__name__
```

**Critical detail:** `evaluation.py:244` creates ROW chains with `metadata={"type": ChainType.ROW, "row_index": i}` and `evaluation.py:255` submits METRIC tasks via `executor.submit(...)` with `name=f"{metric.name}-{i}"`. Names like `faithfulness-0`, `answer_relevancy-3`. When the prompt-level `RagasOutputParserException` raises (`pydantic_prompt.py:557`), `langchain` propagates `on_chain_error` up the callback chain.

### Pattern 2: Post-evaluate reason classification

**What:** After `result.to_pandas()` produces a row of (faithfulness, answer_relevancy, context_precision), classify each NaN cell using (a) the captured exception map and (b) the metric semantics.

**When to use:** Inside `score_query_log()` immediately after the existing dataframe-row-to-ScoreRecord mapping (`score.py:237-257`).

**Example:**
```python
# Source: synthesized from evaluation/harness/score.py + RAGAS faithfulness/answer_relevance/
# context_precision NaN paths verified above [VERIFIED via Read of metric source]
from typing import Optional

def _classify_post_evaluate_nan(
    row_idx: int,
    metric_name: str,
    metric_value: Optional[float],
    tracer: NaNReasonTracer,
) -> Optional[str]:
    """Map a (row_idx, metric, value) cell to a nan_reason string, or None if not NaN.

    Precedence (most specific to least specific):
    1. tracer captured RagasOutputParserException → 'json_parse_failure'
    2. tracer captured LLMDidNotFinishException → 'llm_did_not_finish'
    3. faithfulness=NaN with no captured exception → 'empty_statements'
       (Faithfulness._ascore returns np.nan when statements==[]; faithfulness.py:210-211)
    4. answer_relevancy=NaN with no captured exception → 'empty_questions'
       (calculate_score returns np.nan when all gen_questions==''; answer_relevance.py:120-124)
    5. context_precision=NaN with no captured exception → 'invalid_verdicts'
       (_calculate_average_precision returns np.nan; context_precision.py:116)
    6. NaN with no rule match → 'unknown_nan' (defensive — never silently drop)
    """
    if metric_value is not None:
        return None
    captured = tracer.errors.get((row_idx, metric_name))
    if captured == "RagasOutputParserException":
        return "json_parse_failure"
    if captured == "LLMDidNotFinishException":
        return "llm_did_not_finish"
    # Per-metric semantic NaN paths (when no exception was raised)
    if metric_name == "faithfulness":
        return "empty_statements"
    if metric_name == "answer_relevancy":
        return "empty_questions"
    if metric_name == "context_precision":
        return "invalid_verdicts"
    return "unknown_nan"

# A row's nan_reason is the FIRST non-None classification across the three metrics
# (faithfulness > answer_relevancy > context_precision precedence is arbitrary;
#  pick one and document it; the breakdown bucket only needs ONE reason per row).
```

**Note on success criterion #1 wording:** ROADMAP names exactly three reason values (`empty_contexts`, `empty_statements`, `json_parse_failure`). The taxonomy I've drafted has more — that's deliberate. The blog post only mentions the three named ones, so the rollup will likely show those plus the existing pre-call reasons. The extra `llm_did_not_finish`, `empty_questions`, `invalid_verdicts`, `unknown_nan` are defensive — they prevent silently mapping unrelated NaN paths to `empty_statements` and they make the post-rerun audit (Phase 7) easier. Verify with user during planning whether these extras are wanted; if not, collapse `empty_questions` and `invalid_verdicts` into `empty_statements` and `llm_did_not_finish` into `json_parse_failure` (matching the user's mental taxonomy).

### Anti-Patterns to Avoid

- **Pattern-matching exception strings:** Do NOT do `if "max_tokens" in str(error)`. RAGAS exceptions have stable type names (`RagasOutputParserException`, `LLMDidNotFinishException`); use `type(error).__name__` or `isinstance` checks. String contents change between patch versions.
- **Re-running NaN rows with `raise_exceptions=True`:** Doubles judge cost and is non-deterministic. Capture in-flight via callback instead.
- **Modifying `compare.py`:** The `nan_breakdown` dict (compare.py:110-114) and the markdown footer (compare.py:268-279) already iterate `nan_reason` values dynamically. New reason strings flow through automatically. Don't touch this file.
- **Schema-changing `ScoreRecord`:** `nan_reason: Optional[str]` is already the right shape. Adding new string values is a data-only change. Don't add structured nested fields here — keep the rollup logic dirt simple.
- **Catching `Exception` in the tracer:** Use the tracer's built-in `on_chain_error` hook, which receives `BaseException`. Don't subclass anything else; don't reach into the `Executor`.
- **Calling RAGAS twice (once with raise, once without):** Wastes budget. The callback approach captures everything in one pass.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-row exception capture during evaluate() | Custom subprocess wrapper, signal handler, or thread-local var | `langchain_core.callbacks.BaseCallbackHandler.on_chain_error` | RAGAS routes ALL per-row exceptions through langchain callbacks. The hook is documented and stable. |
| Detecting NaN in dataframe cells | Custom `is_nan()` helper | Existing `_to_float_or_none()` (score.py:171-181) | Already handles None / float NaN / pandas NA / non-numeric strings. Reuse. |
| Per-tier NaN bucket aggregation | New aggregator function | Existing `compare.py::aggregate_tier::nan_breakdown` (lines 110-114) | Already iterates metrics list and increments per-reason counts. New strings flow through. |
| Per-row reason persistence | Sidecar JSON, separate file | Existing `ScoreRecord.nan_reason: Optional[str]` (records.py:63) | Schema already supports any string value. |
| Markdown rendering of breakdown | New formatter | Existing `compare.py:268-279` "NaN breakdown per tier" loop | Already sorts and joins per-reason counts; uses dict.items(). |
| Reading pandas dataframe rows | Custom CSV/JSON conversion | Existing `result.to_pandas()` path (score.py:237) | Already handled with `df.iloc[j].get("metric_name")`. Add reason classification next to it. |

**Key insight:** This phase is ~80% glue code reuse. The new code is the tracer (~50 LOC) + the post-evaluate classifier (~30 LOC) + tests. Resist the urge to refactor anything else — the existing aggregator and renderer ALREADY work correctly because they were built in Phase 131 with this exact extension in mind (see `compare.py` decision references to "honest aggregation").

## Runtime State Inventory

> Skip section: Phase 3 is a code-only change. No runtime state, stored data, OS-registered services, secrets, or build artifacts are touched. The existing `metrics/tier-N-*.json` files on disk continue to be readable (the schema only ever ADDS reason values to a string field that already exists). No data migration is required; old files with `nan_reason: "empty_contexts"` or `null` continue to round-trip through `ScoreRecord` validation unchanged.

**Verified:** `nan_reason: Optional[str]` accepts any string. New values flow through Pydantic validation without schema changes. Existing JSON files on disk under `evaluation/results/metrics/` continue to parse.

## Common Pitfalls

### Pitfall 1: RAGAS Executor swallows exceptions before evaluate() returns
**What goes wrong:** You add a `try/except` around `evaluate()` and never see the exception, because `Executor.wrap_callable_with_index` catches every per-row exception and returns `np.nan` (`ragas/executor.py:71-84`). The dataframe just shows NaN with no clue why.
**Why it happens:** Async batching. RAGAS runs N rows × M metrics concurrently; surfacing exceptions from one row would crash the whole batch with `raise_exceptions=True`. Default is `False` (used by `score.py:232`) precisely so partial results survive.
**How to avoid:** Use the langchain callback hook (`on_chain_error`), which fires BEFORE the swallow. The Executor is downstream of the callback handler chain.
**Warning signs:** "All my NaN rows have the same generic reason" → you're not capturing per-row context. Add row_index to your callback's chain dict and verify it appears in `errors` keys.

### Pitfall 2: Mistaking faithfulness's empty-statements NaN for an exception
**What goes wrong:** You assume every NaN must come from an exception, miss the `if statements == []: return np.nan` branch in `Faithfulness._ascore` (faithfulness.py:210-211), and label the row `unknown_nan` or `json_parse_failure` incorrectly.
**Why it happens:** This NaN path is **silent** — no exception, no log, just a returned NaN. It happens when the judge LLM returns `{"statements": []}` (or the parser successfully decodes an empty list). Reasons include: extremely short answer, non-English answer, or judge model refusing the prompt.
**How to avoid:** Document explicitly that "faithfulness=NaN AND tracer captured no exception for this row+metric" → `empty_statements`. The classifier in Pattern 2 above does this.
**Warning signs:** Tier 4/5 rerun shows N rows with `empty_statements` faithfulness reason — that's expected when answers are very short. Cross-check by reading the actual answer text in the QueryLog; if answer is < 50 chars, this is the likely cause.

### Pitfall 3: ChainType.ROW row_index is in metadata, not the chain name
**What goes wrong:** You parse `serialized["name"]` looking for `"row 0"`, `"row 1"` strings; your tests pass on RAGAS 0.4.3 but break in 0.4.4 if the format changes to `"row-0"` or similar.
**Why it happens:** Names are human-readable; metadata is the structured contract. `evaluation.py:248` sets `metadata={"type": ChainType.ROW, "row_index": i}`. Read from metadata, not name.
**How to avoid:** In `on_chain_start`, pull `(metadata or {}).get("row_index")` for ROW chains; pull `(metadata or {}).get("type")` to dispatch on chain type. Never parse `serialized["name"]` for structured info.
**Warning signs:** Tracer works in tests with stubbed metadata but fails on real evaluate() runs → you're reading the wrong field.

### Pitfall 4: Metric name in METRIC chain includes row index suffix
**What goes wrong:** You expect `metric_name == "faithfulness"` but the chain's name is `"faithfulness-3"` because `evaluation.py:259` uses `name=f"{metric.name}-{i}"`. Your error map keys never match.
**Why it happens:** RAGAS uses `name` for human-readable tracing UIs; the suffix lets users distinguish row 3 from row 4 in trace exports.
**How to avoid:** Strip the `-{i}` suffix: `metric_name = serialized["name"].rsplit("-", 1)[0]`. OR (more robust) detect row_index by walking up parent_run_id to the ROW chain's metadata.
**Warning signs:** Empty `errors` dict despite exceptions actually firing → keys don't match what `_classify_post_evaluate_nan` looks up.

### Pitfall 5: nan_reason precedence when multiple metrics fail on the same row
**What goes wrong:** A row produces NaN for all three metrics. Which reason "wins" for the row-level `ScoreRecord.nan_reason`? Today the field is single-string; without a precedence rule the assignment is order-dependent and tests become flaky.
**Why it happens:** Three metrics × per-row → three potential reasons; only one can fit in `Optional[str]`.
**How to avoid:** Either (a) document a fixed precedence (faithfulness > answer_relevancy > context_precision) and assert it in tests, or (b) extend the schema to hold a per-metric breakdown (`nan_reasons: dict[str, Optional[str]]`). For HARN-05 the simplest path is (a) with documented precedence; (b) is the Phase-9 frozen-doc-friendly answer if reviewers care about per-metric attribution.
**Warning signs:** Same row shows `empty_statements` in one rerun and `invalid_verdicts` in another → no documented precedence rule. Decide and lock it.

### Pitfall 6: Wiring tracer to evaluate() callbacks without preserving RAGAS's own callbacks
**What goes wrong:** You pass `evaluate(..., callbacks=[NaNReasonTracer()])` and break cost tracking, because RAGAS internally appends its own `RagasTracer` and `CostCallbackHandler` to `callbacks` (evaluation.py:228-232). If `callbacks` is the wrong type (BaseCallbackManager vs list), behavior diverges.
**Why it happens:** evaluate() expects either a list it can `.append()` to, or a `BaseCallbackManager` with `.add_handler()`. Passing a list is correct.
**How to avoid:** Pass `callbacks=[NaNReasonTracer()]` as a list. RAGAS will append its own handlers to that list. After evaluate() returns, the tracer instance you passed in is still accessible via the closure variable.
**Warning signs:** Cost tracking breaks (`usage["input_tokens"] == 0` in test that previously had real numbers) → you're shadowing RAGAS's callback list. Pass list, not handler instance directly.

### Pitfall 7: Confusing "no nan_reason captured" with "row scored cleanly"
**What goes wrong:** A row scored with all three metrics non-NaN should have `nan_reason=None`. A row that NaN'd but the tracer didn't capture (some unforeseen path) ALSO ends up with `nan_reason=None` if the classifier gates on captured-only. Now you can't distinguish "scored OK" from "silently failed".
**Why it happens:** Defensive programming gone wrong — if-no-error-default-to-None is the wrong default.
**How to avoid:** Default to `unknown_nan` for any cell that is NaN AND no rule matched. Never let an unexplained NaN reach disk as `nan_reason=None`. Log a WARNING when assigning `unknown_nan` so it surfaces in the run log.
**Warning signs:** `comparison.md` shows `n_NaN=N` for a tier but the per-reason breakdown sums to less than N → some rows are "secretly NaN with no reason". Add an assertion in `aggregate_tier`: `sum(nan_breakdown.values()) == n_nan`.

## Code Examples

Verified patterns from this codebase + RAGAS source:

### Existing: `_short_circuit_nan()` pattern (DO NOT MODIFY — extend, don't replace)
```python
# Source: /Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/score.py:152-168
def _short_circuit_nan(rec: EvalRecord) -> Optional[ScoreRecord]:
    """Return a NaN ScoreRecord without calling RAGAS for known-bad inputs."""
    if rec.error == "max_turns_exceeded":
        return ScoreRecord(question_id=rec.question_id, nan_reason="agent_truncated")
    if not rec.retrieved_contexts:
        return ScoreRecord(question_id=rec.question_id, nan_reason="empty_contexts")
    if rec.error and rec.error.startswith("tier4_import_error"):
        return ScoreRecord(question_id=rec.question_id, nan_reason="tier4_unavailable")
    if rec.error and rec.error.startswith("cached_miss"):
        return ScoreRecord(question_id=rec.question_id, nan_reason="cached_miss")
    return None
```

### NEW: dataframe row → ScoreRecord with reason classification
```python
# Source: synthesis from score.py:237-257 + Pattern 2 above
# Insert in score_query_log() AFTER the existing df.iloc[j] extraction
for j, (orig_idx, _) in enumerate(samples_with_index):
    row = df.iloc[j]
    f = _to_float_or_none(row.get("faithfulness"))
    ar = _to_float_or_none(row.get("answer_relevancy"))
    cp = _to_float_or_none(row.get("context_precision"))

    # Phase 3: classify NaN reason with row precedence (faithfulness > AR > CP)
    nan_reason = (
        _classify_post_evaluate_nan(j, "faithfulness", f, tracer)
        or _classify_post_evaluate_nan(j, "answer_relevancy", ar, tracer)
        or _classify_post_evaluate_nan(j, "context_precision", cp, tracer)
    )

    scores[orig_idx] = ScoreRecord(
        question_id=log.records[orig_idx].question_id,
        faithfulness=f,
        answer_relevancy=ar,
        context_precision=cp,
        nan_reason=nan_reason,
    )
```

### NEW: deterministic test fixture for json_parse_failure
```python
# Source: synthesized from RAGAS pydantic_prompt.py:557 + test_eval_score.py existing patterns
# A stub LLM that returns malformed JSON triggers RagasOutputParserException deterministically
# without spending judge budget.
class _MalformedJsonLLM:
    """Stub BaseRagasLLM that returns text that fails Pydantic parse."""
    async def generate(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        from langchain_core.outputs import Generation, LLMResult
        # Returns text that is NOT valid JSON for any RAGAS prompt's output_model
        return LLMResult(generations=[[Generation(text="not json at all { broken ")]])
    def is_finished(self, result): return True

# Usage: monkeypatch RAGAS's faithfulness.llm to this stub, run evaluate() on a single
# sample with raise_exceptions=False, assert tracer.errors[(0, "faithfulness")] ==
# "RagasOutputParserException"
```

### NEW: deterministic test fixture for empty_statements
```python
# Source: synthesized from RAGAS faithfulness.py:208-211
# Faithfulness short-circuits to NaN when StatementGeneratorOutput.statements == []
class _EmptyStatementsLLM:
    """Stub BaseRagasLLM that returns parseable JSON with empty statements list."""
    async def generate(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        from langchain_core.outputs import Generation, LLMResult
        return LLMResult(generations=[[Generation(text='{"statements": []}')]])
    def is_finished(self, result): return True

# Usage: same as above. Assert ScoreRecord(...).nan_reason == "empty_statements" AND
# tracer.errors does NOT contain (0, "faithfulness") — empty_statements is the no-exception
# path.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single opaque `n_NaN` count per tier | Per-reason `nan_breakdown` dict in `aggregate_tier` (compare.py:110-114) | Phase 131 (existing) | Already in place — proves the rollup architecture is reason-aware |
| Pre-call reasons only (`empty_contexts`, `agent_truncated`, `tier4_unavailable`, `cached_miss`) | Pre-call + post-evaluate reasons (`empty_statements`, `json_parse_failure`, ...) | THIS PHASE | Closes the gap between "RAGAS returned NaN" and "we know why" |
| Re-run with `raise_exceptions=True` to debug NaN | Capture exception types via callback in single run | THIS PHASE recommendation | No double-cost; deterministic; captures original failure mode |

**Deprecated/outdated:**
- The pattern of `f"{metric.name}-{i}"` as the only metric identifier is RAGAS-version-stable in 0.4.x but should be cross-checked when bumping RAGAS. Phase-3 plan should include a regression test (`test_nan_reason_tracer_metric_name_format`) so that future RAGAS upgrades surface the format change at test time, not in production reruns.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The user's mental taxonomy of three reasons (`empty_contexts`, `empty_statements`, `json_parse_failure`) is exhaustive — no requirement to also surface `llm_did_not_finish` / `invalid_verdicts` separately | Architecture Pattern 2 + Pitfall 5 | If user wants finer granularity (separate `llm_did_not_finish` etc.), the classifier's current draft already produces those — no rework, just keep the extra strings rather than collapsing. If user wants coarser granularity (only the 3 named), collapse `llm_did_not_finish` into `json_parse_failure` and `empty_questions`/`invalid_verdicts` into `empty_statements`. Both are 1-line changes. |
| A2 | The single-string `nan_reason` field is sufficient (one reason per row, not per-metric) | Pitfall 5 + Architecture Map | If reviewers ask "which metric failed for THIS row" the schema needs `nan_reasons: dict[str, str]`. Current draft documents precedence (faithfulness > AR > CP) which is honest but not maximally informative. Discuss with user during planning. |
| A3 | RAGAS 0.4.3's chain-name format `f"{metric.name}-{i}"` is stable enough for `.rsplit("-", 1)[0]` parsing through the v1.0 evaluation cycle | Pitfall 4 | If RAGAS bumps and changes the format, the tracer mis-maps metrics. Mitigation: add a unit test asserting `serialized["name"]` matches the expected pattern after a sample evaluate() run; CI catches drift. |
| A4 | The phase does NOT need to backfill historical metrics JSONs on disk with new reasons; only files produced AFTER this phase need the richer breakdown | Runtime State Inventory | If the user wants Phase 9's frozen doc to retroactively re-classify Tier 4 / Tier 5 30/30 NaN rows from the 2026-05-02 baseline, that's an explicit re-score — and it would be cheap (the rows have empty_contexts which is already pre-call). Flag for discuss-phase. |
| A5 | No new pip dependency is required; `langchain-core` is a transitive dep of `ragas` and stable enough to import directly | Standard Stack | If pip resolver puts an incompatible langchain-core version in the env, callback API may break. Verified at research time (`from langchain_core.callbacks import BaseCallbackHandler` works in current `.venv`); add a smoke test asserting the import in test_eval_score.py. |

## Open Questions

1. **Per-row vs per-metric reason granularity**
   - What we know: ROADMAP success criterion #1 says "a per-row `nan_reason` field" (singular). HARN-05 says "distinguish reasons in per-row metrics output" (also singular).
   - What's unclear: Does the user want one reason per row, or one reason per (row, metric) cell?
   - Recommendation: Start with per-row + documented precedence (faithfulness > AR > CP) as it matches the literal ROADMAP text. Surface the precedence in the planning conversation; defer per-metric to Phase 9 (frozen doc) if reviewers ask for it. ESTIMATED 80% likely the user wants per-row given the simplicity bias visible elsewhere in this project.

2. **Should we backfill the existing 2026-05-02 baseline metrics with new reasons?**
   - What we know: Tier 4 and Tier 5 baseline files (`evaluation/results/metrics/tier-{4,5}-2026-05-02T*.json`) all show `nan_reason: "empty_contexts"`. That's already the most specific reason — no upgrade possible since they short-circuited pre-call.
   - What's unclear: Phase 1 (Tier 5 fix) and Phase 2 (Tier 4 fix) reruns will produce NEW NaN cases that PASS the pre-call check but might fail mid-evaluate. Do we backfill those?
   - Recommendation: NO backfill of pre-2026-05-04 data — the empty_contexts label is already truthful. Phase 7 (Full 5-Tier Rerun) will produce fresh files with the richer taxonomy automatically.

3. **How aggressively should we test the live RAGAS callback integration?**
   - What we know: Existing live test (`evaluation/tests/test_eval_smoke_live.py:141`) already exercises a real evaluate() against Gemini judge.
   - What's unclear: Should we add a live test that DELIBERATELY triggers `RagasOutputParserException` (e.g., feed a malformed answer and see judge JSON parse fail)? Or rely on stubbed-LLM unit tests?
   - Recommendation: Stubbed unit tests are sufficient for HARN-05; the failure-mode detection is testable without a real LLM. Add ONE live smoke test that asserts "evaluate() with valid inputs produces no nan_reason of `unknown_nan`" — that catches the case where our classifier misses a real RAGAS NaN path we didn't anticipate.

4. **Is `empty_questions` (answer_relevancy NaN with no exception) even reachable in practice?**
   - What we know: `_calculate_score` in `_answer_relevance.py:120-124` returns NaN when `all(q == "" for q in gen_questions)`. This requires the judge to produce exactly N empty `question` fields in its JSON output.
   - What's unclear: Does Gemini Flash 2.5 ever do this? Or does it always produce some non-empty question?
   - Recommendation: Don't pre-optimize — keep `empty_questions` in the taxonomy with a comment that it's defensive. If after Phase 7 zero rows show this reason, we can collapse it in Phase 9. Adding a defensive bucket costs nothing; missing one costs a Phase-9 re-spin.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 | Score module | ✓ | 3.13 [VERIFIED: .venv path] | — |
| ragas | RAGAS evaluate() + exception types | ✓ | 0.4.3 [VERIFIED: _version.py] | — |
| langchain-core | BaseCallbackHandler subclass | ✓ | (transitive — installed by ragas) [VERIFIED: ragas/callbacks.py imports it] | — |
| pytest | Test execution | ✓ | (in .venv) | — |
| numpy | NaN detection (already used) | ✓ | (transitive) | — |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

This phase is a code-only change against an environment that already has every dep needed.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already in use; see `evaluation/tests/test_eval_score.py`) |
| Config file | (project uses pytest defaults; no pytest.ini in repo root, runs collect from `tests/` and `evaluation/tests/`) |
| Quick run command | `python -m pytest evaluation/tests/test_eval_score.py -x` |
| Full suite command | `python -m pytest evaluation/tests/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HARN-05 | `_short_circuit_nan` continues to emit existing 4 reasons | unit | `python -m pytest evaluation/tests/test_eval_score.py::test_short_circuit_empty_contexts -x` | ✅ exists; regression |
| HARN-05 | `NaNReasonTracer.on_chain_error` records (row, metric, exception_type) | unit | `python -m pytest evaluation/tests/test_eval_score.py::test_nan_reason_tracer_captures_per_row -x` | ❌ Wave 0 |
| HARN-05 | `_classify_post_evaluate_nan` maps `RagasOutputParserException` → `json_parse_failure` | unit | `python -m pytest evaluation/tests/test_eval_score.py::test_classify_json_parse_failure -x` | ❌ Wave 0 |
| HARN-05 | `_classify_post_evaluate_nan` maps faithfulness=NaN+no-exception → `empty_statements` | unit | `python -m pytest evaluation/tests/test_eval_score.py::test_classify_empty_statements -x` | ❌ Wave 0 |
| HARN-05 | `score_query_log` end-to-end with stubbed RAGAS produces 3 distinct reasons across 3 rows | integration | `python -m pytest evaluation/tests/test_eval_score.py::test_score_query_log_distinguishes_reasons -x` | ❌ Wave 0 |
| HARN-05 | `compare.py::aggregate_tier` buckets new reasons (regression — proves no compare.py change needed) | unit | `python -m pytest evaluation/tests/test_eval_compare.py::test_aggregate_tier_with_new_reasons -x` | ❌ Wave 0 |
| HARN-05 | `comparison.md` footer renders new reasons correctly | unit | `python -m pytest evaluation/tests/test_eval_compare.py::test_emit_markdown_with_new_reasons -x` | ❌ Wave 0 (or extend existing test_aggregate_tier_with_nan) |
| HARN-05 | Live smoke check: real Gemini judge run produces no `unknown_nan` rows | smoke (live) | `python -m pytest evaluation/tests/test_eval_smoke_live.py -k smoke_nan_reasons` (NEW) | ❌ Wave 0 (skipif no OPENROUTER_API_KEY) |

### Sampling Rate
- **Per task commit:** `python -m pytest evaluation/tests/test_eval_score.py -x` (~3s, no API)
- **Per wave merge:** `python -m pytest evaluation/tests/ -x` (~10s, no API except live smoke marked `-m live`)
- **Phase gate:** Full suite green + ONE live smoke run against current Tier 4 metrics file produces a populated `nan_breakdown` (not just `empty_contexts`)

### Wave 0 Gaps
- [ ] Add `test_nan_reason_tracer_captures_per_row` — instantiate tracer, manually fire `on_chain_start` then `on_chain_error`, assert errors dict
- [ ] Add `test_classify_json_parse_failure` — feed mock tracer with captured exception, assert classification
- [ ] Add `test_classify_empty_statements` — empty captured, faithfulness=None, assert `empty_statements`
- [ ] Add `test_classify_unknown_nan_warns` — faithfulness=None, no exception, no rule → `unknown_nan` + log warning
- [ ] Add `test_score_query_log_distinguishes_reasons` — full path with stubbed `_MalformedJsonLLM` and `_EmptyStatementsLLM`
- [ ] Add `test_aggregate_tier_with_new_reasons` to `test_eval_compare.py` — proves no compare.py change is needed
- [ ] Optional: add `test_short_circuit_unchanged` regression in `test_eval_score.py` asserting the 4 existing reasons still emit (catches accidental rewrites)
- [ ] Update `evaluation/tests/test_eval_smoke_live.py` (or add a `live`-marked smoke test) that asserts `unknown_nan` count == 0 against a real evaluate() run

## Project Constraints (from CLAUDE.md)

No `CLAUDE.md` was found in the working directory. The planner should treat all phase decisions as Claude's discretion within the bounds of:
- ROADMAP success criteria (3 listed in `additional_context`)
- HARN-05 requirement text
- Existing repo conventions (Python 3.10+, pytest, Pydantic v2, no new top-level packages)
- Git commit style observed in recent commits: `<type>(<scope>): <subject>` with `feat(02-04)`, `test(02-04)`, `docs(02-04)` prefixes

## Sources

### Primary (HIGH confidence)
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/score.py` — `_short_circuit_nan`, `score_query_log`, `_to_float_or_none`, `JUDGE_MAX_TOKENS` constant [VERIFIED: full read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/compare.py` — `aggregate_tier::nan_breakdown`, `emit_markdown` footer rendering [VERIFIED: full read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/records.py` — `ScoreRecord(nan_reason: Optional[str])` schema [VERIFIED: full read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/run.py` — capture-stage QueryLog production (no change needed) [VERIFIED: full read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/smoke_gate.py` — downstream consumer of `nan_reason` (must keep working) [VERIFIED: full read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/exceptions.py` — `RagasOutputParserException`, `LLMDidNotFinishException` definitions [VERIFIED: full read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/executor.py:71-84` — exception swallow + np.nan return path [VERIFIED: read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/evaluation.py:202-309` — evaluate() + callback wiring [VERIFIED: read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/callbacks.py` — `RagasTracer`, `ChainType`, `new_group` (sibling to NaNReasonTracer) [VERIFIED: full read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/metrics/_faithfulness.py:166-214` — `_create_statements`, empty-statements NaN path [VERIFIED: full read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/metrics/_answer_relevance.py:114-129` — answer_relevancy NaN path [VERIFIED: read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/metrics/_context_precision.py:110-131` — context_precision NaN path [VERIFIED: read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/prompt/pydantic_prompt.py:540-558` — `RagasOutputParserException` raise site [VERIFIED: read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/.venv/lib/python3.13/site-packages/ragas/llms/base.py:124-127` — `LLMDidNotFinishException` raise site [VERIFIED: read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/tests/test_eval_score.py` — existing `_short_circuit_nan` test patterns to extend [VERIFIED: full read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/tests/test_eval_compare.py:75-176` — existing `nan_breakdown` aggregator tests [VERIFIED: read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/results/comparison.md` — current `comparison.md` showing per-reason rollup already works [VERIFIED: full read]
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/results/metrics/tier-4-2026-05-05T13_59_50Z.json` — current per-row schema [VERIFIED: full read]

### Secondary (MEDIUM confidence)
- ROADMAP Phase 3 success criteria (.planning/ROADMAP.md:60-66) [CITED: ROADMAP.md as authoritative project spec]
- REQUIREMENTS HARN-05 (.planning/REQUIREMENTS.md:26) [CITED: REQUIREMENTS.md as authoritative project spec]
- pyproject.toml dependency declarations [CITED: confirms ragas/langchain-core availability]

### Tertiary (LOW confidence)
- None. All claims tied to verified file reads.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every dep is already installed, version-pinned, and read directly from the .venv.
- Architecture: HIGH — every code path needed (callback subclassing, exception types, NaN swallow location, dataframe shape, aggregator schema, markdown rendering) was read directly from source files. No reliance on training data or external docs.
- Pitfalls: HIGH — all 7 pitfalls are derived from reading the actual RAGAS 0.4.3 code in this repo's `.venv`. Pitfall 4 (metric name suffix) and Pitfall 5 (per-row vs per-metric) are particularly load-bearing for plan correctness.
- Code examples: HIGH for existing patterns (verbatim file reads), MEDIUM for the NEW `NaNReasonTracer` (synthesized from langchain_core API; verify against a stubbed evaluate() run during Wave 0 testing).

**Research date:** 2026-05-05
**Valid until:** 2026-06-04 (RAGAS 0.4.x patches occasionally change internals; if a RAGAS bump happens before plan execution, re-verify Pitfall 4's metric-name format).

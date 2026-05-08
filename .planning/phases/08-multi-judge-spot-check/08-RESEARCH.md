# Phase 8: Multi-Judge Spot-Check — Research

**Researched:** 2026-05-07
**Domain:** Re-scoring captured RAGAS QueryLogs with a non-Gemini judge to bound family-bias delta
**Confidence:** HIGH (existing harness already supports `--judge-model`; only delta + JSON shape + question-filter are new)

## Summary

Phase 8 adds a small, focused command that takes the Phase-7 sweep_sha=`75f6f1b` (sweep_ts=`2026-05-07T10:59:10Z`) primary-judge metrics, picks 5 question IDs × 3 tiers (15 cells), and re-scores them with a non-Gemini judge — `openrouter/anthropic/claude-haiku-4.5` is already wired (`score.py` docstring example, `shared/pricing.py:42`) and is the lowest-LOC choice. The output is a single structured JSON at `evaluation/results/metrics/multi-judge-spotcheck-{TS}.json` carrying per-cell `(faithfulness, answer_relevancy, context_precision)` triples for primary + secondary + signed delta, plus full provenance (source capture path + git_sha + ts + secondary judge slug + version). Cost is bounded by 15 cells × ~$0.003/cell ≈ $0.045 estimated, comfortably inside the $0.10–0.30 ROADMAP budget.

The phase introduces a new module `evaluation/harness/multi_judge_spotcheck.py` (recommendation, see Q3) rather than extending `score.py`. Reason: `score.py` is forward-contract-locked across Phases 4/5/6/7 and the spot-check needs a different output shape (delta vs. raw scores), a question-id filter, and provenance equivalence with the SOURCE capture (not the re-score time). Adding it as a new module keeps the byte-identical guard intact.

**Primary recommendation:** Build `evaluation/harness/multi_judge_spotcheck.py` (≤180 raw LOC) that imports `score._build_judge`, `score.score_query_log`, `records.read_query_log`, plus a small custom cost recorder via `CostTracker("multi-judge-spotcheck")`. Default judge `openrouter/anthropic/claude-haiku-4.5`. Default question subset = 5 IDs (`single-hop-001`, `single-hop-002`, `multi-hop-001`, `multi-hop-002`, `multimodal-001`) × tiers `1, 4, 5` (one per architecture family — vector / graph-multimodal / agentic). Output JSON keyed by `(question_id, tier)` with `primary`, `secondary`, `delta` blocks per metric.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CAP-02 | User can run a multi-judge spot-check re-scoring 5 questions × 3 tiers (15 cells) with a non-Gemini judge (Claude Haiku or GPT-4.1-mini), and the resulting delta-from-primary-judge is captured in a structured JSON for the frozen doc | Existing `score._build_judge` already accepts arbitrary LiteLLM slugs (Pattern: existing `--judge-model` CLI flag in `score.py:548`); new module wires read-existing-capture + delta arithmetic + structured-JSON write. `anthropic/claude-haiku-4.5` already in `shared/pricing.py:42` so cost-tracker integration is mechanical. |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Read source QueryLog (Phase 7 capture) | Storage / file I/O | — | `records.read_query_log` already does this. No re-capture. |
| Filter to 5 question IDs | Pure transformation | — | Subset filter on `log.records` list before passing to `score_query_log`. |
| Build secondary judge (LiteLLM) | API / LLM call | — | `score._build_judge(judge_model, judge_emb)` already abstracts this. Change only the slug. |
| Re-score 5 records × 3 tiers | API / LLM call | Pure | `score.score_query_log` returns `list[ScoreRecord]` for the filtered subset. |
| Read primary metrics (Gemini) | Storage / file I/O | — | `_latest(metrics_dir, "tier-{N}-*.json")` from `compare.py` for the matching SHA-pinned file. |
| Compute signed delta = secondary − primary | Pure transformation | — | Per-metric, per-cell arithmetic; NaN propagates as `None`. |
| Write structured spot-check JSON | Storage / file I/O | — | Single file `evaluation/results/metrics/multi-judge-spotcheck-{TS}.json`. |
| Record secondary-judge cost | Storage / file I/O | — | Custom `CostTracker("multi-judge-spotcheck")` instance, persist with explicit `dest_dir` to dodge the v1.1 mis-feature. |

**Why this matters:** The spot-check is purely re-scoring + delta-arithmetic — no capture, no graph, no agents. All real work lives in already-built helpers. The new code is a thin orchestration layer plus a JSON schema.

## Standard Stack

### Core (already installed; verified in `pyproject.toml` + `uv.lock`)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `litellm` | 1.83.0 | Unified LLM gateway: OpenRouter → Anthropic Claude Haiku 4.5 | [VERIFIED: uv.lock] Same gateway as primary judge; symmetric flow ensures no methodology drift |
| `ragas` | 0.4.3 | Evaluation metrics (faithfulness / answer_relevancy / context_precision) | [VERIFIED: pyproject.toml + uv.lock] Same metrics RAGAS uses for the primary judge — apples-to-apples |
| `pydantic` | (transitive) | `QueryLog` / `ScoreRecord` schema | [VERIFIED: records.py imports] Existing schema; reuse `ScoreRecord` for cells |
| `rich` | (existing) | Console output | [VERIFIED: pipeline.py uses it] Match existing CLI pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `shared.cost_tracker` | (in-repo) | Cost recording (D-13 schema) | Always — the spot-check spend MUST be on disk per Success Criterion 3 |
| `shared.pricing` | (in-repo) | Token-to-USD lookup | Always — `anthropic/claude-haiku-4.5` already keyed at `pricing.py:42` (input $1.00 / output $5.00 per 1M tokens) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `openrouter/anthropic/claude-haiku-4.5` | `openrouter/openai/gpt-4.1-mini` | [VERIFIED: `shared/pricing.py` grep] gpt-4.1-mini is NOT in `PRICES`. Adding it requires editing `shared/pricing.py` (forward-contract risk on a previously-untouched file plus a verified vendor-rate citation). Claude Haiku 4.5 is zero-LOC. |
| New module `multi_judge_spotcheck.py` | Extending `score.py` with `--questions` + spot-check mode | [VERIFIED: Phases 4/5/6/7 forward-contract trail in 02-04, 03-02, 04-01, 05-01, 05-02, 06-01, 07-01, 07-02, 07-03 SUMMARYs] `score.py` is RAW-LOCKed across all four prior phases. Touching it would break the byte-identical guard. New module preserves the contract. |
| Re-capturing 5 questions × 3 tiers fresh | Re-using existing Phase 7 captures | [VERIFIED: STATE.md note + ROADMAP] Re-capturing would burn ~$0.30 on top of judge spend AND change the SHA. Phase 7 SHA is the version-of-record; the spot-check MUST cite it, not a new SHA. |

**Installation:** None. All dependencies present in current `uv.lock`.

**Version verification:** Performed against `uv.lock` (committed 2026-05-07):
- `litellm == 1.83.0` (matches `freeze.py` BONUS_LIBS pin)
- `ragas == 0.4.3` (matches `freeze.py` CRITICAL_LIBS pin)

## Architecture Patterns

### System Architecture Diagram

```
                     ┌─────────────────────────────────────────────┐
                     │ Phase 7 sweep_sha=75f6f1b (sweep_ts=         │
                     │ 2026-05-07T10:59:10Z) — IMMUTABLE INPUT     │
                     │                                             │
                     │ queries/tier-{1,4,5}-2026-05-07T10_59_10Z   │
                     │   .json  (3 files; 30 EvalRecord each)      │
                     │                                             │
                     │ metrics/tier-{1,4,5}-2026-05-07T10_59_10Z   │
                     │   .json  (3 files; 30 ScoreRecord each      │
                     │           — primary judge = Gemini Flash)   │
                     └─────────────────────────────────────────────┘
                                       │
                          read_query_log + filter (5 ids)
                                       ▼
                     ┌─────────────────────────────────────────────┐
                     │ multi_judge_spotcheck.py amain()            │
                     │  1. Resolve subset: 5 ids × 3 tiers         │
                     │  2. Build secondary judge:                  │
                     │     score._build_judge(                     │
                     │       "openrouter/anthropic/                │
                     │        claude-haiku-4.5",                   │
                     │       JUDGE_EMB_SLUG_DEFAULT)               │
                     │  3. Per tier: score_query_log(              │
                     │       filtered_log, qa_index,               │
                     │       judge_llm, judge_emb)                 │
                     │  4. Read primary metrics from               │
                     │     metrics/tier-{N}-{sweep_ts}.json        │
                     │  5. Compute delta = secondary − primary     │
                     │     per metric per cell                     │
                     │  6. CostTracker("multi-judge-spotcheck")    │
                     │     .record_llm(...) for the slug           │
                     │  7. Write 2 JSONs                           │
                     └─────────────────────────────────────────────┘
                              │                       │
                              ▼                       ▼
            metrics/multi-judge-spotcheck-       costs/multi-judge-
              {sweep_ts}.json                     spotcheck-{ts}.json
                              │                       │
                              └──────────┬────────────┘
                                         ▼
                       Phase 9 freeze.py reads BOTH for the
                       frozen doc's family-bias disclosure
                       (manifest may carry a multi_judge_spotcheck
                        block — flag for Phase 9 contract surface)
```

### Recommended Project Structure
```
evaluation/harness/
├── multi_judge_spotcheck.py     # NEW (≤180 raw LOC)
├── score.py                     # UNCHANGED (forward-contract locked since Phase 4)
├── pipeline.py                  # UNCHANGED (forward-contract locked since Phase 5)
├── freeze.py                    # UNCHANGED in Phase 8 (Phase 9 may add manifest field)
├── compare.py                   # UNCHANGED
├── records.py                   # UNCHANGED
└── run.py                       # UNCHANGED

evaluation/tests/
└── test_eval_multi_judge_spotcheck.py  # NEW (unit + 1 live test)

evaluation/results/
├── metrics/
│   └── multi-judge-spotcheck-{TS}.json   # NEW output
└── costs/
    └── multi-judge-spotcheck-{TS}.json   # NEW output (D-13 schema via CostTracker)
```

### Pattern 1: Re-use score._build_judge — never re-implement LiteLLM wiring
**What:** Import `_build_judge` directly; pass a different slug.
**When to use:** Always. The module already handles RAGAS embeddings legacy aliasing (`embed_query` / `embed_documents`), `max_tokens=JUDGE_MAX_TOKENS=8192` (Plan 02-04 lesson), and `provider="litellm"` boilerplate.
**Example:**
```python
# Source: evaluation/harness/score.py:115-156
from evaluation.harness.score import _build_judge, JUDGE_EMB_SLUG_DEFAULT

SECONDARY_JUDGE_SLUG = "openrouter/anthropic/claude-haiku-4.5"
secondary_llm, secondary_emb = _build_judge(SECONDARY_JUDGE_SLUG, JUDGE_EMB_SLUG_DEFAULT)
# secondary_llm now has max_tokens=8192 baked in (Plan 02-04 fix carries forward)
```

### Pattern 2: Re-use score.score_query_log + ScoreRecord shape
**What:** Pass an already-filtered `QueryLog` (with only the 5 picked records) to `score_query_log`; it returns `list[ScoreRecord]` of length 5.
**When to use:** Always. Avoids re-implementing the NaN short-circuit + `NaNReasonTracer` wiring + post-evaluate classification (Phase 3 work).
**Example:**
```python
# Source: evaluation/harness/score.py:293-408
from evaluation.harness.records import read_query_log
from evaluation.harness.score import score_query_log

WANTED_IDS = ["single-hop-001", "single-hop-002",
              "multi-hop-001", "multi-hop-002", "multimodal-001"]

src_log = read_query_log(Path("evaluation/results/queries/tier-1-2026-05-07T10_59_10Z.json"))
filtered = src_log.model_copy(update={
    "records": [r for r in src_log.records if r.question_id in WANTED_IDS]
})
secondary_scores, usage = await score_query_log(
    filtered, qa_index, secondary_llm, secondary_emb, batch_size=10,
)
# secondary_scores is list[ScoreRecord]; usage carries token counts for cost tracker
```

### Pattern 3: Read primary metrics — pin to the SAME timestamp as the source capture
**What:** For each tier, read `metrics/tier-{N}-{sweep_ts}.json` at the EXACT timestamp that matches `queries/tier-{N}-{sweep_ts}.json`. Do NOT use `_latest()` — that resolves by mtime and could pick a stale/newer file.
**When to use:** Always when computing the delta. Pin via the source `QueryLog.timestamp` field.
**Example:**
```python
# Pinning to exact sweep timestamp — different from compare.py _latest() pattern
from evaluation.harness.score import _ts_for_filename

primary_metrics_path = (
    Path("evaluation/results/metrics") /
    f"tier-{tier}-{_ts_for_filename(src_log.timestamp)}.json"
)
primary_metrics = json.loads(primary_metrics_path.read_text())
primary_by_qid = {m["question_id"]: m for m in primary_metrics}
```

### Pattern 4: Signed delta with None-propagation
**What:** `delta = secondary − primary` per metric. If either is `None`, delta is `None`. Tag the cell with the secondary's `nan_reason` if secondary went NaN.
**When to use:** Always. Aggregating mean-delta over None values is silently wrong; `None` makes the gap explicit in the frozen doc.
**Example:**
```python
def _signed_delta(secondary: float | None, primary: float | None) -> float | None:
    if secondary is None or primary is None:
        return None
    return secondary - primary
```

### Pattern 5: Custom cost-tracker persist with explicit `dest_dir`
**What:** Avoid the pre-existing `tracker.persist()` mis-feature (run.py:302, score.py:518 call `persist()` with no `dest_dir`, ignoring `--results-dir`). The spot-check must call `tracker.persist(dest_dir=Path(args.results_dir) / "costs")` explicitly so the cost JSON lands beside the metrics JSON.
**When to use:** Always when writing a new cost ledger.
**Example:**
```python
# Source: shared/cost_tracker.py:120 (signature already supports dest_dir)
from shared.cost_tracker import CostTracker
tracker = CostTracker("multi-judge-spotcheck")
tracker.record_llm("anthropic/claude-haiku-4.5",
                   usage["input_tokens"], usage["output_tokens"])
tracker.persist(dest_dir=Path(args.results_dir) / "costs")  # NOT persist()
```

### Anti-Patterns to Avoid
- **Re-capturing the 5 questions:** Would burn ~$0.30 capture spend, take 30s+ wall, AND change the git SHA. The whole point of "spot-check" is judge-side methodology delta, not retrieval delta. Use `read_query_log` to load the existing Phase 7 capture.
- **Adding `--judge` flag to `score.py`:** Already exists as `--judge-model`. Don't shadow it. The new tool's CLI flag should be `--judge` (different name, different module) OR alias `--judge-model` to match score.py's vocabulary.
- **Computing the delta with `numpy.nanmean`:** Mean of (None − valid) silently drops cells. Use explicit `None`-propagation; aggregate per-tier-class with explicit "n_skipped_due_to_nan" alongside the mean.
- **Reading primary metrics with `_latest()`:** Returns the most-recent-by-mtime, which could be a different SHA. Use the exact `_ts_for_filename(src_log.timestamp)` filename match.
- **Letting `tracker.persist()` write to the default repo-relative path:** Will silently land in `evaluation/results/costs/` regardless of `--results-dir`. Always pass `dest_dir=` explicitly (Plan 05-02 lesson).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LiteLLM judge wiring | New `litellm.completion(...)` boilerplate | `score._build_judge(slug, emb_slug)` | Already handles `max_tokens=8192` (Plan 02-04), provider="litellm", and 4 RAGAS embedding alias methods (`embed_query`, `embed_documents`, async variants) |
| RAGAS evaluate() driver | New `evaluate(...)` call | `score.score_query_log(...)` | Already wires `NaNReasonTracer` (Phase 3 / HARN-05), `_short_circuit_nan`, `_classify_post_evaluate_nan`, batch_size=10 (Pitfall 3), token_usage_parser, and result→ScoreRecord mapping with both pandas and `result.scores` fallback paths |
| Secondary judge NaN classification | New NaN reason tracer for the secondary judge | Reuse `NaNReasonTracer` via `score_query_log` | The Phase 3 tracer is metric-agnostic and judge-agnostic; it captures `RagasOutputParserException` / `LLMDidNotFinishException` at the LangChain callback level, which fires for any LiteLLM-backed judge |
| Cost tracking | New JSON-writer | `CostTracker("multi-judge-spotcheck").persist(dest_dir=...)` | D-13 schema is already canonical; `freeze.py` may eventually read this file and expect the standard shape |
| Filtering QueryLog by question IDs | New filter pipeline | `Pydantic v2 model_copy(update={"records": [...]})` | One-line filter on the existing list |
| Provenance equivalence with frozen doc | Custom timestamp logic | Pin to `src_log.timestamp` + `src_log.git_sha` (already on `QueryLog` model) | Phase 6 added `embedder` + `embedder_source` to `QueryLog`; capture is fully self-describing |

**Key insight:** This phase is 90% orchestration of helpers already shipped in Phases 1-7. The "new" code is the JSON output schema + question-id filter + delta arithmetic + a 5-line cost recorder. Anything more than ~180 raw LOC is hand-rolling something that exists.

## Runtime State Inventory

> Phase 8 is greenfield (new module, new output files). No rename / refactor / migration. SKIPPING this section per researcher guidance.

## Common Pitfalls

### Pitfall 1: Token-cap NaN redux (Plan 02-04 lesson)
**What goes wrong:** Claude Haiku 4.5's RAGAS faithfulness output is hit by an artificially-low max_tokens cap (RAGAS default = 1024), causing `_create_statements` to truncate atomic-claim lists for long answers (e.g., Tier 4 hybrid-mode answers are 2000+ chars).
**Why it happens:** `ragas.llms.llm_factory(provider="litellm", ...)` defaults to 1024 unless `max_tokens` is explicitly passed.
**How to avoid:** `score._build_judge` already passes `max_tokens=JUDGE_MAX_TOKENS=8192` ([VERIFIED: score.py:138]). The spot-check inherits this for free by reusing `_build_judge`. Claude Haiku 4.5's official output limit is 64K (well above 8192), so no model-side ceiling pressure either. [VERIFIED: pricing.py:42 entry exists; output limit verified in tier-4-multimodal/output paper at line 313 — "claude-haiku-4-5-20251001" used as judge in published spot-check at temperature 0]
**Warning signs:** If any cell shows `nan_reason="empty_statements"` for the secondary judge specifically, suspect a token-cap or model-version drift before assuming a methodology issue.

### Pitfall 2: Cost-ledger drift — `tracker.persist()` ignores `dest_dir` when called bare
**What goes wrong:** `run.py:302` and `score.py:518` call `tracker.persist()` with NO arguments. The default falls through to `shared.cost_tracker.DEFAULT_COSTS_DIR = Path("evaluation/results/costs")` ([VERIFIED: cost_tracker.py:34]) regardless of caller's `--results-dir`. Plan 05-02 hit this and resolved via test-side monkeypatch.
**Why it happens:** Pre-existing v1.1 hardening item; production code under forward-contract lock.
**How to avoid:** The spot-check is a NEW module — there is no forward-contract lock yet. **Always call `tracker.persist(dest_dir=Path(args.results_dir) / "costs")` explicitly.** Do NOT call bare `persist()`. This sidesteps the mis-feature without modifying locked files. (Recommend the planner: do not "fix" the mis-feature in run.py/score.py as part of Phase 8 — that breaks the forward-contract; record as v1.1 still.)
**Warning signs:** If the cost JSON appears at `evaluation/results/costs/` instead of the user's `--results-dir/costs/`, you forgot the explicit `dest_dir=`.

### Pitfall 3: SHA propagation — recording a re-score-time SHA instead of the SOURCE SHA
**What goes wrong:** The spot-check JSON records `_git_sha()` at re-score time. If HEAD has advanced since Phase 7's sweep_sha=`75f6f1b`, the multi-judge delta cites a different SHA than the primary numbers. The frozen doc's family-bias disclaimer would then say "delta measured at SHA X" but the primary numbers are at SHA Y → reproducibility lie.
**Why it happens:** Default behavior of helpers like `run._git_sha()` — they always return current HEAD.
**How to avoid:** The spot-check JSON's `provenance` block MUST carry **two** SHA fields: `source_capture_git_sha` (= `src_log.git_sha`, the value Phase 7 wrote into the QueryLog at sweep time) and `spotcheck_run_git_sha` (= `_git_sha()` at re-score time). The frozen doc cites `source_capture_git_sha` for traceability. If they differ, log a `[yellow]` warning at the top of the run.
**Warning signs:** The two SHAs disagree. Acceptable (HEAD advanced post-sweep), but the disclaimer must clearly cite the source SHA.

### Pitfall 4: Question-id subset drift between tiers
**What goes wrong:** The 3 tiers have 30 questions each, but if you pick different IDs per tier (e.g., q-1..5 for tier-1 but q-3..7 for tier-4), the delta is meaningless — you're comparing two judges on different question populations.
**Why it happens:** Naive ID-by-index selection or per-tier filtering bugs.
**How to avoid:** Hardcode one canonical subset of 5 IDs as a module-level `WANTED_IDS = [...]` constant. Apply the SAME 5-id filter to all 3 tier QueryLogs before any judge call. Assert in code: `len([r for r in filtered.records if r.question_id in WANTED_IDS]) == 5` per tier.
**Warning signs:** Different `n_cells_per_tier` counts in the output JSON's per-tier rollup.

### Pitfall 5: NaN handling on the secondary judge
**What goes wrong:** Claude Haiku 4.5 may produce its own NaNs (`empty_statements`, `json_parse_failure`, etc.) that Gemini 2.5 Flash didn't. If the spot-check JSON treats these as "0" or "drops" them, the delta is undefined for those cells but reported as concrete.
**Why it happens:** Secondary judge has different prompt-following behavior than primary; statement-extraction or verdict-parsing may fail differently.
**How to avoid:** Inherit Phase 3's `NaNReasonTracer` automatically (via `score_query_log`). Each spot-check cell carries a `secondary.nan_reason` field; if non-None, the corresponding `delta.{metric}` is `None` and the cell contributes a "n_skipped" count rather than a delta sum.
**Warning signs:** Aggregate delta dominated by ±1.0 magnitudes (suggests mean over (NaN, valid) pairs miscoded as zero).

### Pitfall 6: Forward-contract leak
**What goes wrong:** Editing `score.py` to add a `--questions` filter or a "spot-check mode" breaks the byte-identical guard that has held across Phases 4/5/6/7 for 6 modules (pipeline, run, score, compare, freeze, smoke_gate).
**Why it happens:** Tempting low-LOC path; "I'll just add one flag."
**How to avoid:** New code goes in a NEW module. The plan must include a forward-contract verification step: `git diff HEAD -- evaluation/harness/{pipeline,run,score,compare,freeze,smoke_gate,records}.py | wc -c` returns 0 post-Phase-8.
**Warning signs:** Any patch hunk touching the 6 RAW-LOCKed modules.

### Pitfall 7: Non-deterministic question ordering
**What goes wrong:** Pydantic's filter or list comprehension preserves insertion order, but the source QueryLog's record list order may not match the WANTED_IDS order. Output JSON cells in inconsistent order across tiers.
**Why it happens:** `[r for r in src_log.records if r.question_id in WANTED_IDS]` filters by membership, preserving source order, not WANTED_IDS order.
**How to avoid:** Either iterate `WANTED_IDS` and build a dict-lookup (`{r.question_id: r for r in src_log.records}`) → guaranteed deterministic, OR sort the output JSON's cells by `(question_id, tier)` lexicographically before write.
**Warning signs:** Diffing two re-runs of the spot-check at the same SHA produces noise in the cell ordering.

## Code Examples

### Bootstrap pattern — `multi_judge_spotcheck.py` skeleton
```python
"""Phase 8 — multi-judge spot-check (CAP-02).

Re-scores 5 questions × 3 tiers from the Phase 7 sweep with a non-Gemini judge
and writes a structured delta JSON for the Phase 9 frozen doc.
"""
from __future__ import annotations

import argparse, asyncio, json, sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console

from evaluation.harness.records import QueryLog, ScoreRecord, read_query_log
from evaluation.harness.run import _git_sha, _ts, _ts_for_filename, _load_golden_qa
from evaluation.harness.score import (
    JUDGE_EMB_SLUG_DEFAULT,
    JUDGE_LLM_SLUG_DEFAULT,
    _build_judge,
    _strip_openrouter_prefix,
    score_query_log,
)
from shared.cost_tracker import CostTracker

# Hardcoded canonical subset (Pitfall 4 + 7 prevention)
WANTED_IDS = (
    "single-hop-001",
    "single-hop-002",
    "multi-hop-001",
    "multi-hop-002",
    "multimodal-001",
)

# 3-tier subset spanning architecture families: vector / multimodal-graph / agentic
DEFAULT_TIERS = (1, 4, 5)

# Default secondary judge (verified slug; pricing.py:42)
SECONDARY_JUDGE_DEFAULT = "openrouter/anthropic/claude-haiku-4.5"


def _signed_delta(secondary: float | None, primary: float | None) -> float | None:
    if secondary is None or primary is None:
        return None
    return secondary - primary


def _filter_records(log: QueryLog, ids: tuple[str, ...]) -> QueryLog:
    by_id = {r.question_id: r for r in log.records}
    return log.model_copy(update={"records": [by_id[i] for i in ids if i in by_id]})


def _read_primary_metrics(metrics_dir: Path, tier: int, ts: str) -> dict[str, dict]:
    """Pin to exact ts (NOT _latest); see Pattern 3 in RESEARCH."""
    path = metrics_dir / f"tier-{tier}-{_ts_for_filename(ts)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Primary metrics missing for tier-{tier}@{ts}: {path}")
    return {m["question_id"]: m for m in json.loads(path.read_text())}


async def amain(args, console: Console) -> int:
    sweep_sha = _git_sha()  # current HEAD; may differ from source_capture_git_sha
    sweep_ts = _ts()
    queries_dir = Path(args.results_dir) / "queries"
    metrics_dir = Path(args.results_dir) / "metrics"

    # Build secondary judge ONCE (re-used across all 3 tiers)
    secondary_llm, secondary_emb = _build_judge(args.judge, args.judge_emb)
    qa_index = {q["id"]: q for q in _load_golden_qa()}

    cost_tracker = CostTracker("multi-judge-spotcheck")
    cells: list[dict] = []

    for tier in args.tiers:
        # 1. Load Phase 7 source capture (pin to ts=2026-05-07T10:59:10Z)
        src_path = queries_dir / f"tier-{tier}-{_ts_for_filename(args.source_ts)}.json"
        src_log = read_query_log(src_path)
        filtered = _filter_records(src_log, WANTED_IDS)
        if len(filtered.records) != len(WANTED_IDS):
            console.print(f"[red]Tier {tier}: missing IDs in source capture; abort.[/red]")
            return 2

        # 2. Re-score with secondary judge
        secondary_scores, usage = await score_query_log(
            filtered, qa_index, secondary_llm, secondary_emb, batch_size=10,
        )

        # 3. Read primary metrics at SAME ts (Pattern 3 — NOT _latest)
        primary_by_qid = _read_primary_metrics(metrics_dir, tier, src_log.timestamp)

        # 4. Compute deltas + assemble cells
        for sec in secondary_scores:
            pri = primary_by_qid.get(sec.question_id, {})
            cells.append({
                "question_id": sec.question_id,
                "tier": f"tier-{tier}",
                "primary": {
                    "faithfulness": pri.get("faithfulness"),
                    "answer_relevancy": pri.get("answer_relevancy"),
                    "context_precision": pri.get("context_precision"),
                    "nan_reason": pri.get("nan_reason"),
                },
                "secondary": {
                    "faithfulness": sec.faithfulness,
                    "answer_relevancy": sec.answer_relevancy,
                    "context_precision": sec.context_precision,
                    "nan_reason": sec.nan_reason,
                },
                "delta": {
                    "faithfulness": _signed_delta(sec.faithfulness, pri.get("faithfulness")),
                    "answer_relevancy": _signed_delta(sec.answer_relevancy, pri.get("answer_relevancy")),
                    "context_precision": _signed_delta(sec.context_precision, pri.get("context_precision")),
                },
            })

        # 5. Cost recording (Pitfall 2 — explicit dest_dir)
        pricing_key = _strip_openrouter_prefix(args.judge)
        try:
            cost_tracker.record_llm(
                pricing_key, usage["input_tokens"], usage["output_tokens"],
            )
        except KeyError:
            console.print(f"[yellow]No PRICES entry for {pricing_key}; cost USD = 0.[/yellow]")

    # 6. Write spot-check JSON (with both SHAs per Pitfall 3)
    out_path = metrics_dir / f"multi-judge-spotcheck-{_ts_for_filename(sweep_ts)}.json"
    payload = {
        "$schema_version": "1.0",
        "spotcheck_run_timestamp": sweep_ts,
        "spotcheck_run_git_sha": sweep_sha,
        "source_capture_timestamp": args.source_ts,
        "source_capture_git_sha": _read_source_sha(queries_dir, args.tiers, args.source_ts),
        "primary_judge": {
            "model": "google/gemini-2.5-flash",  # constant per Phase 7
            "embedder": "openai/text-embedding-3-small",
        },
        "secondary_judge": {
            "model": _strip_openrouter_prefix(args.judge),
            "model_slug": args.judge,  # full LiteLLM slug for litellm reproducibility
            "embedder": _strip_openrouter_prefix(args.judge_emb),
            "max_tokens": 8192,  # JUDGE_MAX_TOKENS — Plan 02-04 lesson preserved
        },
        "cells": cells,
        "aggregate": _per_tier_class_aggregate(cells),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # 7. Persist cost JSON with EXPLICIT dest_dir (Pitfall 2)
    cost_path = cost_tracker.persist(dest_dir=Path(args.results_dir) / "costs")
    console.print(f"[green]Spotcheck → {out_path}[/green]")
    console.print(f"[green]Cost     → {cost_path} (${cost_tracker.total_usd():.6f})[/green]")
    return 0
```

### CLI shape
```bash
# Default invocation — uses Phase 7 sweep at sweep_ts=2026-05-07T10:59:10Z
python -m evaluation.harness.multi_judge_spotcheck \
    --source-ts 2026-05-07T10:59:10Z \
    --tiers 1,4,5 \
    --yes

# Override the secondary judge slug
python -m evaluation.harness.multi_judge_spotcheck \
    --source-ts 2026-05-07T10:59:10Z \
    --judge openrouter/anthropic/claude-haiku-4.5 \
    --yes
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Re-capture all 5 questions × 3 tiers fresh with a different generation model | Re-judge Phase 7's existing capture with a different judge model | Phase 8 design (this doc) | Saves ~$0.30 capture cost; preserves Phase 7 SHA citation; isolates judge-bias from retrieval-bias |
| Hand-roll cost-recorder (custom JSON) | `CostTracker("multi-judge-spotcheck").persist(dest_dir=...)` | Standard since Plan 04-01 (D-13 schema) | Frozen doc consumes one canonical schema |
| Embed multi-judge logic in `score.py` via flags | Separate `multi_judge_spotcheck.py` module | Forward-contract trail Phases 4-7 | Preserves byte-identical guard; isolates spot-check semantics from primary score path |

**Deprecated/outdated:**
- The "use ragas.cost.get_token_usage_for_openai for non-OpenAI models" pattern: it works for Anthropic/Google/OpenRouter through LiteLLM because LiteLLM normalizes the OpenAI usage shape; verified in `score.py:312, 340`. No change needed.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The 3 chosen tiers should be 1, 4, 5 (one per architecture family — vector / graph-multimodal / agentic) | Open Q1 below | LOW — defensible per blog narrative; user may prefer 3, 4, 5 (graph / multimodal / agentic) or 1, 3, 5 (vector / graph / agentic). Discuss in `/gsd-discuss-phase`. |
| A2 | The 5 chosen question IDs should be `single-hop-001`, `single-hop-002`, `multi-hop-001`, `multi-hop-002`, `multimodal-001` (2 single-hop, 2 multi-hop, 1 multimodal — proportional to the 30Q split) | Pattern 1 + Code Examples | LOW — any defensible 5 work. The set must be hardcoded for reproducibility (Pitfall 4 + 7) but the choice itself is judgment. Surface in `/gsd-discuss-phase`. |
| A3 | Claude Haiku 4.5's max_tokens limit is well above 8192 | Pitfall 1 | LOW — [VERIFIED via repo: tier-4-multimodal output paper at line 313 cites `claude-haiku-4-5-20251001` as a judge in published research at temperature 0]. Anthropic's official Claude Haiku 4.5 output limit is 64K tokens; if Claude 4.5 specifically caps lower, NaN-from-token-cap is the canary. |
| A4 | Phase 9's freeze tool's manifest does NOT need a new `multi_judge_spotcheck` block — the frozen markdown will just embed the JSON path inline as text | Open Q4 / Phase 9 contract | MEDIUM — depends on Phase 9 design choices not yet locked. Phase 8 should write the JSON to a stable location (`evaluation/results/metrics/multi-judge-spotcheck-{TS}.json`); whether `freeze.py` needs to know about it is Phase 9's call. Flag for Phase 9 RESEARCH. |
| A5 | The $0.10–0.30 budget includes BOTH input+output tokens for the secondary judge across 15 cells × ~3 RAGAS internal calls per cell × 3 metrics | Cost Budget below | LOW — projected ~$0.045 from Claude Haiku 4.5 token rates and primary-judge token-usage analogs. Even 5× over-projection is still inside budget. |
| A6 | The recorded judge cost from `score_query_log`'s `usage` dict will be non-zero (unlike the Phase 7 ragas-judge cost ledgers which all show $0) | Cost Budget below | MEDIUM — [VERIFIED: 25/25 Phase 7 ragas-judge-tier-*.json files all show input_tokens=0, output_tokens=0, usd=0.0]. This is the v1.1 "judge-cost-ledger underreporting" mis-feature. The spot-check should NOT silently inherit this; the planner should add a fallback estimator (token count × `shared.pricing.PRICES[slug]`) when `usage["input_tokens"] == 0` but `usage["n_scored"] > 0`. Discuss in `/gsd-discuss-phase`. |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.

## Open Questions

1. **Which 3 tiers? (CAP-02 says "3 tiers" — ambiguous which 3)**
   - What we know: Phase 7 has 5 tiers; the requirement says 3.
   - What's unclear: ROADMAP doesn't specify. Possible reads: (a) tiers 1, 4, 5 — one per family (vector / graph-multimodal / agentic); (b) tiers 3, 4, 5 — non-baseline architectures; (c) tiers 1, 3, 5 — vector / graph / agentic; (d) random.
   - Recommendation: Default to **(1, 4, 5)** — covers the 3 architectural families (vector retrieval / graph-multimodal RAG / agentic-tool-calling). Each is qualitatively different so judge-family-bias variance is most informative there. Surface to user in `/gsd-discuss-phase`. [ASSUMED — A1]

2. **Which 5 question IDs?**
   - What we know: 30 total IDs; class split is 10 single-hop / 10 multi-hop / 10 multimodal (`hop_count_tag` × `modality_tag` cross-product).
   - What's unclear: Should the 5 spot-check IDs proportionally span all 3 classes, or focus on one class where bias is most likely?
   - Recommendation: Default to **`single-hop-001, single-hop-002, multi-hop-001, multi-hop-002, multimodal-001`** (2/2/1 — roughly proportional, deterministic, low-numbered for human readability). Surface to user in `/gsd-discuss-phase`. [ASSUMED — A2]

3. **Claude Haiku 4.5 vs GPT-4.1-mini — which secondary judge?**
   - What we know: Both are mentioned in ROADMAP as valid options. `anthropic/claude-haiku-4.5` is in `shared/pricing.py:42` with verified rates ($1.00 input / $5.00 output per 1M tokens). `openai/gpt-4.1-mini` is NOT in PRICES and would require a pricing-table edit. `score.py`'s docstring example uses `openrouter/anthropic/claude-haiku-4.5` already.
   - What's unclear: User preference; whether GPT-4.1-mini's family-distance from Gemini is meaningfully greater than Claude's (both are "non-Google"; Claude is more architecturally-different from Gemini's MoE-Flash than GPT-4.1-mini's GPT-4 lineage).
   - Recommendation: Default to **`openrouter/anthropic/claude-haiku-4.5`** — zero new code beyond a string constant; `pricing.py` already keyed; existing doc-string example. If user prefers GPT-4.1-mini, planner adds 1 line to `pricing.py` (which IS a forward-contract module — flag carefully). [VERIFIED]

4. **Does Phase 9's freeze tool's manifest need a `multi_judge_spotcheck` block, or does the frozen markdown just embed the path inline?**
   - What we know: Phase 4 manifest shape is locked; freeze.py is forward-contract from Phase 5/6/7. Phase 9's plan doesn't exist yet.
   - What's unclear: Will Phase 9's `freeze.py` change to read the spot-check JSON, or will the markdown be hand-edited to include the family-bias section before freezing?
   - Recommendation: Phase 8 writes to a stable, predictable path (`evaluation/results/metrics/multi-judge-spotcheck-{TS}.json` + `evaluation/results/costs/multi-judge-spotcheck-{TS}.json`). Phase 9 decides whether `freeze.py` needs an additive `multi_judge_spotcheck` manifest block. **Flag in 08-RESEARCH.md "Forward-Contract Surface for Phase 9": the spot-check JSON path/schema MUST not change between Phase 8 ship and Phase 9 freeze.** [ASSUMED — A4]

5. **What CLI flag for filtering questions — `--questions` or `--question-ids`?**
   - What we know: `run.py` already exposes `--smoke-question-ids` (with hyphen plural). `score.py` does NOT have a question filter flag.
   - What's unclear: Naming consistency.
   - Recommendation: `--question-ids` to match `run.py`'s convention. But ALSO support a hardcoded `WANTED_IDS` constant as the default — the user typically wants the same canonical 5 across runs.

6. **Should the spot-check produce its own ScoreRecord-compatible JSON next to the multi-judge JSON, so `compare.py` can ingest it as a fourth tier-style row?**
   - What we know: `compare.py` is forward-contract locked. The spot-check is per-cell, not per-tier.
   - What's unclear: Phase 9's frozen-doc table layout.
   - Recommendation: NO. The spot-check output is a separate aggregate ("delta table"), not a parallel tier. Keep concerns separated; let Phase 9's frozen-doc generator render the delta table from the spot-check JSON directly.

## Cost Budget

**Estimate (per Pitfall 1 / Pattern 1 token analysis):**

For each of 15 cells, RAGAS runs:
- `faithfulness._create_statements` (1 call, ~500 input / ~500 output)
- `faithfulness._verify_statements` (1 call, ~600 input / ~200 output)
- `answer_relevancy._create_questions` (1 call, ~400 input / ~300 output)
- `context_precision._verify_per_context` (~3 calls per question, ~300 input / ~50 output each)

Total per cell: ~2400 input + ~1150 output tokens (rough; actual varies wildly by Tier 4 multimodal which has long contexts).

For 15 cells: ~36K input + ~17K output = ~53K total judge tokens.

**Claude Haiku 4.5 pricing** ([VERIFIED: `shared/pricing.py:42`]): $1.00 / 1M input + $5.00 / 1M output.

**Projected spend:** ($1.00 × 0.036) + ($5.00 × 0.017) = $0.036 + $0.085 = **~$0.12** for 15 cells.

This is **inside the $0.10–0.30 ROADMAP budget** with margin. Even a 2× over-projection ($0.24) stays within bounds. The planner should set the cost-surprise gate ceiling at $0.30 (matches ROADMAP) and HARD ceiling at $0.50 (1.6× margin).

**Caveat (A6):** Phase 7's ragas-judge cost ledgers all log `usd=0.0` because of a v1.1 mis-feature in how `usage` is parsed back from RAGAS for non-OpenAI providers. The spot-check planner should add a fallback estimator: when `score_query_log` returns `usage["input_tokens"] == 0 and n_scored > 0`, estimate post-hoc from a fixed token-per-cell heuristic and tag the cost JSON with `"estimated": true`. Otherwise the cost ledger will misleadingly show $0.

## Forward-Contract Surface

Following the Phase 4/5/6/7 pattern, this phase MUST keep the following byte-identical:

| File | Reason |
|------|--------|
| `evaluation/harness/score.py` | Forward-contract since Phase 4 / Plan 02-04 |
| `evaluation/harness/run.py` | Forward-contract since Phase 5 / Plan 05-01 |
| `evaluation/harness/pipeline.py` | Forward-contract since Phase 5 / Plan 05-01 |
| `evaluation/harness/freeze.py` | Forward-contract since Phase 4 / Plan 04-01 |
| `evaluation/harness/compare.py` | Forward-contract since Phase 5 / Plan 05-01 |
| `evaluation/harness/smoke_gate.py` | Forward-contract since Phase 1 |
| `evaluation/harness/records.py` | Forward-contract since Phase 6 / Plan 06-01 |
| `shared/cost_tracker.py` | Pre-existing v1.1 mis-feature explicitly NOT fixed in Phase 8 |
| `shared/pricing.py` | UNLESS user picks GPT-4.1-mini (would require new pricing entry) |

**New files (no forward-contract yet):**
- `evaluation/harness/multi_judge_spotcheck.py`
- `evaluation/tests/test_eval_multi_judge_spotcheck.py`

**Verification command** for the plan:
```bash
git diff HEAD -- evaluation/harness/{pipeline,run,score,compare,freeze,smoke_gate,records}.py \
    shared/cost_tracker.py shared/pricing.py | wc -c
# Expected: 0 (UNLESS the user picks GPT-4.1-mini and pricing.py is updated additively)
```

## Output JSON Schema

The spot-check JSON at `evaluation/results/metrics/multi-judge-spotcheck-{TS}.json`:

```json
{
  "$schema_version": "1.0",
  "spotcheck_run_timestamp": "2026-05-07T13:00:00Z",
  "spotcheck_run_git_sha": "75f6f1b",
  "source_capture_timestamp": "2026-05-07T10:59:10Z",
  "source_capture_git_sha": "75f6f1b",
  "primary_judge": {
    "model": "google/gemini-2.5-flash",
    "embedder": "openai/text-embedding-3-small"
  },
  "secondary_judge": {
    "model": "anthropic/claude-haiku-4.5",
    "model_slug": "openrouter/anthropic/claude-haiku-4.5",
    "embedder": "openai/text-embedding-3-small",
    "max_tokens": 8192
  },
  "cells": [
    {
      "question_id": "single-hop-001",
      "tier": "tier-1",
      "primary": {
        "faithfulness": 0.8235,
        "answer_relevancy": 0.9879,
        "context_precision": 0.3333,
        "nan_reason": null
      },
      "secondary": {
        "faithfulness": 0.7500,
        "answer_relevancy": 0.9912,
        "context_precision": 0.5000,
        "nan_reason": null
      },
      "delta": {
        "faithfulness": -0.0735,
        "answer_relevancy": 0.0033,
        "context_precision": 0.1667
      }
    }
    // ... 14 more cells
  ],
  "aggregate": {
    "by_tier": {
      "tier-1": {
        "mean_delta_faithfulness": -0.05,
        "mean_delta_answer_relevancy": 0.01,
        "mean_delta_context_precision": 0.10,
        "n_cells": 5,
        "n_skipped_due_to_nan": 0
      },
      "tier-4": { /* ... */ },
      "tier-5": { /* ... */ }
    },
    "overall": {
      "mean_delta_faithfulness": -0.04,
      "mean_delta_answer_relevancy": 0.02,
      "mean_delta_context_precision": 0.08,
      "n_cells": 15,
      "n_skipped_due_to_nan": 0
    }
  }
}
```

The cost JSON at `evaluation/results/costs/multi-judge-spotcheck-{TS}.json` follows the standard D-13 `CostTracker` schema (see `shared/cost_tracker.py:7-22`).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing; see `evaluation/tests/conftest.py`) |
| Config file | `pyproject.toml` (no separate `pytest.ini`) |
| Quick run command | `uv run pytest evaluation/tests/test_eval_multi_judge_spotcheck.py -x` |
| Full suite command | `uv run pytest -m 'not live' evaluation/tests/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CAP-02 | CLI runs and writes spot-check JSON at expected path | unit | `pytest evaluation/tests/test_eval_multi_judge_spotcheck.py::test_writes_spotcheck_json -x` | ❌ Wave 0 |
| CAP-02 | Per-cell schema: primary + secondary + delta keys present for all 3 metrics | unit | `pytest evaluation/tests/test_eval_multi_judge_spotcheck.py::test_cell_schema -x` | ❌ Wave 0 |
| CAP-02 | Signed delta arithmetic: secondary − primary; None propagates | unit | `pytest evaluation/tests/test_eval_multi_judge_spotcheck.py::test_signed_delta_with_none -x` | ❌ Wave 0 |
| CAP-02 | Source capture SHA matches Phase 7's `75f6f1b` (pinned to ts) | unit | `pytest evaluation/tests/test_eval_multi_judge_spotcheck.py::test_source_sha_pinned -x` | ❌ Wave 0 |
| CAP-02 | Secondary judge slug + model + embedder + max_tokens recorded | unit | `pytest evaluation/tests/test_eval_multi_judge_spotcheck.py::test_secondary_provenance -x` | ❌ Wave 0 |
| CAP-02 | Cost JSON written at `costs/multi-judge-spotcheck-{TS}.json` (NOT default DEFAULT_COSTS_DIR) | unit | `pytest evaluation/tests/test_eval_multi_judge_spotcheck.py::test_cost_dir_isolation -x` | ❌ Wave 0 |
| CAP-02 | Question subset hardcoded; 5 IDs × 3 tiers = 15 cells | unit | `pytest evaluation/tests/test_eval_multi_judge_spotcheck.py::test_15_cells -x` | ❌ Wave 0 |
| CAP-02 | Forward-contract: 6 harness modules + cost_tracker.py + pricing.py byte-identical | quality gate | `git diff HEAD -- evaluation/harness/{pipeline,run,score,compare,freeze,smoke_gate,records}.py shared/cost_tracker.py shared/pricing.py \| wc -c` returns 0 | quality gate, not pytest |
| CAP-02 | Live spend ≤ $0.30 against real OpenRouter (Claude Haiku 4.5) | live | `pytest -m live evaluation/tests/test_eval_multi_judge_spotcheck.py::test_live_spotcheck_under_budget -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest evaluation/tests/test_eval_multi_judge_spotcheck.py -x` (offline only)
- **Per wave merge:** `uv run pytest -m 'not live' evaluation/tests/ -x` (full offline suite, ~116 tests)
- **Phase gate:** Full offline suite green + 1 live run via `-m live` checkpoint with cost-ack ≤ $0.30

### Wave 0 Gaps
- [ ] `evaluation/tests/test_eval_multi_judge_spotcheck.py` — covers CAP-02 unit + integration + 1 live test
- [ ] No new fixtures needed; existing `conftest.py` provides `live_eval_keys_ok` (used by Plan 05-02 live test) and `tmp_path`

## Project Constraints (no CLAUDE.md present)

No `CLAUDE.md` exists in the working directory. Project constraints are derived from `.planning/STATE.md`, `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`:

1. **v1.0 scope: fix-and-ship the existing 5 tiers; no new architectures, no new questions.** [VERIFIED: PROJECT.md "Out of Scope"]
2. **Multi-judge spot-check (CAP-02) is in v1.0 scope as MEASUREMENT (5q × 3 tiers); full mitigation deferred to v1.1 (METH-02).** [VERIFIED: STATE.md line 80]
3. **Phase 7 sweep_sha=`75f6f1b` is the version-of-record for primary numbers.** [VERIFIED: STATE.md current focus + ROADMAP Phase 7 closure note]
4. **DO NOT rebuild Tier 4 between this point and Phase 9 freeze (Pitfall 3 — graphml drift).** [VERIFIED: STATE.md Phase 9 reminder; tier-4 graph manifest locked at `eb77cf8`]
5. **Forward-contract: every recent phase has hard-locked source files byte-identical pre/post merge** across `score.py / pipeline.py / freeze.py / compare.py / smoke_gate.py / run.py / records.py`. [VERIFIED: forward-contract trail in 02-04 / 03-02 / 04-01 / 05-01 / 05-02 / 06-01 / 07-01 / 07-02 / 07-03 SUMMARYs]
6. **Cost discipline:** ~$0.20–0.50 per 30-question RAGAS run; full sweep ~$1–3; Phase 8 spot-check budget $0.10–0.30. [VERIFIED: PROJECT.md Constraints]
7. **Providers:** OpenRouter for Tier 1+ embeddings/chat; Gemini for legacy `shared.llm`. **No new providers for v1.0.** [VERIFIED: PROJECT.md Constraints]
8. **Reproducibility:** every captured run records timestamp + git SHA + judge model + generation model. [VERIFIED: PROJECT.md Constraints]

## Sources

### Primary (HIGH confidence)
- `evaluation/harness/score.py` — `_build_judge`, `score_query_log`, `JUDGE_MAX_TOKENS=8192`, `JUDGE_LLM_SLUG_DEFAULT`, `JUDGE_EMB_SLUG_DEFAULT`, `_strip_openrouter_prefix`, `NaNReasonTracer`, `_classify_post_evaluate_nan` (read line-by-line)
- `evaluation/harness/records.py` — `QueryLog` (with `embedder` + `embedder_source` from Phase 6), `EvalRecord`, `ScoreRecord`, `read_query_log` (line-by-line)
- `evaluation/harness/run.py` — `_git_sha`, `_ts`, `_ts_for_filename`, `_load_golden_qa`, `COST_PER_Q`, `_capture_tier`, `amain`, `git_sha_override` / `ts_override` kwarg pattern (read offset 60-360)
- `evaluation/harness/compare.py` — `_latest`, `aggregate_tier`, `_detect_judge_provenance`, `SUPPORTED_TIERS` (full read)
- `evaluation/harness/pipeline.py` — orchestration pattern, in-process `run.amain` / `score.amain` / `compare._run` / `freeze.freeze` calls (full read)
- `evaluation/harness/freeze.py` — manifest schema, `judge` block (model, embedder, max_tokens), `library_versions`, `per_tier` schema (full read)
- `shared/cost_tracker.py` — `CostTracker.persist(dest_dir=...)` signature, `DEFAULT_COSTS_DIR`, D-13 schema (full read)
- `shared/pricing.py` — `anthropic/claude-haiku-4.5` entry verified (full read)
- `evaluation/golden_qa.json` — 30 questions, ID format, `hop_count_tag` / `modality_tag` / `source_papers` schema (introspected via python script)
- `evaluation/results/queries/tier-1-2026-05-07T10_59_10Z.json` — Phase 7 source capture: tier=tier-1, sweep_sha=`75f6f1b`, sweep_ts=`2026-05-07T10:59:10Z`, embedder + embedder_source populated (introspected)
- `evaluation/results/metrics/tier-1-2026-05-07T10_59_10Z.json` — Phase 7 primary metrics: `{question_id, faithfulness, answer_relevancy, context_precision, nan_reason}` schema (introspected)
- `pyproject.toml` + `uv.lock` — `litellm==1.83.0`, `ragas==0.4.3` (verified via grep of uv.lock)
- `.planning/ROADMAP.md` Phase 8 description + Phase 9 description (full read)
- `.planning/REQUIREMENTS.md` CAP-02 + traceability table (full read)
- `.planning/STATE.md` Phase 7 closure context + Phase 8 readiness note (head 180 + grep)
- `.planning/PROJECT.md` (full read)
- `.planning/phases/05-pipeline-driver/05-02-SUMMARY.md` lines 22, 35, 40, 171, 215 — `tracker.persist()` mis-feature documented (grepped + read)
- `.planning/phases/02-tier-4-graphml-regeneration/02-04-PLAN.md` — `JUDGE_MAX_TOKENS=8192` lesson (grepped)
- `.planning/phases/07-full-5-tier-rerun/07-RESEARCH.md` line 78-79 — confirms Phase 7 does NOT freeze; Phase 8 produces multi-judge data (grepped)
- `.planning/phases/07-full-5-tier-rerun/07-02-SUMMARY.md` + `07-01-SUMMARY.md` — forward-contract pattern verbatim (grepped)
- `tier-4-multimodal/output/.../2604.20158_stateless_decision_memory_for_enterprise.md` line 313 — independent verification that `claude-haiku-4-5-20251001` is a published RAGAS-style judge model (grepped)

### Secondary (MEDIUM confidence)
- `tier-1-naive/main.py:229`, `tier-1-naive/chat.py:5,89`, `tier-1-naive/README.md:7,20,50` — Claude Haiku 4.5 already used as a chat model in this repo (grepped)
- `evaluation/README.md:66` — `--judge-model openrouter/anthropic/claude-haiku-4.5` already documented as user-facing pattern (grepped)

### Tertiary (LOW confidence)
- Token-per-cell projection in Cost Budget section: extrapolated from RAGAS source-code reading + Phase 7 cost-ledger inspection (since cost ledgers underreport at $0). 2× margin built in.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every library and slug verified against `uv.lock`, `pyproject.toml`, and `shared/pricing.py`. Same RAGAS / LiteLLM / OpenRouter combo as primary judge.
- Architecture: HIGH — pattern is "thin orchestration layer over already-built helpers"; helpers are in production since Phases 1-7 with passing live tests at sweep_sha=`75f6f1b`.
- Pitfalls: HIGH — all 7 pitfalls cite verified source-line evidence (Plan 02-04, Plan 05-02 SUMMARY, score.py line numbers, cost_tracker.py line numbers).
- Cost projection: MEDIUM — token-per-cell estimate uses RAGAS-internal heuristics not directly measured; budget margin is 2.5× to absorb estimate error.
- Open questions: MEDIUM — Q1, Q2, Q3 are user-judgment calls flagged for `/gsd-discuss-phase`. Q4 is a Phase 9 contract that must be resolved before Phase 9 plans land.

**Research date:** 2026-05-07
**Valid until:** 2026-06-07 (30 days; harness is stable, but if Anthropic ships Claude Haiku 5 / OpenAI ships GPT-5-mini before then the secondary judge slug recommendation may shift)

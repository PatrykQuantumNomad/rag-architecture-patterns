# Architecture Research — Eval-Numbers v1.0 Pipeline

**Domain:** Multi-variant RAG eval harness (capture → score → rollup → freeze)
**Researched:** 2026-05-04
**Confidence:** HIGH for the structural pattern (four reference harnesses converge); MEDIUM on the specific freeze-file naming (project-local convention).

## Scope discipline

This file is **integration architecture only** for the v1.0 fix-and-ship work. It does not redesign:

- The five tier CLIs (`tier-{N}-{name}/main.py --ingest|--query`) — already shipped.
- `shared/cost_tracker.py`'s D-13 schema — already correct.
- `evaluation/harness/{run,score,compare}.py` — already implemented (Phase 131).
- `evaluation/harness/adapters/tier_*.py` — already isolate per-tier capture quirks.

What v1.0 needs is the **wrapper that turns three half-coupled CLIs plus a Tier 4 sidecar into a single dated, re-runnable, partial-rerun-safe artifact**. Everything below either points at code that already exists or names a single thin file we add.

## Standard architecture (synthesized from reference harnesses)

Every mature multi-variant LLM eval harness — lm-evaluation-harness (EleutherAI), HELM (Stanford CRFM), Inspect AI (UK AISI), bigcode-evaluation-harness — converges on the **same five-component pipeline**, even though they disagree on syntax:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. CAPTURE       — run model/system on inputs, record outputs  │
│     lm-eval:  --predict_only --log_samples → samples_*.jsonl    │
│     HELM:     helm-run → scenario_state.json                    │
│     Inspect:  eval()    → {timestamp}_{task}_{id}.eval log      │
│     us:       evaluation/harness/run.py → queries/*.json        │
├─────────────────────────────────────────────────────────────────┤
│  2. SCORE         — apply metrics/judges to captured outputs     │
│     lm-eval:  metric pipeline on logged samples                  │
│     HELM:     Metric → per_instance_stats.json                   │
│     Inspect:  scorer → samples[*].score in same log              │
│     us:       evaluation/harness/score.py → metrics/*.json       │
├─────────────────────────────────────────────────────────────────┤
│  3. ROLLUP        — aggregate across questions/instances         │
│     lm-eval:  results.json (per-task aggregate)                  │
│     HELM:     stats.json (per run aggregate)                     │
│     Inspect:  EvalLog.results section                            │
│     us:       evaluation/harness/compare.py (numpy.nanmean)      │
├─────────────────────────────────────────────────────────────────┤
│  4. SUMMARIZE     — combine across variants/models/runs          │
│     lm-eval:   table.md / wandb sweep                            │
│     HELM:      helm-summarize → groups/, runs_to_run_suites.json │
│     Inspect:   inspect_ai.analysis.evals_df()                    │
│     us:        evaluation/harness/compare.py emits comparison.md │
├─────────────────────────────────────────────────────────────────┤
│  5. FREEZE        — write a dated, immutable, committable file   │
│     lm-eval:    user-managed (results dirs by date)              │
│     HELM:       --suite NAME (suite is the freeze handle)        │
│     Inspect:    log_dir per eval-set (durable completion record) │
│     us:         (NEW) eval-numbers-v{x.y}.md committed to repo   │
└─────────────────────────────────────────────────────────────────┘
```

The four reference harnesses also converge on **two architectural splits** that we already half-have and need to lock in:

- **Capture and score are separate phases with separate cost ledgers.** lm-eval's `--predict_only`, HELM's `helm-run` vs `helm-summarize`, Inspect's eval-then-score chain. We already do this (`run.py` writes `queries/`, `score.py` writes `metrics/` + a separate `ragas-judge-tier-{N}` cost JSON). Phase 131 made the right call.
- **Per-instance artifacts are persisted before aggregation.** HELM keeps `per_instance_stats.json` distinct from `stats.json` for exactly the reason we hit on Tier 5: when one question NaNs out, you want to re-score it without re-running the model. Our `metrics/{tier}-{ts}.json` is the per-instance artifact; `comparison.md` is the rollup. Don't fuse them.

### Component responsibilities

| Component | Responsibility | Already exists? |
|-----------|----------------|-----------------|
| **Tier CLI** (`tier-{N}-{name}/main.py`) | Ingest + query for one variant; owns its index/store. | Yes. |
| **Tier eval adapter** (`evaluation/harness/adapters/tier_{N}.py`) | Translate the CLI's per-question call into an `EvalRecord`; bridge cost via shared `CostTracker`. | Yes. |
| **Capture orchestrator** (`evaluation/harness/run.py`) | Loop over (tier × question), produce `queries/tier-{N}-{ts}.json` + `costs/tier-{N}-eval-{ts}.json`. | Yes. |
| **Score orchestrator** (`evaluation/harness/score.py`) | RAGAS judge on captured logs → `metrics/tier-{N}-{ts}.json` + `costs/ragas-judge-tier-{N}-{ts}.json`. NaN short-circuits before judge calls. | Yes. |
| **Rollup + summarize** (`evaluation/harness/compare.py`) | Aggregate latest queries/costs/metrics per tier into `comparison.md`. Pure file I/O + numpy.nanmean. | Yes. |
| **Freeze** (NEW — see Pattern 1) | Wrap `comparison.md` plus a manifest of source-artifact paths into a dated, committed `evaluation/results/eval-numbers-v{x.y}.md`. | **No — to add.** |
| **Pipeline driver** (NEW — see Pattern 2) | Single entry that runs capture → score → compare → freeze for the requested tier set; refuses to clobber a frozen artifact. | **No — to add.** |
| **Tier 4 sidecar** (`tier-4-multimodal/scripts/eval_capture.py`) | User-runs-locally because of MineRU sandbox issues; emits a `queries/tier-4-{ts}.json` that `run.py --tier-4-from-cache` then re-uses. | Yes. |

The two NEW pieces are **thin** — together ≈ 200 lines of Python plus a Makefile target. They reuse the existing component boundaries; they do not replace them.

## Recommended project structure

The existing structure is correct. The v1.0 additions slot in cleanly:

```
evaluation/
├── golden_qa.json                       # 30 hand-authored Q&A (frozen input)
├── harness/
│   ├── run.py                           # CAPTURE     — exists (Phase 131)
│   ├── score.py                         # SCORE       — exists (Phase 131)
│   ├── compare.py                       # ROLLUP+SUM  — exists (Phase 131)
│   ├── records.py                       # Pydantic v2 records (EvalRecord, QueryLog, ScoreRecord)
│   ├── adapters/                        # Per-tier capture adapters
│   │   ├── tier_1.py … tier_5.py
│   │   └── __init__.py
│   ├── pipeline.py                      # NEW — driver: capture→score→compare→freeze
│   └── freeze.py                        # NEW — copies comparison.md → eval-numbers-v{x.y}.md
│                                        #       + writes manifest.json beside it
└── results/
    ├── queries/  tier-{N}-{ts}.json     # CAPTURE outputs   (per-instance, NOT committed)
    ├── costs/    tier-{N}-eval-{ts}.json,
    │             ragas-judge-tier-{N}-{ts}.json
    │                                    # COST LEDGERS      (D-13 schema, NOT committed)
    ├── metrics/  tier-{N}-{ts}.json     # SCORE outputs     (per-instance, NOT committed)
    ├── comparison.md                    # ROLLUP, mutable working copy (committed)
    └── frozen/                          # NEW
        ├── eval-numbers-v1.0.md         # FREEZE artifact (committed, immutable)
        └── eval-numbers-v1.0.manifest.json   # provenance: which queries/costs/metrics paths fed it
```

### Structure rationale

- **`harness/` stays flat** — Phase 131 picked four files that map 1:1 to the canonical capture/score/rollup pipeline. Don't subpackage. The two new files (`pipeline.py`, `freeze.py`) join the same level so `python -m evaluation.harness.pipeline` parallels the existing `python -m evaluation.harness.run`.
- **`results/queries/`, `results/metrics/`, `results/costs/` stay gitignored** — they are reproducible and large. HELM's `benchmark_output/` is also gitignored for the same reason.
- **`results/frozen/` is committed** — this is the eval-portable artifact directory. Treat it like a HELM `--suite`: each version gets a directory-equivalent (we use a name suffix instead because the file count is small). Once written, never edit; supersede with v1.1.
- **`comparison.md` stays committed and mutable** — it is the working "latest" that always reflects the latest unfrozen state, useful in PR diffs. The frozen file is the citation handle.

## Architectural patterns

### Pattern 1: Frozen artifact = copy + manifest (HELM `--suite`, lm-eval results dirs)

**What:** When the user is ready to ship a numbers update, the freeze step does three things:

1. Copy the current `evaluation/results/comparison.md` to `evaluation/results/frozen/eval-numbers-v{x.y}.md`.
2. Capture a manifest beside it that records, for each tier, the **exact source paths** (queries, costs, metrics) and their mtimes/SHAs that fed the rollup.
3. Refuse to overwrite an existing version (force-flag opt-in only).

**When to use:** Always before publishing or citing numbers externally (blog, README, PR title). Internal exploratory runs stay in `comparison.md`.

**Trade-offs:** A copy duplicates content vs symlink, but git-tracking a copy beats git-tracking a symlink across operating systems and lets the file render on GitHub. The manifest is the audit trail; without it the dated md is just a screenshot.

**Example:**

```python
# evaluation/harness/freeze.py
def freeze(version: str, results_dir: Path, force: bool = False) -> Path:
    out = results_dir / "frozen" / f"eval-numbers-v{version}.md"
    if out.exists() and not force:
        raise FileExistsError(f"{out} already frozen — bump version or pass --force")
    src = results_dir / "comparison.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "version": version,
        "frozen_at": _ts(),
        "git_sha": _git_sha(),
        "sources": _collect_sources(results_dir),  # most-recent per tier
    }
    (out.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2))
    return out
```

This mirrors HELM's `--suite NAME` semantics (the suite directory IS the frozen handle) but in our smaller scale we use a suffixed file plus a sidecar manifest because we have one rollup not hundreds.

### Pattern 2: Single-command pipeline driver (Inspect AI `eval_set()`)

**What:** A `pipeline.py` module that calls the existing `run.amain()`, `score.amain()`, and `compare._run()` in sequence for a given tier set, then optionally calls `freeze()`. Each phase is a function call, NOT a subprocess — that way exit codes propagate, asyncio loops are reused, and the cost-surprise prompt fires once per phase.

**When to use:** Default user-facing entry. Manual stage runs stay supported for debugging — the pipeline is a thin orchestrator over them, not a replacement.

**Trade-offs:** Pipeline failures are coarser-grained than per-stage; mitigated by always re-entrant stages (capture → score → compare are each idempotent and consume the latest by mtime, so a failed `--freeze v1.0` is recoverable by re-invoking just freeze).

**Example:**

```bash
# Full sweep, fresh run, ship
python -m evaluation.harness.pipeline --tiers 1,2,3,5 --freeze v1.0 --yes

# Tier 4 takes the cached path (sandbox blocker)
python -m evaluation.harness.pipeline --tiers 4 \
    --tier-4-from-cache evaluation/results/queries/tier-4-*.json --yes

# After Tier 4 lands locally, finalize all five tiers
python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --freeze v1.0 --yes
```

Inspect AI's `eval_set()` plays the same role for them — single call, one durable log directory, automatic stitching of completed work.

### Pattern 3: Most-recent-by-mtime aggregation (existing in compare.py — keep it)

**What:** The rollup picks the most-recent per-tier artifact via `_latest(dir, glob, key=mtime)`. This is the load-bearing primitive that makes partial reruns Just Work.

**When to use:** Always. It is already the pattern in `evaluation/harness/compare.py:_latest()`.

**Trade-offs:** "Latest by mtime" can be wrong if you `cp` an older file in (mtime resets). Acceptable risk because (a) downloads from CI / git checkouts hit this only rarely and (b) the manifest written at freeze time captures the absolute paths used, so audit is possible.

**Why this matters for partial reruns:** If you re-run only Tier 4 capture, you get a new `queries/tier-4-*.json` with newer mtime. `compare.py` automatically picks it up; tiers 1, 2, 3, 5 untouched files still resolve to their last (good) capture. Same for score. **No re-pay for Tier 1's $0.0001 × 30 just because Tier 4 changed.**

### Pattern 4: NaN short-circuit before paying judge cost (existing in score.py — keep it)

**What:** `_short_circuit_nan()` in `score.py` converts known-bad inputs (empty contexts, max-turns-exceeded, tier4_import_error, cached_miss) into `ScoreRecord(nan_reason=...)` **before** building the RAGAS dataset. Judge cost is paid only for records that have at least one retrieved context and no fatal error.

**When to use:** Always. This is the right default for paid-judge harnesses. lm-eval's `--predict_only` plus a separate scorer is a coarser version of the same idea.

**Trade-offs:** Fewer rows scored if upstream capture has bugs (intended — the NaN reasons surface in `comparison.md`'s footer for honest reporting). The Tier 5 `empty_contexts` issue you're investigating is exactly what this pattern is designed to expose, not paper over.

### Pattern 5: Per-tier cost ledger isolation (existing — keep it)

**What:** Each phase × tier pair writes its own D-13 cost JSON: `tier-{N}-eval-{ts}.json` (capture) and `ragas-judge-tier-{N}-{ts}.json` (score). The `CostTracker(name)` collision-avoidance discipline is documented at `run.py:148` and `score.py:358`.

**When to use:** Always. Without per-phase isolation, "how much did Tier 4's judge cost" becomes unanswerable — inspecting one mixed JSON loses the per-tier × per-phase rollup.

**Trade-offs:** More files. Mitigated by the `_latest()` aggregator, which is already glob-aware.

## Data flow — v1.0 pipeline

### Full sweep (every tier, fresh)

```
┌──────────────────────────────────────────────────────────────────────┐
│  python -m evaluation.harness.pipeline --tiers 1,2,3,5 --freeze v1.0 │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─ CAPTURE (asyncio.run boundary) ──────────────────────────────────────┐
│ For each tier in [1,2,3,5]:                                           │
│   adapter loops over 30 golden Q&A                                    │
│   tracker = CostTracker(f"tier-{N}-eval")                             │
│   write queries/tier-{N}-{ts}.json   (per-instance EvalRecords)       │
│   tracker.persist()  → costs/tier-{N}-eval-{ts}.json                  │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─ SCORE (asyncio.run boundary) ────────────────────────────────────────┐
│ For each tier in [1,2,3,5]:                                           │
│   read latest queries/tier-{N}-*.json                                 │
│   NaN-short-circuit empty_contexts / max_turns_exceeded               │
│   judge = OpenRouter Gemini 2.5 Flash via LiteLLM                     │
│   ragas.evaluate([faithfulness, answer_relevancy, context_precision]) │
│   tracker = CostTracker(f"ragas-judge-tier-{N}")                      │
│   write metrics/tier-{N}-{ts}.json   (per-instance ScoreRecords)      │
│   tracker.persist()  → costs/ragas-judge-tier-{N}-{ts}.json           │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─ ROLLUP + SUMMARIZE (sync, pure file I/O) ────────────────────────────┐
│ For each tier in supported set:                                       │
│   _latest(queries/, costs/, metrics/) → aggregate                     │
│   numpy.nanmean for metrics; sum for cost; count for n,n_NaN          │
│ Emit comparison.md (tier rollup table + per-class rollup + footer)    │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─ FREEZE (sync, single file copy + manifest) ──────────────────────────┐
│ Refuse if frozen/eval-numbers-v1.0.md exists.                         │
│ Copy comparison.md → frozen/eval-numbers-v1.0.md                      │
│ Write frozen/eval-numbers-v1.0.manifest.json with source paths/mtimes │
│ git stage both files (don't commit — user does)                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Partial rerun: Tier 4 only (the load-bearing case)

This is the scenario the milestone calls out: **re-run Tier 4 only without re-paying for Tier 1**.

```
                        ┌──────────────────────────────────────────────┐
1. User runs locally:   │ python tier-4-multimodal/scripts/             │
                        │   eval_capture.py                             │
                        │   (writes queries/tier-4-{NEW_TS}.json)       │
                        └──────────────────────────────────────────────┘
                                          │
                                          ▼
                        ┌──────────────────────────────────────────────┐
2. Pipeline, T4 only:   │ python -m evaluation.harness.pipeline         │
                        │   --tiers 4                                   │
                        │   --tier-4-from-cache queries/tier-4-{NEW}    │
                        │   --freeze v1.0   (or omit; bump to v1.1)     │
                        └──────────────────────────────────────────────┘
                                          │
   Capture:   tier-4 only ─────────────►  queries/tier-4-{NEW_TS}.json
              tiers 1,2,3,5 untouched     queries/tier-{1,2,3,5}-* unchanged
                                          │
   Score:     tier-4 only ─────────────►  metrics/tier-4-{NEW_TS}.json
              tiers 1,2,3,5 untouched     metrics/tier-{1,2,3,5}-* unchanged
                                          │
   Compare:   _latest() resolves     ────►  Tier 4 = {NEW_TS} (newer mtime)
              per-tier independently      Tiers 1,2,3,5 = previous (still latest)
                                          │
   Freeze:    new comparison.md       ────► frozen/eval-numbers-v1.0.md (or v1.1)
              + manifest with the
              CURRENT paths chosen
              for each tier
```

**Why it works:**

- Capture is invoked with `--tiers 4` only, so the capture orchestrator's outer loop visits Tier 4 only. The other tiers' files are not read, not written, not invalidated. The `CostTracker(f"tier-4-eval")` produces a fresh cost ledger for Tier 4 only.
- Score is invoked with `--tiers 4` only (the pipeline propagates the flag). Same isolation — judge cost is paid only for Tier 4's 30 questions × 3 metrics.
- Compare is invoked over **all** tiers (the rollup is always over the full leaderboard); but it reads the **most-recent** per-tier file by mtime. Tiers 1,2,3,5 resolve to their previous (frozen-quality) capture/metrics; Tier 4 resolves to the new ones.
- Freeze pins the chosen paths into the manifest. If somebody later asks "which Tier 1 capture did v1.0 cite?", the manifest tells them.

**Cost saved per partial Tier 4 rerun:** ≈ $0.00006 (Tier 1 capture) + $0.0036 (Tier 2 capture) + $0.30 (Tier 3 capture) + $0.03 (Tier 5 capture) + ≈ $0.27 (judge for tiers 1,2,3,5 × 30 q × 3 metrics) ≈ **$0.60 saved per partial rerun.** Tier 3 capture is the dominant savings because graph traversal LLM calls are expensive.

## Build order (suggested phase delivery)

The milestone is "fix Tier 4 graphml regen + diagnose Tier 5 empty_contexts, then re-emit numbers." Map components to fix-phases:

| Phase | Phase intent | Component delivered | Why this order |
|-------|-------------|---------------------|----------------|
| 132-A | Fix Tier 4 graphml regen | (no new component) | Unblocks Tier 4 capture. |
| 132-B | Diagnose Tier 5 empty_contexts | (no new component; diagnosis lives in PITFALLS notes) | Without this, Tier 5 metrics are all NaN → comparison.md is misleading. |
| 132-C | Add `evaluation/harness/freeze.py` | **Freeze + manifest** | Pure-Python, ≈ 60 LOC, no external deps. Test with the existing `comparison.md`. |
| 132-D | Add `evaluation/harness/pipeline.py` | **Single-command driver** | Composes existing run/score/compare + new freeze. ≈ 100 LOC. |
| 132-E | Add `make eval-all` target (or `pyproject.toml` script entry) | UX polish | One-line invocation; matches the milestone's "make eval-all style entry." |
| 132-F | Run full pipeline → freeze v1.0 | First v1.0 numbers artifact | Validates everything end-to-end. |

132-A and 132-B run in parallel (different tiers). 132-C through 132-F are sequential. 132-C, 132-D, 132-E together are << one day's work — the heavy lifting is already done in Phase 131.

## Anti-patterns

### Anti-pattern 1: Re-running every tier when only one changed

**What people do:** Add `--all` to the orchestrator, blow away `queries/`, and re-pay the full sweep cost just because Tier 4 had a bug.

**Why it's wrong:** Tier 3 alone is ≈ $0.30 of LLM cost per sweep. Doing this 5–10 times during a v1.0 fix cycle wastes $1.50–$3.00 of compute and compounds judge cost.

**Do this instead:** Trust the `_latest()` aggregator. Re-run only the broken tier. Let the rollup pick up the new file by mtime. The frozen manifest captures which version of each tier's artifact was cited, so there's a paper trail.

### Anti-pattern 2: Editing the frozen file in place

**What people do:** Notice a typo in `eval-numbers-v1.0.md`, edit it directly, commit.

**Why it's wrong:** The point of a frozen artifact is that anyone citing "v1.0" sees the same content forever. Edits silently rewrite history.

**Do this instead:** Bump to v1.0.1 (typo fix) or v1.1 (new numbers). Keep v1.0 unchanged. The freeze.py refusal-to-overwrite enforces this; force-flag is an emergency escape hatch only, and the manifest records when force was used.

### Anti-pattern 3: One mega cost JSON per run

**What people do:** Use a single `CostTracker("v1.0-run")` across every tier and every phase to "simplify" the cost rollup.

**Why it's wrong:** Cost-per-tier and cost-per-phase become unrecoverable from the merged JSON. Comparing capture cost vs judge cost is the whole point of the cost-tracker design.

**Do this instead:** Stick with the existing `CostTracker(f"tier-{N}-eval")` and `CostTracker(f"ragas-judge-tier-{N}")` discipline (Phase 131 Pitfall 11 in `run.py:148`). The aggregator already handles per-phase rollup.

### Anti-pattern 4: Letting comparison.md drift unsigned

**What people do:** Run capture + score for 2 of 5 tiers, ship `comparison.md` to the blog, never freeze.

**Why it's wrong:** When the blog post links "comparison.md," readers see a moving target. Six weeks later the file is unrecognizable.

**Do this instead:** Always freeze before citing externally. The freeze step exists specifically to give a stable handle. `comparison.md` is the working copy; `eval-numbers-v{x.y}.md` is the citation handle.

### Anti-pattern 5: Conflating capture cost and judge cost in the table

**What people do:** Sum `tier-N-eval-*.json` and `ragas-judge-tier-N-*.json` into a single "Total Cost" column in `comparison.md`.

**Why it's wrong:** Capture cost is what your users would pay if they ran the system in production. Judge cost is your evaluation overhead. Conflating them mis-prices every tier.

**Do this instead:** `compare.py` currently reports capture cost only ("Total Cost (USD)" comes from `tier-N-eval-*.json`). The judge cost lives in the footer for transparency. Keep this split. If users want a TCO column, add it as a separate column with a clear label.

## Integration points

### External services

| Service | Integration pattern | Notes |
|---------|---------------------|-------|
| OpenRouter (capture) | Tier-1/3/5 main.py → `OPENROUTER_API_KEY` from `shared/config.py` | Used for both LLM completions and the text-embedding-3-small embedding model. |
| Gemini File Search (capture) | Tier-2 main.py → `GEMINI_API_KEY` | Vendor-managed retrieval; .store_id cached locally. |
| RAG-Anything / MineRU (capture) | Tier-4 sidecar `eval_capture.py` runs locally, emits cached JSON | Sandbox-incompatible (Phase 130 SC-1 deferral); capture run on developer's box and consumed via `--tier-4-from-cache`. |
| OpenRouter (judge) | RAGAS via LiteLLM → `OPENROUTER_API_KEY` | Same provider as capture; different model slug allowed (separate `--judge-model` flag). |

### Internal boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Tier CLI ↔ Tier eval adapter | Direct Python import (`from tier_3_graph.rag import build_rag`) | Adapter is the per-tier shim; never imports another tier. |
| Tier eval adapter ↔ run.py | `EvalRecord` (Pydantic v2) | The contract. Adapters return; orchestrator persists. |
| run.py ↔ score.py | Filesystem (`queries/*.json`) | **Decoupled on purpose** so score can run on yesterday's capture (and Inspect's eval-retry pattern). |
| score.py ↔ compare.py | Filesystem (`metrics/*.json`) | Same rationale — comparison can re-roll without re-scoring if you change rounding. |
| compare.py ↔ freeze.py | Filesystem (`comparison.md`) | Freeze is a snapshot; refuses to clobber existing version. |
| pipeline.py ↔ everything | Direct Python function calls (NOT subprocess) | Exit codes propagate; asyncio runs share the same event loop where possible; cost-surprise prompts fire once per phase, not five times. |

## Sources

- [GitHub: EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — `--predict_only`, `--log_samples`, results.json + samples_*.jsonl split. (Source priority: official GitHub, HIGH.)
- [GitHub: stanford-crfm/helm](https://github.com/stanford-crfm/helm) — Scenario / Adapter / Executor / Metric / Runner architecture. (HIGH.)
- [HELM tutorial](https://crfm-helm.readthedocs.io/en/latest/tutorial/) — `helm-run` vs `helm-summarize`, `benchmark_output/runs/{suite}/{run_spec}/{scenario_state.json, per_instance_stats.json, stats.json}`, `--suite` as the freeze handle. (HIGH.)
- [HELM benchmark guide](https://github.com/stanford-crfm/helm/blob/main/docs/benchmark.md) — dry-run, `--max-eval-instances`, suite organization. (HIGH.)
- [Inspect AI Eval Sets](https://inspect.aisi.org.uk/eval-sets.html) — log directory as durable completion record, eval-retry, sample re-use across reruns via stable IDs. (HIGH.)
- [Inspect AI Eval Logs](https://inspect.aisi.org.uk/eval-logs.html) — `EvalLog` hierarchy: eval (metadata), plan (config), samples (per-instance), results (aggregated). Filename `{timestamp}_{task}_{id}.eval`. (HIGH.)
- [Inspect AI inspect_eval-retry reference](https://inspect.aisi.org.uk/reference/inspect_eval-retry.html) — built-in retry/resume command. (HIGH.)
- [bigcode-evaluation-harness README](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/docs/README.md) — `--save_generations`, `--load_generations_path`, `--metric_output_path` separation. (HIGH.)
- Phase 131 internal research file (`131-RESEARCH.md`, referenced from `run.py`/`score.py`/`compare.py` docstrings) — captures the existing patterns that this v1.0 architecture extends. (HIGH — already in-repo.)
- [explodinggradients/ragas](https://github.com/explodinggradients/ragas) — RAGAS judge framework being used by `score.py`; `evaluate()` API and `get_token_usage_for_openai` token-usage parser. (HIGH.)

---
*Architecture research for: eval harness v1.0 fix-and-freeze pipeline*
*Researched: 2026-05-04*

# Phase 7: Full 5-Tier Rerun — Research

**Researched:** 2026-05-06
**Domain:** Operational orchestration of an existing eval pipeline against live API + live storage; pre-flight gating; cost containment under a $1–3 budget; SHA invariant preservation across a long-wall sweep; rebuild-then-capture sequencing for Tier 4 specifically.
**Confidence:** HIGH on the harness-level facts (every claim is from source); MEDIUM on cost extrapolation (only 5q live samples exist; 30q × 5 tiers projection has variance); MEDIUM on Tier 4 full-rebuild duration (3-paper smoke measured, 79-paper full has compound variance from vision-LLM 400 errors and OpenRouter throttling).

## Summary

Phase 7 is **not a code-writing phase**. Every piece of machinery already exists and is proven: pipeline.py composes capture → score → compare → freeze in-process with a single sweep-level git SHA + ISO timestamp threaded through `git_sha_override` / `ts_override` kwargs (Phase 5 closed HARN-01/02 at unit + integration + live levels). Per-tier embedder provenance lands in every QueryLog (Phase 6 closed CAP-03). NaN reasons are structured (Phase 3 closed HARN-05). Tier 5's adapter is fixed (Phase 1 closed TIER-01). Tier 4 has a working ingest path and a smoke-only graph (Phase 2 closed TIER-02 with the deferred-to-Phase-7 note). Tier 4's MineRU CLI cache is at 79 papers on disk (75 from Plan 02-01 + 4 host-parsed by Plan 02-02). The only thing left is **execute the sweep against live APIs and verify the outputs**.

What makes this phase research-worthy is the *sequencing and gating discipline*, not new technology choices. Three pre-flight blockers must be cleared in order before the sweep can start: (1) `tier-2-managed/.store_id` is missing on disk (gitignored; never landed) — Tier 2 capture exits 2 immediately under `_check_prereqs` until a fresh File Search store is created; (2) the Tier 4 graphml currently encodes only 3 smoke papers (2886 nodes / 7056 edges per the 2026-05-05T11:14:40Z provenance manifest) — running CAP-01 against this graph ships the blog under a "100-paper corpus" claim that is invalidated by 3-paper graphml ground truth; (3) all 5 single-tier smokes (1, 2, 3, 4, 5 × 5 questions) should green-light with current HEAD before the full sweep is launched, because every smoke failure caught pre-sweep is one fewer round of $1–3 burned re-running. The Tier 4 full rebuild itself is a **15–25h wall, ~$15–35** operation projected from Plan 02-01's 3-paper / ~50min / $1.89 measurement — it is by far the largest cost line in the entire v1.0 scope and dwarfs the $1–3 sweep budget. The roadmap's "$1–3 per full sweep" line covers the *capture+judge sweep only*, NOT the Tier 4 ingest. The plan must price these separately and surface that to the user before the cost-ack checkpoint.

Once pre-flight is green, the sweep itself is one command: `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --tier-4-from-cache <path-to-tier-4-eval-capture-output> --yes`. Pipeline.py captures `sweep_sha` once at top via `run._git_sha()`, propagates to all 5 tiers' QueryLog `git_sha` field, and prints a single banner `Pipeline start git=<sha> ts=<ts>`. On the score side, judge cost is bounded at ~$0.45 (0.003/q × 30q × 5 tiers, conservative); capture is bounded at ~$2.00 (sum of `run.COST_PER_Q × 30`); total ~$2.45 per sweep — fits the $1–3 envelope. **Tier 4 is special-cased**: pipeline.py's run.amain dispatches Tier 4 through `--tier-4-from-cache <path>`, NOT through a live in-pipeline capture (per Phase 130 SC-1 deferral baked into run.py:215-247). The user must run `python tier-4-multimodal/scripts/eval_capture.py --yes` BEFORE invoking pipeline.py and pass the resulting `queries/tier-4-{ts}.json` path to the pipeline's `--tier-4-from-cache` flag. This breaks the "single sweep timestamp" property *for Tier 4 specifically*: Tier 4's QueryLog timestamp will be its eval_capture timestamp, NOT the sweep timestamp; but the pipeline's compare/freeze stages still rely on `_latest()` mtime resolution to pick it up as the "current" Tier 4 capture. **The git SHA invariant CAN still hold across all 5 tiers if eval_capture.py is run at the same HEAD as pipeline.py** — since Tier 4's capture writes its own git SHA at the time of capture, and pipeline.py runs at a single later SHA, the user must `git status` and confirm clean before kicking off Tier 4 capture, then NOT make any commits between Tier 4 capture and pipeline kickoff. This is a discipline gate, not a code gate.

**Primary recommendation:** Structure Phase 7 as **3 plans across 4 conceptual waves**:

- **Plan 07-01 — Pre-flight verification (all-tier smoke gate, no full sweep)** — explicit single-tier 5q smokes for tiers 1/2/3/4/5, gated on `tier-2-managed/.store_id` existence (re-create from `tier-2-managed/main.py --ingest --yes` if missing) and Tier 4 graphml state (current 3-paper graph is acceptable for smoke; full-corpus rebuild is Plan 07-02). Verdict: 5/5 smokes PASS at HEAD before authorizing Plan 07-02.
- **Plan 07-02 — Tier 4 full-corpus rebuild (host-side, 15–25h / $15–35)** — `python tier-4-multimodal/scripts/ingest_from_mineru.py --reset --yes` over the 79-paper cache, followed by `log_graph_stats.py` to write a fresh provenance manifest with the new node/edge counts. **Cost-ack checkpoint REQUIRED** (this is the big cost line). Smoke-test the new graph with `eval_capture.py --smoke-question-ids ... --yes` + `python -m evaluation.harness.smoke_gate --tier 4` before declaring the rebuild done.
- **Plan 07-03 — Live full-sweep execution + post-sweep verification** — `eval_capture.py` for Tier 4 first (writes tier-4 queries+costs JSON locally), then `pipeline.py --tiers 1,2,3,4,5 --tier-4-from-cache <path> --yes` for the rest. Cost-ack checkpoint REQUIRED. Post-sweep grep across all 10 JSONs verifying single-SHA invariant + same-date timestamps + NaN counts <5/30 for tiers 4 and 5 + total spend ≤ $3 (sum of 10 cost JSONs' `totals.usd`). Human-verify checkpoint to confirm CAP-01 closure before triggering /gsd-verify-phase.

The plan must NOT introduce new code beyond a small `verify_full_sweep.py` helper if needed (per Phase 6's lesson about deriving managed-flag in compare.py rather than adding a boolean to QueryLog — keep verification logic out of the harness, in a phase-local script).

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CAP-01 | User can capture a full 5-tier × 30-question RAGAS run on a single date with a single git SHA recorded in every per-tier output JSON | Section "Sweep Architecture" — pipeline.py threads `sweep_sha` via `git_sha_override` kwarg to every tier; Tier 4 special-cased via `--tier-4-from-cache`. Verification commands in Section "Post-Sweep Verification". |

## Project Constraints (from CLAUDE.md)

`./CLAUDE.md` does not exist at repo root. Project conventions inherited from `.planning/STATE.md` Decisions and the 6 already-shipped phases:

- ISO 8601 Z timestamps via `run._ts()` / `run._ts_for_filename()` — single source of truth.
- Git SHA via `run._git_sha()` (short HEAD SHA; "unknown" on failure) — single source of truth.
- `rich.console.Console` for CLI output — match existing per-phase banners.
- `from __future__ import annotations` + module docstring on every harness module.
- Argparse pattern: `build_parser()` separately so tests can introspect; `main(argv)` calls `asyncio.run(amain(args, Console()))`.
- Test fixtures: `tmp_path` + offline mocks; live tests gated under `@pytest.mark.live`. **Pyproject does NOT enforce live-deselect-by-default** (v1.1 hardening item) — `pytest path/to/file.py -v` runs live tests silently. Phase 7's verification commands must always pass `-m 'not live'` for offline checks and `-m live -k <test>` for explicit live invocations.
- Forward-contract guards on the 6 already-shipped harness files: any plan touching them must commit a 0-byte-diff verification step before merge. Phase 7 should NOT touch any of: `freeze.py`, `pipeline.py`, `compare.py`, `score.py`, `run.py` (additively, only tier-4 ingest scripts and possibly a phase-local verifier script).

## Architectural Responsibility Map

| Capability | Owner Module / Path | Phase 7 Action |
|-----------|---------------------|---------------|
| Pre-flight prereq checks (chroma_db, .store_id, lightrag_storage) | `evaluation/harness/run.py::_check_prereqs` | Read, don't edit. Plan 07-01 calls `python -m evaluation.harness.run --tiers 1,2,3,4,5 --limit 0 --yes` (or equivalent) only AFTER `.store_id` is recreated. |
| Tier 4 graphml ingest from MineRU cache | `tier-4-multimodal/scripts/ingest_from_mineru.py` | Run with `--reset --yes` over full 79-paper cache (Plan 07-02). |
| Tier 4 graph stats provenance manifest | `tier-4-multimodal/scripts/log_graph_stats.py` | Run after ingest to write `tier-4-graph-stats-{ts}.json` for Phase 9 (Plan 07-02). |
| Tier 4 live capture (queries + cost JSON) | `tier-4-multimodal/scripts/eval_capture.py` | Run with `--yes` (no `--smoke-question-ids`, no `--limit`) to drive all 30q (Plan 07-03 first step). |
| Sweep capture (Tiers 1, 2, 3, 5) | `evaluation/harness/run.py::amain` (called via pipeline.py) | Driven by pipeline.py with `--tiers 1,2,3,4,5 --tier-4-from-cache <path>`; Tier 4 in `--tiers` triggers passthrough mode. |
| Sweep score (all 5 tiers) | `evaluation/harness/score.py::amain` (called via pipeline.py) | Composed in pipeline.py; `_latest_query_log` per-tier mtime resolution picks up the freshly-written tier-4 capture from eval_capture.py. |
| Sweep compare rollup | `evaluation/harness/compare.py::_run` (called via pipeline.py) | Always tiers="1,2,3,4,5" per pipeline.py D-Q4. |
| Freeze (NOT in Phase 7) | `evaluation/harness/freeze.py::freeze` | Phase 7 should NOT call `--freeze` on pipeline.py. Phase 9 freezes after Phase 8's multi-judge data lands. |
| Post-sweep verification | New phase-local `verify_sweep.py` (recommended) OR inline bash + grep | Single SHA grep across 10 JSONs; NaN count per tier; total spend sum across cost ledgers; same-date check on timestamps. |
| Smoke gate per tier | `evaluation/harness/smoke_gate.py::evaluate_smoke_from_paths` | Plan 07-01 calls per-tier after each tier's capture+score. |

### MUST NOT touch (byte-identical post-Phase 7)

- `evaluation/harness/freeze.py` — Phase 4/5/6 contract LOCKED. No modifications.
- `evaluation/harness/pipeline.py` — Phase 5 contract LOCKED. No modifications.
- `evaluation/harness/compare.py` — Phase 6 contract LOCKED. No modifications.
- `evaluation/harness/score.py` — Phase 6 forward-contract guard intact. No modifications.
- `evaluation/harness/run.py` — Phase 5 + Phase 6 contracts LOCKED. No modifications.
- `evaluation/harness/smoke_gate.py` — Phase 1 + Phase 6 contracts LOCKED. No modifications.

### MAY need targeted creation

- `tier-4-multimodal/scripts/verify_sweep.py` (OPTIONAL) — phase-local helper for post-sweep verification (single-SHA grep, NaN count, total-spend sum). Could also be inline bash + jq invoked from a Plan 07-03 task step. Recommendation: bash + python one-liners are sufficient; do NOT add a Python script unless the plan grows beyond 4 verification steps.

## User Constraints

No `CONTEXT.md` exists for Phase 7 (user did not run `/gsd:discuss-phase`). The following constraints are derived from the roadmap entry, requirements, and accumulated state:

### Locked Decisions (from ROADMAP + STATE)

- **Single eval-date for all 5 tiers; budget ~$1-3 per full rerun, 2-3 reruns total.** (`STATE.md` Decisions)
- **Tier 4 MineRU ingest must run outside the sandbox** (Phase 139 / Phase 2 evidence). The orchestrator is sandboxed; the user must run ingest commands in a host shell. Plan 07-02 is a `checkpoint:user-action` step.
- **Tier 4 capture is special-cased through `eval_capture.py` + `--tier-4-from-cache`** (Phase 130 SC-1 deferral baked into `run.py:215-247`). Pipeline.py honors this; Plan 07-03's first step is `eval_capture.py`.
- **Phase 7 does NOT freeze.** Freeze is Phase 9's job after Phase 8's multi-judge data lands. Pipeline.py supports this naturally — `--freeze` is optional and unset by default.
- **Multi-judge spot-check is Phase 8.** Phase 7 produces only the primary-judge numbers (Gemini 2.5 Flash via OpenRouter).
- **Tier 5 records the SAME embedder as Tier 1** per Plan 06-01 D-ROADMAP-OVERRIDE; the QueryLog will reflect this and compare.py's "Embedder by tier" table will derive `Managed=no` for Tier 5.

### Claude's Discretion (research-recommended)

- **Plan structure**: 3-plan layout (07-01 pre-flight smoke, 07-02 Tier 4 rebuild, 07-03 sweep + verify) — recommended over a single mega-plan because the Tier 4 rebuild is the highest-cost step and deserves its own cost-ack checkpoint independent of the sweep.
- **Whether to write `verify_sweep.py`**: the verification can be done with bash + jq one-liners; a Python helper is justified only if the plan grows beyond 4 verification steps. Recommend inline.
- **Whether to keep current Tier 4 graphml or rebuild**: the roadmap's "100-paper corpus" claim and the Tier 4 graph stats manifest at 3 papers are inconsistent. Plan 07-02 rebuilds. The 3-paper graph stays gitignored at the same path; the rebuild overwrites it.
- **Whether to commit a `tier-4-graph-stats-{ts}.json` manifest for the full rebuild**: YES — Phase 9 cites this manifest in the frozen doc. The smoke-only one is gitignored? No — it IS committed (per Plan 02-01 SUMMARY: "evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json — provenance manifest (committed)"). Plan 07-02 must commit the new manifest alongside the existing one (don't overwrite — both are part of the historical record).

### Deferred Ideas (OUT OF SCOPE)

- Pre-warming `kv_store_llm_response_cache.json` to amortize Tier 4 ingest cost across re-runs (STATE.md "Phase 7 ingest run will hit OpenRouter vision-LLM `400 Invalid URL format` errors" concern). Acceptable to take the cost hit on the first rebuild; future re-runs hit the cache.
- Routing vision-LLM calls direct to OpenAI/Gemini off-OpenRouter to dodge the URL-length 400 errors. Acceptable to accept some multimodal items getting empty descriptions; the graph is robust to ~5–10% multimodal-item drop without invalidating the rollup.
- Parallelizing tier captures (currently strictly sequential in pipeline.py). Out of scope; sequential is the contract per Phase 5 D-Q4.
- Auto-bumping freeze version on `FileExistsError`. Phase 7 doesn't freeze.
- Adding a top-level `pipeline-runs/{ts}.json` manifest aggregating per-stage exit codes. v1.1 deferred per Plan 05-01 D-Q4.

## Pre-Sweep Gating Sequence (CRITICAL)

The sweep cannot start until ALL these gates are green. Phase 7's plan must encode these as explicit task steps:

### Gate 1: Environment & API keys

```bash
test -n "$OPENROUTER_API_KEY"  # exit 0 if set; needed by tiers 1, 3, 4, 5 + judge
test -n "$GEMINI_API_KEY"      # exit 0 if set; needed by Tier 2
test -f .env                   # repo .env loaded by run.py / score.py / eval_capture.py
```

The repo's `.env` (verified at `/Users/patrykattc/work/git/rag-architecture-patterns/.env`) carries both keys (verified). If a fresh clone, the user must populate `.env` from `.env.example`.

### Gate 2: Tier-1 storage exists

```bash
test -d chroma_db/tier-1-naive && ls chroma_db/tier-1-naive/chroma.sqlite3
```

Verified present on disk. If missing: `python tier-1-naive/main.py --ingest`.

### Gate 3: Tier-2 .store_id exists  ⚠️ CURRENTLY MISSING

```bash
test -f tier-2-managed/.store_id && cat tier-2-managed/.store_id  # must produce a UUID
```

**VERIFIED MISSING ON DISK** at `/Users/patrykattc/work/git/rag-architecture-patterns/tier-2-managed/.store_id` (gitignored per `.gitignore` line `tier-2-managed/.store_id`). `evaluation/harness/run.py:119-122` will exit 2 with `[red]tier-2-managed/.store_id missing.[/red]` immediately on any run that includes Tier 2.

**Required fix** (Plan 07-01 task step): `python tier-2-managed/main.py --ingest --yes` — recreates the OpenAI File Search store and writes the `.store_id` sidecar. Cost: TBD from Phase 128 records (likely <$0.50 for a 100-paper ingest; verify against `evaluation/results/costs/tier-2-test-*.json` if present, else surface as cost line in the cost-ack checkpoint).

### Gate 4: Tier-3 lightrag_storage exists

```bash
test -d lightrag_storage/tier-3-graph && ls lightrag_storage/tier-3-graph/graph_chunk_entity_relation.graphml
```

Verified present on disk. If missing: `python tier-3-graph/main.py --ingest --yes`.

### Gate 5: Tier-4 rag_anything_storage exists (smoke-acceptable)

```bash
test -d rag_anything_storage/tier-4-multimodal && \
    ls rag_anything_storage/tier-4-multimodal/graph_chunk_entity_relation.graphml
```

Currently the 3-paper smoke graphml (2886 nodes / 7056 edges from 2026-05-05). **For a CAP-01-honest sweep, Plan 07-02 rebuilds this to the full 79-paper corpus FIRST.** Smoke verification of Tier 4 in Plan 07-01 against the current 3-paper graph is acceptable — the smoke set's 5 questions cite 3 papers (`2005.11401`, `2004.04906`, `2002.08909`) which are all in the smoke graph.

### Gate 6: Per-tier 5q smoke PASS

For each tier 1, 2, 3, 4, 5: run a 5-question smoke and gate on `evaluate_smoke_from_paths` returning verdict=PASS. Tiers 1, 3, 5 use `python -m evaluation.harness.run --tiers <t> --smoke-question-ids single-hop-001,single-hop-002,single-hop-003,multi-hop-001,multi-hop-002 --yes`; Tier 2 uses the same; Tier 4 uses `python tier-4-multimodal/scripts/eval_capture.py --smoke-question-ids ... --yes`. Then `python -m evaluation.harness.score --tiers <t> --yes` and `python -m evaluation.harness.smoke_gate --tier <t>`. Cost ballpark: 5q × 5 tiers × ~$0.014/tier-smoke ≈ $0.07 total (Phase 5 Plan 05-02 measured Tier 5 × 5q at $0.007; tiers 1, 2 are cheaper; tier 3 and 4 are pricier).

**Plan 07-01 acceptance**: 5/5 tier smokes PASS. If any tier smokes FAIL, that tier needs investigation BEFORE the full sweep. The smoke PASS threshold is `populated/measurable >= 0.8 AND faithfulness/context_precision non-NaN on populated rows` per `smoke_gate.py:147-150`.

### Gate 7: Git tree clean (or knowingly dirty)

```bash
git status --porcelain   # if non-empty, the sweep_sha is "dirty"
```

`run._git_sha()` returns the short HEAD SHA regardless of dirtiness. `freeze._git_dirty()` is a manifest field, NOT a precondition (Phase 4 decision). For Phase 7's CAP-01, a dirty tree is technically tolerated, but the "single git SHA recorded in every per-tier output JSON" claim is harder to defend if the tree is dirty. Recommend committing all WIP before the sweep so the SHA references a real commit.

## Tier 4 Full-Corpus Rebuild Specifics (Plan 07-02)

This is the largest cost line in the entire v1.0 scope. STATE.md Decisions explicitly call out: "Phase 7 pre-rerun ingest must process 72 remaining papers (`tier-4-multimodal/output/` minus 3 smoke papers minus 4 Plan-02-02 fresh-MineRU papers); projected wall ~15–25h / cost ~$15–35".

**Verified facts:**

- MineRU cache on disk: 79 papers under `tier-4-multimodal/output/` (verified via `ls | wc -l = 79`).
- Plan 02-01 measured: paper 2005.11401 took ~21min wall. Plan 02-01 final smoke (3 papers): ~50min wall, $1.89 spend, 2886 nodes / 7056 edges.
- Library versions of record (committed in `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json`): raganything==1.2.10, lightrag-hku==1.4.15, mineru==3.1.4 — these are the versions Phase 9's frozen doc cites. Plan 07-02 must NOT bump any of these between rebuild and sweep.
- The 79-paper rebuild **wipes** `rag_anything_storage/tier-4-multimodal/` via `shutil.rmtree` (Pitfall 4 zombie-state guard). The current 3-paper smoke graph is destroyed by this operation. There is no rollback if the rebuild dies mid-way other than re-running `ingest_from_mineru.py --reset --yes` (the `kv_store_llm_response_cache.json` partial-cache from a failed run remains; the second invocation gets cache hits on whatever LLM responses were already paid for).

**Rebuild command:**

```bash
# Run on host shell (NOT in orchestrator sandbox)
cd /Users/patrykattc/work/git/rag-architecture-patterns
source .venv/bin/activate
python tier-4-multimodal/scripts/ingest_from_mineru.py --reset --yes
# ... 15-25h of wall time, ~$15-35 in OpenRouter spend ...

# After ingest completes, write the new provenance manifest
python tier-4-multimodal/scripts/log_graph_stats.py
# → evaluation/results/diagnostics/tier-4-graph-stats-{TS}.json (commit this)
```

**Failure modes during the rebuild:**

| Failure | Symptom | Resolution |
|---------|---------|-----------|
| OpenRouter rate-limit (429) | `_capture_tier` retries via lightrag's internal retry; if exhausted, paper is skipped (Pitfall 1 of `ingest_from_mineru.py`: per-paper exception isolation). | Continue; final paper count <79. Acceptable down to ~70 ingested papers. If <70, re-run ingest. |
| Vision-LLM 400 (URL too long) | OpenRouter rejects base64-encoded large figures. Plan 02-01 SUMMARY: "non-fatal (RAG-Anything continued processing other items) but resulted in a small number of multimodal items getting empty descriptions". | Accept ~5-10% multimodal-item dropout. STATE.md flags this as a known issue. |
| Embedding worker timeout (60s) | Lightrag auto-retries (Plan 02-01 measured: 2 retries, both succeeded). | Continue. |
| OPENROUTER_API_KEY not in os.environ | Lightrag closures read lazily from os.environ, NOT settings. Plan 02-01 fix in commit 5bc3f24 forwards `os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key.get_secret_value()` at script entry. | Already fixed in `ingest_from_mineru.py`; no Phase 7 action needed. |
| Process crash mid-ingest (e.g. ctrl-C, OS reboot) | Storage left in partial state; some papers in graphml, others not. | Run again with `--reset --yes`. The kv_store_llm_response_cache.json from the previous run is wiped by `--reset`. To preserve cache across crash-recovery, omit `--reset` — but then the zombie-state guard refuses unless storage is empty. **Recommendation: accept full re-cost on crash recovery; document as a known limitation.** |

**Smoke gate after rebuild (within Plan 07-02):**

```bash
python tier-4-multimodal/scripts/eval_capture.py \
    --smoke-question-ids single-hop-001,single-hop-002,single-hop-003,multi-hop-001,multi-hop-002 \
    --yes
# → evaluation/results/queries/tier-4-{TS}.json + costs/tier-4-eval-{TS}.json

python -m evaluation.harness.score --tiers 4 --yes
# → metrics/tier-4-{TS}.json + costs/ragas-judge-tier-4-{TS}.json

python -m evaluation.harness.smoke_gate --tier 4
# Verdict must be PASS — same gate Plan 02-04 used post-gap-closure.
```

If smoke FAILs after rebuild, do NOT proceed to Plan 07-03. Investigate (likely: graph quality regression from one of the 4 newly-ingested host papers; or LLM variance ±10% per Plan 02-01 Pitfall 6).

## Sweep Architecture (Plan 07-03)

### Step 1 — Tier 4 live capture (locally, before pipeline)

```bash
python tier-4-multimodal/scripts/eval_capture.py --yes
# → evaluation/results/queries/tier-4-{TS}.json (30 records, all 30 questions)
# → evaluation/results/costs/tier-4-eval-{TS}.json
# Cost: 30 × ~$0.0015 = ~$0.045 (per eval_capture.py docstring)
```

This writes the canonical Tier 4 QueryLog with `git_sha` = HEAD at time of capture. **The user must NOT commit anything between this step and Step 2** — otherwise pipeline.py's `sweep_sha` will not match the Tier 4 QueryLog's `git_sha`, breaking the CAP-01 invariant.

### Step 2 — Pipeline sweep (capture+score+compare for tiers 1, 2, 3, 5; passthrough for tier 4)

```bash
TIER4_PATH=$(ls -t evaluation/results/queries/tier-4-*.json | head -1)
python -m evaluation.harness.pipeline \
    --tiers 1,2,3,4,5 \
    --tier-4-from-cache "$TIER4_PATH" \
    --yes
# Pipeline captures sweep_sha + sweep_ts ONCE, threads through to all 5 tiers.
# Tier 4 in --tiers list triggers passthrough mode — pipeline.py reads the cached
# QueryLog and writes a new tier-4-{sweep_ts}.json with the SAME records but a
# NEW timestamp matching sweep_ts. Cost JSON for tier 4 is also rewritten with
# sweep_sha provenance. (Verify this claim in run.py:233-262 against actual behavior.)
```

⚠️ **Critical detail to verify in implementation**: read `run.py:233-262` carefully. The current code path for `tier == 4 and args.tier_4_from_cache` calls `run_tier4(... from_cache=cache_path, tracker=tracker)` for each question, which presumably reads from the cache file but writes a NEW QueryLog with the `git_sha` set from `git_sha_override` or `_git_sha()`. If pipeline.py threads `git_sha_override = sweep_sha`, then the freshly-written tier-4-{sweep_ts}.json carries sweep_sha — preserving the CAP-01 invariant. **Confirm via a smoke test in Plan 07-01 or first task of Plan 07-03**: run pipeline with `--tiers 4 --tier-4-from-cache <path> --limit 5 --yes` and grep the resulting tier-4-{sweep_ts}.json for `git_sha`. The expected value is HEAD's short SHA (i.e., `_git_sha()` output), NOT the cached file's git_sha.

The cost ballpark for Step 2: per `pipeline._pipeline_cost_estimate`:
- Capture (tiers 1, 2, 3, 5; tier 4 is passthrough so cost is ~0): ~$0.0002 + $0.0001 + $0.01 + $0.001 = $0.0113/q × 30q = $0.339 (sums of `run.COST_PER_Q[1,2,3,5]` × 30; tier 4 cost is in eval_capture.py spend, NOT here).
- Judge: ~$0.003/q × 30q × 5 tiers = $0.45 (covers RAGAS for all 5 tiers).
- Total: ~$0.79 — well under the $1-3 envelope.

Adding the Tier 4 eval_capture.py spend (~$0.045 for 30q) brings the full sweep total to ~$0.84.

⚠️ **The $1-3 budget assumption in STATE.md likely included rough overhead for unexpected retries, judge token-cost variance, or a slightly more expensive judge configuration. The actual envelope is dominated by Tier 3 capture ($0.30) and judge ($0.45).**

### Step 3 — Post-sweep verification

After pipeline.py exits 0, verify:

```bash
# 1. Single SHA across all 10 JSONs (5 capture queries + 5 metrics)
SWEEP_SHA=$(jq -r '.git_sha' "$(ls -t evaluation/results/queries/tier-1-*.json | head -1)")
for t in 1 2 3 4 5; do
    Q=$(ls -t evaluation/results/queries/tier-${t}-*.json | head -1)
    M=$(ls -t evaluation/results/metrics/tier-${t}-*.json | head -1)
    QSHA=$(jq -r '.git_sha' "$Q")
    # metrics JSON does NOT carry git_sha; provenance is in the queries file
    echo "tier-${t}: queries_sha=${QSHA}"
    [[ "$QSHA" == "$SWEEP_SHA" ]] || { echo "FAIL: SHA drift on tier-${t}"; exit 1; }
done
echo "PASS: single SHA $SWEEP_SHA across all 5 tiers"

# 2. Same-date check (all 5 tier captures within same UTC date)
DATES=$(for t in 1 2 3 4 5; do
    Q=$(ls -t evaluation/results/queries/tier-${t}-*.json | head -1)
    jq -r '.timestamp' "$Q" | cut -dT -f1
done | sort -u)
[[ $(echo "$DATES" | wc -l) -eq 1 ]] || { echo "FAIL: tiers span multiple dates"; exit 1; }
echo "PASS: all tiers captured on date $DATES"

# 3. NaN counts per tier from metrics JSONs
for t in 4 5; do
    M=$(ls -t evaluation/results/metrics/tier-${t}-*.json | head -1)
    NAN_COUNT=$(jq '[.[] | select(.nan_reason != null)] | length' "$M")
    echo "tier-${t}: nan_count=${NAN_COUNT}/30"
    [[ "$NAN_COUNT" -lt 5 ]] || { echo "FAIL: tier-${t} NaN >= 5/30"; exit 1; }
done

# 4. Total spend across all 10 cost JSONs
TOTAL=$(jq -s '[.[] | .totals.usd] | add' \
    evaluation/results/costs/tier-{1..5}-eval-*.json \
    evaluation/results/costs/ragas-judge-tier-{1..5}-*.json 2>/dev/null)
# (jq -s reads multiple files; pattern expansion picks up the latest of each glob — verify in plan)
echo "Total spend: $TOTAL"
[[ $(echo "$TOTAL <= 3.0" | bc) -eq 1 ]] || { echo "FAIL: spend $TOTAL > 3.0"; exit 1; }
```

**WARNING about jq glob trick**: `tier-{1..5}-eval-*.json` expands to all matching files including stale ones from prior sweeps. The verifier must filter by mtime DESC and pick the freshest per tier. A safer Python helper avoids this:

```python
# verify_sweep.py — phase-local helper (recommend inlining if <50 LOC)
import json, sys
from pathlib import Path
from collections import defaultdict

results = Path("evaluation/results")
def latest(dir_, pattern):
    files = sorted(dir_.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

errors = []
shas = set(); dates = set(); total = 0.0
for t in (1, 2, 3, 4, 5):
    q = latest(results/"queries", f"tier-{t}-*.json")
    if q is None: errors.append(f"tier-{t}: no queries log"); continue
    qd = json.loads(q.read_text())
    shas.add(qd["git_sha"]); dates.add(qd["timestamp"][:10])
    m = latest(results/"metrics", f"tier-{t}-*.json")
    md = json.loads(m.read_text()) if m else []
    n_nan = sum(1 for r in md if r.get("nan_reason"))
    if t in (4, 5) and n_nan >= 5:
        errors.append(f"tier-{t}: nan_count={n_nan}/30 ≥ 5")
    cap = latest(results/"costs", f"tier-{t}-eval-*.json")
    jud = latest(results/"costs", f"ragas-judge-tier-{t}-*.json")
    for c in (cap, jud):
        if c: total += json.loads(c.read_text())["totals"]["usd"]

if len(shas) != 1: errors.append(f"SHA drift: {shas}")
if len(dates) != 1: errors.append(f"date drift: {dates}")
if total > 3.0: errors.append(f"spend ${total:.4f} > $3.00")

if errors:
    print("\n".join(f"FAIL: {e}" for e in errors)); sys.exit(1)
print(f"PASS: sha={shas.pop()} date={dates.pop()} total=${total:.4f}")
```

**Recommendation**: write this as a phase-local script `tier-4-multimodal/scripts/verify_sweep.py` (or `evaluation/harness/verify_sweep.py` — but that touches the harness; prefer phase-local). ~50 LOC. Plan 07-03's verification step calls it.

### Step 4 — Human-verify checkpoint

Before declaring CAP-01 closed:

- [ ] User reads the final `comparison.md` (regenerated by pipeline.py's compare stage) and confirms numbers look plausible.
- [ ] User reads the new `tier-4-graph-stats-{ts}.json` and confirms node/edge counts match expected scale (~75-79× the smoke 2886-node count is implausible because graph density is non-linear; expected: 50,000–80,000 nodes / 100,000–200,000 edges; **research-honest qualifier: extrapolating linearly from 3-paper graph is unreliable — first full rebuild establishes the actual ground truth**).
- [ ] User reads the embedder-by-tier table in `comparison.md` and confirms it matches Plan 06-01 expectations (Tier 1: openai/text-embedding-3-small via openrouter; Tier 2: gemini-embedding-001 via google-managed; Tier 3: openai/text-embedding-3-small via openrouter; Tier 4: openai/text-embedding-3-small via openrouter; Tier 5: openai/text-embedding-3-small via openrouter — all five "Managed" column values: Tier 2 yes, all others no).

## Cost Envelope Analysis

| Phase | Operation | Estimated Cost | Wall Time | Notes |
|-------|-----------|---------------|-----------|-------|
| Plan 07-01 | Per-tier 5q smoke × 5 tiers | ~$0.07 | ~10 min | T1 ~$0.001 + T2 ~$0.0005 + T3 ~$0.05 + T4 ~$0.0075 + T5 ~$0.014; smokes can run in parallel batches if user is patient |
| Plan 07-01 | `tier-2-managed/main.py --ingest --yes` (recreate `.store_id`) | ~$0.10–0.50 | ~5–15 min | OpenAI File Search ingest of ~100 papers; one-time; cost depends on Phase 128 schedule |
| Plan 07-02 | Tier 4 full-corpus rebuild (79 papers via `ingest_from_mineru.py`) | ~$15–35 | ~15–25h | DOMINANT cost. Off-hours run recommended. Vision-LLM 400 errors expected on ~5-10% of multimodal items; non-fatal. |
| Plan 07-02 | Tier 4 post-rebuild smoke + `log_graph_stats.py` | ~$0.01 | ~5 min | Same eval_capture+score+smoke_gate path Plan 02-04 used |
| Plan 07-03 | `eval_capture.py` for Tier 4 (30q) | ~$0.045 | ~5–10 min | Per eval_capture.py docstring |
| Plan 07-03 | `pipeline.py --tiers 1,2,3,4,5 --tier-4-from-cache <path> --yes` | ~$0.79 | ~30–60 min | Capture $0.339 + judge $0.45; tier-3 dominates capture cost |
| Plan 07-03 | Post-sweep verification | $0 | <1 min | Pure file I/O |
| **Total Phase 7** | | **~$16–37** | **~16–26h** | Tier 4 rebuild dominates; sweep itself fits the $1-3 envelope cleanly |

⚠️ **The roadmap's "$1–3 per full sweep" line refers to Plan 07-03's pipeline.py invocation, NOT the Tier 4 rebuild.** Plan 07-02's rebuild is a one-time cost paid before the budgeted reruns. STATE.md correctly identifies this in the Blockers section: "Phase 7 pre-rerun ingest must process 72 remaining papers ... projected wall ~15–25h / cost ~$15–35".

If the user wants 2-3 reruns of Plan 07-03 (per STATE.md "2-3 reruns total"), they spend $0.79 × 3 = ~$2.37 on the sweep itself plus $0.045 × 3 = $0.135 on Tier 4 capture. The Tier 4 graphml is rebuilt ONCE (Plan 07-02) and reused across all reruns.

## Failure Modes & Recovery

### Mid-sweep tier failure (rate limit, API outage, adapter exception)

`run._capture_tier` swallows per-question exceptions and `run.amain` prints `[yellow]Tier {tier} produced no log[/yellow]` then continues to the next tier. Per Plan 05-01 Pitfall 2: "the exception is swallowed at the adapter level, not raised". Pipeline.py exits 0 with no indication that a tier produced corrupt or partial results.

**Recovery semantics for HARN-02 (single-tier rerun)**:

```bash
# If tier 3 had a partial failure mid-sweep, rerun ONLY tier 3:
python -m evaluation.harness.pipeline --tiers 3 --yes
# This produces a fresh tier-3-{NEW_TS}.json with a NEW sweep_sha.
# CAP-01's "single git SHA across all per-tier outputs" invariant is BROKEN
# because tier-3 now carries a different SHA than tiers 1, 2, 4, 5.
```

**This is the canonical CAP-01 failure mode.** If a partial-sweep failure occurs, the only way to preserve the single-SHA invariant is to **rerun the FULL sweep**, not just the failed tier. Single-tier reruns are a HARN-02 feature for development workflows; for CAP-01-honest captures, full reruns are required when any tier fails.

**Mitigation in Plan 07-03**: detect partial-sweep failures by checking each tier's QueryLog has 30 records:

```bash
for t in 1 2 3 4 5; do
    Q=$(ls -t evaluation/results/queries/tier-${t}-*.json | head -1)
    N=$(jq '.records | length' "$Q")
    [[ "$N" -eq 30 ]] || { echo "FAIL: tier-${t} has only $N/30 records"; exit 1; }
done
```

If any tier has <30 records, the full sweep must be rerun. Costs another ~$0.79 + $0.045.

### Cache-invalidating Tier 4 graphml drift

If the Tier 4 graph is rebuilt between reruns (e.g. user reruns `ingest_from_mineru.py --reset --yes` thinking it'll be cheap), the graphml node/edge counts change (LLM variance ±10% per Plan 02-01 Pitfall 6) and `tier-4-graph-stats-{ts}.json` produces a new manifest. Phase 9's frozen doc cites a specific graphml manifest; the manifest cited must match the graphml that produced the captured Tier 4 numbers. **Plan 07-02's manifest is the version of record once Plan 07-03 runs against it.** Subsequent rebuilds must NOT happen between Plan 07-02 and Phase 9's freeze.

### Single-date invariant under long wall

The user might start the sweep at 23:00 UTC and have it finish at 02:00 UTC the next day. Per the success criterion: "queries/tier-{N}-{TS}.json ... with timestamps within the same date". Pipeline.py captures `sweep_ts = run._ts()` ONCE at top and threads it through all 5 tiers' QueryLog timestamps via `ts_override`. **All 5 QueryLog timestamps will be IDENTICAL** (same ISO 8601 string, with second precision). The "same date" check trivially holds.

**However**, the per-tier *cost JSONs* are written by `tracker.persist()` which uses its own ISO timestamp (NOT `ts_override` — verified in score.py:518 + run.py:273; v1.1 hardening item per Plan 05-02). For a sweep that crosses midnight UTC, the per-tier cost JSONs may have date components from both days. The `git_sha` in each cost JSON IS consistent (cost JSONs don't carry git_sha; they carry `tier` + `timestamp` + `queries[]` + `totals`). For CAP-01: the ROADMAP says "git SHA recorded in every per-tier output JSON" — this maps to the QueryLog files (5 capture + 5 metrics; metrics JSON doesn't carry SHA either, only queries does). The user-facing claim in Success Criterion 2 ("grep all 10 output JSONs ... find the same git SHA recorded in each") is technically only meaningful for the 5 capture JSONs (queries) since cost+metrics JSONs don't carry git_sha. **Plan 07-03's verifier must be honest about this**: grep over `.git_sha` in the 5 queries files, NOT all 10. The claim "no SHA drift across tiers" is provable on 5/10 JSONs.

Recommend the plan rewords this as "single git SHA recorded in every per-tier QueryLog (5 files)". The `--freeze` manifest from Phase 4 captures the sweep SHA for posterity and is the authoritative cross-tier provenance, but Phase 7 doesn't run `--freeze`.

### Tier ordering matters?

Pipeline.py sorts `tiers = sorted(int(t) for t in args.tiers.split(","))`, so `--tiers 5,3,1` becomes `[1, 3, 5]`. Captures run sequentially in tier-number order. Score stage iterates the same order. Compare stage hard-codes "1,2,3,4,5". **Tier ordering does NOT matter for correctness**; it only affects when each tier's API spend hits the cost ledger. If the user wants Tier 3 (the most expensive) to run first to fail-fast on rate limits, they cannot reorder via pipeline.py — they would need to run `pipeline --tiers 3 --yes` first to validate, then `pipeline --tiers 1,2,3,4,5 --yes` for the full sweep. **This is a fail-fast pattern worth recommending in the plan**: a "smoke-tier-3-first" task step before the full sweep.

### Parallelization within budget?

Currently sequential. Per Plan 05-01 D-Q4: out of scope for v1.0. Tiers 1, 2, 5 are cheap and IO-bound; could parallelize. Tier 3 is the expensive one (LightRAG hybrid mode = many LLM calls per question). Tier 4 is passthrough (no live work). **For Phase 7 v1.0, accept sequential.** Total wall time for the sweep itself is bounded by ~15-30 min (Phase 5 Plan 05-02 measured Tier 5 × 5q = 158s wall; extrapolating to 30q × 5 tiers ≈ ~15-30min wall depending on Tier 3 hybrid mode latency).

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 7 OPERATIONAL FLOW                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Plan 07-01: PRE-FLIGHT (host shell, sandbox-permitted)
  ┌─────────────────────────────────────────────────────────────┐
  │ Gate 1-5: env + storage + .store_id (recreate if missing)   │
  │   ↓                                                          │
  │ Gate 6: 5q smoke × 5 tiers → smoke_gate verdict=PASS×5      │
  │   ↓                                                          │
  │ Gate 7: git status clean (recommend, not strict)             │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ↓ (PASS×5)
Plan 07-02: TIER 4 FULL REBUILD (host shell, NOT sandbox; cost-ack required)
  ┌─────────────────────────────────────────────────────────────┐
  │ shutil.rmtree rag_anything_storage/tier-4-multimodal/        │
  │   ↓                                                          │
  │ ingest_from_mineru.py --reset --yes  (~15-25h, $15-35)       │
  │   over 79-papers MineRU cache                                │
  │   ↓                                                          │
  │ log_graph_stats.py → tier-4-graph-stats-{ts}.json           │
  │   ↓                                                          │
  │ tier-4 5q smoke against new graphml → smoke_gate PASS        │
  │   ↓                                                          │
  │ Commit: tier-4-graph-stats-{ts}.json + SUMMARY               │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ↓ (smoke PASS)
Plan 07-03: LIVE FULL SWEEP (host shell; cost-ack required)
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 1: eval_capture.py --yes (Tier 4 30q live, ~$0.045)     │
  │   → queries/tier-4-{ts1}.json  (carries HEAD git_sha)        │
  │   ↓                                                          │
  │ Step 2: pipeline.py --tiers 1,2,3,4,5                        │
  │            --tier-4-from-cache <path> --yes  (~$0.79)        │
  │   ↓ run.amain captures sweep_sha + sweep_ts ONCE             │
  │   ↓ Tiers 1, 2, 3, 5: live capture writes new QueryLog       │
  │     each carrying git_sha=sweep_sha, timestamp=sweep_ts      │
  │   ↓ Tier 4: passthrough reads cached records, rewrites a     │
  │     NEW tier-4-{sweep_ts}.json with git_sha=sweep_sha        │
  │     (verify: read run.py:233-262 carefully)                  │
  │   ↓ score.amain: 5 tiers × 30q × 3 metrics → metrics + judge │
  │   ↓ compare._run: regenerates comparison.md (5-tier rollup   │
  │     + per-class table + embedder-by-tier table)              │
  │   ↓                                                          │
  │ Step 3: verify_sweep helper (phase-local script or inline)   │
  │   - all 5 tier QueryLogs share single git_sha               │
  │   - all 5 share single date                                  │
  │   - tier-4 + tier-5 NaN counts < 5/30                        │
  │   - sum of cost JSONs ≤ $3                                   │
  │   - all 5 QueryLogs have 30 records                          │
  │   ↓                                                          │
  │ Step 4: human-verify checkpoint (read comparison.md +        │
  │   embedder table + graph stats manifest)                     │
  │   ↓                                                          │
  │ Commit: SUMMARY (no source code changes)                     │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ↓
                   CAP-01 CLOSED → Phase 8 (multi-judge)
```

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4 (per pyproject `[dependency-groups].test`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (markers only; no addopts) |
| Quick run command | `pytest evaluation/tests/ -m 'not live' -x` |
| Full suite command | `pytest evaluation/tests/ -m 'not live' -x` (offline only — Phase 7 should not introduce new live tests beyond the pre-flight smokes already invoked outside pytest) |

### Phase Requirements → Test Map

Phase 7 is largely a no-test-file phase: the verification is via running existing harness modules + grep'ing JSON outputs. The "tests" are operational gates, not pytest functions.

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CAP-01 | Pre-flight 5q smoke gates per tier | manual-cli (existing) | `python -m evaluation.harness.run --tiers <t> --smoke-question-ids ...; python -m evaluation.harness.score --tiers <t>; python -m evaluation.harness.smoke_gate --tier <t>` | ✅ existing |
| CAP-01 | Pipeline sweep produces single-SHA captures | manual-cli (existing) | `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --tier-4-from-cache <path> --yes` | ✅ existing (pipeline.py shipped Phase 5) |
| CAP-01 | Single-SHA + same-date verification across 5 QueryLogs | bash + jq OR phase-local script | `for t in 1 2 3 4 5; do jq -r '.git_sha' "$(ls -t evaluation/results/queries/tier-${t}-*.json \| head -1)"; done \| sort -u \| wc -l` (must equal 1) | ❌ NEW (phase-local script or inline) |
| CAP-01 | NaN count per tier ≤ 5/30 | bash + jq | `jq '[.[] \| select(.nan_reason != null)] \| length' "$(ls -t evaluation/results/metrics/tier-{4,5}-*.json \| head -1)"` (must be < 5) | ❌ NEW (phase-local script or inline) |
| CAP-01 | Total spend ≤ $3 across 10 cost JSONs | bash + jq | `jq -s '[.[] \| .totals.usd] \| add' <freshest 10 cost files>` (must be ≤ 3.0) | ❌ NEW (phase-local script or inline) |

### Sampling Rate

- **Per task commit:** Tasks in Plans 07-01/02/03 are operational, not code-bearing. After each operational step that produces a JSON artifact, run the relevant verification one-liner from Section "Step 3 — Post-sweep verification".
- **Per wave merge:** `pytest evaluation/tests/ -m 'not live' -x` to confirm no offline regression introduced by any phase-local script or accidental harness edit.
- **Phase gate:** Plan 07-03 Step 4 human-verify checkpoint is the formal gate before `/gsd-verify-phase`.

### Wave 0 Gaps

- [x] No new test files needed for the existing harness (already covered by Phases 1-6 test surface).
- [ ] **(Optional)** Phase-local verifier script: `tier-4-multimodal/scripts/verify_sweep.py` (~50 LOC) per Section "Step 3" code sketch. Justification: bash + jq one-liners are brittle for the freshest-per-tier glob pattern; a Python helper using `Path.glob` + mtime sort is more reliable.
- [ ] **(Optional)** Smoke-test for verify_sweep.py: `evaluation/tests/test_verify_sweep.py` if the script lands. ~20 LOC, offline-only via tmp_path fixtures. Skip if the verification stays inline bash.

## Common Pitfalls

### Pitfall 1: Running pipeline.py BEFORE tier-2-managed/.store_id exists

**What goes wrong:** `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 ...` exits 2 immediately on the prereq check (`run.py:119-122`). User has paid the cost-surprise prompt time but no API spend; still annoying.

**Why it happens:** `.store_id` is gitignored and may not exist in the user's working tree.

**How to avoid:** Plan 07-01 Gate 3 explicitly tests `test -f tier-2-managed/.store_id` and runs `python tier-2-managed/main.py --ingest --yes` if missing.

**Warning signs:** `[red]tier-2-managed/.store_id missing.[/red]` in pipeline.py output.

### Pitfall 2: Committing changes between Tier 4 eval_capture.py and pipeline.py invocation

**What goes wrong:** `eval_capture.py` writes `git_sha=<HEAD_a>` into tier-4 QueryLog. User makes a fix, commits. Pipeline.py reads HEAD which is now `<HEAD_b>` and threads `<HEAD_b>` as sweep_sha. Tier 4's NEW QueryLog (rewritten by passthrough) carries `<HEAD_b>`, but the user might think it carries `<HEAD_a>` because the *records* are from `<HEAD_a>`-time capture. The records and the QueryLog SHA disagree.

**Why it happens:** Git is fast; capture commands take time. Easy to slip a commit between them.

**How to avoid:** Plan 07-03 Step 1 runs `eval_capture.py`, then Step 2 IMMEDIATELY runs pipeline.py without intervening commits. Document this as a critical sequencing constraint.

**Warning signs:** Post-sweep verification's "single SHA across QueryLogs" check would still PASS (because passthrough rewrites tier-4's SHA to match sweep_sha), but the underlying records were captured at a different SHA. The CAP-01 invariant holds at the QueryLog level but the *semantic* invariant is subtly broken — the answer/contexts in tier-4 records were generated against an older code state. If pipeline.py's run.py edits to Tier 4 adapter happened between capture and pipeline.py invocation, this matters; if not, it's cosmetic.

### Pitfall 3: Tier 4 graphml drift between rebuild and freeze

**What goes wrong:** User runs Plan 07-02 (rebuild), then Plan 07-03 (sweep), then `ingest_from_mineru.py --reset --yes` AGAIN (e.g. testing a hypothesis). New graphml has different node/edge counts. Phase 9 freezes a manifest citing the new graph stats, but `comparison.md` was generated against the old graph.

**How to avoid:** Document in Plan 07-02 SUMMARY: "DO NOT rerun `ingest_from_mineru.py --reset` between Plan 07-03 and Phase 9 freeze". Phase 9's freeze captures the manifest cited; if a re-rebuild is desired (e.g. for v1.1 reruns), the freeze is for v1.1, not v1.0.

**Warning signs:** `tier-4-graph-stats-*.json` glob in `evaluation/results/diagnostics/` shows multiple manifests with different node counts.

### Pitfall 4: Vision-LLM 400 errors during Tier 4 rebuild interpreted as fatal

**What goes wrong:** User sees `400 Invalid URL format` in red during ingest_from_mineru.py and aborts with ctrl-C, thinking the rebuild is broken.

**Why it happens:** OpenRouter's URL length limit is hit by base64-encoded large figures. RAG-Anything continues processing other items; the error is non-fatal.

**How to avoid:** Document in Plan 07-02 task description: "expect ~5-10% multimodal-item dropout from vision-LLM 400 errors; non-fatal; do NOT abort the run". STATE.md's Blockers section already lists this; the plan should restate it inline.

**Warning signs:** Red `400` errors interspersed with normal progress; final ingested_count <79 but >70.

### Pitfall 5: Cost-tracker writes to default dir instead of --results-dir

**What goes wrong:** Plan 07-03 wants to use a tmp_path or alternate results-dir. Pipeline.py honors `--results-dir` for queries+metrics, but `tracker.persist()` writes to the hardcoded `shared.cost_tracker.DEFAULT_COSTS_DIR = Path('evaluation/results/costs')` regardless. Cost JSONs land in the wrong directory.

**Why it happens:** v1.1 hardening item per Plan 05-02 SUMMARY. `run.py:273` and `score.py:518` call `tracker.persist()` with no `dest_dir` arg.

**How to avoid:** Plan 07-03 uses the default `--results-dir evaluation/results` (do NOT customize). All artifacts land in the canonical location. Document the limitation in the plan.

**Warning signs:** `evaluation/results/queries/` has fresh files but `evaluation/results/costs/` doesn't (or vice versa).

### Pitfall 6: `pytest -v` runs live tests silently

**What goes wrong:** Verifier runs `pytest evaluation/tests/test_eval_pipeline.py -v` to confirm no regression; this silently invokes the `@pytest.mark.live` test, paying real API cost (~$0.014 per Plan 05-02 measurement).

**Why it happens:** `pyproject.toml` declares the `live` marker but does NOT include `addopts = "-m 'not live'"`. v1.1 hardening item.

**How to avoid:** Plan 07-* verification commands MUST always include `-m 'not live'`. If a live invocation is needed, use `-m live -k <test_name>` explicitly with cost-ack.

**Warning signs:** Unexpected OpenRouter charges in the daily cost report.

### Pitfall 7: Misreading Tier 4 capture as live in pipeline.py

**What goes wrong:** Reader of pipeline.py thinks "--tiers 4" triggers a live in-process Tier 4 capture. Actually, `run.py:233-247` returns None with a yellow skip notice if `args.tier_4_from_cache` is unset.

**Why it happens:** Phase 130 SC-1 deferred Tier 4 to user-side eval_capture.py.

**How to avoid:** Plan 07-03 docstring + plan-frontmatter explicitly: "Tier 4 in --tiers REQUIRES --tier-4-from-cache pointing at a freshly-written eval_capture.py output. Pipeline.py does NOT live-capture Tier 4."

**Warning signs:** `[yellow]Tier 4 SKIPPED — no --tier-4-from-cache supplied.[/yellow]`

### Pitfall 8: Stale tier-N-from-cache files

**What goes wrong:** User passes `--tier-4-from-cache evaluation/results/queries/tier-4-2026-05-02T17_44_57Z.json` (the OLD pre-rebuild capture). Pipeline rewrites it as the current Tier 4 capture, but the records were generated against the OLD 3-paper graph, not the new 79-paper one.

**Why it happens:** Many tier-4-*.json files exist in queries/ from prior phases; easy to grab the wrong one.

**How to avoid:** Plan 07-03 Step 1 captures freshly; Step 2 explicitly resolves the freshest via `TIER4_PATH=$(ls -t evaluation/results/queries/tier-4-*.json | head -1)`. This guarantees the freshest mtime is used.

**Warning signs:** Tier 4 NaN count high (because records were against 3-paper graph; many questions reference papers not in graph). If post-sweep verification shows NaN > 5/30 on tier-4, suspect stale cache.

## Code Examples

### Pre-flight verification one-liners (Plan 07-01)

```bash
# All-tier env + storage gate
test -n "$OPENROUTER_API_KEY" && echo "OK: OPENROUTER_API_KEY set"
test -n "$GEMINI_API_KEY" && echo "OK: GEMINI_API_KEY set"
test -d chroma_db/tier-1-naive && echo "OK: chroma_db/tier-1-naive exists"
test -f tier-2-managed/.store_id && echo "OK: tier-2 .store_id exists" \
    || python tier-2-managed/main.py --ingest --yes
test -d lightrag_storage/tier-3-graph && echo "OK: lightrag_storage/tier-3-graph exists"
test -d rag_anything_storage/tier-4-multimodal && echo "OK: rag_anything_storage/tier-4-multimodal exists"

# Per-tier 5q smoke loop (offline-fast for tiers 1, 2; live-slow for 3, 4, 5)
for t in 1 2 3 5; do
    python -m evaluation.harness.run --tiers $t \
        --smoke-question-ids single-hop-001,single-hop-002,single-hop-003,multi-hop-001,multi-hop-002 \
        --yes
    python -m evaluation.harness.score --tiers $t --yes
    python -m evaluation.harness.smoke_gate --tier $t || { echo "FAIL: tier $t"; exit 1; }
done
# Tier 4 separately via eval_capture.py
python tier-4-multimodal/scripts/eval_capture.py \
    --smoke-question-ids single-hop-001,single-hop-002,single-hop-003,multi-hop-001,multi-hop-002 \
    --yes
python -m evaluation.harness.score --tiers 4 --yes
python -m evaluation.harness.smoke_gate --tier 4 || { echo "FAIL: tier 4"; exit 1; }
```

### Tier 4 full rebuild (Plan 07-02 — host shell, NOT sandbox)

```bash
# Verify NOT in sandbox (parse_missing_papers.py uses the same heuristic)
[[ "$ALL_PROXY" != *socks5h* ]] && [[ -z "$CLAUDE_CODE_SANDBOX" ]] || \
    { echo "ABORT: in sandbox; rebuild must run on host"; exit 1; }

# Verify cache state
PAPER_COUNT=$(ls tier-4-multimodal/output/ | wc -l)
[[ "$PAPER_COUNT" -ge 75 ]] || { echo "ABORT: cache has $PAPER_COUNT papers, need ≥75"; exit 1; }

# Cost-ack checkpoint (manual): user reads this and confirms
echo "About to rebuild Tier 4 graph from $PAPER_COUNT papers."
echo "Estimated wall: 15-25h. Estimated cost: \$15-35."
echo "Press Ctrl-C now to abort, or wait 30s..."; sleep 30

# Run ingest
python tier-4-multimodal/scripts/ingest_from_mineru.py --reset --yes

# Capture provenance
python tier-4-multimodal/scripts/log_graph_stats.py
# → evaluation/results/diagnostics/tier-4-graph-stats-{ts}.json

# Smoke verification
python tier-4-multimodal/scripts/eval_capture.py \
    --smoke-question-ids single-hop-001,single-hop-002,single-hop-003,multi-hop-001,multi-hop-002 \
    --yes
python -m evaluation.harness.score --tiers 4 --yes
python -m evaluation.harness.smoke_gate --tier 4
# Verdict must be PASS

# Commit the new manifest (storage is gitignored, but the diagnostics file is committed)
git add evaluation/results/diagnostics/tier-4-graph-stats-*.json
git commit -m "feat(07-02): Tier 4 full-corpus rebuild — N nodes / E edges from 79 papers"
```

### Live full sweep (Plan 07-03)

```bash
# Step 1: Tier 4 live capture (must run before pipeline.py)
python tier-4-multimodal/scripts/eval_capture.py --yes
TIER4_PATH=$(ls -t evaluation/results/queries/tier-4-*.json | head -1)
echo "Tier 4 capture: $TIER4_PATH"

# Step 2: Pipeline sweep (no commits between Step 1 and Step 2!)
python -m evaluation.harness.pipeline \
    --tiers 1,2,3,4,5 \
    --tier-4-from-cache "$TIER4_PATH" \
    --yes
# (pipeline.py prints sweep_sha + sweep_ts banner; one cost-prompt; ~30-60min wall)

# Step 3: Post-sweep verification (phase-local script or inline)
python tier-4-multimodal/scripts/verify_sweep.py
# OR inline: see Section "Post-Sweep Verification" code blocks

# Step 4: Human-verify checkpoint
cat evaluation/results/comparison.md | head -50
cat evaluation/results/diagnostics/tier-4-graph-stats-*.json | jq '.graphml_node_count, .graphml_edge_count' | head -2
# User reads, confirms numbers plausible, marks checkpoint approved
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sweep orchestration | A new orchestrator script | `python -m evaluation.harness.pipeline` | Phase 5 shipped this; it's the contract for HARN-01. Re-implementing breaks D-Q4 (compare always rolls all 5 tiers). |
| Tier 4 ingest | Custom MineRU subprocess loop | `tier-4-multimodal/scripts/ingest_from_mineru.py` | Phase 2 shipped with bypass for the parser-version probe + OPENROUTER_API_KEY env forwarding. Re-implementing re-introduces those traps. |
| Graph provenance | Manually noting node/edge counts | `tier-4-multimodal/scripts/log_graph_stats.py` | Pydantic-typed; Phase 9 reads via `GraphStats(**json.load(f))`. Manual = schema drift. |
| Smoke gate verdict | Custom thresholds + counts | `evaluation/harness/smoke_gate.py::evaluate_smoke_from_paths` | Phase 1 D-04 contract: ratio≥0.8 measurable + non-NaN faithfulness/context_precision. Reusing across tiers means one threshold across the project. |
| Cost-prompt UX | New prompts in plan-task scripts | `pipeline.py --yes` honors via `args.yes` plumb-through | Phase 5 D-Q4: single cost prompt per sweep. Don't fragment. |
| Single-SHA propagation | Env-var hack | `pipeline.py`'s existing `git_sha_override` kwarg | Phase 5 Plan 05-01 D-Q1 locked the targeted-kwarg approach. Don't introduce env-var coupling. |
| NaN reason classification | Re-parsing RAGAS exceptions | `score.py::NaNReasonTracer` + `_classify_post_evaluate_nan` | Phase 3 closed HARN-05; reasons are structured. Plan 07-03 verification reads `nan_reason` directly from metrics JSON. |
| Embedder-by-tier table | Authoring from memory | `compare.py::emit_markdown` (Phase 6 D-Q1) | Already auto-emits the table from QueryLog provenance. Phase 9 is pure copy-paste. |

**Key insight:** Phase 7 reuses everything. The only "new code" justified is a ~50-LOC phase-local `verify_sweep.py` if the post-sweep verification grows beyond 4 inline checks. Even that is optional.

## Runtime State Inventory

This is an operational phase that touches significant runtime state. Explicit answers per category:

| Category | Items | Action Required |
|----------|-------|-----------------|
| Stored data — ChromaDB | `chroma_db/tier-1-naive/` (Tier 1 + Tier 5 read here) | None — verified present; preserve. |
| Stored data — OpenAI File Search | OpenAI-side store referenced by `tier-2-managed/.store_id` | **MISSING locally** — `.store_id` not on disk. **Plan 07-01 must recreate** via `python tier-2-managed/main.py --ingest --yes`. May incur OpenAI File Search cost (~$0.10–0.50). |
| Stored data — LightRAG Tier 3 | `lightrag_storage/tier-3-graph/` (graphml + 8 kv_store_*.json + 3 vdb_*.json) | None — verified present; preserve. |
| Stored data — LightRAG Tier 4 | `rag_anything_storage/tier-4-multimodal/` (currently 3-paper smoke graphml; will be wiped + rebuilt to 79-paper full graph in Plan 07-02) | **REBUILD** in Plan 07-02 via `ingest_from_mineru.py --reset --yes`. The 3-paper graph is destroyed. |
| Stored data — Mem0/Redis/etc | None — project doesn't use these | None. |
| Live service config — OpenRouter | API key in `.env` | None — verified `.env` carries `OPENROUTER_API_KEY`. |
| Live service config — OpenAI File Search | Store created by `tier-2-managed/main.py --ingest`; lives server-side at OpenAI | Plan 07-01 recreates. The remote store is created on demand; no manual API console step needed. |
| Live service config — Gemini Direct | API key in `.env` | None — verified `.env` carries `GEMINI_API_KEY`. |
| OS-registered state | None — no Task Scheduler / launchd / systemd registrations | None. |
| Secrets/env vars | `OPENROUTER_API_KEY`, `GEMINI_API_KEY` (both in `.env`) | None — both present. |
| Build artifacts / installed packages | `.venv/` with the 4 critical libs at pinned versions (`lightrag-hku==1.4.15`, `raganything==1.2.10`, `openai-agents==0.14.6`, `ragas>=0.4.3`, `mineru==3.1.4`, `litellm>=1.0`) | None — verified pinned in pyproject.toml. **DO NOT bump between Plan 07-02 rebuild and Phase 9 freeze**, or the manifest provenance breaks. |
| Diagnostics manifests committed | `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json` (3-paper smoke) | Plan 07-02 writes a NEW manifest alongside (does NOT overwrite). Both are part of the historical record. |
| Eval results queries+costs+metrics | `evaluation/results/queries/`, `costs/`, `metrics/` (all gitignored except costs/.gitkeep) | Plan 07-03 sweep produces 5 fresh queries + 5 fresh metrics + 10 fresh costs JSONs. Old files remain on disk (mtime resolution picks freshest). User may want to manually clear stale files post-Phase-9 to keep the directory tidy, but that's housekeeping, not a phase requirement. |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python 3.10+ | All harness | ✓ | 3.13.x (per Phase 4 Plan 04-01 STATE) | — |
| OpenRouter API key | Tiers 1, 3, 4, 5 + judge | ✓ | — | Skip via `_check_prereqs`; tier exits 2 |
| Gemini API key | Tier 2 | ✓ | — | Skip via `_check_prereqs`; tier exits 2 |
| OpenAI File Search store (`.store_id`) | Tier 2 | ✗ MISSING | — | Recreate via `tier-2-managed/main.py --ingest --yes` |
| chroma_db/tier-1-naive/ | Tiers 1, 5 | ✓ | — | Recreate via `tier-1-naive/main.py --ingest` |
| lightrag_storage/tier-3-graph/ | Tier 3 | ✓ | — | Recreate via `tier-3-graph/main.py --ingest --yes` |
| rag_anything_storage/tier-4-multimodal/ | Tier 4 | ✓ (3-paper smoke) | 2886 nodes / 7056 edges | Plan 07-02 rebuilds to 79-paper |
| MineRU CLI on host | Plan 07-02 ingest (sandbox-blocked) | ✓ on host (per Plan 02-02) | mineru==3.1.4 | None — must run on host. |
| MineRU output cache | Plan 07-02 ingest input | ✓ | 79 papers under tier-4-multimodal/output/ | Re-parse missing via `parse_missing_papers.py` |
| `lightrag-hku==1.4.15` | All tiers + ingest | ✓ | 1.4.15 | none |
| `raganything==1.2.10` | Tier 4 + ingest | ✓ | 1.2.10 | none |
| `openai-agents==0.14.6` | Tier 5 | ✓ | 0.14.6 | none |
| `ragas>=0.4.3,<0.5` | Score | ✓ | 0.4.3 | none |
| `litellm>=1.0,<2` | Score (judge LLM/embedder) | ✓ | 1.83.0 (per Phase 4 manifest) | none |
| `chromadb==1.5.8` (BONUS) | Tier 1 | ✓ | 1.5.8 | none |

**Missing dependencies with no fallback:** None — `.store_id` is recreatable.

**Missing dependencies with fallback:**
- `tier-2-managed/.store_id` — recreate via `main.py --ingest`. Plan 07-01 must include this as a task step.

## Open Questions

1. **Does pipeline.py's Tier 4 passthrough write a NEW QueryLog with sweep_sha, or read+forward the cached one verbatim?**
   - What we know: `run.py:233-262` is the Tier 4 branch. It calls `run_tier4(... from_cache=cache_path, tracker=tracker)` per question. After the loop, it constructs a new `QueryLog(timestamp=..., git_sha=..., model=..., embedder=..., embedder_source=..., records=records)` and `write_query_log` to disk.
   - What's unclear without running it: whether the records loaded from cache get re-tracked through `tracker` (paying double-counted cost on capture-cost JSON?) and whether the OUTPUT path uses `_ts_for_filename(timestamp)` where timestamp is sweep_ts (=`ts_override`) or the cached-record timestamp.
   - Recommendation: Plan 07-01 Task includes a smoke step `pipeline.py --tiers 4 --tier-4-from-cache <existing-tier-4-file> --limit 5 --yes` and inspects the resulting tier-4-{ts}.json: does it carry HEAD's git_sha? Does cost JSON carry zero-or-real spend? Verify before paying for the full sweep.

2. **What is the realistic cost ceiling for Plan 07-02's Tier 4 full rebuild?**
   - What we know: Plan 02-01 measured 3-paper smoke at $1.89. STATE.md projects 79 papers at $15-35.
   - What's unclear: vision-LLM 400 errors mean some multimodal items are dropped → cost SAVED on those items. But the entity-extraction LLM passes are the bottleneck and apply to all items. STATE.md's $15-35 projection includes a 2× safety margin.
   - Recommendation: cost-ack checkpoint in Plan 07-02 quotes "$15-35; OpenRouter monthly budget should accommodate". User decides whether to proceed.

3. **What is the expected Tier 4 NaN count post-rebuild?**
   - What we know: Plan 02-04 closed gap with JUDGE_MAX_TOKENS=8192; Tier 4 5q smoke = 5/5 faithfulness=1.0 (zero NaN). Full 30-question run could see NaN from questions whose source papers are in the 4 host-parsed papers (1909.01066, 2002.06177, 2309.15217, 2410.05779) which were NOT in the 75-paper smoke but ARE in the 79-paper rebuild.
   - What's unclear: whether the 4 newly-ingested papers produce graph-quality regressions. Plan 02-02 noted the MineRU output layout is slightly different (no intermediate `<8charhash>` segment); ingest_from_mineru.py's `_find_content_list` uses rglob so this should be transparent.
   - Recommendation: Plan 07-02 smoke gate after rebuild reads the same 5q smoke set; PASS verdict gates Plan 07-03. If smoke FAILs, investigate before sweep.

4. **Should the verifier helper be a Python script or inline bash?**
   - What we know: Phase 6 lesson — "static-source assertion test" pattern + "don't add layers when bash is enough". Phase 5 lesson — sketch-LOC validate before locking.
   - What's unclear: how the user prefers to operate. Inline bash is fine for one-shot checks; Python is reusable across reruns.
   - Recommendation: write inline bash for v1.0 of Plan 07-03. Promote to Python helper if the user runs the sweep 2-3 times and the inline blocks become repetitive.

## Sources

### Primary (HIGH confidence — read directly)

- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/pipeline.py` (190 LOC) — Phase 5 driver
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/run.py` (421 LOC) — capture stage with sweep-SHA kwargs
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/score.py` (580 LOC) — RAGAS scorer
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/compare.py` (425 LOC) — rollup with embedder table
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/freeze.py` (97 LOC) — frozen-doc writer (NOT used in Phase 7)
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/smoke_gate.py` (299 LOC) — D-04 verdict for tier smokes
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/records.py` (89 LOC) — Pydantic schemas
- `/Users/patrykattc/work/git/rag-architecture-patterns/tier-4-multimodal/scripts/ingest_from_mineru.py` (398 LOC) — Tier 4 rebuild helper
- `/Users/patrykattc/work/git/rag-architecture-patterns/tier-4-multimodal/scripts/log_graph_stats.py` (~280 LOC) — provenance writer
- `/Users/patrykattc/work/git/rag-architecture-patterns/tier-4-multimodal/scripts/parse_missing_papers.py` — host-side MineRU helper (sandbox detection pattern reusable)
- `/Users/patrykattc/work/git/rag-architecture-patterns/tier-4-multimodal/scripts/eval_capture.py` (244 LOC) — Tier 4 live-capture entry point
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/ROADMAP.md` — Phase 7 success criteria + decimal-phase progress
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/REQUIREMENTS.md` — CAP-01 verbatim + traceability
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/STATE.md` — accumulated decisions + blockers (Tier 4 cost projection $15-35; vision-LLM 400 issue; pytest live-deselect gap)
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/phases/02-tier-4-graphml-regeneration/02-01-SUMMARY.md` — 3-paper smoke rebuild metrics + Pitfall 4/Pitfall 6 references
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/phases/02-tier-4-graphml-regeneration/02-02-SUMMARY.md` — 4 host-parsed papers (1909.01066, 2002.06177, 2309.15217, 2410.05779) — Phase 7 cache state
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/phases/05-pipeline-driver/05-RESEARCH.md` — pipeline.py composition contract + 14 pitfalls
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/results/comparison.md` (current) — 5-tier rollup with tier-4 + tier-5 at 30/30 NaN baseline (regenerable post-sweep)
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json` — 3-paper smoke ground truth (will be augmented by Plan 07-02's full-corpus manifest)
- `/Users/patrykattc/work/git/rag-architecture-patterns/.gitignore` — `tier-2-managed/.store_id` confirmed gitignored; `evaluation/results/queries/` + `metrics/` gitignored
- `/Users/patrykattc/work/git/rag-architecture-patterns/pyproject.toml` — pinned versions (lightrag-hku==1.4.15, raganything==1.2.10, openai-agents==0.14.6, mineru==3.1.4, ragas>=0.4.3, litellm>=1.0)

### Secondary (referenced)

- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/golden_qa.json` — 30 questions, 19 single-hop / 11 multi-hop, 20 text / 10 multimodal, 18 unique source papers
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/tests/conftest.py` — fixtures `live_eval_keys_ok`, `tier1_index_present`, `tier4_storage_present` (reusable in any phase-local test)
- Existing tier-{1..5} capture+metrics+cost JSONs from prior phases (regenerable; mtime resolution picks freshest)

### Tertiary

- None. No external library research required — Phase 7 is pure operational orchestration of repo-internal code.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Pipeline.py's Tier 4 passthrough writes a NEW QueryLog with `git_sha=sweep_sha` and `timestamp=sweep_ts` (NOT verbatim cached values). | Sweep Architecture Step 2 + Open Question 1 | If wrong, the single-SHA invariant breaks — tier-4 QueryLog carries the eval_capture.py SHA, not pipeline's sweep SHA. Verifier must check this on the first sweep; if confirmed broken, the plan needs a code-edit task to fix. **Plan 07-01 should include a low-cost smoke step to verify** (~$0.01 for a 5q tier-4 passthrough test). [ASSUMED — confirm by reading run.py:233-262 line by line OR by running a smoke] |
| A2 | The 79-paper Tier 4 rebuild costs $15-35 and takes 15-25h wall. | Cost Envelope Analysis | If wall time exceeds 30h or cost exceeds $50, user may abort and request a smaller-corpus rebuild. The 79-paper rebuild is the largest operation in v1.0; if it overshoots, scope decisions must be revisited. [ASSUMED — extrapolation from 3-paper smoke; first full rebuild is the actual measurement] |
| A3 | The `tier-2-managed/main.py --ingest --yes` step costs <$0.50. | Pre-Sweep Gating Sequence Gate 3 | If higher (e.g. $2-5 for 100 papers via OpenAI File Search), the cost-ack checkpoint must surface this. [ASSUMED — based on OpenAI File Search pricing of ~$0.10/GB indexed; 100-paper PDF corpus is ~50-100MB; ceiling ~$0.50] [VERIFIED: pricing reference at openai.com/api/pricing — File Search storage pricing] |
| A4 | Total spend across all 10 Phase 7 cost JSONs (5 capture + 5 judge) sits at ~$0.84 for one full sweep, fitting the $1-3 envelope. | Cost Envelope Analysis | If actual spend exceeds $3, the sweep is over budget per Success Criterion 5. The breakdown (cap $0.34 + judge $0.45 + tier4-capture $0.045) has slack; Tier 3's $0.30 is the largest variable. [ASSUMED — extrapolation from Phase 5 Plan 05-02 measured Tier 5 5q at $0.007 + run.COST_PER_Q + judge factor; first full-30q-sweep validates] |
| A5 | The 4 host-parsed papers (1909.01066, 2002.06177, 2309.15217, 2410.05779) ingest cleanly via `ingest_from_mineru.py` despite a slightly different MineRU output directory layout (no intermediate `<8charhash>` segment per Plan 02-02 SUMMARY). | Tier 4 Full-Corpus Rebuild | If `_find_content_list` (which uses `**/*_content_list.json` rglob) fails on these papers, ingest count drops to 75. Plan 02-02 SUMMARY claims rglob handles this transparently; verify on the first rebuild. [ASSUMED — Plan 02-02 SUMMARY explicit claim; not yet exercised by an end-to-end ingest] |
| A6 | The Phase 7 sweep should NOT freeze. Phase 9 owns freeze. | Plan structure | If user wants to freeze a v0.x interim doc after Plan 07-03 (e.g. for a partial blog draft), the plan should expose `--freeze v0.7` as an option. Recommendation: don't expose; Phase 9 is the canonical freeze point. [ASSUMED — ROADMAP and STATE consistent on this] |
| A7 | The `_capture_tier` adapter-exception swallowing semantics (Plan 05-01 Pitfall 2) means partial-tier failures pass silently. | Failure Modes & Recovery | This is a known harness limitation; Plan 07-03's record-count check (every tier has 30 records) catches it post-sweep but doesn't prevent it mid-sweep. Acceptable for v1.0; tracked in Plan 05-01 as v1.1 hardening. [VERIFIED — read run.py:316-318] |
| A8 | The 30-question golden_qa.json is locked at HEAD (no mid-sweep edits). | All sections | If the golden_qa is edited mid-sweep (e.g. typo fix to a question), some tiers' captures reference the old text and others the new. v1.0 scope explicitly locks 30 questions per REQUIREMENTS.md "Out of Scope" section. [VERIFIED — REQUIREMENTS.md "Adding new questions to the golden set | 30 is locked"] |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-tier independent SHA via lazy `_git_sha()` calls inside `_capture_tier` | Single sweep SHA via `git_sha_override` kwarg threaded from pipeline.py | Phase 5 Plan 05-01 (2026-05-06) | CAP-01's "single git SHA" invariant is now mechanically enforceable |
| Manual aggregation of per-tier numbers into comparison.md | `compare._run` reads queries+costs+metrics via `_latest()` mtime + writes 9-column rollup + 7-column class table + embedder-by-tier table | Phase 4 + Phase 6 (2026-05-05/06) | Phase 9's freeze is pure copy-paste; no Phase 9 logic for embedder disclosure (Phase 6 D-Q1) |
| RAGAS NaN as opaque single field | Per-row `nan_reason` ∈ {empty_contexts, empty_statements, json_parse_failure, ...} | Phase 3 (2026-05-05) | Phase 7 verification can grep `.nan_reason` for structured root-cause counts; Phase 9 freezes the breakdown |
| Tier 5 hard-coded `retrieved_contexts=[]` | Walk `RunResult.new_items` for `ToolCallOutputItem` | Phase 1 (2026-05-04) | Tier 5 NaN target dropped from 30/30 to <5/30 (Phase 7 Success Criterion 4) |
| Tier 4 graph from arbitrary partial state | Wipe-and-rebuild via `ingest_from_mineru.py --reset --yes` from MineRU cache | Phase 2 (2026-05-05; smoke-only deferred to Phase 7) | Phase 7's "100-paper corpus" claim becomes defensible after Plan 07-02 |
| 75-paper MineRU cache (some golden_qa papers missing) | 79-paper cache (4 host-parsed by Plan 02-02) | Phase 2 Plan 02-02 (2026-05-05) | All 18 unique source papers in golden_qa.json are now MineRU-parseable |

**Deprecated/outdated:**
- The roadmap mentions "100-paper corpus"; actual cache is 79 papers (78 unique source IDs in cache, 18 of which are referenced by golden_qa). The blog claim should say "≈80-paper multimodal corpus" or similar.
- The 3-paper smoke graph at 2886 nodes / 7056 edges is the current state; Phase 7 deprecates it via the 79-paper full rebuild.

## Metadata

**Confidence breakdown:**

- Sweep architecture (pipeline.py + run.py + score.py + compare.py composition): HIGH — read all modules end-to-end with line numbers; Phase 5 Plan 05-01 closed at unit + integration + live levels.
- Pre-flight gating sequence: HIGH — verified all 6 gates against actual file existence on disk; gate 3 (tier-2 .store_id) confirmed missing.
- Tier 4 full-rebuild specifics: MEDIUM — extrapolation from 3-paper smoke; first 79-paper rebuild produces the actual measurement. Plan 07-02 is structured to absorb variance via cost-ack and smoke verification.
- Cost envelope: MEDIUM — capture cost is well-modeled by `run.COST_PER_Q × 30`; judge cost has 30% variance from token-usage parser limitations (v1.1 hardening item).
- Failure modes: HIGH — every claim traceable to specific line numbers in run.py / score.py / pipeline.py / ingest_from_mineru.py.
- Common pitfalls: HIGH — every pitfall has a specific source-line citation or STATE.md cross-reference.
- Tier 4 passthrough behavior (Open Q1): MEDIUM — read the code path but did NOT execute a smoke to verify written-file contents; Plan 07-01 includes a low-cost verification step to upgrade this to HIGH before paying for the full sweep.

**Research date:** 2026-05-06
**Valid until:** 2026-06-06 (30 days; the harness modules are in stable LOCKED forward-contract state per Phases 4/5/6 SUMMARYs; the only invalidator is a substantial edit to one of the 6 harness files, which would re-trigger this research).

## RESEARCH COMPLETE

**Phase:** 7 — Full 5-Tier Rerun
**Confidence:** HIGH on harness facts; MEDIUM on cost extrapolation and Tier 4 rebuild duration

**Recommended approach for the planner to lock:**

1. **Structure as 3 plans across 4 conceptual waves**:
   - Plan 07-01 — Pre-flight verification: env + storage + 5q smoke gate × 5 tiers; recreate `tier-2-managed/.store_id`. Cost ~$0.50–1.00. Wall ~30min.
   - Plan 07-02 — Tier 4 full-corpus rebuild on host (79 papers via `ingest_from_mineru.py --reset --yes`). Cost-ack checkpoint REQUIRED at $15-35 / 15-25h projection. Smoke-verify rebuild before declaring done. Commit new graph-stats manifest.
   - Plan 07-03 — Live full-sweep + post-sweep verification. Step 1 `eval_capture.py` for Tier 4. Step 2 `pipeline.py --tiers 1,2,3,4,5 --tier-4-from-cache <path> --yes`. Step 3 verify single-SHA + same-date + NaN counts + total spend. Step 4 human-verify checkpoint reading `comparison.md` + new `tier-4-graph-stats-{ts}.json`.

2. **Pre-sweep gates are non-negotiable**: 6 gates listed in Section "Pre-Sweep Gating Sequence". Plan 07-01 must encode each as an explicit task step with verification command. Gate 3 (tier-2 .store_id missing) is the most critical because it's invisible until pipeline.py exits 2.

3. **Tier 4 has TWO live-capture entry points** (D-CAPTURE-ENTRYPOINTS per Phase 6 Plan 06-01): `eval_capture.py` for the standalone path AND pipeline.py's passthrough mode via `--tier-4-from-cache`. Plan 07-03 uses BOTH: eval_capture.py to write the canonical 30q QueryLog, then pipeline.py's passthrough to rewrite it with sweep_sha provenance.

4. **No new harness code**. The plan should explicitly forbid edits to `pipeline.py` / `run.py` / `score.py` / `compare.py` / `freeze.py` / `smoke_gate.py` and verify byte-identical via `git diff` post-execution. Optional phase-local `verify_sweep.py` (~50 LOC) under `tier-4-multimodal/scripts/` if inline bash verification grows beyond 4 checks.

5. **Cost-ack checkpoints**: Plan 07-02 ($15-35; the BIG line) and Plan 07-03 ($1 envelope; smaller). Both before live spend.

6. **Open Question 1 must be resolved early** (Tier 4 passthrough writes new sweep_sha or cached SHA?): Plan 07-01 includes a 5q `pipeline.py --tiers 4 --tier-4-from-cache <existing-tier-4> --limit 5 --yes` smoke step that grep's the resulting tier-4-{ts}.json for `.git_sha`. Verify before paying for the full sweep. ~$0.01 cost.

7. **Single-SHA invariant has a subtle limit**: only the 5 QueryLog (queries) JSONs carry `git_sha`. Cost JSONs carry `tier` + `timestamp` only; metrics JSONs carry no top-level provenance (per-row `question_id` only). Plan 07-03's verifier must grep across the 5 queries files, not all 10. ROADMAP wording should be re-read with this lens.

8. **CAP-01 NaN target is <5/30 per tier** for tier 4 + tier 5 specifically (success criteria 3 + 4). Tiers 1, 2, 3 currently have 0 NaN (per current `comparison.md`); Phase 7 should preserve that.

9. **Phase 7 does NOT freeze**. Phase 9 owns freeze after Phase 8 multi-judge data lands. `pipeline.py --freeze` is unset.

10. **Live-test deselect-by-default gap** (v1.1 hardening item): all Phase 7 verification commands MUST include `-m 'not live'` to avoid silent live-test invocation. Document in plan task descriptions.

### Key Findings (3-bullet executive summary)

- **Phase 7 is operational, not code-bearing**. Every harness module is shipped, locked, and forward-contract-guarded. The work is sequencing + gating + verification, not implementation.
- **Tier 4 full-corpus rebuild is the dominant cost line** ($15-35 / 15-25h), separate from the $1-3 sweep envelope. Plan 07-02 must surface this as its own cost-ack checkpoint distinct from Plan 07-03's sweep checkpoint.
- **`tier-2-managed/.store_id` is currently missing on disk** (gitignored, never landed in this clone). Plan 07-01 must recreate via `tier-2-managed/main.py --ingest --yes` BEFORE any sweep attempt — pipeline.py exits 2 immediately on the prereq check otherwise.

### File Created

`.planning/phases/07-full-5-tier-rerun/07-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Sweep architecture (pipeline composition) | HIGH | All modules read end-to-end with line numbers; Phase 5 Plan 05-02 closed live verification |
| Pre-flight gating | HIGH | All 6 gates verified against on-disk state; gate 3 confirmed missing |
| Tier 4 rebuild | MEDIUM | 3-paper smoke measured; 79-paper extrapolation has compound variance |
| Cost envelope | MEDIUM | Capture/judge well-modeled; total dependent on Tier 3 hybrid mode wall variance |
| Pitfalls | HIGH | Every pitfall traceable to source line OR STATE.md decision row |
| Open Q1 (Tier 4 passthrough SHA semantic) | MEDIUM | Read code path but did not execute; Plan 07-01 includes ~$0.01 verification step |

### Open Questions (carried into the plan)

1. Tier 4 passthrough writes new sweep_sha or cached SHA? — verify with $0.01 smoke in Plan 07-01.
2. Realistic cost ceiling for 79-paper rebuild ($15-35 vs higher)? — first rebuild measures.
3. Expected post-rebuild Tier 4 NaN count? — first rebuild's smoke verifies.
4. Phase-local `verify_sweep.py` script vs inline bash? — start inline; promote if reused.

### Ready for Planning

Research complete. Planner can now create 3 PLAN.md files (07-01-PLAN.md, 07-02-PLAN.md, 07-03-PLAN.md) per the recommended structure above.

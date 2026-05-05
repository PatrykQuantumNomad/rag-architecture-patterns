---
phase: 02-tier-4-graphml-regeneration
verified: 2026-05-05T00:00:00Z
status: passed
score: 5/5
overrides_applied: 0
---

# Phase 2: Tier 4 Graphml Regeneration — Verification Report

**Phase Goal:** Tier 4 has a clean, verifiable LightRAG graph rebuilt from MineRU-parsed JSON (parsed outside the sandbox per Phase 139 evidence), verified on a 5-question smoke test before any rerun budget is committed.

**Verified:** 2026-05-05

**Status:** PASSED

**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Storage was deleted (not moved) and rebuilt from MineRU JSON parsed on the host | VERIFIED | `ingest_from_mineru.py` line 336: `shutil.rmtree(RAG_STORAGE)` on `--reset`; zombie-state guard refuses to run on non-empty storage without `--reset`. Commit `39f02cd` message confirms wipe + 3-paper smoke ingest. `kv_store_full_docs.json` has exactly 3 keys: `2005.11401`, `2004.04906`, `2002.08909` matching the smoke paper set parsed by MineRU outside the sandbox. |
| 2 | `graph_chunk_entity_relation.graphml` exists, is non-empty, and node/edge counts are logged | VERIFIED | File at `rag_anything_storage/tier-4-multimodal/graph_chunk_entity_relation.graphml` is 4.4 MB. Direct count: 2886 `<node>` elements, 7056 `<edge>` elements. Provenance manifest `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json` (committed at `39f02cd`) records `graphml_node_count=2886`, `graphml_edge_count=7056`. |
| 3 | Running `run --tiers 4 --tier-4-from-cache` against 5 smoke questions shows populated `retrieved_contexts` | VERIFIED | `evaluation/results/queries/tier-4-2026-05-05T13_59_50Z.json` exists with 5 records. All 5 have non-empty `retrieved_contexts`: n_ctx distribution `[1, 7, 1, 5, 1]`. `run.py` has `--tier-4-from-cache` flag at line 349. `eval_capture.py` has `--smoke-question-ids` flag (commit `17142f8`) importing `DEFAULT_SMOKE_IDS` from `evaluation.harness.run`. |
| 4 | Smoke-test 5 questions produce <5/5 `empty_contexts` NaNs | VERIFIED | `evaluation/results/metrics/tier-4-2026-05-05T13_59_50Z.json` shows 5/5 `faithfulness=1.0`, 5/5 `context_precision` non-NaN, 5/5 `answer_relevancy` non-NaN, all `nan_reason=null`. This is the post-Plan-02-04 re-score. Plan 02-04 confirmed `SmokeGateResult` verdict=PASS with `non_nan_faithfulness_count=5`. Tier 5 regression check also confirmed PASS (Phase 1 metrics `tier-5-2026-05-04T18_48_17Z.json`: 5/5 faithfulness non-NaN). |
| 5 | MineRU + LightRAG + RAG-Anything library versions are recorded for reproducibility | VERIFIED | `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json` (committed) records: `raganything_version=1.2.10`, `lightrag_version=1.4.15`, `mineru_version=3.1.4`. `log_graph_stats.py` captures versions via `importlib.metadata.version()` at runtime. |

**Score:** 5/5 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tier-4-multimodal/scripts/ingest_from_mineru.py` | Wipe-and-rebuild ingest helper | VERIFIED | ~390 LOC, `shutil.rmtree` wipe on `--reset`, zombie-state guard, per-paper exception isolation. Commit `90fff3b`. |
| `tier-4-multimodal/scripts/log_graph_stats.py` | Pydantic-typed provenance writer | VERIFIED | `GraphStats` model, `collect_graph_stats`, `write_stats`, `importlib.metadata` version capture. Commit `620f326`. |
| `rag_anything_storage/tier-4-multimodal/graph_chunk_entity_relation.graphml` | Non-empty rebuilt graph | VERIFIED | 4.4 MB, 2886 nodes, 7056 edges confirmed via direct `grep -c` on file. |
| `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json` | Committed provenance manifest | VERIFIED | Committed at `39f02cd`. Contains node/edge counts, library versions, git SHA, timestamp. |
| `tier-4-multimodal/scripts/parse_missing_papers.py` | Host-side MineRU top-up for 4 missing papers | VERIFIED | ~250 LOC, sandbox-detection pre-flight, idempotent skip. Commit `874ef14`. 4 output dirs present in `tier-4-multimodal/output/`. |
| `tier-4-multimodal/scripts/eval_capture.py` | `--smoke-question-ids` flag + `_filter_qa` helper | VERIFIED | `DEFAULT_SMOKE_IDS` imported from `evaluation.harness.run`; `_filter_qa` pure helper extracted. Commits `17142f8`, `693495e`. |
| `evaluation/harness/score.py` | `JUDGE_MAX_TOKENS=8192` constant wired into `_build_judge` | VERIFIED | Line 70: `JUDGE_MAX_TOKENS = 8192`. Line 131: `max_tokens=JUDGE_MAX_TOKENS` passed to `llm_factory`. Commits `cdbc376` (RED), `7fc6d66` (GREEN). |
| `evaluation/results/queries/tier-4-2026-05-05T13_59_50Z.json` | 5-question smoke capture with populated contexts | VERIFIED | 5 records, all `retrieved_contexts` non-empty, n_ctx `[1, 7, 1, 5, 1]`. |
| `evaluation/results/metrics/tier-4-2026-05-05T13_59_50Z.json` | Re-scored metrics: 5/5 faithfulness=1.0 | VERIFIED | 5/5 `faithfulness=1.0`, all `nan_reason=null`. Gitignored but present on disk and re-scorable from the committed capture. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ingest_from_mineru.py` | `rag_anything_storage/tier-4-multimodal/` | `shutil.rmtree` + `rag.insert_content_list` | WIRED | Wipe guard at line 334-337; insert loop confirmed by provenance manifest recording 2886 nodes from 3 papers. |
| `log_graph_stats.py` | `graph_chunk_entity_relation.graphml` | `networkx.read_graphml` + `GraphStats` | WIRED | `collect_graph_stats` reads graphml via networkx; `write_stats` emits JSON to `evaluation/results/diagnostics/`. Manifest committed. |
| `eval_capture.py` | `evaluation.harness.run.DEFAULT_SMOKE_IDS` | `from evaluation.harness.run import DEFAULT_SMOKE_IDS` | WIRED | Line 53 of `eval_capture.py` confirmed. Single source of truth enforced. |
| `eval_capture.py` | `rag.lightrag.aquery` | Direct call (bypasses VLM wrapper) | WIRED | Rule-1 auto-fix in commit `693495e`; pre-fix produced `n_ctx=[0,0,0,0,0]`, post-fix `[1,7,1,5,1]`. |
| `score._build_judge` | `JUDGE_MAX_TOKENS` | `max_tokens=JUDGE_MAX_TOKENS` kwarg to `llm_factory` | WIRED | Lines 70 and 131 of `score.py` confirmed via `grep`. |
| `smoke_gate` | `evaluation/results/metrics/tier-4-*.json` | `_latest()` mtime resolution | WIRED | `smoke_gate` reads mtime-sorted metrics; post-02-04 re-score overwrote with 5/5 faithfulness=1.0; verdict=PASS confirmed in `02-04-SUMMARY.md`. |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `queries/tier-4-2026-05-05T13_59_50Z.json` | `retrieved_contexts` | `rag.lightrag.aquery(..., only_need_context=True)` against live graphml | Yes — n_ctx `[1,7,1,5,1]`, no repr leak | FLOWING |
| `metrics/tier-4-2026-05-05T13_59_50Z.json` | `faithfulness`, `context_precision`, `answer_relevancy` | RAGAS `evaluate()` against 5 populated records via `llm_factory` with `max_tokens=8192` | Yes — 5/5 non-NaN, faithfulness all 1.0 | FLOWING |
| `diagnostics/tier-4-graph-stats-*.json` | `graphml_node_count`, `graphml_edge_count`, library versions | `networkx.read_graphml` on live graphml + `importlib.metadata.version()` | Yes — 2886/7056 counts match direct grep on graphml | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Evidence | Result | Status |
|----------|----------|--------|--------|
| graphml node count matches provenance | `grep -c "node id=" graphml` = 2886; manifest records 2886 | Match | PASS |
| graphml edge count matches provenance | `grep -c "<edge " graphml` = 7056; manifest records 7056 | Match | PASS |
| `kv_store_full_docs` has 3 smoke paper entries | Python parse: keys = `['2005.11401', '2004.04906', '2002.08909']` | 3/3 | PASS |
| metrics file 5/5 faithfulness non-NaN | Python parse: `non_nan_faithfulness_count=5`, `nan_faithfulness_count=0` | 5/5 | PASS |
| `JUDGE_MAX_TOKENS=8192` wired in `_build_judge` | `grep -n "JUDGE_MAX_TOKENS"` shows line 70 (constant) and line 131 (kwarg) | Both present | PASS |
| All Phase 2 commits exist in git log | `git log --oneline --no-walk` for all 11 documented hashes | 11/11 found | PASS |
| Tier-5 regression check intact | `tier-5-2026-05-04T18_48_17Z.json`: 5/5 faithfulness non-NaN | 5/5 | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| TIER-02 | 02-01-PLAN, 02-02-PLAN | Tier 4 graphml rebuilt from MineRU-parsed JSON | SATISFIED | `ingest_from_mineru.py` + provenance manifest + 2886/7056 graph |
| TIER-03 (Tier 4 portion) | 02-03-PLAN, 02-04-PLAN | Tier 4 smoke test passing | SATISFIED | 5/5 smoke questions with populated contexts, RAGAS verdict PASS |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `ingest_from_mineru.py` | 159 | `return []` | INFO | Legitimate: early-return in `_discover_papers` when `mineru_output_root` does not exist; not a stub — the caller uses the result only when the path exists. Not a data-flow issue. |

No blockers. No warnings. The single `return []` is a valid defensive guard in a helper function, not a rendering stub.

---

## Human Verification Required

None. All must-have truths were verified programmatically against live files, git history, and captured metric data.

The smoke gate verdict of PASS for both Tier 4 and Tier 5 is recorded verbatim in `02-04-SUMMARY.md` and supported by the on-disk metrics files. Regenerating the verdict requires only `python -m evaluation.harness.score --tiers 4 --yes` against the committed capture file.

---

## Deferred Items

The following items are explicitly out of scope for Phase 2 and addressed in later phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | 72 of 79 papers not yet ingested into the graphml (only 3-paper smoke-only graph exists) | Phase 7 | Phase 7 goal: "All 5 tiers × 30 questions captured and scored"; success criteria confirm full ingest pre-requisite |
| 2 | Judge cost ledger underreports ($0 recorded vs real spend via RAGAS token-usage-parser gap) | Phase 7 prep / v1.1 | Documented in 02-03 and 02-04 SUMMARY; tracked as methodology improvement |
| 3 | `n_ctx=1` for 3/5 smoke questions (RAG-Anything split-by-`-----` heuristic) | Phase 7 | Phase 7 may switch to smarter splitter; not a Phase 2 correctness issue |

---

## Gaps Summary

No gaps. All 5 ROADMAP success criteria are VERIFIED with concrete codebase evidence.

**Must-have #1 (storage wipe + host MineRU rebuild):** Confirmed via `shutil.rmtree` in `ingest_from_mineru.py`, the wipe happened as part of the `--reset` execution path, and the resulting storage directory has only 3 paper entries in `kv_store_full_docs.json`. The host-side MineRU parse for 4 additional golden-QA papers is documented in `parse_missing_papers.py` and confirmed by 4 output directories in `tier-4-multimodal/output/`.

**Must-have #2 (graphml non-empty + counts logged):** The graphml file is 4.4 MB with 2886 nodes and 7056 edges confirmed by direct grep. The committed provenance manifest records identical counts with library version pins.

**Must-have #3 (run with --tier-4-from-cache shows populated contexts):** The 5-record queries file has all contexts populated (n_ctx `[1,7,1,5,1]`). The `--tier-4-from-cache` flag exists in `run.py` and `--smoke-question-ids` in `eval_capture.py`.

**Must-have #4 (<5/5 empty_contexts NaNs):** Post-Plan-02-04 re-score: 0/5 NaN faithfulness. Zero `nan_reason` fields. Smoke gate verdict=PASS confirmed. Tier 5 regression check intact.

**Must-have #5 (library versions recorded):** `raganything==1.2.10`, `lightrag-hku==1.4.15`, `mineru==3.1.4` in committed provenance manifest. Captured at runtime via `importlib.metadata`.

---

_Verified: 2026-05-05_
_Verifier: Claude (gsd-verifier)_

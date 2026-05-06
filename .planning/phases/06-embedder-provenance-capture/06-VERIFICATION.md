---
phase: 06-embedder-provenance-capture
verified: 2026-05-06T00:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
---

# Phase 6: Embedder Provenance Capture — Verification Report

**Phase Goal:** Every per-tier capture JSON records the embedding model used by that tier so the frozen doc's embedder-confound table is generated from data, not authored from memory. Closes requirement CAP-03.
**Verified:** 2026-05-06
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Any `queries/tier-{N}-*.json` produced after this phase carries top-level `embedder` + `embedder_source` fields | VERIFIED | `QueryLog` in `records.py:47-48` declares both as `Optional[str] = None`; `run.py:288-296` threads them into every `QueryLog` constructor call; offline parametrized test `test_run_amain_writes_embedder_per_tier_offline` (5 sub-cases) PASS |
| 2 | Tier 1,3,4,5 record `openai/text-embedding-3-small` / `openrouter`; Tier 2 records `gemini-embedding-001` / `google-managed`; Tier 5 matches Tier 1 (D-ROADMAP-OVERRIDE) | VERIFIED | Constants confirmed in source: `embed_openai.py:31-32`, `main.py:97-98`, `rag.py:56-57` (T3), `rag.py:53-54` (T4), `tools.py:49+57` (T5 re-imports T1's EMBED_MODEL, declares own `EMBEDDER_SOURCE="openrouter"`); `test_per_tier_embedder_constants_importable` asserts all five pairs and PASSES |
| 3 | `compare.py` aggregates embedder fields into the rollup and `emit_markdown` emits both a per-tier embedder line AND a dedicated "Embedder by tier" table; `freeze.py` carries embedder fields in its per_tier manifest entry | VERIFIED | `compare.py:133-134` passes through in `aggregate_tier`; `compare.py:268-270` emits per-tier line; `compare.py:277-285` emits "Embedder by tier" table; `freeze.py:59-60` writes `embedder`/`embedder_source` into each per_tier manifest entry; three compare tests + one freeze test all PASS |

**Score: 3/3 truths verified**

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `evaluation/harness/records.py` | `QueryLog` carries `embedder` + `embedder_source` Optional fields | VERIFIED | Lines 47-48; both fields default to `None` (D-BACKCOMPAT) |
| `evaluation/harness/run.py` | Threads `embedder`/`embedder_source` into `QueryLog` for all 5 tier branches | VERIFIED | Lines 173-185 (T1), 188-206 (T2), 211-231 (T3), 235-262 (T4), 264-282 (T5); single `QueryLog(...)` call at 288-296 used by all branches |
| `tier-4-multimodal/scripts/eval_capture.py` | Threads `embedder`/`embedder_source` into `QueryLog` at the live capture path | VERIFIED | Lines 46-51 import `EMBEDDER_SOURCE` from `tier_4_multimodal.rag`; lines 205-206 pass `embedder=DEFAULT_EMBED_MODEL, embedder_source=EMBEDDER_SOURCE` |
| `tier-1-naive/embed_openai.py` | Declares `EMBED_MODEL="openai/text-embedding-3-small"` + `EMBEDDER_SOURCE="openrouter"` | VERIFIED | Lines 31-32 |
| `tier-2-managed/main.py` | Declares `EMBED_MODEL="gemini-embedding-001"` + `EMBEDDER_SOURCE="google-managed"` | VERIFIED | Lines 97-98 |
| `tier-3-graph/rag.py` | Declares `DEFAULT_EMBED_MODEL` (matches T1 value) + `EMBEDDER_SOURCE="openrouter"` | VERIFIED | Lines 56-57; exported in `__all__` at lines 171-172 |
| `tier-4-multimodal/rag.py` | Declares `DEFAULT_EMBED_MODEL` + `EMBEDDER_SOURCE="openrouter"` | VERIFIED | Lines 53-54; exported in `__all__` at lines 237-238 |
| `tier-5-agentic/tools.py` | Re-imports `EMBED_MODEL` from T1; declares own `EMBEDDER_SOURCE="openrouter"` | VERIFIED | Lines 47-50 import `EMBED_MODEL` from `tier_1_naive.embed_openai`; line 57 declares `EMBEDDER_SOURCE: str = "openrouter"` with D-ROADMAP-OVERRIDE comment |
| `evaluation/harness/compare.py` | `aggregate_tier` passes `embedder`/`embedder_source` through; `emit_markdown` emits per-tier line + "Embedder by tier" table | VERIFIED | Lines 133-134 (aggregate), 268-270 (per-tier line), 277-285 (table block); managed derived as `embedder_source == "google-managed"` |
| `evaluation/harness/freeze.py` | `per_tier` manifest entries carry `embedder` + `embedder_source` | VERIFIED | Lines 59-60 in the per-tier dict construction inside `freeze()` |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tier_1_naive.embed_openai.EMBED_MODEL` | `run.py` T1 branch | `from tier_1_naive.embed_openai import EMBED_MODEL as T1_EMBEDDER` | WIRED | `run.py:174`; assigned to `embedder` at line 184 |
| `tier_1_naive.embed_openai.EMBEDDER_SOURCE` | `run.py` T1 branch | `from tier_1_naive.embed_openai import EMBEDDER_SOURCE as T1_EMBEDDER_SOURCE` | WIRED | `run.py:175`; assigned to `embedder_source` at line 185 |
| `tier_2_managed.main.EMBED_MODEL` | `run.py` T2 branch | `from tier_2_managed.main import EMBED_MODEL as T2_EMBEDDER` | WIRED | `run.py:190`; assigned to `embedder` at line 205 |
| `tier_3_graph.rag.DEFAULT_EMBED_MODEL` | `run.py` T3 branch | `from tier_3_graph.rag import DEFAULT_EMBED_MODEL, EMBEDDER_SOURCE as T3_EMBEDDER_SOURCE` | WIRED | `run.py:211-214`; assigned to `embedder` at line 230 |
| `tier_4_multimodal.rag.DEFAULT_EMBED_MODEL` | `run.py` T4 branch | `from tier_4_multimodal.rag import DEFAULT_EMBED_MODEL as T4_EMBEDDER` | WIRED | `run.py:235-238`; assigned at line 261 |
| `tier_4_multimodal.rag.EMBEDDER_SOURCE` | `eval_capture.py` | `from tier_4_multimodal.rag import ... EMBEDDER_SOURCE` | WIRED | `eval_capture.py:50`; passed as `embedder_source=EMBEDDER_SOURCE` at line 206 |
| `tier_1_naive.embed_openai.EMBED_MODEL` | `tier_5_agentic.tools` | `from tier_1_naive.embed_openai import EMBED_MODEL` | WIRED | `tools.py:49`; re-exported; `run.py:269` imports it as `T5_EMBEDDER` |
| `tier_5_agentic.tools.EMBEDDER_SOURCE` | `run.py` T5 branch | `from tier_5_agentic.tools import EMBEDDER_SOURCE as T5_EMBEDDER_SOURCE` | WIRED | `run.py:270`; assigned to `embedder_source` at line 282 |
| `QueryLog.embedder`/`embedder_source` | `compare.aggregate_tier` | `.get("embedder")` / `.get("embedder_source")` on parsed JSON | WIRED | `compare.py:133-134`; passed through to `capture_provenance` dict at lines 384-385 |
| `capture_provenance` | `emit_markdown` | passed as argument | WIRED | `compare.py:392-398`; consumed at lines 261-285 |
| `aggregate_tier` result | `freeze.per_tier` manifest | `row.get("embedder")` / `row.get("embedder_source")` | WIRED | `freeze.py:59-60` |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `QueryLog.embedder` in output JSON | `embedder` local var in `_capture_tier` | Per-tier module-level constant (`EMBED_MODEL` or `DEFAULT_EMBED_MODEL`) | Yes — static literal, not a DB query; appropriate for this metadata type | FLOWING |
| `QueryLog.embedder_source` in output JSON | `embedder_source` local var in `_capture_tier` | Per-tier `EMBEDDER_SOURCE` constant | Yes — static literal | FLOWING |
| "Embedder by tier" table in `comparison.md` | `capture_provenance[*].embedder` | `aggregate_tier` -> queries JSON -> `QueryLog.embedder` | Yes — flows from real capture JSON on disk | FLOWING |
| `per_tier[tier].embedder` in freeze manifest | `row.get("embedder")` | `aggregate_tier` result | Yes — same pipeline as compare.py | FLOWING |

Note: D-NO-LIVE-SMOKE: no live capture JSON exists yet (Phase 7 will produce them). The flow is verified structurally via unit tests that write fixture JSONs and confirm the field appears at the output end. Static-literal data source is appropriate — embedder identity is a compile-time fact, not a runtime computation.

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `QueryLog` schema round-trips embedder fields | `uv run pytest test_eval_records.py::test_query_log_carries_embedder_field` | PASSED | PASS |
| Legacy JSON (no embedder) loads with None | `uv run pytest ...::test_query_log_legacy_json_loads_with_none_embedder` | PASSED | PASS |
| All five per-tier constants importable with correct values | `uv run pytest ...::test_per_tier_embedder_constants_importable` | PASSED | PASS |
| `run.amain` writes correct embedder for all 5 tiers (parametrized) | `uv run pytest ...::test_run_amain_writes_embedder_per_tier_offline` | 5/5 PASSED | PASS |
| `eval_capture.py` wires embedder kwargs | `uv run pytest ...::test_eval_capture_writes_embedder_for_tier_4` | PASSED | PASS |
| `compare.py` emits per-tier embedder line | `uv run pytest ...::test_compare_emits_embedder_in_provenance_footer` | PASSED | PASS |
| `compare.py` em-dash for legacy captures | `uv run pytest ...::test_compare_emits_em_dash_for_legacy_embedder` | PASSED | PASS |
| `compare.py` emits "Embedder by tier" table with T5==T1 and T2 managed=yes | `uv run pytest ...::test_compare_emits_embedder_by_tier_table` | PASSED | PASS |
| `freeze.py` manifest carries embedder per tier (all 5, D-ROADMAP-OVERRIDE for T5) | `uv run pytest ...::test_freeze_manifest_carries_embedder_per_tier` | PASSED | PASS |

**All 13 phase-6 behavioral tests pass under `uv run`.**

Environment note: bare `python` (pyenv 3.11.7) lacks `google-genai` and `langchain_core`, causing two import-chain failures. `uv run` uses the project venv (Python 3.13.1) where all dependencies are present. This is a local environment setup artifact, not a code defect.

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CAP-03 | 06-01-PLAN.md | Embedder model + provenance gateway recorded in every capture JSON | SATISFIED | `QueryLog.embedder` + `QueryLog.embedder_source` fields added; all 5 tier branches in `run.py` + `eval_capture.py` (T4 second path) populate them; compare + freeze carry them forward |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tier-5-agentic/tools.py` | 56 | Comment containing `NOT "openai-hosted-managed"` matches naive grep for rejected string | Info | This is intentional documentation locking D-ROADMAP-OVERRIDE in source; production code contains no wrong value. SUMMARY.md acknowledges this and notes the audit clears with `grep -vE 'NOT[[:space:]]+"openai-hosted'` |

No stubs, no placeholder implementations, no empty returns in paths that matter. All `Optional[str] = None` defaults are intentional backward-compatibility fields (D-BACKCOMPAT), not hollow stubs — they get populated by real code in Phase 7 live captures.

---

## Human Verification Required

None. All must-haves are verifiable via source inspection and offline tests. Phase 6 is intentionally pure-offline (D-NO-LIVE-SMOKE) — embedder identifiers are static literals known at code-edit time. Visual or runtime verification of capture JSON contents is deferred to Phase 7's natural 5-tier rerun.

---

## Gaps Summary

No gaps. All three must-haves are behaviorally verified:

1. **Must-have 1 (schema + wiring):** `QueryLog` carries both fields; both capture entrypoints (`run.py` and `eval_capture.py`) thread them correctly for all 5 tiers. Parametrized offline test confirms the written JSON contains correct values for every tier.

2. **Must-have 2 (per-tier constants, D-ROADMAP-OVERRIDE applied):** All five tier modules declare `EMBEDDER_SOURCE` (and `EMBED_MODEL` / `DEFAULT_EMBED_MODEL` where applicable) with the correct values. Tier 5 correctly records `openai/text-embedding-3-small` / `openrouter` — identical to Tier 1 — NOT the stale ROADMAP wording about an OpenAI hosted vector store. The override is documented inline in both ROADMAP.md and the PLAN frontmatter.

3. **Must-have 3 (compare + freeze):** `compare.py` passes embedder fields through `aggregate_tier`, emits a per-tier line in the capture-provenance footer, and emits a dedicated "Embedder by tier" table. `freeze.py` writes `embedder` + `embedder_source` into each `per_tier` manifest entry. Three compare tests + one freeze test confirm all rendering paths including the D-ROADMAP-OVERRIDE row for Tier 5 and the `managed=yes` row for Tier 2.

---

_Verified: 2026-05-06T00:00:00Z_
_Verifier: Claude (gsd-verifier)_

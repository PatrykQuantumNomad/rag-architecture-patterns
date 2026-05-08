# Phase 8 — Validation Architecture

**Source:** Extracted from `08-RESEARCH.md` §Validation Architecture per Nyquist gate
(VALIDATION.md must exist as a separate artifact — checker BLOCKER #1).

**Phase requirement closed by this validation plan:** CAP-02 (multi-judge spot-check).

---

## Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing; see `evaluation/tests/conftest.py`) |
| Config file | `pyproject.toml` (no separate `pytest.ini`) |
| Quick run command | `uv run pytest evaluation/tests/test_eval_multi_judge_spotcheck.py -x` |
| Full offline suite | `uv run pytest -m 'not live' evaluation/tests/ -x --ignore=evaluation/tests/test_eval_adapters.py` |
| Live invocation | `uv run pytest -m live evaluation/tests/test_eval_multi_judge_spotcheck.py::test_live_spotcheck_under_budget -x` |
| Cost ceiling (live) | $0.50 HARD; $0.30 SOFT (ROADMAP SC-3 envelope) |

---

## Phase Requirements → Test Map

Every CAP-02 sub-behavior maps to one (or more) named test. The
**Owning Plan / Task** column states which plan asserts the behavior, plus
where the same behavior is re-verified in Plan 08-02 live mode.

| # | Req ID | Behavior | Test Type | Test Name | Owning Plan / Task | Live Re-verify |
|---|--------|----------|-----------|-----------|--------------------|----------------|
| 1 | CAP-02 | CLI runs and writes spot-check JSON at expected path | integration | `test_writes_spotcheck_json` (alias: `test_amain_writes_spotcheck_json`) | 08-01 Task 1 | 08-02 Task 2 step "spot-check JSON exists and has 15 cells" (SC-1) |
| 2 | CAP-02 | Per-cell schema: primary + secondary + delta keys present for all 3 metrics | unit | `test_cell_schema` (alias: covered inside `test_amain_writes_spotcheck_json` per-cell loop) | 08-01 Task 1 | 08-02 Task 2 step "Each cell has expected schema" (SC-2) |
| 3 | CAP-02 | Signed delta arithmetic: secondary − primary; None propagates | unit | `test_signed_delta_with_none` | 08-01 Task 1 | 08-02 Task 2 step "delta block keys / no None on PASS" (SC-2) |
| 4 | CAP-02 | Source capture SHA matches Phase 7's `75f6f1b` (pinned to ts; not `_git_sha()`) | unit | `test_source_sha_pinned` (alias: `test_dual_sha_provenance` covers same invariant) | 08-01 Task 1 + Task 2 (`_read_source_sha` helper test) | 08-02 Task 2 step `assert spotcheck["source_capture_git_sha"] == "75f6f1b"` (SC-4) |
| 5 | CAP-02 | Secondary judge slug + model + embedder + max_tokens=8192 recorded | unit | `test_secondary_provenance` (alias: covered inside `test_amain_writes_spotcheck_json` field assertions) | 08-01 Task 1 | 08-02 Task 2 step `assert spotcheck["secondary_judge"]["max_tokens"] == 8192` (SC-4) |
| 6 | CAP-02 | Cost JSON written at `costs/multi-judge-spotcheck-{TS}.json` (NOT `DEFAULT_COSTS_DIR`) | unit | `test_cost_dir_isolation` (alias: `test_amain_writes_cost_ledger_to_explicit_dest_dir`) | 08-01 Task 1 | 08-02 Task 2 step "NO production-default cost path written" (SC-3) |
| 7 | CAP-02 | Question subset hardcoded; 5 IDs × 3 tiers = 15 cells | integration | `test_15_cells` (alias: covered inside `test_amain_writes_spotcheck_json`) | 08-01 Task 1 | 08-02 Task 2 step `assert len(cells) == 15` (SC-1 + SC-2) |
| 8 | CAP-02 | Pre-flight: all 5 wanted IDs are present in tiers 1, 4, 5 at sweep_sha=75f6f1b | unit (offline grep) | `test_phase_7_captures_have_all_5_ids` | 08-01 Task 0 | n/a — pre-flight only, NEVER runs against API |
| 9 | CAP-02 | Forward-contract: 9 RAW-LOCKed files byte-identical pre/post | quality gate | `git diff HEAD -- … \| wc -c == 0` (NOT pytest) | 08-01 Task 1 + Task 2 verify blocks; 08-02 Task 2 verify block | 08-02 Task 2 final guard pass |
| 10 | CAP-02 | Live spend ≤ $0.30 (SOFT, ROADMAP SC-3); ≤ $0.50 (HARD, runaway protection) | live | `test_live_spotcheck_under_budget` | 08-02 Task 2 | n/a — this IS the live test |

**Test-name aliases:** Plan 08-01 currently uses `test_amain_writes_spotcheck_json`,
`test_amain_writes_cost_ledger_to_explicit_dest_dir`, etc. The Validation Architecture
test map (this document) uses canonical short names like `test_writes_spotcheck_json`.
Either set is acceptable as long as the **plan's test-name cross-reference table**
(see 08-01 Task 1 `<behavior>`) maps each plan-level test name to the canonical
VALIDATION.md test name. Do **NOT** rename existing offline tests if they already
implement the right behavior — alias mapping is sufficient (per checker WARNING #7).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Read source QueryLog (Phase 7 capture) | Storage / file I/O | — | `records.read_query_log` already does this. No re-capture. |
| Filter to 5 question IDs | Pure transformation | — | Subset filter on `log.records` list before passing to `score_query_log`. |
| Build secondary judge (LiteLLM) | API / LLM call | — | `score._build_judge(judge_model, judge_emb)` already abstracts this. Change only the slug. |
| Re-score 5 records × 3 tiers | API / LLM call | Pure | `score.score_query_log` returns `list[ScoreRecord]` for the filtered subset. |
| Read primary metrics (Gemini) | Storage / file I/O | — | `_read_primary_metrics(metrics_dir, tier, ts)` pins to source ts (NOT `_latest()` mtime). |
| Read source capture SHA (`source_capture_git_sha`) | Storage / file I/O | — | `_read_source_sha(queries_dir, tiers, source_ts)` reads `src_log.git_sha` from the source QueryLog — **NEVER** falls back to `_git_sha()` (current HEAD). |
| Compute signed delta = secondary − primary | Pure transformation | — | Per-metric, per-cell arithmetic; NaN propagates as `None`. |
| Write structured spot-check JSON | Storage / file I/O | — | Single file `evaluation/results/metrics/multi-judge-spotcheck-{TS}.json`. |
| Record secondary-judge cost | Storage / file I/O | — | Custom `CostTracker("multi-judge-spotcheck")` instance, persist with explicit `dest_dir` to dodge the v1.1 mis-feature. |

**Why this matters:** The spot-check is purely re-scoring + delta-arithmetic — no
capture, no graph, no agents. All real work lives in already-built helpers. The new
code is a thin orchestration layer plus a JSON schema. The
`_read_source_sha` helper is **separately tracked** because it directly maps to test
case #4 above (Phase 7 sweep_sha=`75f6f1b` MUST appear in `source_capture_git_sha`,
NOT current HEAD); see Plan 08-01 Task 2 `<action>` step 2 for the helper signature.

---

## Sampling Rate

- **Per task commit (Plan 08-01):** `uv run pytest evaluation/tests/test_eval_multi_judge_spotcheck.py -x` (offline only).
- **Per wave merge (Plan 08-01 GREEN gate):** `uv run pytest -m 'not live' evaluation/tests/ -x --ignore=evaluation/tests/test_eval_adapters.py` (full offline suite, ≥125 PASS expected after this plan).
- **Phase gate (Plan 08-02):** Full offline suite green + 1 live run via `-m live` checkpoint with cost-ack ≤ $0.30 SOFT / ≤ $0.50 HARD.
- **Forward-contract gate (every commit, both plans):** `git diff HEAD -- evaluation/harness/{pipeline,run,score,compare,freeze,smoke_gate,records}.py shared/cost_tracker.py shared/pricing.py | wc -c` MUST return `0`. **Non-zero ⇒ escalate as `## CHECKPOINT REACHED — forward-contract violation`, do NOT commit, do NOT proceed.**

---

## Wave 0 Gaps

- [ ] `evaluation/tests/test_eval_multi_judge_spotcheck.py` — covers CAP-02 unit + integration (Plan 08-01) + 1 live test (Plan 08-02).
- [ ] No new fixtures needed; existing `conftest.py` provides `live_eval_keys_ok` (used by Plan 05-02 live test) and `tmp_path`.

---

## Cost-Ceiling Two-Tier Enforcement (BLOCKER #2 fix)

ROADMAP SC-3 requires **spend stayed within $0.10–0.30**. Plan 08-02's live test
asserts a **HARD ceiling at $0.50** (runaway protection). To bridge these:

| Tier | Threshold | Mechanism | Outcome |
|------|-----------|-----------|---------|
| SOFT | total_usd > $0.30 | `pytest.warns` or printed escalation in `test_live_spotcheck_under_budget` | Test exits as `## CHECKPOINT REACHED — SC-3 envelope exceeded` with full cost breakdown; user decides: accept (`PASS-WITH-DEVIATION`), abort, or rerun. |
| HARD | total_usd > $0.50 | `assert total_usd <= 0.50` (pytest fails outright) | Plan FAILS; cost runaway prevented. |

**Decision matrix:**
- `total_usd ≤ $0.30` → plan PASS, SC-3 satisfied cleanly.
- `$0.30 < total_usd ≤ $0.50` → plan exits PASS-WITH-DEVIATION via human checkpoint; SC-3 marked as DEVIATED in SUMMARY (deviation rationale required).
- `total_usd > $0.50` → plan FAIL; rerun forbidden without diagnosis.

This two-tier rule is **encoded inside `test_live_spotcheck_under_budget`**
(Plan 08-02 Task 2) and reflected in Plan 08-02's `must_haves.truths` for SC-3.

---

## Cross-Reference

- Source artifact: `08-RESEARCH.md` §Validation Architecture (lines 631–662)
- Phase Requirements: `.planning/REQUIREMENTS.md` (CAP-02)
- ROADMAP success criteria: `.planning/ROADMAP.md` Phase 8 (SC-1, SC-2, SC-3, SC-4)
- Forward-contract trail: Plans 02-04, 03-02, 04-01, 05-01, 05-02, 06-01, 07-01, 07-02, 07-03 SUMMARYs

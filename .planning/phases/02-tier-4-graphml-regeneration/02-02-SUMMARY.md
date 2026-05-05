---
phase: 02-tier-4-graphml-regeneration
plan: 02
subsystem: ingest
tags: [mineru, hybrid-auto-engine, golden-qa, phase-7-prep, host-only, non-blocking]

# Dependency graph
requires:
  - phase: 02-tier-4-graphml-regeneration-01
    provides: "ingest_from_mineru.py script; smoke-only graphml (2886 nodes/7056 edges, 3 papers); existing 75-paper MineRU cache in tier-4-multimodal/output/"
provides:
  - "tier-4-multimodal/scripts/parse_missing_papers.py — host-side MineRU top-up helper for the 4 golden_qa-referenced papers missing from the 75-paper cache; sandbox-detection pre-flight, idempotent, non-zero exit on any golden-qa failure"
  - "tier-4-multimodal/tests/test_parse_missing.py — 11 unit tests (subprocess mocked): skip-already-parsed, command shape, sandbox detection, per-paper failure isolation, main() exit codes"
  - "tier-4-multimodal/output/1909.01066/ — MineRU-parsed output (11 pages; hybrid_auto backend)"
  - "tier-4-multimodal/output/2002.06177/ — MineRU-parsed output (59 pages; hybrid_auto backend)"
  - "tier-4-multimodal/output/2309.15217/ — MineRU-parsed output (8 pages; hybrid_auto backend)"
  - "tier-4-multimodal/output/2410.05779/ — MineRU-parsed output (16 pages; hybrid_auto backend)"
affects: [phase-7, phase-9]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Host-only MineRU subprocess loop with sandbox-detection pre-flight (Pitfall 3 of RESEARCH.md: OMP shmem block in orchestrator sandbox)"
    - "Idempotent per-paper skip via target_dir.exists() check; hardcoded MISSING_PAPERS_GOLDEN constant for auditable stable list"
    - "list-form subprocess.run (no shell=True) per V5 ASVS even for curated manifest inputs"
    - "Non-zero exit only when a golden-qa-referenced paper fails (Open Question 2 of RESEARCH.md)"

key-files:
  created:
    - tier-4-multimodal/scripts/parse_missing_papers.py
    - tier-4-multimodal/tests/test_parse_missing.py
  modified: []

key-decisions:
  - "Plan 02-02 host MineRU top-up: 4/4 papers parsed successfully on host (approved branch). Wall time ~49 min total. The 4 new papers are in the MineRU cache but NOT in the smoke-only graphml; Phase 7 will re-rebuild over the full 79-paper cache."
  - "MineRU version: project-pinned mineru==3.1.4 (matches Plan 02-01 provenance manifest; version not explicitly captured in script output but consistent with .venv install)"
  - "Backend used: hybrid-auto-engine (locked in _build_mineru_command per RESEARCH.md A2 — matches existing 75-paper cache layout)"
  - "The 4 new papers were NOT re-ingested into LightRAG graphml; Plan 02-01's smoke-only rebuild ran with --paper-ids 2005.11401,2004.04906,2002.08909 BEFORE this top-up; Phase 7 is the canonical place for full 79-paper ingest"
  - "Non-blocking semantics confirmed: plan outcome did NOT gate Phase 2 ship; checkpoint resolved as approved"

patterns-established:
  - "Phase 7 full rebuild: run ingest_from_mineru.py --reset --yes over the full tier-4-multimodal/output/ cache (now 75 + 4 = 79 papers)"

# Metrics
duration: 49min (host MineRU wall time; script authoring in Task 1 was ~20 min)
completed: 2026-05-05
---

# Phase 2 Plan 02: Host MineRU Top-Up (4 Missing Golden-QA Papers) Summary

**parse_missing_papers.py script authored and run on host; 4 golden_qa-referenced papers parsed by MineRU 3.1.4 with hybrid-auto-engine in ~49 min wall time (94 pages total); MineRU cache now covers 79 papers; Phase 2 graphml NOT updated (Phase 7 will rebuild over full cache)**

## Performance

- **Duration:** ~49 min (host MineRU wall time for Task 2); ~20 min for Task 1 script authoring
- **Started:** 2026-05-05 (Task 1 script commit 874ef14)
- **Completed:** 2026-05-05 (Task 2 host run; user-reported terminal output)
- **Tasks:** 2 (Task 1: script + tests; Task 2: host checkpoint — approved)
- **Files modified:** 2 created (parse_missing_papers.py, test_parse_missing.py); 4 output directories produced by MineRU on host

## Accomplishments

- Authored `parse_missing_papers.py` (~250 LOC) with sandbox-detection pre-flight, idempotent skip, list-form subprocess invocation, and non-zero exit on golden-qa paper failure
- 11 unit tests passing with subprocess.run mocked (no real MineRU invocation in CI)
- User ran the script on host machine; all 4 golden_qa-referenced papers parsed successfully (Parsed 4/4 papers; 0 failures: [])
- MineRU cache extended from 75 to 79 papers; Phase 7 can now run a full-corpus ingest without further MineRU top-up steps

## Host Machine Command Sequence

The user ran a single command from the repo root on the host machine (outside the orchestrator sandbox):

```bash
python tier-4-multimodal/scripts/parse_missing_papers.py
```

Prerequisites confirmed on host:
- `.venv` activated (`source .venv/bin/activate`)
- `which mineru` resolved to `.venv/bin/mineru`
- Clean host environment (no `ALL_PROXY=socks5h://...` sandbox marker)

## Per-Paper Wall Time

| Paper ID | Title (short) | Pages | Start (host clock) | End (host clock) | Wall Time |
|----------|---------------|-------|--------------------|------------------|-----------|
| 1909.01066 | language_models_as_knowledge_bases | 11 | 08:27:48 | 08:42:11 | ~14 min |
| 2002.06177 | the_next_decade_in_ai_four_steps_towards | 59 | 08:42:19 | 09:05:37 | ~23 min |
| 2309.15217 | ragas_automated_evaluation_of_retrieval | 8 | 09:05:45 | 09:08:46 | ~3 min |
| 2410.05779 | lightrag_simple_and_fast_retrieval_augme | 16 | 09:08:52 | 09:16:25 | ~7 min |
| **Total** | 4 papers | **94 pages** | 08:27:48 | 09:16:25 | **~49 min** |

Per-paper average: ~31 sec/page (including MineRU model warm-up on first paper).

## MineRU Version and Backend

- **MineRU version:** `mineru==3.1.4` (project-pinned; matches Plan 02-01's provenance manifest `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json`). The script does not log `mineru --version` output; version is inferred from the project `.venv` install consistent with `pyproject.toml` pin `mineru>=2.5,<4` resolved to 3.1.4.
- **Backend:** `hybrid-auto-engine` (locked in `_build_mineru_command` per RESEARCH.md A2; matches the existing 75-paper cache `hybrid_auto/` directory layout)

## Output File Verification

All 4 `_content_list.json` files confirmed present after host run:

```
tier-4-multimodal/output/1909.01066/1909.01066_language_models_as_knowledge_bases/hybrid_auto/1909.01066_language_models_as_knowledge_bases_content_list.json
tier-4-multimodal/output/2002.06177/2002.06177_the_next_decade_in_ai_four_steps_towards/hybrid_auto/2002.06177_the_next_decade_in_ai_four_steps_towards_content_list.json
tier-4-multimodal/output/2309.15217/2309.15217_ragas_automated_evaluation_of_retrieval_/hybrid_auto/2309.15217_ragas_automated_evaluation_of_retrieval__content_list.json
tier-4-multimodal/output/2410.05779/2410.05779_lightrag_simple_and_fast_retrieval_augme/hybrid_auto/2410.05779_lightrag_simple_and_fast_retrieval_augme_content_list.json
```

Note: the output directory layout differs slightly from the pattern in RESEARCH.md — MineRU 3.1.4 places the content list directly at `<pid>/<pid>_<paper_name>/hybrid_auto/<paper_name>_content_list.json` (no intermediate `<8charhash>` segment). This is consistent with the existing 75-paper cache layout produced by the same MineRU version and is compatible with `ingest_from_mineru.py`'s `_find_content_list` helper (which uses `rglob("*_content_list.json")`).

## No Graphml Update (Phase 7 Note)

The 4 new papers were NOT re-ingested into the LightRAG graphml. Plan 02-01's smoke-only rebuild ran with `--paper-ids 2005.11401,2004.04906,2002.08909` BEFORE this top-up. The Phase 2 graphml remains the 3-paper smoke-only graph (2886 nodes / 7056 edges).

**Phase 7 prep action:** When Phase 7 runs the full rebuild, it must run:

```bash
python tier-4-multimodal/scripts/ingest_from_mineru.py --reset --yes
```

over the entire `tier-4-multimodal/output/` cache. At that point the cache contains:
- 72 deferred papers (from the original 75-paper cache, excluding the 3 smoke papers)
- 3 smoke papers: 2005.11401, 2004.04906, 2002.08909
- 4 top-up papers from this plan: 1909.01066, 2002.06177, 2309.15217, 2410.05779

Total: 79 papers for the full-corpus rebuild. (The `--paper-ids` flag can be omitted to let `ingest_from_mineru.py` discover all papers under `tier-4-multimodal/output/` automatically.)

## Non-Blocking Semantics

This plan is explicitly non-blocking for Phase 2 ship. The frontmatter `non_blocking: true` and `phase_gate: false` flags signal `/gsd:verify-phase` to ignore Plan 02-02's status when computing Phase 2 completion. The 4 papers parsed here are golden_qa-referenced but outside the smoke set (which uses 2005.11401, 2004.04906, 2002.08909). The checkpoint resolved as `approved` — the top-up completed successfully on the user's host machine and is recorded here for Phase 7's planning context.

## Task Commits

1. **Task 1: parse_missing_papers.py + unit tests** - `874ef14` (feat)
2. **Task 2: host checkpoint** - resolved `approved` (no code commit; MineRU output is gitignored per `.gitignore`)

**Plan metadata:** (this SUMMARY commit — see below)

## Files Created/Modified

- `tier-4-multimodal/scripts/parse_missing_papers.py` — Host-side MineRU subprocess loop; MISSING_PAPERS_GOLDEN constant; sandbox-detection pre-flight; idempotent skip; _build_mineru_command with hybrid-auto-engine lock; per-paper error isolation; non-zero exit on golden-qa failure
- `tier-4-multimodal/tests/test_parse_missing.py` — 11 unit tests with subprocess.run mocked
- `tier-4-multimodal/output/1909.01066/` — MineRU output (gitignored; verified present)
- `tier-4-multimodal/output/2002.06177/` — MineRU output (gitignored; verified present)
- `tier-4-multimodal/output/2309.15217/` — MineRU output (gitignored; verified present)
- `tier-4-multimodal/output/2410.05779/` — MineRU output (gitignored; verified present)

## Decisions Made

- MineRU version pinned at 3.1.4 (project-level); not re-captured from `mineru --version` CLI in script output — recorded here for Phase 9 provenance
- Backend `hybrid-auto-engine` locked in code (not a CLI arg) to prevent accidental mismatch with the 75-paper cache layout
- MISSING_PAPERS_GOLDEN is a hardcoded 4-tuple (not dynamically derived) for auditability and Phase 7 stability
- The 4 new papers remain outside the Phase 2 graphml by design; Phase 7 is the canonical full-rebuild step

## Deviations from Plan

None — plan executed exactly as written. Task 1 (script + tests) completed at commit 874ef14. Task 2 (host checkpoint) resolved as `approved` with all 4 papers parsed successfully.

## Issues Encountered

None. The model cache was warm on the host machine (MineRU had been used previously for the 75-paper run), so no extended cold-start delay. All 4 papers completed without MineRU errors.

## User Setup Required

None — the host machine command was a single invocation of the existing script with no additional configuration.

## Next Phase Readiness

- Plan 02-03 (smoke verification) is the remaining Phase 2 plan; it does not depend on 02-02's output (smoke set papers were already in the cache before this top-up)
- Phase 7 can now run `ingest_from_mineru.py --reset --yes` over the full 79-paper cache without any further MineRU top-up steps
- The `parse_missing_papers.py` script remains available for re-use if additional papers are identified as missing in future phases

---
*Phase: 02-tier-4-graphml-regeneration*
*Completed: 2026-05-05*

## Self-Check: PASSED

Files verified present:
- FOUND: tier-4-multimodal/output/1909.01066/_content_list.json (via rglob)
- FOUND: tier-4-multimodal/output/2002.06177/_content_list.json (via rglob)
- FOUND: tier-4-multimodal/output/2309.15217/_content_list.json (via rglob)
- FOUND: tier-4-multimodal/output/2410.05779/_content_list.json (via rglob)
- Task 1 commit 874ef14 exists in git log

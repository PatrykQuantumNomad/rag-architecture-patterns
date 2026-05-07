"""Phase 8 / Plan 08-01 — offline tests for evaluation.harness.multi_judge_spotcheck.

This file is intentionally split into two logical sections per the TDD trail:

    Section A (Task 0 — pre-flight): a single offline assertion that the
    Phase 7 sweep_sha=`75f6f1b` source captures actually contain the 5
    wanted question IDs in tiers 1, 4, 5. Runs WITHOUT importing the
    module-under-test so that Task 1's RED commit (which DOES import
    multi_judge_spotcheck before it exists) cannot break the pre-flight.

    Section B (Task 1 — RED) is APPENDED in a follow-up commit.

Test-name cross-reference (per VALIDATION.md WARNING #7 / 08-01 Task 1
behavior table). Plan-level test names (left) are authoritative; the
canonical names in `08-VALIDATION.md` (right) are aliases:

    test_signed_delta_with_none                          ← test_signed_delta_with_none
    test_filter_records_deterministic_order              ← (helper-level)
    test_filter_records_missing_id_aborts                ← (helper-level)
    test_read_primary_metrics_pins_to_ts                 ← test_source_sha_pinned (TS half)
    test_read_source_sha_returns_log_git_sha_not_head    ← test_source_sha_pinned (SHA half)
    test_estimate_cost_fallback                          ← (cost block of test_15_cells)
    test_amain_writes_spotcheck_json                     ← test_writes_spotcheck_json + test_cell_schema + test_secondary_provenance + test_15_cells
    test_amain_writes_cost_ledger_to_explicit_dest_dir   ← test_cost_dir_isolation
    test_amain_aborts_on_missing_id                      ← (no canonical alias)
    test_dual_sha_provenance                             ← test_source_sha_pinned (SHA half)
    test_phase_7_captures_have_all_5_ids (Task 0)        ← (pre-flight only — WARNING #5 closure)
    test_module_imports                                  ← (locks D-1, D-2, D-3 module constants)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# D-2 lock (planner verification 2026-05-07 — IDs ARE present per spec):
WANTED_IDS = {
    "single-hop-001",
    "single-hop-002",
    "multi-hop-001",
    "multi-hop-002",
    "multimodal-001",
}
# D-1 lock — one tier per architecture family:
PHASE_7_TIERS = (1, 4, 5)
# Phase 7 sweep invariant:
PHASE_7_SWEEP_TS = "2026-05-07T10:59:10Z"
PHASE_7_TS_FILENAME = "2026-05-07T10_59_10Z"
PHASE_7_SWEEP_SHA = "75f6f1b"


def test_phase_7_captures_have_all_5_ids(capsys):
    """Pre-flight: assert all 5 wanted IDs are present in tiers 1, 4, 5 at sweep_sha=75f6f1b.

    Closes checker WARNING #5 — produces evidence (3 stdout lines, one per tier)
    that D-2 holds BEFORE any other test in this file runs. Skips cleanly on a
    fresh clone if Phase 7 captures aren't on disk.
    """
    queries_dir = _REPO_ROOT / "evaluation" / "results" / "queries"
    paths = {
        tier: queries_dir / f"tier-{tier}-{PHASE_7_TS_FILENAME}.json"
        for tier in PHASE_7_TIERS
    }
    missing_files = [p for p in paths.values() if not p.exists()]
    if missing_files:
        pytest.skip(
            "Phase 7 source capture missing — run /gsd-execute-phase 7 first. "
            f"Missing: {missing_files}"
        )

    failures: list[str] = []
    for tier in PHASE_7_TIERS:
        data = json.loads(paths[tier].read_text(encoding="utf-8"))
        ids_present = {r["question_id"] for r in data["records"]}
        missing_ids = WANTED_IDS - ids_present
        if missing_ids:
            failures.append(
                f"tier-{tier}: missing IDs {sorted(missing_ids)}"
            )
        if data["git_sha"] != PHASE_7_SWEEP_SHA:
            failures.append(
                f"tier-{tier}: git_sha={data['git_sha']!r} (expected {PHASE_7_SWEEP_SHA!r})"
            )
        # Evidence print (visible under -v / -s)
        print(
            f"[Pre-flight] tier-{tier}: "
            f"{len(WANTED_IDS - missing_ids)}/5 wanted IDs present "
            f"at sweep_sha={data['git_sha']}"
        )

    assert not failures, (
        "Phase 7 capture pre-flight failed:\n  " + "\n  ".join(failures)
    )

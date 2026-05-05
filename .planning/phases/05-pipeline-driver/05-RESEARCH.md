# Phase 5: Pipeline Driver — Research

**Researched:** 2026-05-05
**Domain:** In-process orchestration of an existing four-stage Python harness (capture → score → compare → freeze) into a single CLI module with single-tier-rerun semantics.
**Confidence:** HIGH (every integration contract was read from source; no external library research required — this phase is pure composition of repo-internal code with locked signatures.)

## Summary

Phase 5 ships `evaluation/harness/pipeline.py` — a composition layer that calls four already-shipped, locked entry points in-process: `run.amain()` (async capture), `score.amain()` (async scoring), `compare._run()` (sync rollup), and `freeze.freeze()` (sync immutable snapshot). The integration contract is fully knowable from source: I read all four modules end-to-end. There is **one architectural friction point** worth surfacing — `run._capture_tier()` calls `_git_sha()` lazily *per tier* inside the loop (run.py:159), not once at sweep start. Phase 5's "single git SHA captured at start" success criterion can be honored either by (a) a 1-line targeted change to `run.amain()` to capture SHA once and pass it into `_capture_tier()` via a new keyword argument, or (b) having `pipeline.py` snapshot SHA into an env var that `_git_sha()` consults first. Option (a) is recommended — it is honest about the change, surface-stable, and runs in <10 lines of run.py edits.

`compare._run()` and `freeze.freeze()` are sync functions called from an async pipeline driver — they must be wrapped in `asyncio.to_thread(...)` to avoid blocking the event loop. `_cost_surprise()` consolidation is straightforward: pipeline.py honors its own `--yes` flag at the top, then always passes `args.yes=True` into the synthesized argparse.Namespace it hands to `run.amain()` so the inner prompt is suppressed. HARN-02 (single-tier rerun preserving other tiers) works for free because both `score.amain` and `compare._run` already use `_latest()` mtime resolution to pick per-tier files independently — re-running `--tiers 4` writes new tier-4 files and the existing tier-{1,2,3,5} files remain on disk and get picked up by their own `_latest()` calls. This must be **verified** by reading `score._latest_query_log` (line 105) and `compare._latest` (line 48) — both glob on `tier-{N}-*.json` and sort by mtime DESC, fully tier-isolated. No coordination needed.

**Primary recommendation:** Land pipeline.py at ~150-200 raw LOC with the single-SHA propagation done via a small additive change to `run.amain()` (accept optional `git_sha_override` kwarg; default = old `_git_sha()` lazy behavior). Reuse `run._git_sha()` and `run._ts()` for the canonical SHA + ISO timestamp at sweep start. Wrap sync stages in `asyncio.to_thread`. Honor `--yes` once. Make `--freeze` optional (capture+score+compare without freeze is a valid sweep). Treat tier-4-from-cache as a passthrough flag plumbed into the synthesized run.amain args.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HARN-01 | `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5` runs capture → score → rollup → freeze in one command with single git SHA + ISO timestamp captured at start | Section 2 (SHA propagation), Section 3 (in-process composition), Section 6 (argparse) |
| HARN-02 | `--tiers 4` re-run does not invalidate other tiers' captures (relies on `_latest()` mtime resolution) | Section 4 (single-tier rerun walkthrough); verified via reading `compare._latest` (compare.py:48) + `score._latest_query_log` (score.py:105) |

## Project Constraints (from CLAUDE.md)

No `./CLAUDE.md` was found at repo root. Project conventions are inherited from `.planning/STATE.md` Decisions and the four already-shipped modules:

- ISO 8601 Z timestamps via `run._ts()` / `run._ts_for_filename()` — single source of truth.
- Git SHA via `run._git_sha()` (short HEAD SHA; "unknown" on failure) — single source of truth.
- `rich.console.Console` for CLI output — match existing per-phase banners.
- All harness modules use `from __future__ import annotations` + module docstring.
- Argparse pattern: separate `build_parser()` so tests can introspect flags; `main(argv)` calls `asyncio.run(amain(args, Console()))`.
- Test fixtures: `tmp_path` + offline mocks; live tests gated under `@pytest.mark.live`. **Pyproject does NOT enforce live-deselect-by-default** (v1.1 hardening item) — `pytest path/to/file.py -v` runs live tests silently. Phase 5's test surface must respect this gap.

## Architectural Responsibility Map

| Capability | Owner Module | Pipeline.py Role |
|-----------|--------------|-----------------|
| Per-tier capture (run async loop, write `queries/tier-N-*.json` + `costs/tier-N-eval-*.json`) | `run.py` (`amain`, `_capture_tier`) | Synthesize an `argparse.Namespace` and `await run.amain(args, console)` |
| RAGAS scoring (load latest query log, write `metrics/tier-N-*.json` + `costs/ragas-judge-tier-N-*.json`) | `score.py` (`amain`, `score_query_log`) | Synthesize `argparse.Namespace` and `await score.amain(args, console)` |
| Comparison rollup (load latest queries+costs+metrics, write `comparison.md`) | `compare.py` (`_run`, `aggregate_tier`, `aggregate_class_rollup`, `emit_markdown`) | `await asyncio.to_thread(compare._run, args)` (sync function) |
| Immutable snapshot (copy `comparison.md` → `frozen/eval-numbers-vX.Y.md` + sidecar manifest) | `freeze.py` (`freeze`, locked Phase 4 contract) | `await asyncio.to_thread(freeze.freeze, version=..., force=..., results_dir=..., source=None)` |
| **Cross-stage SHA + ISO timestamp single source** | **pipeline.py** | Capture once via `run._git_sha()` + `run._ts()` at start; propagate to run.amain (via new optional kwarg) and emit a banner line |
| **Single cost-surprise prompt** | **pipeline.py** | Call `run._cost_surprise()` once at top with full tier list; pass `args.yes=True` into all downstream invocations |
| **Tier-4-from-cache passthrough** | **pipeline.py** | Accept `--tier-4-from-cache <path>`; thread into the args.Namespace handed to run.amain |
| **Stage gating** (e.g. `--no-freeze`, `--freeze` optional) | **pipeline.py** | Gate which stages execute |

### MUST NOT touch (byte-identical post-Phase 5)

- `evaluation/harness/freeze.py` — Phase 4 forward contract LOCKED at `freeze(version, force, results_dir, source) -> Path`. Pipeline.py imports and calls; zero edits.
- `evaluation/harness/compare.py` — Pure file I/O + numpy. Pipeline.py calls `compare._run(args)` via `asyncio.to_thread`. Zero edits.
- `evaluation/harness/score.py` — Pipeline.py synthesizes args matching `score.build_parser()` and calls `await score.amain(args, console)`. Zero edits.

### MAY need targeted edit (minimum surface)

- `evaluation/harness/run.py` — **One additive change** to honor "single git SHA at start": accept an optional `git_sha_override: str | None = None` parameter on `_capture_tier` (or have `amain` capture SHA once and pass it down). Default behavior preserved via `git_sha = git_sha_override or _git_sha()` on line 159. Estimated +3 LOC. Pipeline.py would synthesize this override into args. **Justification:** the alternative (env-var hack like `RAG_PIPELINE_GIT_SHA_OVERRIDE`) couples two modules through environment state, which is harder to test and easier to break. A new optional kwarg with backwards-compatible default is the smallest honest change.

  **If the planner prefers zero changes to run.py:** an alternative is to monkeypatch `evaluation.harness.run._git_sha` at pipeline.py module scope before invoking `run.amain` — works but is fragile (relies on Python attribute mutation, breaks if run.py adopts `from .helpers import _git_sha`). The kwarg approach is recommended.

## Single-SHA / Single-Timestamp Propagation Strategy

**Current behavior** (verified by reading `run.py:146-272`):
- `_capture_tier()` is called once per tier inside `amain`'s `for tier in tiers:` loop (line 315).
- Each call enters `_capture_tier` and computes `timestamp = _ts()` (line 158) and `git_sha = _git_sha()` (line 159) **at the top of the function** — i.e. once per tier, not once per sweep.
- For a 5-tier sweep, that produces 5 distinct timestamps and (rarely) 5 distinct SHAs (e.g. if user commits mid-sweep).

**Phase 5 requirement** (HARN-01 + Success Criterion 1): "one git SHA recorded at start and propagated to every per-tier output JSON".

**Recommended approach: additive kwarg on run.amain + run._capture_tier**

```python
# run.py — proposed minimal change (+3 LOC effective)
async def _capture_tier(tier, qa, args, console, *, git_sha_override=None, ts_override=None):
    timestamp = ts_override or _ts()
    git_sha = git_sha_override or _git_sha()
    ...

async def amain(args, console):
    sweep_sha = getattr(args, "git_sha_override", None)
    sweep_ts = getattr(args, "ts_override", None)
    ...
    for tier in tiers:
        log = await _capture_tier(
            tier, qa, args, console,
            git_sha_override=sweep_sha, ts_override=sweep_ts,
        )
```

Then pipeline.py:

```python
# pipeline.py
sweep_sha = run._git_sha()
sweep_ts = run._ts()
console.print(f"[cyan]Pipeline started: git={sweep_sha} ts={sweep_ts}[/cyan]")
run_args = build_run_args(...)
run_args.git_sha_override = sweep_sha
run_args.ts_override = sweep_ts
exit_code = await run.amain(run_args, console)
```

**Why this beats alternatives:**

| Approach | Verdict |
|----------|---------|
| Env var (`RAG_PIPELINE_GIT_SHA_OVERRIDE`) read inside `_git_sha()` | Couples through process state. Hard to test in pytest without per-test cleanup. Risk of stale env var leaking between sweeps. |
| Monkeypatch `run._git_sha` from pipeline.py | Fragile — breaks if run.py refactors imports. Test-double smell in production code path. |
| Have pipeline.py call `run.amain` with all tiers in one call (no SHA injection) | Doesn't actually solve the problem — `_capture_tier` still calls `_git_sha()` per tier internally. |
| **Optional kwarg with backwards-compat default** (recommended) | Smallest change, fully testable, run.amain remains usable standalone with old behavior. |

**Timestamp note:** The same kwarg pattern handles ISO timestamp propagation. Without it, each tier gets its own timestamp; with it, all 5 per-tier output files share a single sweep-level timestamp suffix. Note that the *filenames* embed `_ts_for_filename(timestamp)` (run.py:265) so propagating ts means all 5 files get the same suffix — careful: that's actually fine because each file is `tier-{N}-{ts}.json`, distinguished by tier number, not timestamp.

## In-Process Composition Pattern

The four stages have asymmetric sync/async signatures:

| Stage | Function | Sync/Async | How pipeline.py calls it |
|-------|----------|------------|--------------------------|
| Capture | `run.amain(args, console) -> int` | async | `code = await run.amain(args, console)` |
| Score | `score.amain(args, console) -> int` | async | `code = await score.amain(args, console)` |
| Compare | `compare._run(args) -> int` | **sync** | `code = await asyncio.to_thread(compare._run, args)` |
| Freeze | `freeze.freeze(version, force, results_dir, source) -> Path` | **sync** | `path = await asyncio.to_thread(freeze.freeze, version, force, results_dir, source)` |

**Why `asyncio.to_thread` for compare and freeze:** both do file I/O (compare reads/writes JSON + markdown; freeze does `shutil.copy2`). Calling them directly inside an async function would block the event loop — harmless for a CLI tool that has nothing else running, BUT **bad pattern hygiene** (per run.py's own docstring: "single asyncio.run boundary; sync work via asyncio.to_thread inside adapters" — Pitfall 5). Honor the project convention.

**Pattern:**

```python
# pipeline.py — recommended composition pattern
async def amain(args, console):
    # 1. Capture sweep-level identity ONCE
    sweep_sha = run._git_sha()
    sweep_ts = run._ts()
    console.rule(f"Pipeline start git={sweep_sha} ts={sweep_ts}")

    # 2. Cost surprise ONCE (early exit if user declines)
    tiers = [int(t) for t in args.tiers.split(",")]
    if not args.yes:
        qa = run._load_golden_qa()
        n_q = min(args.limit or len(qa), len(qa))
        if not run._cost_surprise(tiers, n_q, console):
            return 1

    # 3. Capture (async) — pass --yes through to suppress inner prompt
    if not args.no_capture:
        run_args = _build_run_args(args, sweep_sha, sweep_ts, yes=True)
        code = await run.amain(run_args, console)
        if code != 0:
            return code

    # 4. Score (async)
    if not args.no_score:
        score_args = _build_score_args(args, yes=True)
        code = await score.amain(score_args, console)
        if code != 0:
            return code

    # 5. Compare (sync — wrapped)
    if not args.no_compare:
        compare_args = _build_compare_args(args)
        code = await asyncio.to_thread(compare._run, compare_args)
        if code != 0:
            return code

    # 6. Freeze (sync — wrapped, optional)
    if args.freeze:
        path = await asyncio.to_thread(
            freeze.freeze,
            version=args.freeze,
            force=False,  # honor refuse-clobber default
            results_dir=Path(args.results_dir),
            source=None,  # default = results_dir/comparison.md
        )
        console.print(f"[green]Frozen: {path}[/green]")

    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    return asyncio.run(amain(args, Console()))
```

**Key invariant:** `asyncio.run` is called exactly once at the top of `main` — same boundary discipline as run.py, score.py. Avoids "event loop already running" errors.

## HARN-02: Single-Tier Rerun Semantics

Walking through `pipeline.py --tiers 4 --tier-4-from-cache <path>`:

1. **Capture stage** (`run.amain`): synthesizes args with `tiers="4"`, `tier_4_from_cache=<path>`. `run._capture_tier(4, ...)` writes:
   - `evaluation/results/queries/tier-4-{sweep_ts}.json` (NEW file; existing `tier-4-2026-05-02T11_42_15Z.json` is untouched)
   - `evaluation/results/costs/tier-4-eval-{sweep_ts}.json` (NEW file)

2. **Score stage** (`score.amain`): tiers="4". `_latest_query_log(queries_dir, 4)` (score.py:105) globs `tier-4-*.json`, sorts by mtime DESC. The new file from step 1 has the freshest mtime → it wins. Writes:
   - `evaluation/results/metrics/tier-4-{sweep_ts}.json` (NEW)
   - `evaluation/results/costs/ragas-judge-tier-4-{sweep_ts}.json` (NEW)

3. **Compare stage** (`compare._run`): tiers="1,2,3,4,5" (NOT just 4 — see argparse design below). For each tier, `compare._latest(results_dir / "queries", f"tier-{tier}-*.json")` (compare.py:48) finds the freshest file. Result:
   - tier-1, tier-2, tier-3, tier-5 → resolve to OLD files (from a previous full sweep)
   - tier-4 → resolves to the new file from step 1
   - `comparison.md` is regenerated, mixing fresh tier-4 + cached older tiers.

4. **Freeze stage** (skipped — typical single-tier reruns don't freeze; the user freezes only after a clean full sweep).

**This works correctly without modification.** Verified:
- `score.py:105-112` — `_latest_query_log` glob is `tier-{tier}-*.json`, fully tier-isolated.
- `compare.py:48-53` — `_latest` glob is parameter-driven, fully tier-isolated.
- Neither function deletes, moves, or otherwise mutates non-targeted tier artifacts.

**Pipeline.py contract for HARN-02:** when capture stage runs `--tiers 4`, the score stage MUST be invoked with `--tiers 4` too (otherwise the score stage sees no fresh queries log and re-scores stale cached tiers — wasteful and potentially confusing). The compare stage MUST be invoked with `--tiers 1,2,3,4,5` (the full set, OR `args.tiers_compare` if user wants partial) so the rollup contains all 5 tier rows including the freshly re-captured tier 4.

**Recommendation:**
- `--tiers` (capture+score): exactly the tiers user wants to (re-)capture.
- Compare stage: defaults to ALL 5 tiers (so single-tier reruns produce a complete `comparison.md`). Could be overridden via a separate `--compare-tiers` flag, but keeping it simple at "always all 5" is cleaner for v1.0.

## Cost-Surprise Consolidation

**Current behavior:**
- `run.amain` (line 307-310) calls `_cost_surprise(tiers, len(qa), console)` if not `args.yes`.
- `score.amain` (line 463-476) has its own inline cost prompt (no helper extraction); also gated on `args.yes`.

For a full sweep `--tiers 1,2,3,4,5` without consolidation: pipeline.py would prompt → run.amain would prompt → score.amain would prompt. Three prompts. Success Criterion 4 says "fire once".

**Recommended approach (least-invasive):**

1. Pipeline.py honors its own `--yes` flag at the top.
2. If `not args.yes`, pipeline.py calls `run._cost_surprise(tiers, n_q, console)` directly. If declined, exit 1.
3. Pipeline.py **always** sets `args.yes=True` on the synthesized `run_args` and `score_args` Namespaces — this suppresses the inner prompts unconditionally.

**Why this is least-invasive:**
- Zero changes to run.py / score.py.
- `_cost_surprise()` is already a callable helper (run.py:131-143) — pipeline.py imports it cleanly.
- Score.py's inline prompt isn't extracted to a helper, but pipeline.py doesn't need to call it — pipeline owns the prompt UX.

**Limitation:** the score-stage cost (~$0.003/q × 30q × 5 tiers ≈ $0.45) is NOT included in the consolidated prompt's ballpark. The run-stage `COST_PER_Q` (run.py:69) sums to ~$0.013/q × 30q × 5 tiers ≈ $1.95 capture + ~$0.45 score ≈ $2.40 total. Pipeline.py SHOULD show a combined estimate ("$~2.00 capture + ~$0.45 judge ≈ $2.45 total") in its prompt. Easy to compute inline using existing `run.COST_PER_Q` constant + the conservative judge factor (`0.003 * n_records * len(tiers)`).

**Implementation sketch:**

```python
def _pipeline_cost_estimate(tiers, n_q):
    capture = sum(run.COST_PER_Q.get(t, 0) * n_q for t in tiers)
    judge = 0.003 * n_q * len(tiers)
    return capture, judge, capture + judge

if not args.yes:
    cap, jud, total = _pipeline_cost_estimate(tiers, n_q)
    console.print(f"[yellow]Capture ~${cap:.4f} + judge ~${jud:.4f} ≈ ${total:.4f}[/yellow]")
    try:
        ans = input("Continue? [y/N]: ").strip().lower()
    except EOFError:
        ans = "n"
    if ans not in {"y", "yes"}:
        return 1
```

## Argparse Surface for pipeline.py

Mapping each flag to a success criterion or HARN requirement:

| Flag | Default | Justification |
|------|---------|---------------|
| `--tiers <csv>` | `1,2,3,4,5` | HARN-01 + HARN-02 — which tiers to (re-)capture+score. Drives `_cost_surprise` count. Comma-list parsed identically to run.py / score.py / compare.py. |
| `--limit <int>` | `None` | Smoke / dev override. Threaded into run.amain + score.amain (both already support it). |
| `--smoke-question-ids <csv>` | `None` | Reuses run.py's existing flag for Phase 1/2 smoke 5-question subset; pipeline.py threads through to run.amain only (score's `--limit` already truncates by index, and the queries log carries the right subset). |
| `--tier-4-from-cache <path>` | `None` | Success Criterion 2 — required for any sweep that includes tier 4. Threaded through to run.amain. |
| `--results-dir <path>` | `evaluation/results` | Single source of truth for queries/costs/metrics/frozen tree. Threaded into all 4 stages. |
| `--judge-model <slug>` | `score.JUDGE_LLM_SLUG_DEFAULT` | Threaded to score.amain. Phase 5 may want alt-judge dry runs. |
| `--judge-emb <slug>` | `score.JUDGE_EMB_SLUG_DEFAULT` | Threaded to score.amain. |
| `--batch-size <int>` | `10` | Threaded to score.amain (Pitfall 3 of 131-RESEARCH — bounds RAGAS concurrency). |
| `--mode <hybrid\|naive\|...>` | `hybrid` | Threaded to run.amain (Tier 3 LightRAG mode). |
| `--tier1-k <int>` | `5` | Threaded to run.amain (Tier 1 retrieval K). |
| `--freeze <version>` | `None` (no-op) | Success Criterion 1 — when set (e.g. `--freeze v1.0`), runs freeze stage at end. When unset, pipeline ends after compare. |
| `--yes` | `False` | Success Criterion 4 — single cost-surprise consolidation. Pipeline.py honors at top, threads `args.yes=True` into run/score args. |
| `--no-capture` | `False` | Skip capture stage (re-score+compare existing logs). Useful for judge-model A/B without re-paying capture cost. Optional v1.0 — could defer to v1.1 if LOC budget is tight. |
| `--no-score` | `False` | Skip score stage (compare-only over existing metrics). Cheap rollup-only refresh. Optional v1.0. |
| `--no-compare` | `False` | Skip compare stage. Rare — debug only. **Probably defer to v1.1.** |
| `--no-freeze` | (implicit when `--freeze` is unset) | Don't add an explicit flag; absence of `--freeze` is the no-freeze mode. |

**Recommended minimum v1.0 flag set:** `--tiers`, `--limit`, `--smoke-question-ids`, `--tier-4-from-cache`, `--results-dir`, `--judge-model`, `--judge-emb`, `--batch-size`, `--mode`, `--tier1-k`, `--freeze`, `--yes`. That's 12 flags — same order of magnitude as score.py's parser. Skip `--no-capture` / `--no-score` / `--no-compare` initially; add only if user tests demand it. (Phase 5 deliverable is "the tool that makes a single eval-date rerun mechanically possible" per ROADMAP — the optional skip-stage flags are not on the critical path.)

## Failure Semantics

Per-stage exit codes propagate up through the async chain:

| Stage | Failure trigger | pipeline.py behavior |
|-------|-----------------|----------------------|
| Capture (`run.amain`) | Tier prereq missing (no chroma_db, no .store_id, no graphml) → returns 2 | Pipeline aborts, returns 2. Score/compare/freeze NOT run. |
| Capture | One tier's adapter throws mid-sweep | Currently `_capture_tier` returns None and run.amain prints `[yellow]Tier {tier} produced no log[/yellow]` then continues to next tier (run.py:316-318). **The exception is swallowed at the adapter level, not raised.** Pipeline sees exit code 0. **Pitfall:** partial captures pass to score stage; `_latest_query_log` may resolve to a stale prior file for the failed tier, masking the failure. **Mitigation:** pipeline.py should record `n_logs_written` per tier from run.amain's behavior — but this requires touching run.amain to return a richer status. **Recommendation: accept the existing semantics for v1.0** and document in pipeline.py docstring that partial-capture failures fall through to whatever the latest-mtime file happens to be. |
| Capture | `--tier-4-from-cache` missing while tier 4 is requested | run.amain prints yellow skip and continues other tiers (run.py:215-223). Returns 0. Pipeline continues to score, compare. comparison.md will reflect tier-4 from prior sweep (or em-dash if first run). |
| Score (`score.amain`) | OPENROUTER_API_KEY missing → returns 2 | Pipeline aborts, returns 2. Compare/freeze NOT run. |
| Score | RAGAS evaluate raises mid-tier with `raise_exceptions=False` (default) | Per-row NaN; tier-level scores written. Returns 0. |
| Compare (`compare._run`) | results_dir missing → returns 2 | Pipeline aborts, returns 2. Freeze NOT run. |
| Freeze (`freeze.freeze`) | `FileExistsError` (already frozen) | Raises out of `asyncio.to_thread`. Pipeline catches `(FileExistsError, FileNotFoundError, RuntimeError)` mirroring `freeze.main:90` and returns 2. |
| Freeze | `RuntimeError` (critical lib not installed) | Same — returns 2. |
| Freeze | `FileNotFoundError` (no comparison.md) | Same — returns 2. |

**Exit codes summary:**
- 0 = full sweep success.
- 1 = user declined cost prompt.
- 2 = a stage failed (specific stage prints the reason; pipeline.py prints which stage halted).

**Pitfall to surface in plan:** if freeze fails with `FileExistsError` mid-pipeline, the capture/score/compare have already succeeded — `comparison.md` is fresh, but no immutable snapshot exists. User should either bump version (`--freeze v1.0.1`) or `--force` (NOT exposed in pipeline.py initially — user can call `python -m evaluation.harness.freeze --version 1.0 --force` directly if they really want). Pipeline.py recommended behavior: on `FileExistsError`, print exactly the freeze module's `[red]Freeze refused: ...[/red]` message and exit 2 — comparison.md remains, user re-runs freeze separately.

## Test Surface

Mirroring the test conventions of `test_eval_run.py`, `test_eval_score.py`, `test_eval_freeze.py`:

### Unit tests (offline, ~6-10 cases)

1. **Argparse**: `build_parser().parse_args(["--help"])` raises SystemExit(0); required `--tiers` validation; default values match recommended flag set.
2. **Stage gating**: `--no-capture` skips capture (mock run.amain → assert not called). Same for `--no-score`, `--no-compare`. Absence of `--freeze` skips freeze.
3. **Cost-surprise single-prompt**: monkeypatch `input()` to return "y"; assert `run._cost_surprise` is called exactly once across the full sweep (mock all three amain functions).
4. **Cost-surprise abort**: monkeypatch `input()` to return "n"; assert pipeline returns 1 and run.amain / score.amain / compare._run are NEVER called.
5. **SHA propagation**: capture sweep_sha + sweep_ts; assert run.amain receives an args Namespace with `git_sha_override == sweep_sha` and `ts_override == sweep_ts`.
6. **Stage exit propagation**: have run.amain return 2 (mock); assert pipeline returns 2 and score.amain/compare._run never called.
7. **Freeze passthrough**: `--freeze v1.0` invokes `freeze.freeze(version="v1.0", force=False, ...)` exactly once.
8. **Freeze refusal handling**: monkeypatch `freeze.freeze` to raise `FileExistsError`; pipeline returns 2.
9. **--yes consolidation**: `--yes` flag bypasses cost prompt entirely AND threads `yes=True` into run_args + score_args.
10. **Tier-4-from-cache passthrough**: assert the synthesized `run_args.tier_4_from_cache` matches the user-supplied value.

### Integration tests (offline, ~2-3 cases)

11. **End-to-end with stub stages**: mock all 4 entry points to return 0 / a known Path; run pipeline.py end-to-end with `--tiers 1 --freeze v1.0 --yes`; assert order of calls (run → score → compare → freeze) and that the synthesized args carry consistent values.
12. **HARN-02 single-tier rerun simulation**: build a fake results_dir with tier-{1,2,3,5} pre-existing files (from "prior sweep"); mock run.amain to write a new tier-4 file with fresh mtime; **don't mock compare._run** — let it actually run against the fixture; assert comparison.md contains all 5 tier rows and tier-4's row reflects the fresh data. (Reuses `test_eval_freeze.py::_build_fixture` pattern.)

### Live smoke (gated `@pytest.mark.live`, ~1 case, ~$0.05-0.15 cost)

13. **Live tier-1 single-tier sweep**: `--tiers 1 --limit 5 --yes` — uses real OpenRouter API for 5 questions, no freeze. Verifies the full chain end-to-end. Skip-cleanly if `OPENROUTER_API_KEY` is unset (use existing `live_eval_keys_ok` fixture from conftest.py:21). **Note** the v1.1 hardening item: pyproject lacks `addopts = "-m 'not live'"` so a bare `pytest test_eval_pipeline.py` will run this — the plan should explicitly use `pytest -m 'not live' evaluation/tests/test_eval_pipeline.py` for offline runs and `pytest -m live evaluation/tests/test_eval_pipeline.py -k pipeline` for the live smoke.

### CLI quality gate (post-test, sketch ~5 sub-checks, mirroring Plan 04-01's gate)

14. `pipeline --help` exits 0, prints all 12 expected flags.
15. `pipeline --tiers 99` exits 2 with unsupported-tier message (delegates to run.amain's existing validation).
16. `pipeline --tiers 4` (no `--tier-4-from-cache`) — capture skips tier 4 with yellow notice; score/compare run; pipeline exits 0. (Not strictly a "failure" — verifies graceful skip behavior.)
17. `pipeline --tiers 1 --limit 0 --yes` — n_q=0; either passes through 0 to all stages OR pipeline.py guards on `n_q == 0`. Recommend the latter — print warning and exit 1.
18. **Importable**: `from evaluation.harness.pipeline import amain, build_parser, main; assert callable(amain)`.

## Pitfalls Discovered While Reading the Existing Modules

1. **`_git_sha()` lazy capture per-tier** (run.py:159) — Already discussed at length. Phase 5's "single SHA" requirement collides with this; recommended fix is the `git_sha_override` kwarg.

2. **`_capture_tier` swallows adapter exceptions silently** — When an adapter raises mid-question, the loop continues; when it returns None, run.amain prints yellow and moves to the next tier. Pipeline.py cannot distinguish "tier failed mid-question, partial log written" from "tier wrote a complete log". **Implication for HARN-02:** if a single-tier rerun's capture partially fails, score stage will resolve `_latest()` to whatever was written (possibly an incomplete log), and the rollup will silently include corrupt tier data. **Mitigation:** out of scope for v1.0; document as known limitation.

3. **score.amain has inline cost prompt, not extracted to a helper** (score.py:463-476) — pipeline.py cannot reuse it the way it can reuse `run._cost_surprise`. This is fine because pipeline.py owns the consolidated prompt and just sets `args.yes=True` on score_args. But the planner should know: there is NO `score._cost_surprise` to import; score's prompt is suppressed by the `args.yes` flag.

4. **`_latest()` ties on mtime, not on content** — If a single-tier rerun produces a NEW tier-4 file, `_latest()` correctly picks the newer mtime. BUT: if the user has been doing manual file moves (e.g. `cp old_capture.json queries/tier-4-2026-01-01T00_00_00Z.json`) without preserving mtimes, `_latest()` may pick a "newer ts in filename but older mtime" file. This is a pre-existing harness quirk — pipeline.py inherits it. **Pitfall:** document; do not fix in Phase 5.

5. **`asyncio.run` boundary** — pipeline.py must call `asyncio.run` exactly once. If a developer is tempted to "factor cleanly" by wrapping each stage in its own `asyncio.run(...)`, they will hit "RuntimeError: There is no current event loop in thread 'MainThread'" or similar. This is also called out in run.py's docstring (Pitfall 5 of 131-RESEARCH). **Pipeline.py top-level entry point is `main()` → `asyncio.run(amain(...))` — NEVER `asyncio.run` inside amain.**

6. **`compare._run` is sync; needs `asyncio.to_thread`** — Calling `compare._run(args)` directly from `pipeline.amain` (async) blocks the event loop. Same for `freeze.freeze`. Use `asyncio.to_thread`.

7. **`freeze.freeze` raises on dirty git tree?** — Re-read freeze.py line by line: it does NOT raise on dirty tree. `_git_dirty()` is a *manifest field*, not a precondition. Pipeline.py inherits this — a dirty tree DOES freeze, with `git_dirty: true` recorded in the manifest. This matches Phase 4 decision (manifest-only, not refusal). Good — no special handling needed.

8. **Tier-4-from-cache flag plumbing** — `run.py:351` defines `--tier-4-from-cache` with `default=None`. Pipeline.py must thread it through verbatim. **Subtlety:** if user passes `--tier-4-from-cache <path>` to pipeline.py BUT does NOT include tier 4 in `--tiers`, the flag is silently ignored. Pipeline.py should warn (yellow) when this happens.

9. **`--smoke-question-ids` AND `--limit` interaction** — run.py:295-305: smoke filter applies BEFORE `--limit`. Pipeline.py must replicate this ordering when synthesizing run_args (just pass both through; run.amain handles).

10. **Score stage requires queries to exist** — score.amain returns 1 if no query logs are found (`return 1` at score.py:455). When pipeline.py runs capture+score back-to-back, this can't happen unless capture wrote nothing. **Pitfall:** if `--no-capture` is offered as a v1.0 flag and user runs `pipeline --no-capture --tiers 4` against an empty queries dir, score stage exits 1 — pipeline propagates 1. Document or guard.

11. **Multi-judge spot-check coexistence (CAP-02)** — Phase 8 will introduce a parallel "spot check" pipeline that re-scores existing captures with an alternate judge. **Implication for Phase 5:** don't bake "judge model" assumptions into pipeline.py's API surface. The `--judge-model` flag is the right level of abstraction; Phase 8 will reuse `score.amain` directly without going through pipeline.py.

12. **Embedder-provenance phase ordering** — Phase 6 (Embedder Provenance Capture) modifies `run._capture_tier` to record per-tier embedder model in the queries JSON. Phase 5 and Phase 6 are listed as parallel-friendly in ROADMAP. **Implication:** if Phase 5 lands first, Phase 6 must NOT break pipeline.py's run_args synthesis. The minimal touch surface (`git_sha_override` / `ts_override` kwargs) is orthogonal to Phase 6's per-tier embedder field — no conflict expected, but the planner should note in PLAN.md that pipeline.py should not own embedder metadata.

13. **Tests are run from `evaluation/tests/` AND `tests/` (top-level)** — conftest.py at evaluation/tests/conftest.py adds repo-root to sys.path; pipeline.py tests SHOULD live at `evaluation/tests/test_eval_pipeline.py` to inherit the same fixtures.

14. **`pytest live-deselect-by-default` not configured** — From STATE.md v1.1 hardening: a bare `pytest path/to/file.py` runs live tests silently. Phase 5 plan must explicitly invoke `pytest -m 'not live' evaluation/tests/test_eval_pipeline.py` for offline runs to avoid accidental API spend.

## LOC Budget Recommendation

**Estimate methodology** — heeding Plan 04-01's lesson, this is a **raw LOC** estimate (counting blank lines, imports, docstrings) not SLOC. Verify via `wc -l` on a sketch before locking.

**Component breakdown (raw LOC):**

| Component | Estimate |
|-----------|----------|
| Module docstring (10-line description with usage examples mirroring run.py / score.py / freeze.py docstring style) | ~25 |
| Imports + sys.path bootstrap | ~15 |
| `_pipeline_cost_estimate(tiers, n_q)` helper | ~10 |
| `_build_run_args(args, sweep_sha, sweep_ts, yes)` builder | ~25 |
| `_build_score_args(args, yes)` builder | ~20 |
| `_build_compare_args(args)` builder | ~10 |
| `async def amain(args, console)` orchestration body (incl. cost prompt + 4 stage calls + console banners) | ~60 |
| `def build_parser()` (12 flags × ~3 lines each) | ~50 |
| `def main(argv)` + `if __name__ == "__main__"` | ~5 |
| **Total** | **~220 raw LOC** |

**Recommended LOC budget: 180-260 raw LOC.**

If the plan is going to set a `max_lines` hard cap (Plan 04-01 used 95), Phase 5's reasonable cap is **260** (room for in-line comments, blank-line separation, no semicolon-stacking compromises). Below 200 risks the same compression pain as Plan 04-01 (3 iterations to fit). **Verify the sketch with `printf '%s\n' "$SKETCH" | wc -l` before locking.** This is the explicit Plan 04-01 lesson.

If pipeline.py grows above 280 raw LOC, the planner should consider whether a helper module (e.g. `evaluation/harness/_pipeline_args.py` with the three `_build_*_args` builders) is justified. For v1.0 single-file is preferable.

## Open Questions

1. **Should pipeline.py expose `--force` for freeze?**
   - What we know: freeze.py has `--force` to overwrite an existing frozen artifact.
   - What's unclear: from the user's POV during a full sweep, having pipeline silently overwrite a frozen v1.0 doc is dangerous. Users who *want* to overwrite should run freeze.py directly.
   - Recommendation: do **not** expose `--force` in pipeline.py for v1.0. Refuse-clobber is the safer default; user invokes `python -m evaluation.harness.freeze --force ...` explicitly when they really mean it.

2. **Should pipeline.py auto-bump freeze version on `FileExistsError`?**
   - Recommendation: NO. Auto-versioning hides the user's intent. Print the freeze module's refusal message verbatim and exit 2.

3. **Does pipeline.py need its own `--output-dir` separate from `--results-dir`?**
   - run.py uses `--output-dir` (default `evaluation/results`); compare.py / freeze.py use `--results-dir` (default `evaluation/results`). They're synonyms in the codebase but spelled differently.
   - Recommendation: pipeline.py uses `--results-dir` (matches the newer compare.py / freeze.py convention); the synthesized `run_args.output_dir` and `score_args.output_dir` both get set to `args.results_dir`.

4. **Is there a way for pipeline.py to record a top-level "pipeline run manifest"?**
   - I.e. a file `evaluation/results/pipeline-runs/{sweep_ts}.json` recording {tiers, n_q, sweep_sha, sweep_ts, exit_code_per_stage, freeze_version}.
   - Out of scope for v1.0 per ROADMAP. **Defer to v1.1.**

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python 3.10+ | All harness modules | (assumed yes) | 3.13.1 (per STATE.md Plan 04-01) | — |
| `rich` | Console output | (yes — used by run/score/freeze) | per pyproject [shared] | — |
| All four harness modules importable | Composition | (yes — confirmed by reading source) | freeze.py 95 LOC, run.py 384 LOC, score.py 579 LOC, compare.py 404 LOC | — |
| Tier-specific runtime prereqs (chroma_db/, .store_id, lightrag_storage/) | Capture stage `_check_prereqs` | (varies; checked at run.amain entry) | — | run.amain returns exit 2 with clear message; pipeline propagates |
| OPENROUTER_API_KEY | Score stage | (varies) | — | score.amain returns exit 2 with clear message; pipeline propagates |
| 4 critical libs (lightrag-hku, raganything, openai-agents, ragas) | Freeze stage `_library_versions` | (yes per STATE.md Plan 04-01) | 1.4.15 / 1.2.10 / 0.14.6 / 0.4.3 | RuntimeError + exit 2 |

**No new external dependencies for pipeline.py.** It is pure composition.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4 (per pyproject `[dependency-groups].test`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (markers only; no addopts) |
| Quick run command | `pytest evaluation/tests/test_eval_pipeline.py -m 'not live' -x` |
| Full suite command | `pytest evaluation/tests/ -m 'not live' -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HARN-01 | Single SHA propagated across all per-tier outputs | unit | `pytest evaluation/tests/test_eval_pipeline.py::test_sha_propagation -x` | ❌ Wave 0 |
| HARN-01 | Capture → score → compare → freeze sequence | integration | `pytest evaluation/tests/test_eval_pipeline.py::test_full_sweep_order -x` | ❌ Wave 0 |
| HARN-01 | Single cost-surprise prompt | unit | `pytest evaluation/tests/test_eval_pipeline.py::test_single_cost_prompt -x` | ❌ Wave 0 |
| HARN-02 | Single-tier rerun preserves other tiers | integration | `pytest evaluation/tests/test_eval_pipeline.py::test_single_tier_rerun_preserves_others -x` | ❌ Wave 0 |
| Phase 4 contract | freeze() called with locked signature | unit | `pytest evaluation/tests/test_eval_pipeline.py::test_freeze_passthrough -x` | ❌ Wave 0 |
| Live smoke | Real OpenRouter call works end-to-end | live | `pytest -m live evaluation/tests/test_eval_pipeline.py::test_pipeline_live_tier1_smoke` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest evaluation/tests/test_eval_pipeline.py -m 'not live' -x` (offline only).
- **Per wave merge:** `pytest evaluation/tests/ -m 'not live' -x` (full offline suite).
- **Phase gate:** Full offline suite green + 1 live smoke (~$0.05) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `evaluation/tests/test_eval_pipeline.py` — covers HARN-01 + HARN-02 + freeze-passthrough + cost-prompt + stage-gating
- [ ] (Optional) helper fixture builder — a small `_build_pipeline_fixture` à la `test_eval_freeze.py::_build_fixture` to construct a fake results_dir for the integration test
- [ ] (No framework install needed — pytest already in `[dependency-groups].test`)

## Sources

### Primary (HIGH confidence — read directly)
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/run.py` (384 LOC) — `amain`, `_capture_tier`, `_git_sha`, `_ts`, `_check_prereqs`, `_cost_surprise`, `COST_PER_Q`, `DEFAULT_SMOKE_IDS`, `build_parser`, `main`
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/score.py` (579 LOC) — `amain`, `score_query_log`, `_build_judge`, `JUDGE_LLM_SLUG_DEFAULT`, `JUDGE_EMB_SLUG_DEFAULT`, `JUDGE_MAX_TOKENS`, `_latest_query_log`, inline cost prompt, `build_parser`, `main`
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/compare.py` (404 LOC) — `_run`, `aggregate_tier`, `aggregate_class_rollup`, `_latest`, `emit_markdown`, `_detect_judge_provenance`, `SUPPORTED_TIERS`, `build_parser`, `main`
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/freeze.py` (95 LOC) — `freeze(version, force, results_dir, source) -> Path` LOCKED Phase 4 contract
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/STATE.md` — Phase 4 closure notes, v1.1 hardening items
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/REQUIREMENTS.md` — HARN-01 / HARN-02 verbatim
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/ROADMAP.md` — Phase 5 success criteria

### Secondary (referenced for pattern reuse)
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/tests/test_eval_freeze.py` — `_build_fixture` pattern reusable for integration test
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/tests/test_eval_run.py` — Mock pattern for `_capture_tier`
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/tests/conftest.py` — `live_eval_keys_ok` fixture
- `/Users/patrykattc/work/git/rag-architecture-patterns/pyproject.toml` — Python ≥3.10, pytest 8.4, [tool.pytest.ini_options].markers (NO `addopts = "-m 'not live'"`)

### Tertiary
- None. No external library research required — Phase 5 is pure repo-internal composition.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Adding `git_sha_override` / `ts_override` kwargs to `run._capture_tier` is acceptable to the user (vs. zero-touch alternatives like env-var or monkeypatch) | Architectural Responsibility Map → MAY need targeted edit | If user vetoes any run.py change, fall back to monkeypatch approach (Section 2 Alternative). Adds ~5 LOC of pipeline.py boilerplate; same correctness. |
| A2 | Score.amain's inline cost prompt cannot be reused; pipeline owns the consolidated prompt | Cost-Surprise Consolidation | If user wants a unified prompt helper, would need a small refactor to score.py to extract `_cost_surprise` (out of scope for v1.0 by recommendation). |
| A3 | `--no-capture` / `--no-score` / `--no-compare` flags are NOT critical-path for v1.0 | Argparse Surface | If user wants them in v1.0, +30 raw LOC for the flags + branches. Easy to add. |
| A4 | Compare stage always operates on ALL 5 tiers regardless of `--tiers` | HARN-02 walkthrough | If user wants `--tiers 4` to also produce a 1-row comparison.md, adds a `--compare-tiers` flag. Recommend deferring. |
| A5 | Live smoke test budget (~$0.05) is acceptable | Test Surface | If user prefers offline-only validation, drop test #13. Reduces verification confidence but stays at unit + integration coverage. |
| A6 | LOC estimate of 220 raw / 260 hard cap is reasonable | LOC Budget | Could be off ±50 LOC. **Mandatory:** verify via `wc -l` on a sketch before locking the cap, per Plan 04-01 lesson. |

## Metadata

**Confidence breakdown:**
- Integration contract (run/score/compare/freeze surfaces): HIGH — read all four modules end-to-end with line numbers.
- Composition pattern (asyncio.to_thread, single asyncio.run): HIGH — explicit guidance in run.py docstring + standard Python idiom.
- HARN-02 single-tier rerun semantics: HIGH — verified `_latest()` is fully tier-isolated by reading `compare.py:48` + `score.py:105`.
- SHA propagation kwarg approach: HIGH — minimum-surface change, fully testable.
- LOC estimate: MEDIUM — sketch-driven; sensitive to argparse verbosity and docstring length. Verify before locking.
- Test surface: HIGH — mirrors three already-shipped test modules (test_eval_run, test_eval_score, test_eval_freeze).
- Pitfalls: HIGH — every pitfall traceable to a specific line in an existing module.

**Research date:** 2026-05-05
**Valid until:** 2026-06-05 (no external dependencies; only invalidated by edits to run.py / score.py / compare.py / freeze.py — at which point re-read those files).

## RESEARCH COMPLETE

**Recommended approach for the planner to lock:**

1. **Land `evaluation/harness/pipeline.py` at ~220 raw LOC** (hard cap 260; verify sketch with `wc -l` per Plan 04-01 lesson).

2. **Capture sweep identity ONCE** at top of `pipeline.amain`: `sweep_sha = run._git_sha()` and `sweep_ts = run._ts()`. Print as a banner.

3. **Make a single targeted edit to `evaluation/harness/run.py`**: add optional `git_sha_override` / `ts_override` kwargs to `_capture_tier` with backwards-compatible defaults (~3 effective LOC). Pipeline.py threads sweep_sha + sweep_ts via these kwargs through a synthesized args.Namespace. Zero edits to score.py, compare.py, freeze.py.

4. **In-process composition with mixed sync/async**: `await run.amain(...)`, `await score.amain(...)`, `await asyncio.to_thread(compare._run, ...)`, `await asyncio.to_thread(freeze.freeze, ...)`. Single `asyncio.run` boundary in `main()`.

5. **Single cost-surprise prompt** by calling `run._cost_surprise()` at pipeline.py top and threading `args.yes=True` into all downstream Namespaces. Show combined ballpark (capture + judge ≈ $X) using `run.COST_PER_Q` + conservative judge factor.

6. **HARN-02 works for free**: `compare._latest` and `score._latest_query_log` are already tier-isolated mtime-DESC. Compare stage operates on all 5 tiers regardless of `--tiers`. Verified by source reading.

7. **Argparse v1.0 surface**: 12 flags (`--tiers`, `--limit`, `--smoke-question-ids`, `--tier-4-from-cache`, `--results-dir`, `--judge-model`, `--judge-emb`, `--batch-size`, `--mode`, `--tier1-k`, `--freeze`, `--yes`). Defer `--no-capture` / `--no-score` / `--no-compare` to v1.1.

8. **Failure semantics**: each stage's exit code propagates. Catch `(FileExistsError, FileNotFoundError, RuntimeError)` from freeze. Aborted cost prompt → exit 1. Stage failure → exit 2.

9. **Test surface**: 10 unit tests + 2 integration tests + 1 live smoke (~$0.05), placed at `evaluation/tests/test_eval_pipeline.py` mirroring existing fixture patterns. Use `pytest -m 'not live'` explicitly per v1.1 hardening note.

10. **Phase 5 unblocks Phase 7** — the full 5-tier rerun on a single eval-date will execute via `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --freeze v1.0 --yes` as the single command of record.

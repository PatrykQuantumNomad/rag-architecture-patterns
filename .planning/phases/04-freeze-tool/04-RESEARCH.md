# Phase 4: Freeze Tool — Research

**Researched:** 2026-05-05
**Domain:** Pure-Python CLI / sidecar manifest writer (no new external deps)
**Confidence:** HIGH

## Summary

Phase 4 adds `evaluation/harness/freeze.py` — a ~60-LOC pure-Python module that copies the current `evaluation/results/comparison.md` to `evaluation/results/frozen/eval-numbers-vX.Y.md` and writes a sidecar `eval-numbers-vX.Y.manifest.json` capturing git SHA, freeze timestamp, the EXACT capture/score/metrics files that fed the rollup (with mtimes), per-tier model, judge model, and pinned versions of `lightrag-hku`, `raganything`, `openai-agents`, `ragas`. Re-running with the same `--version` MUST refuse via `FileExistsError` semantics; `--force` overrides.

All raw materials already exist in the repo: `_git_sha()` / `_ts()` / `_ts_for_filename()` helpers in `evaluation/harness/run.py`, `_latest()` mtime resolution in `compare.py`, judge-model detection via `_detect_judge_provenance()` in `compare.py`, library versions via `importlib.metadata.version()` (verified working against `.venv` for all four packages), and `argparse` / `Console` patterns mirrored across `run.py` / `score.py` / `compare.py`. The 60-LOC budget is achievable because the work decomposes into ~6 small operations, all reusing existing code.

**Primary recommendation:** Build `freeze.py` as a single-module CLI with one public function `freeze(version: str, force: bool = False) -> Path` that returns the path of the written markdown. The CLI is a 6-line wrapper around `freeze()`. Reuse `_git_sha`, `_ts`, `_latest` via direct import from `evaluation.harness.run` and `evaluation.harness.compare`. Use `Path.write_text` after `Path.exists()` check (with `--force` short-circuit) — the FileExistsError-on-`open('x')` idiom is alternative but explicit `exists()` + custom error message is clearer for the prescribed wording. Library versions come from `importlib.metadata.version()` against the live venv at freeze time.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HARN-03 | User can produce a frozen artifact via `evaluation/harness/freeze.py` that refuses to clobber an existing `frozen/eval-numbers-vX.Y.md` — frozen docs are immutable once written | Path.exists() + sys.exit(2) with prescribed error wording (Common Pitfall 1); `--force` flag to override (Architecture Pattern 4) |
| HARN-04 | User can read a sidecar `frozen/eval-numbers-vX.Y.manifest.json` recording the git SHA, capture timestamps per tier, judge model, generation models per tier, and all relevant library versions (lightrag-hku, raganything, openai-agents, ragas) | Manifest schema (Sidecar Manifest Schema); `_git_sha()` reuse from run.py; `importlib.metadata.version()` proven working in venv (Standard Stack); per-tier capture timestamps + model already in QueryLog headers; judge model via `_detect_judge_provenance()` |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Markdown copy (comparison.md → frozen/) | freeze.py | — | Pure file I/O; new module |
| Manifest JSON emission | freeze.py | — | Pure file I/O; new module |
| Source file resolution (which capture/score/metrics fed rollup) | freeze.py reuses compare.py `_latest()` | compare.py | freeze.py imports `_latest`; do NOT re-implement |
| Git SHA + ISO 8601 Z timestamp | run.py helpers (`_git_sha`, `_ts`) | freeze.py | run.py is the canonical source per STATE.md; freeze.py imports |
| Library version pinning lookup | `importlib.metadata.version()` (stdlib) | freeze.py | Authoritative — reads installed dist-info, not pyproject text |
| Judge model detection | compare.py `_detect_judge_provenance()` | freeze.py | Already exists and is correct; reuse |
| Idempotence / clobber-refusal | freeze.py | — | New behavior owned here; CLI flag `--force` is the override |
| In-process invocation (Phase 5) | `freeze(version, force) -> Path` | freeze.py CLI wrapper | Phase 5 imports the function; CLI is a 6-line wrapper around it |

## Standard Stack

### Core (all stdlib + already-pinned project deps)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `argparse` | stdlib (Python 3.10+) | CLI flag parsing | Mirrors run.py / score.py / compare.py; no new dep [VERIFIED: pyproject.toml requires-python=">=3.10"] |
| `pathlib.Path` | stdlib | File ops, exists check, write_text | Used by every existing harness module [VERIFIED: grep `from pathlib import Path` in run.py / score.py / compare.py] |
| `json` | stdlib | Manifest serialization | Used by every harness module [VERIFIED: ubiquitous in harness] |
| `shutil.copy2` | stdlib | Copy comparison.md preserving mtime | Stdlib; preserves source mtime so frozen file's mtime reflects when comparison was generated, not when freeze ran [CITED: docs.python.org/3/library/shutil.html#shutil.copy2] |
| `importlib.metadata.version` | stdlib (Python 3.8+) | Read installed package versions | Authoritative for actual-installed-version (NOT pyproject text which is a constraint) [VERIFIED: returned `1.4.15 / 1.2.10 / 0.14.6 / 0.4.3` against `.venv` in this session] |
| `subprocess` | stdlib | Used inside `_git_sha()` | Already in run.py — no new usage [VERIFIED: run.py:30] |
| `datetime` | stdlib | Used inside `_ts()` | Already in run.py [VERIFIED: run.py:32] |
| `rich.console.Console` | rich>=14,<16 | Colored CLI output | Already pinned in `[shared]` extra [VERIFIED: pyproject.toml line ~28] |

**Installation:** No new packages. `freeze.py` runs from the existing `[evaluation]` + `[shared]` extras already required by `compare.py`.

**Version verification (2026-05-05 against `/Users/patrykattc/work/git/rag-architecture-patterns/.venv`):**
- `lightrag-hku == 1.4.15` ✓
- `raganything == 1.2.10` ✓
- `openai-agents == 0.14.6` ✓
- `ragas == 0.4.3` ✓
- `litellm == 1.83.0` (BONUS: judge LLM driver — recommend including in manifest)
- `chromadb == 1.5.8` (BONUS: Tier 1 + Tier 5 vector store — recommend including in manifest)

### Helpers Reused From Existing Code (do NOT re-implement)

| Helper | Location | What It Returns | Reuse Verdict |
|--------|----------|-----------------|---------------|
| `_git_sha()` | `evaluation/harness/run.py:72-79` | Short HEAD SHA (e.g. `"ce5c2ad"`) or `"unknown"` if git unavailable | **Import as-is.** Uses `cwd=_REPO_ROOT`, returns "unknown" on any subprocess failure. Does NOT detect dirty tree (see Pitfall 4). [VERIFIED: read run.py lines 72-79] |
| `_ts()` | `evaluation/harness/run.py:82-84` | ISO 8601 UTC second-precision (e.g. `"2026-05-05T19:30:00Z"`) | **Import as-is.** Single source of truth per STATE.md. [VERIFIED: read run.py lines 82-84] |
| `_ts_for_filename(ts)` | `evaluation/harness/run.py:87-89` | `ts.replace(":", "_")` for macOS-friendly filenames | Likely NOT needed — frozen filenames use `--version` not timestamps. [VERIFIED: read run.py lines 87-89] |
| `_latest(dir, pattern)` | `evaluation/harness/compare.py:48-52` | Most-recent mtime-DESC `Path` matching glob, or `None` | **Import as-is.** Identical resolution logic ensures `freeze.py` records the SAME files `compare.py` consumed. [VERIFIED: read compare.py lines 48-52] |
| `_detect_judge_provenance(results_dir)` | `evaluation/harness/compare.py:310-325` | `(judge_model_slug, judge_emb_slug)` tuple from latest `ragas-judge-*.json` | **Import as-is.** Falls back to `"google/gemini-2.5-flash"` / `"openai/text-embedding-3-small"` per 131-RESEARCH defaults. [VERIFIED: read compare.py lines 310-325] |
| `aggregate_tier(tier, results_dir)` | `evaluation/harness/compare.py:69-136` | Returns dict with `queries_path`, `cost_path`, `metrics_path`, `timestamp`, `git_sha`, `model` per tier — exactly the manifest fields we need | **Import and call** for each tier instead of re-globbing. Returns `None` for missing tiers (handle gracefully). [VERIFIED: read compare.py lines 116-136] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `importlib.metadata.version()` | Parse `pyproject.toml` deps with `tomllib` | pyproject lists CONSTRAINTS (`>=1.4,<2`); installed metadata lists ACTUAL pinned version. Manifest needs the pin, not the constraint. Use `importlib.metadata`. |
| `importlib.metadata.version()` | Parse `uv.lock` | uv.lock is authoritative for the lockfile but `importlib.metadata` reads the same dist-info that's actually loaded at runtime. Equivalent for a synced venv; `importlib.metadata` is stdlib (no toml parser needed). |
| `shutil.copy2()` | `Path.read_text()` + `Path.write_text()` | copy2 preserves mtime + permissions in one stdlib call. write_text resets mtime to "now" (loses signal of when the rollup was actually generated). Prefer copy2. |
| `Path.exists()` + manual error | `Path.open('x')` (exclusive-create) | open('x') raises FileExistsError automatically. Cleaner, but the prescribed error wording ("eval-numbers-v1.0.md already frozen — bump version or pass --force") is more naturally produced by an explicit `if path.exists() and not force: print(...); return 1` block. Either works; recommend explicit-exists-check for prescribed wording. |
| Capture file mtimes via `os.path.getmtime` | `Path.stat().st_mtime` | Equivalent. `stat()` is the pathlib idiom. |

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│  USER:  python -m evaluation.harness.freeze --version 1.0               │
│         │                                                                │
│         ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ argparse build_parser() — wrapping freeze() public function        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│         │                                                                │
│         ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ freeze(version: str, force: bool = False) -> Path                  │ │
│  │   1. Resolve out_md = frozen/eval-numbers-v{version}.md            │ │
│  │   2. If out_md.exists() and not force → exit 2 with prescribed msg │ │
│  │   3. Read source: comparison.md (must exist; bail if missing)      │ │
│  │   4. For each tier in [1..5]: aggregate_tier(t, results_dir)       │ │
│  │      → harvest queries_path / cost_path / metrics_path / model     │ │
│  │   5. judge_model, judge_emb = _detect_judge_provenance(results_dir)│ │
│  │   6. lib_versions = {p: importlib.metadata.version(p) for p in ...}│ │
│  │   7. shutil.copy2(comparison.md, out_md)                           │ │
│  │   8. Write out_md.with_suffix('.manifest.json') (manifest schema)  │ │
│  │   9. Return out_md                                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  PHASE 5 ENTRY: from evaluation.harness.freeze import freeze            │
│                 freeze(version="1.0", force=False)  # in-process call   │
└─────────────────────────────────────────────────────────────────────────┘

UPSTREAM (consumed read-only by freeze.py):
  evaluation/results/comparison.md       — copied verbatim
  evaluation/results/queries/tier-N-*.json   — provenance via _latest()
  evaluation/results/metrics/tier-N-*.json   — provenance via _latest()
  evaluation/results/costs/tier-N-eval-*.json — provenance via _latest()
  evaluation/results/costs/ragas-judge-*.json — judge model via _detect_judge_provenance()
  .venv installed dist-info              — library versions via importlib.metadata

DOWNSTREAM (produced by freeze.py):
  evaluation/results/frozen/eval-numbers-v{version}.md
  evaluation/results/frozen/eval-numbers-v{version}.manifest.json
```

### Recommended Project Structure

```
evaluation/harness/
├── freeze.py            # NEW — ~60 LOC pure-Python (no new deps)
├── run.py               # existing — provides _git_sha, _ts, _ts_for_filename
├── score.py             # existing
├── compare.py           # existing — provides _latest, aggregate_tier, _detect_judge_provenance
└── ...

evaluation/results/
└── frozen/              # NEW directory created by freeze.py first run
    ├── eval-numbers-v1.0.md
    └── eval-numbers-v1.0.manifest.json

evaluation/tests/
└── test_eval_freeze.py  # NEW — TDD-friendly unit tests using tmp_path fixture
```

### Pattern 1: Public Function + Thin CLI Wrapper (for in-process Phase 5 reuse)

**What:** Expose `freeze(version, force) -> Path` as the public entry point. CLI is a 6-line wrapper.

**When to use:** Whenever a CLI tool will also be called as a function from another module (Phase 5 pipeline.py).

**Example:**
```python
# Source: this research, mirrors run.py amain() / build_parser() / main() pattern (run.py:274-380)
def freeze(version: str, force: bool = False, results_dir: Path = None) -> Path:
    """Public entry — Phase 5 pipeline calls this directly."""
    results_dir = results_dir or _REPO_ROOT / "evaluation" / "results"
    out_md = results_dir / "frozen" / f"eval-numbers-v{version}.md"
    if out_md.exists() and not force:
        raise FileExistsError(
            f"eval-numbers-v{version}.md already frozen — bump version or pass --force"
        )
    # ... resolve sources, copy, write manifest ...
    return out_md


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = freeze(version=args.version, force=args.force)
    except FileExistsError as e:
        Console().print(f"[red]{e}[/red]")
        return 2
    Console().print(f"[green]Wrote {path}[/green]")
    return 0
```

### Pattern 2: Reuse `aggregate_tier()` to Record Source Files

**What:** Instead of re-globbing queries/costs/metrics inside freeze.py, call `aggregate_tier(t, results_dir)` for each tier. The returned dict already contains `queries_path`, `cost_path`, `metrics_path`, `timestamp`, `git_sha`, `model` — exactly what HARN-04 needs.

**When to use:** When you need to record EXACTLY which files compare.py would consume, with zero risk of mtime drift between rollup time and freeze time (assuming the user runs `compare` immediately before `freeze`, which is what Phase 5 will do).

**Example:**
```python
# Source: aggregate_tier() returns these fields per compare.py:116-136
from evaluation.harness.compare import aggregate_tier, _detect_judge_provenance, SUPPORTED_TIERS

per_tier = {}
for t in SUPPORTED_TIERS:
    row = aggregate_tier(t, results_dir)
    if row is None:
        per_tier[f"tier-{t}"] = {"status": "missing"}
        continue
    per_tier[f"tier-{t}"] = {
        "queries_path": row["queries_path"],
        "queries_mtime": Path(row["queries_path"]).stat().st_mtime,
        "cost_path": row["cost_path"],
        "metrics_path": row["metrics_path"],
        "timestamp": row["timestamp"],
        "git_sha": row["git_sha"],
        "generation_model": row["model"],
    }
```

### Pattern 3: `importlib.metadata` for Authoritative Pinned Versions

**What:** Use `importlib.metadata.version("package-name")` to read the actual installed version, NOT `pyproject.toml` (which holds constraints, not pins).

**When to use:** Whenever a manifest must record "what was installed when this artifact was produced".

**Example:**
```python
# Source: docs.python.org/3/library/importlib.metadata.html#distribution-versions
# VERIFIED working in this session against the project .venv
from importlib.metadata import version, PackageNotFoundError

PINNED_LIBS = ("lightrag-hku", "raganything", "openai-agents", "ragas")

def _library_versions() -> dict[str, str]:
    out = {}
    for pkg in PINNED_LIBS:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
    return out
```

### Pattern 4: Refuse-to-Clobber via Explicit `Path.exists()`

**What:** Check `out_md.exists()` BEFORE writing. If exists and `not force`, exit non-zero with the prescribed error wording. If exists and `force`, proceed (silently overwrites both `.md` and `.manifest.json`).

**When to use:** Whenever a CLI must produce immutable artifacts by default with an explicit override.

**Example:**
```python
# Source: this research; mirrors what `mv -i` / `cp -n` do at the OS level
if out_md.exists() and not force:
    msg = f"eval-numbers-v{version}.md already frozen — bump version or pass --force"
    print(msg, file=sys.stderr)
    return 2
```

**Exit codes (mirrors compare.py / run.py / score.py conventions):**
- `0` = success (file written)
- `1` = user-aborted (n/a here unless we add a confirm prompt — DO NOT add one)
- `2` = pre-flight failure (already frozen, comparison.md missing, etc.)

### Anti-Patterns to Avoid

- **Re-implementing `_latest()` inside freeze.py.** Tasks must `from evaluation.harness.compare import _latest, aggregate_tier, _detect_judge_provenance`. Re-implementation drifts from the canonical resolver and breaks HARN-04's "exactly which files fed comparison.md" guarantee.
- **Reading versions from `pyproject.toml` text.** Constraints ≠ pins. `importlib.metadata` reads the actual loaded dist-info.
- **Writing manifest BEFORE the markdown.** If the markdown copy fails (disk full, permission denied), a manifest pointing to a non-existent file is worse than no manifest. Order: copy md → write manifest → return. (Acceptable inverse: write manifest atomically as `.tmp` first, copy md, rename `.tmp` — but the simple ordering is fine for a 60-LOC tool.)
- **Treating dirty git tree as identical to clean tree.** `_git_sha()` returns the HEAD SHA regardless of working-tree state. The manifest should also record dirty/clean (Pitfall 4). Add `git_dirty: bool` to the manifest.
- **Reading `comparison.md` and re-emitting modified content.** The whole point of freeze is "copy-pasteable" — the frozen markdown is byte-identical to comparison.md. Any modification (e.g. injecting a "frozen at" footer) breaks copy-paste-into-blog determinism. Use `shutil.copy2`, not `read_text` + `write_text`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Determine "latest" capture/score/metrics file per tier | New `find_latest_for_tier()` helper | `evaluation.harness.compare._latest` | Already battle-tested; consistency with rollup |
| Aggregate per-tier provenance (timestamp/git_sha/model) | New globbing logic | `evaluation.harness.compare.aggregate_tier` | Already returns exactly the fields HARN-04 needs |
| Detect the judge LLM model | Re-read latest `ragas-judge-*.json` | `evaluation.harness.compare._detect_judge_provenance` | Already exists with sensible defaults |
| Get git SHA | New `subprocess.check_output` | `evaluation.harness.run._git_sha` | Single source of truth per STATE.md |
| Get ISO 8601 UTC Z timestamp | New `datetime.now(...)` call | `evaluation.harness.run._ts` | Single source of truth per STATE.md |
| Read installed package version | Parse `uv.lock` or `pyproject.toml` | `importlib.metadata.version` (stdlib) | Reads actual loaded dist-info |
| Copy a file preserving mtime | `read_text()` + `write_text()` | `shutil.copy2` | Stdlib, one call, preserves metadata |
| Determine "did the user pass --force?" | Custom env var check | argparse `action="store_true"` | Mirrors `--yes` patterns in run.py / score.py |

**Key insight:** This phase has zero novel infrastructure to invent. Every primitive freeze.py needs already exists in `run.py` / `compare.py` / stdlib. The 60-LOC budget is achievable precisely because we are wiring existing pieces, not writing new ones.

## Sidecar Manifest Schema

Concrete schema for `eval-numbers-v{version}.manifest.json`. Every field has a verified source.

```json
{
  "$schema_version": "1.0",
  "version": "1.0",
  "frozen_at": "2026-05-05T19:30:00Z",
  "git_sha": "0f1f1a8",
  "git_dirty": false,
  "source_markdown": "evaluation/results/comparison.md",
  "source_markdown_mtime": "2026-05-05T14:13:00Z",
  "frozen_markdown": "evaluation/results/frozen/eval-numbers-v1.0.md",
  "judge": {
    "model": "google/gemini-2.5-flash",
    "embedder": "openai/text-embedding-3-small",
    "max_tokens": 8192
  },
  "per_tier": {
    "tier-1": {
      "status": "present",
      "generation_model": "google/gemini-2.5-flash",
      "capture_timestamp": "2026-05-02T17:26:59Z",
      "capture_git_sha": "ce5c2ad",
      "queries_path": "evaluation/results/queries/tier-1-2026-05-02T17_26_59Z.json",
      "queries_mtime": "2026-05-02T13:27:00Z",
      "cost_path": "evaluation/results/costs/tier-1-eval-20260502T172659Z.json",
      "cost_mtime": "2026-05-02T13:27:00Z",
      "metrics_path": "evaluation/results/metrics/tier-1-2026-05-02T17_26_59Z.json",
      "metrics_mtime": "2026-05-02T14:36:00Z"
    },
    "tier-2": { "...": "..." },
    "tier-3": { "...": "..." },
    "tier-4": { "status": "missing" },
    "tier-5": { "status": "present", "...": "..." }
  },
  "library_versions": {
    "lightrag-hku": "1.4.15",
    "raganything": "1.2.10",
    "openai-agents": "0.14.6",
    "ragas": "0.4.3",
    "litellm": "1.83.0",
    "chromadb": "1.5.8"
  },
  "python_version": "3.11.7"
}
```

**Field provenance (every entry has a verified source):**

| Field | Source |
|-------|--------|
| `version` | CLI `--version` arg |
| `frozen_at` | `evaluation.harness.run._ts()` |
| `git_sha` | `evaluation.harness.run._git_sha()` |
| `git_dirty` | NEW helper — `subprocess.check_output(["git","status","--porcelain"])` non-empty (see Pitfall 4) |
| `source_markdown` | constant `"evaluation/results/comparison.md"` (relative to `_REPO_ROOT`) |
| `source_markdown_mtime` | `comparison_md.stat().st_mtime` formatted to ISO 8601 Z |
| `frozen_markdown` | computed `frozen/eval-numbers-v{version}.md` |
| `judge.model`, `judge.embedder` | `compare._detect_judge_provenance(results_dir)` — returns the tuple |
| `judge.max_tokens` | constant `score.JUDGE_MAX_TOKENS` (8192) — import from score.py |
| `per_tier[tier-N].status` | `"present"` if `aggregate_tier()` returned non-None; else `"missing"` |
| `per_tier[tier-N].generation_model` | `aggregate_tier(t, ...).model` (already pulled from QueryLog header) |
| `per_tier[tier-N].capture_timestamp` | `aggregate_tier(t, ...).timestamp` (from QueryLog) |
| `per_tier[tier-N].capture_git_sha` | `aggregate_tier(t, ...).git_sha` (from QueryLog — note: this is the SHA at CAPTURE time, possibly ≠ freeze-time SHA) |
| `per_tier[tier-N].{queries,cost,metrics}_path` | `aggregate_tier()` returns these as already-resolved string paths |
| `per_tier[tier-N].{queries,cost,metrics}_mtime` | `Path(p).stat().st_mtime` formatted to ISO 8601 Z |
| `library_versions[*]` | `importlib.metadata.version(name)` (verified working) |
| `python_version` | `sys.version.split()[0]` (e.g. `"3.11.7"`) |

**Why TWO git_shas (top-level + per-tier-capture):** The top-level `git_sha` is when freeze ran; `per_tier.capture_git_sha` is when each tier was captured. Phase 7 will run all 5 tiers under one SHA, so they should match — but Phase 4 has no way to enforce that, and a mismatch is a real provenance signal worth surfacing (e.g. tier-4 was re-captured after a cherry-pick).

**Path serialization:** Store paths RELATIVE to `_REPO_ROOT` (not absolute). This makes the manifest readable when the repo is cloned to a different machine. Use `Path(p).relative_to(_REPO_ROOT)` defensively wrapped in try/except (return absolute on ValueError).

## CLI Flag Surface

Mirrors patterns in `run.py` / `score.py` / `compare.py`.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--version` | string, REQUIRED | — | Version slug for filename (e.g. `1.0` produces `eval-numbers-v1.0.md`) |
| `--force` | bool, store_true | `False` | Override clobber-refusal — overwrites both `.md` and `.manifest.json` |
| `--results-dir` | string | `evaluation/results` | Mirrors compare.py — parent of `queries/`, `metrics/`, `costs/`, `frozen/` |
| `--source` | string | `{results-dir}/comparison.md` | Source markdown to copy. Default discovers via results-dir. |

**Out of scope (DO NOT add):**
- `--yes` / cost-surprise prompt — freeze is offline, costs nothing
- `--tiers` — freeze always covers all 5 tiers (status="missing" for absent ones)
- `--dry-run` — argued YAGNI; `--force=False` already gives a no-op preview via the refuse-to-clobber path

## In-Process Function Signature for Phase 5

Phase 5 (Pipeline Driver) calls `freeze()` as the final stage in-process (per ROADMAP.md / STATE.md decisions). Sketch:

```python
# In evaluation/harness/freeze.py
def freeze(
    version: str,
    force: bool = False,
    results_dir: Path | None = None,
    source: Path | None = None,
) -> Path:
    """Write frozen md + sidecar manifest. Return path to the frozen md.

    Raises FileExistsError if version already frozen and force=False.
    Raises FileNotFoundError if source markdown missing.
    """
    ...
```

**In Phase 5's `pipeline.amain()`, the call site is:**
```python
from evaluation.harness.freeze import freeze
# ... after capture/score/compare ...
if args.freeze:
    try:
        frozen_path = freeze(version=args.freeze, force=args.force_freeze)
        console.print(f"[green]Frozen → {frozen_path}[/green]")
    except FileExistsError as e:
        console.print(f"[red]Freeze refused: {e}[/red]")
        return 2
```

The CLI `main()` wrapper translates `FileExistsError` to exit code 2 + colored stderr; Phase 5 catches and translates to its own exit code.

## 60-LOC Budget Estimate

The Success Criteria budget says "approximately 60 LOC of pure-Python". Sketch decomposition:

| Block | Est. LOC | Notes |
|-------|---------|-------|
| Module docstring + imports | 8 | Standard header |
| Constants (PINNED_LIBS tuple, _REPO_ROOT) | 3 | One-liners |
| `_git_dirty()` helper | 5 | New (subprocess + boolean) |
| `_iso_z_from_mtime(p: Path) -> str` helper | 3 | datetime.fromtimestamp(...).strftime |
| `_library_versions()` helper | 6 | dict comprehension + try/except |
| `_relative(p: Path) -> str` helper | 4 | Path.relative_to + try/except |
| Manifest assembly (per-tier loop + judge + libs) | 18 | Calls aggregate_tier, _detect_judge_provenance, _library_versions |
| `freeze()` public function shell + clobber check + copy + manifest write | 12 | Path.exists check, shutil.copy2, json.dumps, Path.write_text |
| `build_parser()` | 8 | 4 add_argument calls |
| `main()` | 6 | Parse → try/except → exit code |
| `if __name__ == "__main__":` | 1 | `raise SystemExit(main())` |
| **TOTAL** | **~74** | Slightly over but within "approximately 60" tolerance |

**To hit 60 exactly:** Inline the helpers into `freeze()` (loses readability but is mechanical). Recommendation: don't chase the 60 number at the cost of clarity — the phase's "approximately 60 LOC" wording implies tolerance for ±15 LOC.

## Common Pitfalls

### Pitfall 1: Frozen Markdown Path Drift Between Comparison and Freeze

**What goes wrong:** User runs `compare.py` at T1 (picks files A, B, C via mtime), edits a metrics file at T2, runs `freeze.py` at T3. `freeze.py` calls `_latest()` again and resolves to a DIFFERENT file set (X, Y, C) than comparison.md was generated from. The manifest then lies about which files fed the rollup.

**Why it happens:** `_latest()` is mtime-DESC; mtimes can drift between rollup and freeze.

**How to avoid:** Phase 5 (Pipeline Driver) calls `compare._run()` immediately before `freeze()` in-process. For standalone `freeze.py` use, document in the module docstring: "Run `python -m evaluation.harness.compare` immediately before this command, or call `pipeline.py` instead." Add a manifest field `source_markdown_mtime` so any downstream reader can detect skew (`source_markdown_mtime` significantly older than `frozen_at` → manual user investigation required).

**Warning signs:** `source_markdown_mtime` more than ~1 hour before `frozen_at` (heuristic).

### Pitfall 2: Symlinks vs. Copy

**What goes wrong:** Tempting to symlink `eval-numbers-v1.0.md → comparison.md` for "perfect" copy-paste consistency. This breaks: (a) git treats symlinks differently across platforms (Windows in particular); (b) mutating comparison.md silently mutates the "frozen" doc — defeating the point.

**Why it happens:** Premature optimization for "byte-identical" guarantee.

**How to avoid:** Always `shutil.copy2()`. Frozen means frozen — a real file copy.

**Warning signs:** Code review surfaces `os.symlink` or `Path.symlink_to` usage. Reject.

### Pitfall 3: `--force` Half-Overwrites

**What goes wrong:** `--force` overwrites `eval-numbers-v1.0.md` but the `.manifest.json` write fails (e.g. permission error mid-write). Now the markdown is the new content but the manifest still describes the OLD version. State is mangled.

**Why it happens:** Two sequential file writes without atomicity.

**How to avoid:** Either (a) accept the small risk (60-LOC budget), or (b) write to `.tmp` files first then `Path.rename()` both at the end. Recommended approach for v1.0: accept the risk + add a final consistency assertion (`assert manifest_path.exists()` after write). If write fails, the manifest path's exception propagates and the user sees a clear traceback.

**Warning signs:** Tests must cover "manifest write fails after md write succeeds" — verify whatever cleanup behavior is documented.

### Pitfall 4: Dirty Git Tree at Freeze Time

**What goes wrong:** User runs `freeze --version 1.0` while sitting on a dirty working tree. `_git_sha()` returns the HEAD SHA but the actual code that produced comparison.md may include uncommitted changes. The manifest looks reproducible but isn't.

**Why it happens:** `_git_sha()` does NOT detect dirty trees per its current implementation [VERIFIED: read run.py:72-79].

**How to avoid:** Add a `_git_dirty()` helper to freeze.py (5 lines) that runs `git status --porcelain` and returns `bool(output.strip())`. Record `git_dirty: true/false` in the manifest. (Phase 4 should NOT auto-block on dirty tree — Phase 7 / Phase 9 may legitimately freeze a working-tree-modified comparison while iterating; surfacing the bit is enough.)

**Warning signs:** `git_dirty: true` in any frozen manifest that ships to the blog → review checklist item for Phase 9.

### Pitfall 5: Race Between mtime Read and Path Capture

**What goes wrong:** `_latest()` resolves a path at T1, freeze.py reads `Path(p).stat().st_mtime` at T2, file is replaced between T1 and T2. mtime in manifest doesn't match the file `_latest()` actually picked.

**Why it happens:** Filesystem operations aren't atomic.

**How to avoid:** This is an extremely narrow race (microseconds) that only matters if another process is concurrently writing capture files. In practice, freeze.py runs after capture/score/compare have ALL completed — the race surface is empty. Document as "known low-risk" and skip mitigation. If Phase 5's pipeline becomes parallel-aware later, revisit.

**Warning signs:** Concurrent-write CI runs. Add a `pytest.mark.skipif(parallel)` guard if it ever surfaces.

### Pitfall 6: Library Version Lookup Fails Silently

**What goes wrong:** `importlib.metadata.version("openai-agents")` raises `PackageNotFoundError` because the venv was built without the `[tier-5]` extra. Manifest gets `"openai-agents": null` (or missing key) and the blog doc claims "openai-agents version unknown".

**Why it happens:** The `[tier-N]` extras are opt-in; not every venv has all of them.

**How to avoid:** Wrap each lookup in try/except → record `"not-installed"` literal string in the manifest (not `null`). This makes the manifest self-describing. The phase's success criteria say "pinned versions of `lightrag-hku`, `raganything`, `openai-agents`, `ragas`" — Phase 4 should fail loudly (exit 2) if ANY of these four are missing because they are gating deps for the v1.0 numbers. Other entries (litellm, chromadb) can be best-effort.

**Warning signs:** Manifest with `"not-installed"` for any of the four critical libs → Phase 7 / Phase 9 blocker.

### Pitfall 7: Frozen Path is Absolute (Breaks Across Machines)

**What goes wrong:** Manifest contains `"queries_path": "/Users/patryk/work/git/.../tier-1-...json"`. Anyone cloning the repo to a different machine sees a useless path.

**Why it happens:** `aggregate_tier()` returns paths via `str(queries_path)` where `queries_path` is a `Path` constructed from `_REPO_ROOT.glob(...)` — full absolute path.

**How to avoid:** Run all manifest paths through `Path(p).relative_to(_REPO_ROOT)` defensively. Wrap in try/except (`ValueError` if path is outside the repo — record absolute as fallback with `"path_outside_repo": true` flag).

**Warning signs:** Grep manifest for `/Users/`, `/home/`, `C:\\` literal strings → fail.

### Pitfall 8: Test Coverage for Pure I/O Module

**What goes wrong:** Freeze is "trivial pure I/O" so no one writes tests. Six months later, refactoring breaks the manifest schema and Phase 9 ships broken numbers.

**Why it happens:** Underestimating the consumer (Phase 9 reads this manifest verbatim).

**How to avoid:** Plan tests for: (1) happy path (writes both files, returns path); (2) refuse-to-clobber raises FileExistsError; (3) `--force` overwrites; (4) missing comparison.md raises FileNotFoundError; (5) manifest schema fields (assert specific keys present); (6) library_versions reads the real `.venv` (xfail if `openai-agents` missing — local dev edge case); (7) per-tier `status: missing` when no queries file present. Tests use `tmp_path` fixture + monkey-patched `_REPO_ROOT`.

**Warning signs:** test_eval_freeze.py absent or under 80 LOC. Plan it before merging.

## Code Examples

Verified patterns drawn from the existing codebase plus stdlib.

### Reading library versions (verified working in this session)

```python
# Source: docs.python.org/3/library/importlib.metadata.html#distribution-versions
# Verified 2026-05-05 against /Users/patrykattc/work/git/rag-architecture-patterns/.venv:
#   lightrag-hku=1.4.15, raganything=1.2.10, openai-agents=0.14.6, ragas=0.4.3
from importlib.metadata import version, PackageNotFoundError

CRITICAL_LIBS = ("lightrag-hku", "raganything", "openai-agents", "ragas")

def _library_versions(critical: tuple[str, ...] = CRITICAL_LIBS) -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in critical:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
    return out
```

### Refuse-to-clobber

```python
# Source: this research; mirrors `cp -n` semantics
import sys
from pathlib import Path

def _ensure_not_frozen(out_md: Path, version: str, force: bool) -> None:
    if out_md.exists() and not force:
        msg = f"eval-numbers-v{version}.md already frozen — bump version or pass --force"
        print(msg, file=sys.stderr)
        raise SystemExit(2)
```

### Per-tier provenance via aggregate_tier

```python
# Source: aggregate_tier() defined at evaluation/harness/compare.py:69-136
from evaluation.harness.compare import aggregate_tier, SUPPORTED_TIERS

def _per_tier_block(results_dir: Path) -> dict:
    out = {}
    for t in SUPPORTED_TIERS:
        row = aggregate_tier(t, results_dir)
        if row is None:
            out[f"tier-{t}"] = {"status": "missing"}
            continue
        out[f"tier-{t}"] = {
            "status": "present",
            "generation_model": row["model"],
            "capture_timestamp": row["timestamp"],
            "capture_git_sha": row["git_sha"],
            "queries_path": row["queries_path"],
            "cost_path": row["cost_path"],
            "metrics_path": row["metrics_path"],
        }
    return out
```

### Detecting dirty tree

```python
# Source: this research; uses subprocess like _git_sha() in run.py:72-79
import subprocess
from pathlib import Path

def _git_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        )
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return False
```

### Copy preserving mtime

```python
# Source: docs.python.org/3/library/shutil.html#shutil.copy2
import shutil
from pathlib import Path

def _freeze_markdown(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)  # preserves mtime + permissions
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pkg_resources.get_distribution(...).version` (setuptools) | `importlib.metadata.version(...)` (stdlib) | Python 3.8 | Stdlib, no setuptools dep, 5x faster cold-start [CITED: docs.python.org/3/library/importlib.metadata.html] |
| Manual `os.path.join` + string concat | `pathlib.Path` | Python 3.4+ | Used everywhere in this repo's harness |
| `parse_file()` (Pydantic v1) | `model_validate_json()` (Pydantic v2) | Pydantic 2.0 | Already adopted in records.py |

**Deprecated/outdated:**
- `pkg_resources` — use `importlib.metadata` (we follow the new API)

## Open Questions

1. **Should `--force` also re-write a fresh manifest, or preserve the old one?**
   - What we know: HARN-03 says frozen markdown is immutable by default; `--force` is the override.
   - What's unclear: When `--force` overwrites the markdown, does the user expect the manifest to also reflect "this was re-frozen at T2"? Or should the manifest be append-only?
   - Recommendation: `--force` overwrites BOTH md and manifest. The manifest's `frozen_at` field is the freeze-time of the LATEST write. Append-only manifests are over-engineering for v1.0.

2. **Should Phase 4 fail (exit 2) if any of the 4 critical libs is `not-installed`?**
   - What we know: Pitfall 6 says yes for the four critical libs; the success criteria say the manifest must "find pinned versions of lightrag-hku, raganything, openai-agents, ragas".
   - What's unclear: What if a user freezes from a venv that was synced with `--extra evaluation` only (no tier extras)? They get not-installed for tier-N libs.
   - Recommendation: Hard-fail if ANY of the four is missing, with error pointing to `uv sync --all-extras`. This forces a reproducible-numbers discipline.

3. **Where should the schema version live: in the manifest OR in a Pydantic model?**
   - What we know: Records use Pydantic models (records.py).
   - What's unclear: Pydantic models for the manifest are nice for evolution but add ~30 LOC and risk blowing the 60-LOC budget.
   - Recommendation: Skip Pydantic for v1.0; embed `"$schema_version": "1.0"` as a string literal at the top of the manifest dict. Phase 9 can convert to Pydantic if downstream consumers need validation.

4. **How does Phase 9 (the blog handoff doc) consume this manifest?**
   - What we know: Phase 9 reads the manifest to populate provenance lines in the frozen markdown body.
   - What's unclear: Does Phase 9 inject manifest-derived content INTO the frozen markdown, or does the frozen markdown stay byte-identical to comparison.md and Phase 9 emits a SECOND (curated) markdown?
   - Recommendation: Defer to Phase 9 research. Phase 4 commits to byte-identical copy of comparison.md. Phase 9 either (a) extends comparison.md FIRST (so the frozen copy has all phase-9 content), or (b) writes a separate `eval-numbers-v1.0-curated.md`. This is a Phase 9 decision, not a Phase 4 one.

5. **Should the frozen directory be in `.gitignore` or committed to git?**
   - What we know: `.gitignore` excludes `evaluation/results/queries/` and `evaluation/results/metrics/` but does NOT exclude `evaluation/results/frozen/` (verified via reading the .gitignore in this session). `comparison.md` IS committed.
   - What's unclear: Implicit but not stated.
   - Recommendation: Frozen artifacts SHOULD be committed (they ARE the deliverable). Confirm with planner — if planner agrees, plan includes a `.gitkeep` in `frozen/` for first-run discoverability.

## Project Constraints (from STATE.md / ROADMAP.md / pyproject.toml)

These are NOT from CLAUDE.md (none exists), but they have equivalent authority because they are codified in the planning docs:

- **No new external deps.** Phase budget says "no new external deps". All work uses stdlib + already-pinned project deps.
- **~60 LOC pure-Python.** Phase budget says "approximately 60 LOC". Treat as ±15 LOC tolerance, not a hard limit.
- **Lives at `evaluation/harness/freeze.py`.** Path is locked.
- **In-process callable for Phase 5.** `freeze(version, force) -> Path` must be importable.
- **Use `_git_sha` / `_ts` / `_ts_for_filename` from `evaluation.harness.run`** — these are the SoT per STATE.md "Decisions" section.
- **Frozen markdown is byte-identical to `comparison.md`.** No injected footers. Use `shutil.copy2`.
- **Pinned versions to capture: `lightrag-hku`, `raganything`, `openai-agents`, `ragas`** — exact 4 listed in HARN-04. Recommend ALSO capturing `litellm` + `chromadb` as bonus (cheap; manifest is JSON dict; no LOC pressure).
- **Python 3.10+** per pyproject.toml `requires-python = ">=3.10"`.
- **Tests live at `evaluation/tests/test_eval_freeze.py`.** Mirrors the existing test layout (`test_eval_compare.py`, `test_eval_score.py`, etc.).
- **Out of scope per ROADMAP/REQUIREMENTS:**
  - Adding new tiers
  - Re-architecting the harness (only `freeze.py` + `pipeline.py` are allowed new files)
  - Changing comparison.md's content or shape (Phase 4 doesn't touch compare.py)

## Runtime State Inventory

> Phase 4 is greenfield (a new file is added). However, it interacts with existing runtime artifacts.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — Phase 4 produces NEW artifacts under `evaluation/results/frozen/`; does not mutate existing data stores. The directory does not yet exist; freeze.py creates it via `Path.mkdir(parents=True, exist_ok=True)`. | None |
| Live service config | None — no external services configured. | None |
| OS-registered state | None — no daemons, tasks, or schedulers. | None |
| Secrets / env vars | None new. freeze.py does NOT call any LLM, does NOT read API keys. | None |
| Build artifacts | None — pure-Python module, no compiled outputs. | None |

**Verified by:** Reading run.py / score.py / compare.py source. Phase 4 has NO runtime state migration.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | freeze.py | ✓ | 3.11.7 (.venv) | — |
| `git` CLI | `_git_sha()` and `_git_dirty()` | ✓ | system git | `_git_sha()` already returns `"unknown"` on subprocess failure; `_git_dirty()` should mirror this and return False |
| `lightrag-hku` 1.4.15 | manifest library_versions | ✓ | 1.4.15 | record `"not-installed"` |
| `raganything` 1.2.10 | manifest library_versions | ✓ | 1.2.10 | record `"not-installed"` |
| `openai-agents` 0.14.6 | manifest library_versions | ✓ | 0.14.6 | record `"not-installed"` |
| `ragas` 0.4.3 | manifest library_versions | ✓ | 0.4.3 | record `"not-installed"` |
| `evaluation/results/comparison.md` | source markdown to copy | ✓ | mtime 2026-05-05 14:13 | error exit 2 if missing |
| `evaluation/results/queries/` | aggregate_tier() globs here | ✓ | populated 13 files | tier reported `"status": "missing"` |
| `evaluation/results/metrics/` | aggregate_tier() globs here | ✓ | populated 8 files | tier reported `"status": "missing"` |
| `evaluation/results/costs/` | aggregate_tier() globs here | ✓ | populated 50+ files | tier reported `"status": "missing"` |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.x [VERIFIED: pyproject.toml `[dependency-groups].test = ["pytest>=8.4,<9"]`] |
| Config file | `pyproject.toml [tool.pytest.ini_options]` (no separate pytest.ini) |
| Quick run command | `.venv/bin/pytest evaluation/tests/test_eval_freeze.py -v` |
| Full suite command | `.venv/bin/pytest evaluation/tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HARN-03 | First freeze succeeds | unit | `pytest evaluation/tests/test_eval_freeze.py::test_freeze_writes_md_and_manifest -x` | ❌ Wave 0 |
| HARN-03 | Re-freeze same version refuses with prescribed wording | unit | `pytest evaluation/tests/test_eval_freeze.py::test_freeze_refuses_clobber -x` | ❌ Wave 0 |
| HARN-03 | `--force` overwrites | unit | `pytest evaluation/tests/test_eval_freeze.py::test_freeze_force_overwrites -x` | ❌ Wave 0 |
| HARN-04 | Manifest contains git_sha + frozen_at | unit | `pytest evaluation/tests/test_eval_freeze.py::test_manifest_top_level_fields -x` | ❌ Wave 0 |
| HARN-04 | Manifest per-tier paths + mtimes recorded | unit | `pytest evaluation/tests/test_eval_freeze.py::test_manifest_per_tier_provenance -x` | ❌ Wave 0 |
| HARN-04 | Manifest library_versions populated for 4 critical libs | unit | `pytest evaluation/tests/test_eval_freeze.py::test_manifest_library_versions -x` | ❌ Wave 0 |
| HARN-04 | Manifest judge model + embedder fields | unit | `pytest evaluation/tests/test_eval_freeze.py::test_manifest_judge_block -x` | ❌ Wave 0 |
| HARN-03+04 | freeze() callable in-process (Phase 5 contract) | unit | `pytest evaluation/tests/test_eval_freeze.py::test_freeze_in_process_returns_path -x` | ❌ Wave 0 |
| HARN-04 | Missing comparison.md raises FileNotFoundError | unit | `pytest evaluation/tests/test_eval_freeze.py::test_freeze_no_source_errors -x` | ❌ Wave 0 |
| HARN-04 | Missing tier reports `status: missing` | unit | `pytest evaluation/tests/test_eval_freeze.py::test_manifest_missing_tier -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/pytest evaluation/tests/test_eval_freeze.py -v` (~10 tests, <1s, all offline)
- **Per wave merge:** `.venv/bin/pytest evaluation/tests/ -v -m "not live"` (full offline suite)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `evaluation/tests/test_eval_freeze.py` — covers HARN-03 + HARN-04 (10 tests above)
- [ ] No new conftest fixtures needed; existing `evaluation/tests/conftest.py` already has tmp_path-style helpers (verify when planning)
- [ ] Framework install: already present (pytest 8.4 in dev group)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Frozen artifacts SHOULD be committed to git (no .gitignore entry for `frozen/`) | Open Q5 | If wrong, planner must add `.gitignore` entry; no other code impact |
| A2 | The 60-LOC budget tolerates ±15 LOC | 60-LOC Budget Estimate | If hard-60 enforced, plan inlines helpers (loses readability) |
| A3 | `--force` overwrites BOTH .md and .manifest.json (not just .md) | Open Q1 | Behavior decision; trivially reversible |
| A4 | Phase 4 hard-fails if any of the 4 critical libs is `not-installed` | Open Q2 + Pitfall 6 | If wrong, manifest emits `"not-installed"` and Phase 7/9 see it; user might miss the signal |
| A5 | Frozen markdown is byte-identical copy of comparison.md (no injection) | STATE.md decision + Anti-Patterns | If wrong, freeze.py becomes a templating tool — adds 20+ LOC and breaks copy-paste-into-blog determinism |
| A6 | Phase 9 (handoff doc) is responsible for any provenance content INSIDE the markdown body; Phase 4 only writes the SIDECAR manifest | Open Q4 | If wrong, Phase 4 scope grows; planner must redirect to Phase 9 |
| A7 | `_detect_judge_provenance()`'s defaults (`"google/gemini-2.5-flash"` / `"openai/text-embedding-3-small"`) are acceptable when no `ragas-judge-*.json` exists | Sidecar Manifest Schema | If wrong, manifest could record stale defaults; mitigation = require at least one judge cost file before freezing |
| A8 | `subprocess.check_output(["git","status","--porcelain"])` is available everywhere `_git_sha()` works | Pitfall 4 + Code Examples | If wrong, fallback to `git_dirty: false` masks the signal but doesn't crash |

## Sources

### Primary (HIGH confidence)
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/run.py` (full read) — `_git_sha`, `_ts`, `_ts_for_filename` definitions
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/compare.py` (full read) — `_latest`, `aggregate_tier`, `_detect_judge_provenance`, comparison.md emission
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/score.py` (lines 60-150, 405-417) — `JUDGE_MAX_TOKENS`, `_persist_metrics`, judge wiring
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/records.py` (full read) — QueryLog / ScoreRecord schemas
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/results/comparison.md` (full read) — current rollup content
- `/Users/patrykattc/work/git/rag-architecture-patterns/pyproject.toml` (full read) — pinned dep versions, Python version, pytest config
- `/Users/patrykattc/work/git/rag-architecture-patterns/uv.lock` (greps for the four critical libs) — pinned versions confirmed match pyproject
- `/Users/patrykattc/work/git/rag-architecture-patterns/.gitignore` (full read) — confirmed `frozen/` is NOT excluded
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/STATE.md` — locked decisions on `_git_sha` / `_ts` SoT
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/REQUIREMENTS.md` — HARN-03/HARN-04 wording
- `/Users/patrykattc/work/git/rag-architecture-patterns/.planning/ROADMAP.md` — Phase 4 success criteria
- Live `.venv` `importlib.metadata.version()` runs verifying all 4 critical libs return correct strings
- `https://docs.python.org/3/library/importlib.metadata.html#distribution-versions` — official stdlib API contract
- `https://docs.python.org/3/library/shutil.html#shutil.copy2` — preserves mtime + permissions
- Sample artifact files (one of each: tier-1 query log, tier-1 metrics, tier-1 cost, judge cost) — JSON shapes verified

### Secondary (MEDIUM confidence)
- N/A — Phase 4 is entirely codebase-internal; no WebSearch needed.

### Tertiary (LOW confidence)
- N/A.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every helper exists in the repo and was read directly; library versions verified live
- Architecture: HIGH — pattern matches existing `run.py` / `compare.py` / `score.py` exactly
- Pitfalls: HIGH — pulled from real repo evidence (.gitignore, _git_sha behavior, _latest mtime semantics)

**Research date:** 2026-05-05
**Valid until:** 2026-06-04 (30 days — domain is stable, no fast-moving deps)

## RESEARCH COMPLETE

**Phase:** 4 - Freeze Tool
**Confidence:** HIGH

### Key Findings
1. Every primitive freeze.py needs already exists in run.py / compare.py / stdlib — `_git_sha`, `_ts`, `_latest`, `aggregate_tier`, `_detect_judge_provenance`. Zero new infrastructure.
2. `importlib.metadata.version()` works against `.venv` for ALL FOUR critical libs (lightrag-hku 1.4.15, raganything 1.2.10, openai-agents 0.14.6, ragas 0.4.3) — verified live in this session.
3. Per-tier provenance is best harvested by calling `aggregate_tier(t, results_dir)` and reading the dict it already returns (queries_path, cost_path, metrics_path, timestamp, git_sha, model). Do NOT re-glob.
4. Frozen markdown must be byte-identical to comparison.md — use `shutil.copy2`, never read+rewrite. Manifest is the SIDECAR for provenance, not an injection into the markdown body.
5. The 60-LOC budget is achievable but tight; ~74 LOC is realistic with helper extraction. ±15 LOC tolerance recommended.

### Confidence Assessment
| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | Every helper read directly; live venv version probe succeeded |
| Architecture | HIGH | Mirrors existing run.py / compare.py / score.py patterns 1:1 |
| Pitfalls | HIGH | Drawn from actual .gitignore, actual _git_sha behavior, actual _latest mtime semantics |

### Open Questions (Must-Decide-Before-Planning)
1. **Should `--force` overwrite both .md AND .manifest.json (recommended) or only .md?**
2. **Should freeze.py exit 2 if any of the 4 critical libs is not-installed (recommended) or record "not-installed" string and continue?**
3. **Should the frozen/ directory be committed to git (recommended — .gitignore does not exclude it) or excluded?**
4. **What does Phase 9 (handoff doc) inject into the frozen markdown — and does that content live in comparison.md (so freeze byte-copies it) or in a separate Phase-9-curated file?**
5. **Strict 60 LOC vs. ±15 tolerance — which interpretation does the planner enforce?**

### File Created
`/Users/patrykattc/work/git/rag-architecture-patterns/.planning/phases/04-freeze-tool/04-RESEARCH.md`

### Ready for Planning
Research complete. Planner can now create PLAN.md files.

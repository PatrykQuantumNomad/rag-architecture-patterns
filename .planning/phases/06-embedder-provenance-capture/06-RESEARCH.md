# Phase 6: Embedder Provenance Capture — Research

**Researched:** 2026-05-06
**Domain:** Pydantic schema extension + capture-site instrumentation across two capture entry points (harness/run.py + tier-4 eval_capture.py) feeding the existing compare.py rollup.
**Confidence:** HIGH (every fact below was read from source — no external library research needed; this phase is pure instrumentation of repo-internal code.)

## Summary

Phase 6 extends the existing `QueryLog` Pydantic model in `evaluation/harness/records.py` with two new top-level optional fields (`embedder` + `embedder_source`) and threads the literal embedder identifier through the five tier-capture sites that already exist. The schema today carries `tier, timestamp, git_sha, model, records` (verified 2026-05-06 against five live capture JSONs in `evaluation/results/queries/` — see Section "Current Capture Schema (Verified)"). A `compare.py` rollup change surfaces the new fields in the per-tier provenance footer. **No new files; no new abstractions; ~25-40 net LOC across five files** (records.py +2 fields, run.py +5 LOC for embedder/source per tier, eval_capture.py +3 LOC, compare.py +4-6 LOC for the footer line, plus an Optional-tolerant default to preserve backwards compatibility for older committed Tier-4 captures).

The phase description in `.planning/ROADMAP.md` contains **one factual error** that the planner must override: it claims Tier 5 records "OpenAI's hosted vector-store embedder with an explicit `managed=true` flag". The actual code in `tier-5-agentic/tools.py:47-50` and `tier-5-agentic/tools.py:90-101` shows Tier 5 reuses **Tier 1's ChromaDB collection** (`chroma_db/tier-1-naive/`) and embeds queries via OpenRouter `openai/text-embedding-3-small` — the **same** embedder Tiers 1, 3, 4 use. There is no hosted vector store. Section "Tier-by-Tier Embedder Reality (Verified from Source)" documents what each tier actually uses; the `managed=true` flag is meaningful **only for Tier 2** (Google File Search managed indexing — `tier-2-managed/main.py:191`).

**Primary recommendation:** Land a single PLAN.md (one-file phase, pure-offline TDD red→green) that (1) extends `QueryLog` with `embedder: Optional[str]` + `embedder_source: Optional[str]` (both nullable for backwards compatibility with older captures); (2) hard-codes the per-tier (embedder, source) tuple at the top of `_capture_tier()` in run.py and at the construction site in `tier-4-multimodal/scripts/eval_capture.py`; (3) extends `compare.py`'s capture-provenance footer to emit the embedder line per tier; (4) adds a v1.0 frozen "embedder by tier" disclosure block built from the same data. **No live smoke required** — the capture sites all write JSON synchronously and the new fields are static literals known at code-edit time, so a pure-offline TDD test suite covers 100% of the behavior. **One PLAN.md, ~6-8 tasks, single atomic commit per task.**

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CAP-03 | User can verify per-tier embedding model is recorded in capture JSON so the embedder-confound disclosure in the frozen doc is data-backed (not narrative) | Section "Current Capture Schema (Verified)" + Section "Tier-by-Tier Embedder Reality" + Section "Recommended Schema Extension" + Section "Compare.py Rollup Integration" |

## Project Constraints (from CLAUDE.md)

No `./CLAUDE.md` exists at repo root. Project conventions inherited from `.planning/STATE.md` Decisions and from the five already-shipped harness modules:

- **ISO 8601 Z** timestamps via `run._ts()` / `run._ts_for_filename()` — single source of truth.
- **Git SHA** via `run._git_sha()` (short HEAD SHA; "unknown" on failure) — single source of truth.
- **`from __future__ import annotations`** + module docstring at top of every harness module.
- **Argparse pattern**: separate `build_parser()` so tests can introspect flags; `main(argv)` calls `asyncio.run(amain(args, Console()))`.
- **Pure-offline TDD red→green** is the strong project preference (Plans 03-01, 04-01, 05-01 — wall times 5/18/12 min respectively). Live smokes are appended only when API behavior is itself the verification target (Plans 03-03, 05-02). Phase 6 has no API behavior to verify — all embedder strings are static literals.
- **Atomic commits per task**: one commit per task, `red` and `green` are separate commits per Plan 04-01 / 05-01 precedent.
- **`evaluation/results/queries/` is gitignored** (verified `.gitignore` line "evaluation/results/queries/"). Older capture JSONs on disk are local-only artifacts; no committed-state assumption is safe.
- **Test fixtures**: `tmp_path` + offline mocks. Live tests gated under `@pytest.mark.live`. **Pyproject does NOT enforce live-deselect-by-default** (v1.1 hardening item) — `pytest path/to/file.py -v` runs live tests silently. Phase 6 has no live tests so this gap doesn't apply.
- **Pydantic v2 idioms** throughout (`model_validate_json`, NOT the deprecated `parse_file`) — see records.py:74-75.

## Architectural Responsibility Map

| Capability | Owner Module | Phase 6 Edit |
|-----------|--------------|--------------|
| Per-tier capture (write `queries/tier-N-*.json`) — Tiers 1, 2, 3, 5 | `evaluation/harness/run.py` (`_capture_tier`) | **Add 2 lines per tier branch** (set `embedder` + `embedder_source` literals) and **2 args** to the `QueryLog(...)` constructor |
| Per-tier capture (Tier 4 separate entry point — see Pitfall 1) | `tier-4-multimodal/scripts/eval_capture.py` (`_capture`) | **Same 2 lines** added at the QueryLog construction site (line 195-201) |
| Capture record schema (Pydantic v2 `QueryLog`) | `evaluation/harness/records.py` (`QueryLog`) | **Add 2 fields**: `embedder: Optional[str] = None` + `embedder_source: Optional[str] = None` |
| Comparison rollup — provenance footer | `evaluation/harness/compare.py` (`emit_markdown` + `_run`) | **Extend the `capture_provenance` dict** built at line 357-365 with `embedder`/`embedder_source`; **extend `emit_markdown` footer** at line 261-264 to emit the embedder line per tier |
| Frozen sidecar manifest — per-tier provenance block | `evaluation/harness/freeze.py` (`freeze`) | **Extend the per-tier manifest entry** at line 58-66 with the new fields (read via `aggregate_tier`'s pass-through) |

### MUST NOT touch (byte-identical post-Phase 6)

- `evaluation/harness/score.py` — Phase 3 / Phase 4 forward contract. Score reads only `records[*].question_id` + `records[*].answer` + `records[*].retrieved_contexts` + `records[*].error`. The `embedder` field at the top level is invisible to it.
- `evaluation/harness/pipeline.py` — Phase 5 forward contract LOCKED. Pipeline only synthesizes `argparse.Namespace` for run.amain and score.amain; the embedder field flows through unchanged via in-process `QueryLog` instances.
- `evaluation/harness/smoke_gate.py` — Phase 1 / Phase 2 forward contract. Reads `records[*]` only; top-level new fields are invisible.
- All five **adapter** files (`evaluation/harness/adapters/tier_{1..5}.py`) — they construct `EvalRecord`, not `QueryLog`. The QueryLog assembly happens at the `_capture_tier()` boundary in run.py.

### MAY need targeted edit (minimum surface)

- `evaluation/harness/run.py:181-256` — five tier branches. Each tier branch already pulls in a `model` constant (e.g. `T1_MODEL`, `T2_MODEL`); we colocate two new constants `T1_EMBEDDER` + `T1_EMBEDDER_SOURCE` and pass them through. **Estimated +10-12 LOC across all five tier branches.**
- `evaluation/harness/run.py:261-267` — the `QueryLog(...)` constructor call. **+2 args.**
- `tier-4-multimodal/scripts/eval_capture.py:195-201` — same QueryLog constructor call. **+2 args.** Embedder constant is already at line 46 (`DEFAULT_EMBED_MODEL`); we add a sibling `EMBEDDER_SOURCE = "openrouter"` constant.
- `evaluation/harness/compare.py:357-365` (`capture_provenance` dict build) — **+2 fields.**
- `evaluation/harness/compare.py:261-264` (footer emit loop) — **+1 conditional emit line per tier.**
- `evaluation/harness/freeze.py:58-66` (per-tier manifest entry) — **+2 dict assignments.** Reads from `aggregate_tier`'s row dict, which already pass-throughs `model` / `timestamp` / `git_sha` from the QueryLog (compare.py:130-132); we add `embedder` / `embedder_source` to that pass-through and to the manifest entry.

## Current Capture Schema (Verified)

Top-level keys observed in **all five** existing capture JSONs (verified 2026-05-06 against `evaluation/results/queries/tier-{1..5}-*.json`):

```json
{
  "tier": "tier-1",
  "timestamp": "2026-05-02T17:26:59Z",
  "git_sha": "ce5c2ad",
  "model": "google/gemini-2.5-flash",
  "records": [ /* list[EvalRecord] */ ]
}
```

The Pydantic source is `evaluation/harness/records.py:40-47`:

```python
class QueryLog(BaseModel):
    """A single tier × timestamp capture run, holding all 30 EvalRecords."""
    tier: str
    timestamp: str  # ISO 8601 UTC, "Z" suffix — matches CostTracker D-13
    git_sha: str
    model: str
    records: list[EvalRecord]
```

**There is no `embedder` field today.** Phase 6 must add it.

**Confirmed top-level field exhaustive list across all five tiers:** `tier, timestamp, git_sha, model, records` — only those five. No nested provenance dict; no metadata sub-block. The schema extension lands at the same level as `model`.

## Tier-by-Tier Embedder Reality (Verified from Source)

| Tier | Embedder Identifier | Source Tag | `managed` Flag? | Source File | Line |
|------|---------------------|-----------|-----------------|-------------|------|
| 1 | `openai/text-embedding-3-small` | `openrouter` | `false` | `tier-1-naive/embed_openai.py` | 31 (`EMBED_MODEL`) |
| 2 | `gemini-embedding-001` | `google-managed` | **`true`** | `tier-2-managed/main.py` | 191 (`tracker.record_embedding("gemini-embedding-001", ...)`) |
| 3 | `openai/text-embedding-3-small` | `openrouter` | `false` | `tier-3-graph/rag.py` | 56 (`DEFAULT_EMBED_MODEL`) |
| 4 | `openai/text-embedding-3-small` | `openrouter` | `false` | `tier-4-multimodal/rag.py` | 53 (`DEFAULT_EMBED_MODEL`) |
| 5 | `openai/text-embedding-3-small` | `openrouter` | `false` | `tier-5-agentic/tools.py` | 47-50 + 90-101 (imports + reuses Tier 1's `EMBED_MODEL`) |

**Where the embedder string lives today:**

- **Tier 1**: `tier_1_naive.embed_openai.EMBED_MODEL = "openai/text-embedding-3-small"` (a module-level string constant). Run.py line 169 already imports the LLM `DEFAULT_MODEL`; mirror that pattern with `EMBED_MODEL`.
- **Tier 2**: hardcoded string `"gemini-embedding-001"` in `tier-2-managed/main.py:191` — there is no `EMBED_MODEL` constant exported by tier-2-managed today. The harness adapter (`evaluation/harness/adapters/tier_2.py`) does NOT reference the embedder. **Recommendation: introduce an `EMBED_MODEL = "gemini-embedding-001"` constant** in `tier-2-managed/main.py` (or a new lightweight `tier-2-managed/embed.py`) and import it from run.py — this keeps the rule "embedder identity is a single source of truth per tier" intact, mirroring the Tier 1/3/4 pattern. Alternative: hardcode the string in run.py's tier-2 branch — less clean but +0 LOC in tier-2 code. **Recommend the constant approach** for symmetry.
- **Tier 3**: `tier_3_graph.rag.DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"` — already imported by run.py:201. Reuse.
- **Tier 4 (harness/run.py path, cached-only)**: Hardcoded `model = "google/gemini-2.5-flash"` at run.py:241 because Tier 4 capture is cached-mode in run.py. **Embedder is the same `openai/text-embedding-3-small`** because Tier 4's RAG storage was built with that embedder (verified `tier-4-multimodal/rag.py:53`). The cached log itself currently does NOT carry an embedder field (Phase 6 fixes this).
- **Tier 4 (eval_capture.py path, the live route)**: Already imports `DEFAULT_EMBED_MODEL` at line 46. The QueryLog construction at line 195-201 must be extended.
- **Tier 5**: `tier_5_agentic.tools` imports `EMBED_MODEL` from `tier_1_naive.embed_openai` (line 47-50). Tier 5 is **a Tier-1-embedder reuse**, not a new embedder. The harness adapter `evaluation/harness/adapters/tier_5.py` does NOT touch the embedder. Run.py's tier-5 branch (line 244-256) imports `DEFAULT_MODEL` from `tier_5_agentic.agent`; we add a sibling import `from tier_1_naive.embed_openai import EMBED_MODEL` (since Tier 5 re-uses Tier 1's embedder).

**Definitive resolution of the ROADMAP "Tier 5 hosted vector-store embedder" claim:** The claim is **incorrect**. Tier 5 does NOT use OpenAI's hosted vector store. Tier 5 uses an OpenAI Agents SDK agent that calls `@function_tool`-decorated functions (`search_text_chunks`, `lookup_paper_metadata`); `search_text_chunks` queries Tier 1's local ChromaDB collection. Embedding happens via OpenRouter `openai/text-embedding-3-small`. The phase plan must therefore record `embedder = "openai/text-embedding-3-small"` and `embedder_source = "openrouter"` for Tier 5 — exactly the same as Tier 1.

**Definitive resolution of the `managed=true` flag question:** The flag is meaningful **only for Tier 2** (Gemini File Search is a managed-indexing service — Google handles embedding internally and we cannot inspect the dim/algorithm). Tiers 1, 3, 4, 5 all use the same OpenRouter-routed `openai/text-embedding-3-small` and there is nothing "managed" about them — we own the indexing.

## Recommended Schema Extension

Add **two** top-level optional fields to `QueryLog` in `records.py`:

```python
# evaluation/harness/records.py — proposed minimal change (+2 fields)
class QueryLog(BaseModel):
    """A single tier × timestamp capture run, holding all 30 EvalRecords."""
    tier: str
    timestamp: str
    git_sha: str
    model: str
    embedder: Optional[str] = None        # NEW — e.g. "openai/text-embedding-3-small"
    embedder_source: Optional[str] = None # NEW — e.g. "openrouter" | "google-managed"
    records: list[EvalRecord]
```

**Why two fields, not one combined string:**

1. The `embedder` string is the *model identifier* — what the blog reader needs to compare across tiers ("Tier 2 used `gemini-embedding-001`; Tier 1 used `openai/text-embedding-3-small`").
2. The `embedder_source` string is the *provenance gateway* — `openrouter` vs `google-managed` distinguishes who handled the indexing (we did, vs Google did). This is the data behind the "embedder confound" disclosure: if Tier 2 underperforms Tier 1, is it because the *embedder model* is worse, or because *managed indexing has a different chunking strategy we cannot inspect*?
3. **Two fields is honest separation**; a single combined string like `"openrouter/openai/text-embedding-3-small"` would conflate model identity with gateway identity (and we already see this pattern bite us — Plan 04-01 / Plan 132 had to deal with the `openrouter/` prefix-strip in `_strip_openrouter_prefix` at `evaluation/harness/adapters/tier_5.py:50-56`).

**Why `Optional[str] = None` and not required-with-default-string:**

Backwards compatibility. `evaluation/results/queries/` is gitignored, so older locally-cached JSONs won't have the new fields. A required field with a default would still serialize to JSON with the default, but **deserialization** of an older JSON via `QueryLog.model_validate_json()` (records.py:75) would silently inject the default and the rollup couldn't tell "missing because old capture" from "missing because forgotten". `Optional[str] = None` forces the rollup code (compare.py / freeze.py) to render an em-dash placeholder for legacy captures — honest disclosure. Phase 7 always re-captures, so the new fields will be present in every Phase 7 output JSON.

**Pydantic v2 deserialization tolerance verified by reading records.py:74-75.** `QueryLog.model_validate_json` honors `Optional[T] = None` by accepting either presence-with-value, presence-with-null, or absence — all three serialize back to the same instance. No migration required for older JSONs on disk; they will simply load with `embedder=None`.

## Per-Tier Embedder Constants (Recommended Locations)

Single source of truth per tier, imported from existing modules where possible:

| Tier | Embedder Constant | Source Tag Constant | Where to Define |
|------|-------------------|---------------------|-----------------|
| 1 | `EMBED_MODEL` | `EMBEDDER_SOURCE = "openrouter"` | `tier-1-naive/embed_openai.py` (EMBED_MODEL exists; add EMBEDDER_SOURCE constant — **+1 LOC**) |
| 2 | `EMBED_MODEL = "gemini-embedding-001"` | `EMBEDDER_SOURCE = "google-managed"` | New constants in `tier-2-managed/main.py` (sibling to existing `INDEX_USD_PER_1M_TOKENS` at line 91 — **+2 LOC**) |
| 3 | `DEFAULT_EMBED_MODEL` | `EMBEDDER_SOURCE = "openrouter"` | `tier-3-graph/rag.py` (DEFAULT_EMBED_MODEL exists at line 56; add EMBEDDER_SOURCE — **+1 LOC**) |
| 4 | `DEFAULT_EMBED_MODEL` | `EMBEDDER_SOURCE = "openrouter"` | `tier-4-multimodal/rag.py` (DEFAULT_EMBED_MODEL exists at line 53; add EMBEDDER_SOURCE — **+1 LOC**) |
| 5 | `EMBED_MODEL` (re-export from tier-1) | `EMBEDDER_SOURCE = "openrouter"` | `tier-5-agentic/tools.py` (EMBED_MODEL re-import exists at line 47-50; add EMBEDDER_SOURCE constant — **+1 LOC**) |

**Total tier-side LOC: +6 across five files** (one constant added per tier; Tiers 1, 3, 4, 5 add only EMBEDDER_SOURCE because the embedder constant already exists; Tier 2 adds both).

**Alternative: hardcode all five (embedder, source) tuples in run.py and eval_capture.py only — zero edits to tier modules.** This is **+2 LOC less** total but couples run.py to the embedder identity of every tier. Given that the embedder identifiers are already constants in 4 of 5 tier modules, the symmetric "constant per tier" approach is honest and discoverable. **Recommend the constant approach.**

## run.py Changes — Per-Tier Branch

For each tier branch in `_capture_tier()` (run.py:168-256), pull the (embedder, source) tuple from the tier's module and pass it into the `QueryLog(...)` constructor at line 261-267:

```python
# run.py — proposed minimal change in tier-1 branch (lines 168-179):
elif tier == 1:
    from evaluation.harness.adapters.tier_1 import (
        DEFAULT_MODEL as T1_MODEL,
        run_tier1,
    )
    from tier_1_naive.embed_openai import EMBED_MODEL as T1_EMBEDDER, EMBEDDER_SOURCE as T1_EMBEDDER_SOURCE  # NEW
    for i, q in enumerate(qa):
        ...
        rec = await run_tier1(...)
        records.append(rec)
    model = T1_MODEL
    embedder = T1_EMBEDDER             # NEW
    embedder_source = T1_EMBEDDER_SOURCE  # NEW

# Mirror across tier-2, tier-3, tier-4 (cached path), tier-5 branches.

# At the QueryLog assembly (run.py:261-267):
log = QueryLog(
    tier=f"tier-{tier}",
    timestamp=timestamp,
    git_sha=git_sha,
    model=model,
    embedder=embedder,                 # NEW
    embedder_source=embedder_source,   # NEW
    records=records,
)
```

**Estimated total run.py LOC delta: +12-15** (1 import line + 2 assignment lines per tier × 5 tiers, minus the Tier 4 cached-mode quirk which already has a hardcoded `model`).

## eval_capture.py Changes (Tier 4 Live Path)

The live Tier 4 capture entry point is `tier-4-multimodal/scripts/eval_capture.py` (NOT harness/run.py). The QueryLog construction at line 195-201 must be extended. `DEFAULT_EMBED_MODEL` is already imported at line 46:

```python
# tier-4-multimodal/scripts/eval_capture.py — line 46-47 (existing):
from tier_4_multimodal.rag import build_rag, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL
# Add EMBEDDER_SOURCE to the import once it's defined in tier_4_multimodal.rag:
from tier_4_multimodal.rag import EMBEDDER_SOURCE  # NEW

# At line 195-201 (existing QueryLog construction):
log = QueryLog(
    tier="tier-4",
    timestamp=timestamp,
    git_sha=_git_sha(),
    model=DEFAULT_LLM_MODEL,
    embedder=DEFAULT_EMBED_MODEL,    # NEW
    embedder_source=EMBEDDER_SOURCE, # NEW
    records=records,
)
```

**Estimated eval_capture.py LOC delta: +3** (one import + two QueryLog kwargs).

## compare.py Rollup Integration

Today `aggregate_tier()` (compare.py:69-136) reads the queries JSON and builds a row dict that includes `model`, `timestamp`, `git_sha` (lines 130-132). The capture-provenance footer emit loop is at compare.py:259-264. Two changes:

**Change 1** — extend `aggregate_tier()` row dict (compare.py:130-132):

```python
# compare.py — proposed minimal change at lines 130-136:
return {
    "tier": tier,
    ...
    "timestamp": queries.get("timestamp"),
    "git_sha": queries.get("git_sha"),
    "model": queries.get("model"),
    "embedder": queries.get("embedder"),                # NEW (None for legacy)
    "embedder_source": queries.get("embedder_source"),  # NEW (None for legacy)
    ...
}
```

**Change 2** — extend the `capture_provenance` dict (compare.py:357-365) and the `emit_markdown` footer block (compare.py:259-264):

```python
# compare.py — _run() at line 357-365:
capture_provenance.append({
    "tier_label": r["tier_label"],
    "timestamp": r.get("timestamp", "—"),
    "model": r.get("model", "—"),
    "git_sha": r.get("git_sha", "—"),
    "embedder": r.get("embedder"),                # NEW
    "embedder_source": r.get("embedder_source"),  # NEW
})

# compare.py — emit_markdown() at lines 259-264 (existing per-tier line) + a new embedder line:
for prov in capture_provenance:
    lines.append(
        f"- `{prov['tier_label']}`: captured {prov['timestamp']} "
        f"(model `{prov['model']}`, git `{prov['git_sha']}`)"
    )
    # NEW — embedder line, em-dashed when missing (legacy captures pre-Phase 6):
    emb = prov.get("embedder") or "—"
    src = prov.get("embedder_source") or "—"
    lines.append(f"  - embedder: `{emb}` (source: `{src}`)")
```

**Optional Change 3 (recommended)** — add a dedicated "Embedder by Tier" disclosure block above the NaN breakdown, so the frozen doc has a clean lift-out table:

```markdown
**Embedder by tier:**

| Tier | Embedder Model | Source | Managed |
|------|----------------|--------|---------|
| tier-1 | openai/text-embedding-3-small | openrouter | no |
| tier-2 | gemini-embedding-001 | google-managed | yes |
| tier-3 | openai/text-embedding-3-small | openrouter | no |
| tier-4 | openai/text-embedding-3-small | openrouter | no |
| tier-5 | openai/text-embedding-3-small | openrouter | no |
```

The "Managed" column is derived (`source == "google-managed"`) — no third field needed in the schema.

**Estimated compare.py LOC delta: +8-12** (Change 1: +2 LOC, Change 2: +5 LOC, Change 3: +5-8 LOC if we lift the table out).

## freeze.py Sidecar Manifest Integration

`freeze.py:53-66` builds the per-tier manifest entry. Two new fields land in the entry dict:

```python
# freeze.py — proposed minimal change at lines 58-66:
entry: dict = {
    "status": "present",
    "generation_model": row.get("model"),
    "embedder": row.get("embedder"),               # NEW
    "embedder_source": row.get("embedder_source"), # NEW
    "capture_timestamp": row.get("timestamp"),
    "capture_git_sha": row.get("git_sha"),
    ...
}
```

**Estimated freeze.py LOC delta: +2.**

The sidecar manifest already records the *judge* embedder at line 72 (`"embedder": judge_emb`). The **per-tier** embedder is a different concept — capture-side, not judge-side. The two coexist cleanly because the `judge` block is nested under `"judge": {...}` while the per-tier embedder lives in `per_tier[tier-N]`.

## Backwards Compatibility

**Legacy QueryLog JSONs in `evaluation/results/queries/` (pre-Phase 6) lack the new fields.** Three behaviors must hold:

1. **Loading**: `QueryLog.model_validate_json(legacy.read_text())` succeeds with `embedder=None` + `embedder_source=None`. **Verified mechanism**: Pydantic v2's default behavior with `Optional[str] = None` accepts absence in the input. No code change needed beyond the schema declaration.
2. **compare.py rollup**: The capture-provenance footer emits `—` for the embedder line when `prov['embedder']` is `None`. This preserves the existing "honest disclosure" pattern (compare.py already emits `—` for missing tier rows at line 188-189).
3. **freeze.py manifest**: The `embedder` / `embedder_source` keys land as JSON `null` when the source QueryLog is legacy. A `tier-{N}` entry already has a `"status": "missing"` short-circuit at compare.py:57; we extend that to include the new fields-or-null path.

**Recommendation: do NOT migrate older JSONs.** Phase 7 will fully re-capture; the legacy JSONs become irrelevant once Phase 7 runs. Adding a migration step would be scope creep that v1.0 explicitly forbids.

**Phase 7 dependency**: Phase 7 will read the **new-shape** captures (because Phase 7's pipeline.py runs the Phase-6-modified run.py). The frozen v1.0 doc thus carries embedder data for every tier — Success Criterion 3 of Phase 6 satisfied at the Phase 7 ship.

## Tier 5 `managed=true` Flag — Resolution

The phase description in ROADMAP.md says:

> User can confirm Tier 1, 3, 4 record `text-embedding-3-small` (matching pinned config); Tier 2 records Google's managed embedder; Tier 5 records OpenAI's hosted vector-store embedder with an explicit `managed=true` flag

**This is wrong about Tier 5.** Tier 5 does NOT use a hosted vector store. Tier 5 reuses Tier 1's local ChromaDB and embeds via OpenRouter `openai/text-embedding-3-small` (verified `tier-5-agentic/tools.py:47-50` + `tier-5-agentic/tools.py:90-101`).

**Two options for the planner**:

**Option A (recommended)** — drop the `managed=true` flag from the schema entirely; treat "managed" as a *derived* property in the rollup table (`source == "google-managed"` ⇒ managed). This keeps the Pydantic schema minimal (two fields, not three), avoids redundancy with `embedder_source`, and matches the actual code — Tier 5's source is `openrouter`, identical to Tiers 1/3/4. **Tier 5 records: `embedder=openai/text-embedding-3-small, embedder_source=openrouter`.** No new flag.

**Option B** — keep a third nullable boolean field `embedder_managed: Optional[bool] = None` in `QueryLog`. Set `True` for Tier 2 only. Set `False` for Tiers 1/3/4/5. Adds a third field without new information (it duplicates `embedder_source == "google-managed"`).

**Option A is honest, minimal, and matches what the code actually does.** The frozen-doc embedder table can compute the "Managed" column from `source` without needing a flag in the JSON. This also fixes the ROADMAP's incorrect Tier 5 description: Tier 5's row in the frozen embedder table will read `openai/text-embedding-3-small` / `openrouter` / `no` — same as Tier 1, which is the **truth** of the Tier 5 implementation.

**Recommend Option A.** The planner should land Phase 6's schema with two fields (`embedder`, `embedder_source`) and document the resolution in the plan SUMMARY so the frozen doc disclosure stays honest.

## Test Strategy — Pure-Offline TDD red→green

Phase 6 has zero new API behavior. Every embedder string is a static literal known at code-edit time. Live testing would burn cost without verifying anything new (the live tests in Plans 03-03 and 05-02 verify *runtime API behavior*; here there is no runtime behavior).

**Recommended test plan** (matches Plan 03-01 / Plan 05-01 precedent — pure-offline TDD):

| Test | What It Verifies | Where |
|------|------------------|-------|
| `test_query_log_carries_embedder_field` | `QueryLog(embedder="x", embedder_source="y")` round-trips through `model_dump_json` / `model_validate_json` | `evaluation/tests/test_eval_records.py` |
| `test_query_log_legacy_json_loads_with_none_embedder` | `model_validate_json('{...legacy fields, no embedder...}')` succeeds with `embedder=None` | `evaluation/tests/test_eval_records.py` |
| `test_run_amain_writes_embedder_per_tier_offline` | Stub the five tier adapters → run amain → assert each tier's queries JSON top-level has the expected `(embedder, embedder_source)` tuple. Five sub-cases (one per tier). Use `monkeypatch` to stub adapter calls (matches `test_eval_run.py` precedent). | `evaluation/tests/test_eval_run.py` |
| `test_eval_capture_writes_embedder_for_tier_4` | Same approach, scoped to `tier-4-multimodal/scripts/eval_capture.py` | `tier-4-multimodal/tests/test_eval_capture.py` (new file or extend existing) |
| `test_compare_emits_embedder_in_provenance_footer` | Construct a fake QueryLog with `embedder="x"` + `embedder_source="y"` → invoke `emit_markdown` → assert the footer string contains `embedder: \`x\` (source: \`y\`)` | `evaluation/tests/test_eval_compare.py` |
| `test_compare_emits_em_dash_for_legacy_embedder` | Construct a legacy QueryLog dict (no `embedder` key) → invoke compare → assert footer contains `embedder: \`—\` (source: \`—\`)` | `evaluation/tests/test_eval_compare.py` |
| `test_compare_emits_embedder_table` (if Option C lifted out) | Assert the rollup table block is rendered with the four expected rows | `evaluation/tests/test_eval_compare.py` |
| `test_freeze_manifest_carries_embedder_per_tier` | Build a fake `evaluation/results/` tree with QueryLogs carrying embedder fields → invoke freeze → assert manifest's `per_tier[tier-N]` carries `embedder` + `embedder_source` | `evaluation/tests/test_eval_freeze.py` (extend) |

**Total tests: 7-8 new unit tests + 1-2 existing-test diffs (where compare/freeze tests already mock QueryLogs).** Estimated wall time per Plan 03-01 / 05-01 precedent: ~5-10 min for pure-offline TDD red→green.

**No live smoke is needed.** If the planner wants belt-and-braces verification post-Phase 7, the natural Phase-7 single-tier live smoke (Plan 05-02 precedent at $0.007 / 158s) would already exercise the new field flow because Phase 7 runs the same pipeline.py that runs the modified run.py — but a Phase-6-specific live test adds zero new signal beyond what Phase 7 will exercise anyway.

## Recommended Plan Structure

**One PLAN.md** (`06-01-embedder-provenance-capture.md`), single wave, ~6-8 tasks, all pure-offline TDD red→green. Matches the Phase 4 precedent (`04-01-PLAN.md` was a single plan that closed Phase 4 entirely).

**Task breakdown sketch:**

1. **Task 1 (RED)** — Extend `records.py::QueryLog` with `Optional[embedder, embedder_source]`. Add `test_query_log_carries_embedder_field` + `test_query_log_legacy_json_loads_with_none_embedder` (both fail without the schema change).
2. **Task 1 (GREEN)** — Add the two fields. Tests pass.
3. **Task 2** — Add `EMBEDDER_SOURCE` constants in tier-1, tier-2, tier-3, tier-4, tier-5 modules (+ `EMBED_MODEL = "gemini-embedding-001"` in tier-2-managed). RED: tier-imports test fails. GREEN: constants added.
4. **Task 3** — Thread the (embedder, source) tuples through `run.py::_capture_tier`. RED: `test_run_amain_writes_embedder_per_tier_offline` fails. GREEN: passes.
5. **Task 4** — Thread embedder through `tier-4-multimodal/scripts/eval_capture.py`. RED: `test_eval_capture_writes_embedder_for_tier_4` fails. GREEN: passes.
6. **Task 5** — Extend `compare.py::aggregate_tier` row dict + `emit_markdown` footer + (optional) embedder table. RED: `test_compare_emits_embedder_in_provenance_footer` + `test_compare_emits_em_dash_for_legacy_embedder` fail. GREEN: pass.
7. **Task 6** — Extend `freeze.py::freeze` per-tier manifest. RED: `test_freeze_manifest_carries_embedder_per_tier` fails. GREEN: passes.
8. **Task 7 (optional, if planner wants explicit verification gate)** — End-to-end fixture test: synthesize five fake tier QueryLogs with embedder fields → run compare → run freeze → assert the chain is intact. Mirrors Plan 05-01's `test_e2e_with_stub_stages` (offline composition test).

**Atomic commits:** one commit per task (RED + GREEN may be one combined commit per Plan 03-01 precedent, or two separate commits per Plan 04-01 / 05-01 precedent — planner picks).

**Estimated wall time:** ~15-25 min total (matches Plan 03-01 + Plan 05-01 averages for pure-offline multi-task plans).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| New schema validation | Custom JSON validator for `embedder` field | Pydantic v2 `Optional[str] = None` on QueryLog | Already the project's schema layer; backwards-compat behavior is documented (records.py:74) |
| New "managed?" flag in schema | Third boolean field `embedder_managed` | Derive from `embedder_source == "google-managed"` in compare.py | Avoids redundancy; one less field to keep in sync |
| New legacy-migration tool | Walk old JSONs and rewrite them with embedder=None explicit | Just rely on Pydantic's `Optional` default | `evaluation/results/queries/` is gitignored — legacy artifacts are local and Phase 7 re-captures everything |
| New embedder-confound rollup module | `evaluation/harness/embedder_disclosure.py` | Inline 5-row table in compare.py's existing footer | Existing footer pattern handles the disclosure; one module is the cost ceiling for v1.0 |

**Key insight:** This phase is *the smallest possible instrumentation change* that closes CAP-03. The existing harness already passes through `model`, `timestamp`, `git_sha` end-to-end — `embedder` + `embedder_source` follow the exact same path. No new abstraction is needed; the planner should resist any temptation to build an "embedder registry" or "provenance manager" pattern.

## Common Pitfalls

### Pitfall 1: Tier 4 capture has TWO entry points — both must be edited

**What goes wrong:** A planner instruments `evaluation/harness/run.py` and considers Tier 4 covered. But Tier 4's live path is `tier-4-multimodal/scripts/eval_capture.py` (verified Plan 02-03 evidence + STATE.md), which builds its own QueryLog at line 195-201. If only run.py is edited, Tier-4 captures done via the live path will lack the embedder field.

**Why it happens:** Phase 130 SC-1 deferred Tier 4's live test to the user (sandbox kernel-level OMP shmem block on MineRU). The harness's run.py supports a cached-mode-only Tier 4 path; the live path lives in the tier-4 scripts directory. Two entry points = two QueryLog construction sites.

**How to avoid:** Edit BOTH `evaluation/harness/run.py:261-267` AND `tier-4-multimodal/scripts/eval_capture.py:195-201`. Add a unit test for each path (`test_run_amain_writes_embedder_per_tier_offline` covers run.py; `test_eval_capture_writes_embedder_for_tier_4` covers eval_capture.py).

**Warning signs:** The plan has tasks for run.py edits but not for eval_capture.py edits. Or: only one tier-4 test exists (run.py's cached path), not two.

### Pitfall 2: Legacy JSONs on disk break compare.py if the field is required

**What goes wrong:** Planner makes `embedder` a required field with a string default like `embedder: str = "unknown"`. Older committed Tier-4 JSONs (e.g., `tier-4-2026-05-05T13_59_50Z.json` — verified to exist) deserialize cleanly but inject `"unknown"` silently. The rollup now claims those tiers used an "unknown" embedder, which is a false provenance disclosure.

**Why it happens:** `evaluation/results/queries/` is gitignored, but local artifacts on disk will exist for every developer who has run captures before Phase 6. Phase 7 will re-capture; until then, `compare.py` may run against legacy JSONs (e.g., during Phase 6's own test fixtures or developer ad-hoc runs).

**How to avoid:** Use `Optional[str] = None`. Have compare.py emit `—` (em-dash) when the field is `None`. This is the existing "honest disclosure" pattern (compare.py:188-189 already does this for missing tier rows).

**Warning signs:** A test name like `test_query_log_default_embedder_is_unknown` — that's the symptom of a required-with-default schema.

### Pitfall 3: Tier 2's embedder identifier is a hardcoded string today (no constant exists)

**What goes wrong:** Planner imports `EMBED_MODEL` from `tier-2-managed/main.py` and the import fails because the constant doesn't exist. Or: planner hardcodes `"gemini-embedding-001"` directly in run.py's tier-2 branch, breaking the "single source of truth per tier" rule.

**Why it happens:** Tier 2's only embedder reference is at `tier-2-managed/main.py:191` — a hardcoded argument to `tracker.record_embedding()`. There is no module-level `EMBED_MODEL` constant for Tier 2 like there is for Tiers 1, 3, 4.

**How to avoid:** Phase 6 must add `EMBED_MODEL = "gemini-embedding-001"` and `EMBEDDER_SOURCE = "google-managed"` as module-level constants in `tier-2-managed/main.py` BEFORE attempting the run.py thread-through. Do this in Task 2 of the plan (add constants across all five tiers).

**Warning signs:** A plan task that says "import EMBED_MODEL from tier-2" without a prior task that *adds* EMBED_MODEL to tier-2.

### Pitfall 4: Tier 5's embedder constant should re-import from Tier 1, not duplicate

**What goes wrong:** Planner adds a `EMBED_MODEL = "openai/text-embedding-3-small"` constant in `tier-5-agentic/tools.py` as if Tier 5 owned the embedder. Now there are two sources of truth for the same string and a future bump to e.g. `text-embedding-3-large` requires touching two modules.

**Why it happens:** The DRY temptation when adding constants symmetrically across tiers.

**How to avoid:** Tier 5 already imports `EMBED_MODEL from tier_1_naive.embed_openai` (tools.py:47-50). The Phase 6 change at the Tier-5 site is to add `EMBEDDER_SOURCE = "openrouter"` only — the embedder string is re-exported from Tier 1's module by reference. The harness adapter `evaluation/harness/adapters/tier_5.py` does NOT need to know about the embedder; only run.py needs to read it for the QueryLog assembly.

**Warning signs:** A plan task adds the same string `"openai/text-embedding-3-small"` to two tier modules (Tier 1 AND Tier 5). The Tier 5 add should be a re-import, not a duplication.

### Pitfall 5: The ROADMAP's Tier 5 description must be overridden in the plan

**What goes wrong:** Planner takes ROADMAP literally and tries to record `embedder = "openai-hosted-managed"` + `managed = true` for Tier 5. Test assertions match the ROADMAP wording. The test passes but the data is wrong — Tier 5 actually uses OpenRouter `openai/text-embedding-3-small`.

**Why it happens:** The phase description in `.planning/ROADMAP.md` was authored before the Tier 5 implementation was fully verified. The actual Tier-5 code (tier-5-agentic/tools.py:47-101) reuses Tier 1's ChromaDB and OpenRouter-routed embedder.

**How to avoid:** The plan SUMMARY must explicitly call out the override and cite the source-of-truth file (`tier-5-agentic/tools.py:47-50`). Test assertions must match the **code reality**, not the ROADMAP wording. The frozen doc embedder table will then read truthfully (Tier 5 source: openrouter, managed: no — same as Tier 1).

**Warning signs:** A plan task or test that uses the string `"openai-hosted-managed"` or `"openai-hosted"` for Tier 5. That's the ROADMAP wording, which is wrong.

### Pitfall 6: compare.py's `emit_markdown` must tolerate Optional fields without crashing

**What goes wrong:** `prov['embedder']` raises KeyError when the row dict was built from a legacy JSON that lacked the field. The compare command exits 1 with a Python traceback.

**Why it happens:** Defensive `.get()` is required because legacy JSONs deserialize to `QueryLog` with `embedder=None`, and `aggregate_tier` may build a row dict that doesn't carry the key explicitly.

**How to avoid:** Always use `prov.get("embedder")` not `prov["embedder"]`. Render `—` when `None`. Mirror the existing `_fmt_float` em-dash pattern (compare.py:182-185).

**Warning signs:** A test fixture that always sets the field; no test for the legacy-missing path.

## Code Examples

### Pydantic v2 Optional field with backwards-compat default

```python
# Source: evaluation/harness/records.py:40-47 (existing) + Phase 6 additions
from typing import Optional
from pydantic import BaseModel

class QueryLog(BaseModel):
    """A single tier × timestamp capture run, holding all 30 EvalRecords."""
    tier: str
    timestamp: str
    git_sha: str
    model: str
    embedder: Optional[str] = None        # NEW
    embedder_source: Optional[str] = None # NEW
    records: list  # Pydantic infers list[EvalRecord] from import context

# Round-trip verification (test pattern from test_eval_records.py:13-30):
log = QueryLog(
    tier="tier-1",
    timestamp="2026-05-06T12:00:00Z",
    git_sha="abc1234",
    model="google/gemini-2.5-flash",
    embedder="openai/text-embedding-3-small",
    embedder_source="openrouter",
    records=[],
)
roundtrip = QueryLog.model_validate_json(log.model_dump_json())
assert roundtrip.embedder == "openai/text-embedding-3-small"
assert roundtrip.embedder_source == "openrouter"

# Legacy-tolerant load:
legacy_json = '{"tier":"tier-1","timestamp":"2026-05-02T17:26:59Z","git_sha":"ce5c2ad","model":"google/gemini-2.5-flash","records":[]}'
legacy = QueryLog.model_validate_json(legacy_json)
assert legacy.embedder is None  # field absent in JSON, default None applied
assert legacy.embedder_source is None
```

### Compare.py footer emit with em-dash fallback

```python
# Source: evaluation/harness/compare.py:259-264 (existing pattern) + Phase 6 additions
for prov in capture_provenance:
    lines.append(
        f"- `{prov['tier_label']}`: captured {prov['timestamp']} "
        f"(model `{prov['model']}`, git `{prov['git_sha']}`)"
    )
    emb = prov.get("embedder") or "—"
    src = prov.get("embedder_source") or "—"
    lines.append(f"  - embedder: `{emb}` (source: `{src}`)")
```

## Open Questions

1. **Should Phase 6 add the embedder-disclosure table to comparison.md, or leave it to Phase 9 (Frozen Handoff Doc)?**
   - **What we know**: compare.py already emits the per-tier provenance footer (line 256-266). Adding a small embedder table there is +5-8 LOC.
   - **What's unclear**: Phase 9 owns the frozen v1.0 doc; the embedder confound disclosure is one of Phase 9's success criteria (DOC-04). If compare.py emits the table, Phase 9 lifts it from comparison.md verbatim. If not, Phase 9 must build it from the manifest sidecar.
   - **Recommendation**: emit the table in compare.py during Phase 6 (Task 5 of the plan). It's +8 LOC and makes Phase 9 a pure copy-paste. The frozen-doc author (or freeze.py) reads `comparison.md` and copies the table.

2. **Should `tier-2-managed` get the `EMBED_MODEL` constant in `main.py` or in a new `embed.py`?**
   - **What we know**: Tier 2's main.py already has the hardcoded string at line 191. Tier 1's pattern is `tier-1-naive/embed_openai.py` (separate embed module).
   - **What's unclear**: Symmetry argues for a new `tier-2-managed/embed.py`; minimal-change argues for adding the constant to `tier-2-managed/main.py`.
   - **Recommendation**: add to `tier-2-managed/main.py` (sibling to existing `INDEX_USD_PER_1M_TOKENS` constant at line 91). Tier 2 has no separate embed call — Google File Search is managed — so a separate `embed.py` would be empty boilerplate. Two new lines in main.py is the smallest honest change.

3. **Should the schema use `embedder` or `embedder_model` as the field name?**
   - **What we know**: `model` is the LLM identifier; `embedder` is the embedding model. Both are "models" but they live in different layers.
   - **What's unclear**: Symmetry argues for `embedder_model`; brevity argues for `embedder`.
   - **Recommendation**: `embedder` (single word). Matches the project's naming convention in `score.py` (`judge_emb` not `judge_embedder_model`) and `freeze.py` (`"embedder": judge_emb` in the manifest at line 72). The existing project uses `embedder` for the embedding model identifier; mirror that.

## Sources

### Primary (HIGH confidence — verified against source files in this session)

- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/records.py` (records.py:40-47) — QueryLog schema (read end-to-end)
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/run.py` (run.py:146-272) — `_capture_tier` per-tier branches (read end-to-end)
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/compare.py` (compare.py:69-307) — `aggregate_tier`, `emit_markdown`, `_run` (read end-to-end)
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/freeze.py` (freeze.py:42-77) — `freeze` + per-tier manifest entry (read end-to-end)
- `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/pipeline.py` (pipeline.py:93-146) — Phase 5 in-process composition (verified no edits needed)
- `/Users/patrykattc/work/git/rag-architecture-patterns/tier-1-naive/embed_openai.py:31` — `EMBED_MODEL = "openai/text-embedding-3-small"` (Tier 1 source of truth)
- `/Users/patrykattc/work/git/rag-architecture-patterns/tier-2-managed/main.py:191` — `tracker.record_embedding("gemini-embedding-001", ...)` (Tier 2 hardcoded — needs new constant)
- `/Users/patrykattc/work/git/rag-architecture-patterns/tier-3-graph/rag.py:56` — `DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"` (Tier 3 source of truth)
- `/Users/patrykattc/work/git/rag-architecture-patterns/tier-4-multimodal/rag.py:53` — `DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"` (Tier 4 source of truth)
- `/Users/patrykattc/work/git/rag-architecture-patterns/tier-5-agentic/tools.py:47-50` + `tier-5-agentic/tools.py:90-101` — Tier 5 reuses Tier 1's EMBED_MODEL via OpenRouter; **definitively NOT a hosted vector store** (resolves ROADMAP error)
- `/Users/patrykattc/work/git/rag-architecture-patterns/tier-4-multimodal/scripts/eval_capture.py:46-201` — Tier 4 live entry point (separate from harness/run.py)
- `/Users/patrykattc/work/git/rag-architecture-patterns/.gitignore` — confirms `evaluation/results/queries/` is gitignored (legacy JSONs are local-only)
- Five existing capture JSONs in `evaluation/results/queries/` — verified top-level keys are `{tier, timestamp, git_sha, model, records}` only

### Secondary (MEDIUM confidence — read but used for context only)

- `evaluation/harness/score.py:69-71, 339, 555` — judge embedder pattern (separate concept from per-tier embedder, but informs naming)
- `evaluation/tests/test_eval_records.py:1-30` — existing Pydantic test pattern (template for new tests)
- `.planning/REQUIREMENTS.md` (CAP-03 + DOC-04) — phase requirement scope
- `.planning/STATE.md` — Phase 5 close + Phase 6 readiness statement
- `.planning/ROADMAP.md` Phase 7 dependency block — confirms Phase 7 reads new-shape captures

### Tertiary (LOW confidence — none required)

- No external library research needed. All embedder strings are repo-internal constants. Pydantic v2 backwards-compat behavior is documented (records.py:74-75) and verified against the project's existing test pattern (test_eval_records.py).

## Metadata

**Confidence breakdown:**
- Schema extension: **HIGH** — read records.py + the five capture JSONs end-to-end; Pydantic v2 Optional default behavior verified in source.
- Per-tier embedder reality: **HIGH** — read all five tier modules' embedder source files; ROADMAP error resolved.
- compare.py / freeze.py integration: **HIGH** — read both modules end-to-end; integration sites identified line-by-line.
- Plan structure / wall time: **HIGH** — three prior pure-offline TDD plan precedents (03-01, 04-01, 05-01) provide consistent ~5-18 min wall times.
- Tier 5 ROADMAP description override: **HIGH** — verified by reading tools.py + agent.py; no hosted vector store anywhere in Tier 5 code path.
- Backwards compatibility: **HIGH** — Pydantic v2 `Optional[T] = None` default behavior is documented and project-tested (records.py:74-75 confirms `model_validate_json` is the canonical loader).

**Research date:** 2026-05-06
**Valid until:** 2026-06-05 (30 days — schema changes are stable; only invalidated if a tier swaps embedders, which is out of v1.0 scope per ROADMAP "Out of Scope" line "Switching providers")

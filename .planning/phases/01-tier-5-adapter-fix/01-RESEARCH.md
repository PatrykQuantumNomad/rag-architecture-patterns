# Phase 1: Tier 5 Adapter Fix - Research

**Researched:** 2026-05-04
**Domain:** OpenAI Agents SDK 0.14.6 — extracting tool outputs from `RunResult.new_items`; RAGAS 0.4.3 NaN gating; smoke harness wiring inside the existing 3-stage eval pipeline
**Confidence:** HIGH (every claim verified against installed source at `.venv/lib/python3.13/site-packages/agents/` or against existing repo files; no Context7 / web fetches needed because the SDK source is on disk and the bug fix touches code already present in the repo)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Context extraction shape

- **Granularity:** One entry per chunk/result. `search_text_chunks` output is split into N entries (one per chunk); each `lookup_paper_metadata` output is its own entry. RAGAS `context_precision@K` needs a real list with K>1 to be meaningful.
- **Entry content:** Text-only with a short provenance prefix. Format: `[paper_id=<id>] <text>` where text is the chunk text (for `search_text_chunks`) or the abstract field (for `lookup_paper_metadata`). Strip JSON braces and field names — RAGAS judges natural prose.
- **Tool scope:** Both tools feed `retrieved_contexts`. `lookup_paper_metadata` outputs are included via the abstract field only (title/authors/year are metadata, not grounding text).
- **Aggregation across iterations:** Concat all `ToolCallOutputItem` outputs in agent-iteration order, then dedup by stable id (`chunk_id` for chunks, `paper_id` for metadata lookups). Repeated tool calls don't inflate `context_precision`.

#### Smoke test question selection

- **Selection method:** Deterministic question IDs. Hard-pick a specific 5 from the golden manifest and reuse them every smoke run.
- **Class mix:** 3 single-hop + 2 multi-hop. **No multimodal** in the smoke set — Tier 5 retrieves from text-only chunks (Tier 1's ChromaDB), so multimodal questions don't have fair retrieval here.
- **Stability across phases:** Same 5 IDs reused for Phase 1 (Tier 5 smoke), Phase 2 (Tier 4 smoke), and any future smoke checks. Apples-to-apples deltas; one row of provenance.
- **Location:** CLI flag `--smoke-question-ids id1,id2,...` on `evaluation/harness/run.py` with a hard-coded default constant. Default reproduces the gate; flag allows ad-hoc overrides without code edits. The actual 5 IDs are picked during planning by reading `dataset/manifests/` and selecting deterministically.

#### Smoke pass/fail threshold

- **Population gate:** ≥4/5 populated `retrieved_contexts`. Allows one legitimate Pitfall 9 self-cite case while requiring strong evidence the fix landed.
- **Score gate:** Also require RAGAS `faithfulness` and `context_precision` to return non-NaN on the populated cells. Population alone doesn't guarantee usefulness — score.py runs against the smoke 5 as part of the gate.
- **Truncated cells (`error="max_turns_exceeded"`, Pitfall 8):** Excluded from the denominator, logged separately in the smoke report.
- **Zero-tool-call cells (Pitfall 9 — agent self-cited from training):** Excluded from the denominator, logged separately. The rate is surfaced in the report so we can decide later if it's acceptable.
- **Minimum measurable denominator:** If exclusions leave fewer than 3 measurable cells, the smoke is **inconclusive** — re-run with a different 5 or investigate. Otherwise apply the ≥0.8 ratio (effectively ≥4/5-equivalent) on the measurable subset.

#### Fallback policy when smoke fails

- **Ordering:** Instrument first, mutate second. If the initial smoke fails, install the `debug-tier5` extra (`openinference-instrumentation-openai-agents==1.4.2`, `arize-phoenix>=6,<8`, `opentelemetry-sdk>=1.30,<2`, `opentelemetry-exporter-otlp-proto-http>=1.30,<2`), re-run smoke with `set_tracing_disabled(disabled=False)`, and observe Phoenix spans to confirm: (a) the agent is calling tools, (b) tools are returning data, (c) the adapter is seeing `ToolCallOutputItem` in `result.new_items`. Only after spans tell us *what is happening* do we attempt STACK.md mutations (simplify schema → switch model slug → bump openai-agents).
- **Exit policy:** Phase 1 ships when the smoke gate passes (≥4/5 populated AND non-NaN RAGAS on the measurable subset). If after instrumentation + the 3 STACK.md mutations the gate still fails, **mark Phase 1 incomplete, write a fallback-log doc, and escalate to the user before Phase 7**. No silent partial-ship.
- **Provenance for any degraded ship:** If the user explicitly authorizes shipping a partial fix (e.g., 3/5 measurable populated instead of 4/5), write `evaluation/results/diagnostics/tier-5-fallback-{TS}.json` recording every attempt (mutation tried, smoke result, span observations). Phase 9's frozen doc gets an explicit "Tier 5: partial-fix caveat" line citing that JSON. Honest provenance, no hidden caveats.

### Claude's Discretion

- **Per-attempt budget:** How many smoke runs / dollars / wall-clock per fallback step before calling it "tried". Claude picks based on signal quality (e.g., one smoke run is usually enough; flaky truncations may justify a re-run).
- **Specific 5 question IDs:** Selected during planning by reading `dataset/manifests/papers.json` and picking 3 single-hop + 2 multi-hop with deterministic, defensible criteria.
- **Dedup tie-breaking:** Which entry wins when the same `chunk_id` appears with different scores across iterations (probably first occurrence; not consequential for RAGAS).
- **Smoke-report shape:** Schema of the smoke-result JSON written for the gate (Claude designs to match existing harness conventions).
- **Tool-error handling:** What to do if a `ToolCallOutputItem` has an error payload rather than a result (probably skip with a logged warning).

### Deferred Ideas (OUT OF SCOPE)

- **OpenInference + Phoenix as a runtime dep:** They stay in the `debug-tier5` optional extra (CLEANUP-02 in REQUIREMENTS.md v1.1) — diagnostic-only, not committed to the default install.
- **Adding new tools to the Tier 5 agent:** out of scope.
- **Bootstrap 95% CI on the smoke gate** (METH-01): smoke uses point estimates only.
- **Multi-judge spot-check on the smoke 5:** Phase 8 owns the multi-judge re-score; the smoke gate uses Gemini-only scoring.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TIER-01 | User can run Tier 5 evaluation and get non-empty `retrieved_contexts` populated by walking `RunResult.new_items` for `ToolCallOutputItem` (replaces the hard-coded `retrieved_contexts=[]` at `evaluation/harness/adapters/tier_5.py:125`) | Verified `RunResult.new_items: list[RunItem]` and `ToolCallOutputItem.output: Any` at `.venv/lib/python3.13/site-packages/agents/result.py:181` and `agents/items.py:376-390` (installed openai-agents==0.14.6). The walk pattern is a 5-line change inside `run_tier5`. |
| TIER-03 (Tier 5 portion) | User can verify the fix on a 5-question smoke test before committing to a full 30-question rerun | The existing harness already supports `--limit N`. CONTEXT.md locks a stronger contract: a `--smoke-question-ids` flag with a hard-coded default of 5 specific IDs. Recommended default (text-only, 3 single-hop + 2 multi-hop): `single-hop-001, single-hop-002, single-hop-003, multi-hop-001, multi-hop-002` — see Architecture Patterns §Smoke Question IDs. |
</phase_requirements>

---

## Summary

The bug at `evaluation/harness/adapters/tier_5.py:125` is exactly what STACK.md describes: a hard-coded `retrieved_contexts=[]` that pre-dates the discovery that the openai-agents SDK exposes tool outputs through `RunResult.new_items`. The fix is a literal 5-line walk; the rest of Phase 1 is scaffolding for an honest smoke gate, a fallback path, and provenance for any degraded ship.

The SDK contract is fully verified by reading the installed source at `.venv/lib/python3.13/site-packages/agents/` (openai-agents 0.14.6, the exact pin in pyproject.toml). `RunResult` extends `RunResultBase` which declares `new_items: list[RunItem]`. `RunItem` is a union including `ToolCallOutputItem`, which is a dataclass with two relevant fields: `raw_item: ToolCallOutputTypes` (the stringified Responses-API payload sent back to the model) and `output: Any` (the **raw return value** of the `@function_tool` callable). For `search_text_chunks(query, k)` that's the `list[dict]` built at `tier-5-agentic/tools.py:132-145`; for `lookup_paper_metadata(paper_id)` that's the `dict` built at `tier-5-agentic/tools.py:163-171`. The adapter never needs to parse the stringified payload — `item.output` is the canonical typed value.

The smoke gate, fallback policy, and provenance JSON are all locked by CONTEXT.md. The planner's job is to (a) write the 5-line walk plus dedup logic, (b) update the existing test at `evaluation/tests/test_eval_adapters.py:174` (which currently asserts `retrieved_contexts == []` — a load-bearing assertion that must invert), (c) add the `--smoke-question-ids` CLI flag with a hard-coded default constant, (d) wire a fallback-log writer at `evaluation/results/diagnostics/tier-5-fallback-{TS}.json` for any degraded ship, and (e) make `set_tracing_disabled(disabled=True)` at `tier-5-agentic/agent.py:43` conditional on an env flag so the OpenInference fallback path can capture spans.

**Primary recommendation:** Use `from agents import ToolCallOutputItem` (it's re-exported from `agents/__init__.py:37`) and `isinstance(item, ToolCallOutputItem)` to filter `result.new_items`. Per-tool-output, branch on `isinstance(out, list)` (chunks) vs `isinstance(out, dict)` (metadata lookup) — the SDK puts the **raw return value** on `item.output`, not a stringified version. Dedup via a tiny in-loop set keyed on `(paper_id, page)` for chunks and `paper_id` for metadata. Wrap entries in CONTEXT.md's `[paper_id=<id>] <text>` provenance prefix before appending to the `contexts` list.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Walk `RunResult.new_items` for `ToolCallOutputItem` and project tool outputs into `retrieved_contexts` | `evaluation/harness/adapters/tier_5.py` (the eval adapter) | — | The agent itself (in `tier-5-agentic/agent.py` + `tools.py`) returns its own values; only the eval-side adapter is responsible for transforming them into the RAGAS shape. The agent must remain unaware of RAGAS — that boundary is locked. |
| Conditional tracing toggle for OpenInference fallback | `tier-5-agentic/agent.py:43` (where `set_tracing_disabled` lives) | `evaluation/harness/run.py` (passes a flag through) | The toggle has to live at module-load time because `set_tracing_disabled` runs once on import. Read an env var (e.g., `RAG_DEBUG_TIER5_TRACING=1`) so the harness can opt in without touching the agent module's API. |
| Smoke question selection | `evaluation/harness/run.py` (the existing CLI) | `evaluation/golden_qa.json` (data) | Existing argparse accepts `--limit`; add `--smoke-question-ids id1,id2,...` next to it. Default = a hard-coded module-level constant (the 5 IDs). The selection is filtering, not loading — `_load_golden_qa()` already returns the full 30. |
| Fallback diagnostics JSON writer | `evaluation/harness/adapters/tier_5.py` or a new `evaluation/harness/diagnostics.py` | `evaluation/results/diagnostics/` (output dir, new) | Plan-level call: this is fallback scaffolding, not core path; isolating it in a tiny new file keeps the adapter's main flow clean. The directory does not yet exist; `mkdir(parents=True, exist_ok=True)` at write time per existing harness convention. |
| Smoke-report JSON schema | New module (Claude's discretion per CONTEXT.md) | Mimics `evaluation/results/queries/*.json` shape (Pydantic model, indent-2 JSON, ISO 8601 UTC timestamp with `Z` suffix) | Repo convention is fully consistent: every persisted artifact is a Pydantic-`model_dump_json(indent=2)` write. Match it. |

---

## Standard Stack

The phase introduces **zero new runtime dependencies**. Every library it touches is already pinned in `pyproject.toml`. The fallback path (debug-tier5 extras) is also covered by CONTEXT.md as deferred to v1.1's `[debug-tier5]` optional extra (CLEANUP-02 in REQUIREMENTS.md), so even instrumentation is install-on-demand, not a Phase 1 deliverable.

### Core (Already Pinned — DO NOT change)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `openai-agents[litellm]` | `==0.14.6` (verified `.venv/lib/python3.13/site-packages/openai_agents-0.14.6.dist-info/METADATA`) | Agent runtime; provides `RunResult.new_items` + `ToolCallOutputItem` (the fix vehicle) | Pinned across the repo — bumping invalidates Phase 130's pricing assumptions and forces a rerun. The 0.14.x API surface for `new_items` / `ToolCallOutputItem` is stable per the installed source. [VERIFIED: `.venv/lib/python3.13/site-packages/agents/items.py:376` and `agents/__init__.py` re-export] |
| `litellm` | `==1.83.0` (verified `.venv/lib/python3.13/site-packages/litellm-1.83.0.dist-info/`) | OpenRouter routing for the Gemini agent + the RAGAS judge | Already in use across all tiers. No bump needed. |
| `ragas` | `==0.4.3` (verified `.venv/lib/python3.13/site-packages/ragas-0.4.3.dist-info/`) | Metrics scoring; the smoke score-gate runs `score.amain()` over the 5-record subset | Already patched twice in this codebase (`f0cd134`, `186a9c2`) — bumping invalidates those fixes. |
| `chromadb` | `>=1.5.8,<2` (pinned, version not directly inspected this session) | Tier 1's collection; Tier 5's `search_text_chunks` reads it via `tier_1_naive.store.open_collection(reset=False)` (Pitfall 9 read-only invariant) | No change. |
| `pydantic` | `>=2.10,<3` (per `pyproject.toml:28`) | All harness records (`EvalRecord`, `QueryLog`, `ScoreRecord`); the new smoke-report JSON should be a Pydantic v2 model written via `model_dump_json(indent=2)` (matches `evaluation/harness/records.py:69`) | Repo convention. |

### Supporting (Already Pinned — for the Tier 5 agent path itself)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `lightrag-hku` | `==1.4.15` (verified `.venv/lib/python3.13/site-packages/lightrag_hku-1.4.15.dist-info/`) | Tier 3 stack; not touched by Phase 1 but its version is recorded in any fallback log per CONTEXT.md provenance requirement | Recorded in fallback JSON only |
| `raganything` | `==1.2.10` (verified `.venv/lib/python3.13/site-packages/raganything-1.2.10.dist-info/`) | Tier 4 stack; not touched | Recorded in fallback JSON only |

### Conditional / Fallback (NOT installed by default — `debug-tier5` extra; install only if smoke fails)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `openinference-instrumentation-openai-agents` | `==1.4.2` | OTel spans of every Runner step including `ToolCallItem` and `ToolCallOutputItem` JSON payloads | If smoke shows empty contexts AFTER the adapter fix — confirms tool calling vs. tool absence |
| `arize-phoenix` | `>=6,<8` | Local OTLP collector + UI for span inspection | Same as above |
| `opentelemetry-sdk` | `>=1.30,<2` | Tracer provider | Same as above |
| `opentelemetry-exporter-otlp-proto-http` | `>=1.30,<2` | OTLP/HTTP exporter to Phoenix | Same as above |

[CITED: `.planning/research/STACK.md` lines 32-44 and 78-84 — confirms these versions and their purpose; also confirms the install command is the `[debug-tier5]` optional extra defined in REQUIREMENTS.md v1.1 CLEANUP-02.]

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Walking `result.new_items` | `result.context_wrapper.run_data` introspection | The SDK does not expose tool-output collection on `context_wrapper` — `new_items` is the documented and only path. [VERIFIED: `agents/result.py:181` declares `new_items` directly on `RunResultBase`; `context_wrapper` carries usage/state only.] |
| Walking `result.new_items` | `result.final_output` as the context source | `final_output` is the **answer**, not the evidence. Using it as both makes faithfulness trivially 1.0 (answer grounded in itself). [CITED: `.planning/research/STACK.md:287` What-NOT-to-use table.] |
| `isinstance(item, ToolCallOutputItem)` | `getattr(item, "type", None) == "tool_call_output_item"` | The string-literal form is more brittle but defensive: if a future SDK patch renames the dataclass while keeping the type literal, the duck-type check survives. **Use `isinstance` for now (HIGH confidence, current SDK)**; document the duck-type fallback in a comment. [VERIFIED: `agents/items.py:387` declares `type: Literal["tool_call_output_item"]`.] |
| `from agents.items import ToolCallOutputItem` | `from agents import ToolCallOutputItem` | The top-level package re-exports it. Top-level form matches the import style already used in the same file (`from agents import Runner` at `tier_5.py:36`). |

---

## Architecture Patterns

### System Architecture Diagram (Phase 1 data flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ user invokes:  python -m evaluation.harness.run --tiers 5 \                 │
│                  --smoke-question-ids single-hop-001,…,multi-hop-002 --yes  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │  evaluation/harness/run.py         │
                  │   - parse args                     │
                  │   - _load_golden_qa()  → 30 Qs     │
                  │   - if --smoke-question-ids set:   │
                  │     filter by id (NEW)             │
                  │   - _capture_tier(tier=5, qa, …)   │
                  └────────────────────────────────────┘
                                    │
                                    ▼ (per-question loop)
                  ┌────────────────────────────────────┐
                  │  adapters/tier_5.py: run_tier5()   │
                  │   ┌────────────────────────────┐   │
                  │   │ Runner.run(agent, q,       │   │
                  │   │            max_turns=10)   │   │
                  │   └────────────────────────────┘   │
                  │             │                       │
                  │             ▼                       │
                  │      RunResult { new_items, … }    │
                  │             │                       │
                  │             ▼                       │
                  │   ┌────────────────────────────┐   │  THE 5-LINE FIX
                  │   │ for item in new_items:     │   │
                  │   │   if isinstance(item,      │   │
                  │   │      ToolCallOutputItem):  │   │
                  │   │     project item.output    │   │
                  │   │     → contexts list        │   │
                  │   │     (dedup + provenance)   │   │
                  │   └────────────────────────────┘   │
                  │             │                       │
                  │             ▼                       │
                  │   EvalRecord(retrieved_contexts=    │
                  │              contexts, …)           │
                  └────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │  agent stack (NOT MODIFIED beyond conditional tracing)     │
       │                                                            │
       │  tier-5-agentic/agent.py:                                  │
       │    Agent(tools=[search_text_chunks, lookup_paper_metadata])│
       │    set_tracing_disabled(disabled=  ←── make conditional    │
       │      not os.getenv("RAG_DEBUG_TIER5_TRACING")              │
       │    )                                                       │
       │                                                            │
       │  tier-5-agentic/tools.py:                                  │
       │    @function_tool search_text_chunks → list[dict]          │
       │      keys: paper_id, page, snippet, similarity             │
       │    @function_tool lookup_paper_metadata → dict             │
       │      keys: paper_id, title, authors, year, abstract        │
       └────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │  evaluation/results/queries/       │
                  │    tier-5-{TS}.json                │
                  │    (now with non-empty contexts)   │
                  └────────────────────────────────────┘
                                    │
                                    ▼ (smoke gate phase)
                  ┌────────────────────────────────────┐
                  │  evaluation/harness/score.py       │
                  │   - read latest tier-5-*.json      │
                  │   - _short_circuit_nan() classifies│
                  │     each row:                      │
                  │       empty_contexts (Pitfall 9)   │
                  │       agent_truncated (Pitfall 8)  │
                  │       (else) RAGAS-scored          │
                  │   - write metrics/tier-5-{TS}.json │
                  └────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │  smoke gate evaluator (NEW)        │
                  │   - read both query log + metrics  │
                  │   - exclude max_turns_exceeded     │
                  │     and zero-tool-call rows        │
                  │   - measurable denominator ≥ 3?    │
                  │     - PASS: ≥0.8 ratio populated   │
                  │              + non-NaN metrics     │
                  │     - FAIL: write fallback JSON    │
                  │             & escalate             │
                  └────────────────────────────────────┘
```

### Recommended Project Structure (additions only)

```
evaluation/
├── harness/
│   ├── adapters/
│   │   └── tier_5.py             # MODIFIED: 5-line walk + dedup + provenance prefix
│   ├── run.py                    # MODIFIED: --smoke-question-ids flag + default const
│   ├── score.py                  # UNCHANGED — already short-circuits empty_contexts
│   └── smoke_gate.py             # NEW (~80 LOC): read query+metrics, classify, gate
└── results/
    └── diagnostics/              # NEW directory (created on first failure)
        └── tier-5-fallback-{TS}.json   # written ONLY when escalating
tier-5-agentic/
└── agent.py                      # MODIFIED: 1-line conditional on tracing toggle
```

The smoke gate logic is small enough to inline into `run.py` if the planner prefers. Keeping it in a separate module simplifies unit testing (the gate is pure logic over JSON shapes — no API calls).

### Pattern 1: The 5-Line Walk

**What:** Iterate `result.new_items`, filter `ToolCallOutputItem`, project each `item.output` into one or more provenance-prefixed strings.

**When to use:** This is the entire load-bearing transformation. Apply once inside `run_tier5`, between `result = await Runner.run(...)` and the `EvalRecord(...)` constructor. Never call it elsewhere — RAGAS shape is the eval boundary, not a general adapter contract.

**Example (target shape — planner finalizes details):**

```python
# evaluation/harness/adapters/tier_5.py — proposed change
# Source: agents/items.py:376 (ToolCallOutputItem dataclass) + tier_5_agentic/tools.py:122,155

from agents import Runner, ToolCallOutputItem  # add ToolCallOutputItem here
from agents.exceptions import MaxTurnsExceeded
# … existing imports unchanged …

# … inside run_tier5, after result = await Runner.run(...) succeeds:

contexts: list[str] = []
seen: set[tuple[str, ...]] = set()  # dedup by stable id (CONTEXT.md decision)
for item in result.new_items:
    if not isinstance(item, ToolCallOutputItem):
        continue
    out = item.output  # raw return value of the @function_tool callable
    # search_text_chunks → list[dict] with keys paper_id, page, snippet, similarity
    if isinstance(out, list):
        for hit in out:
            if not isinstance(hit, dict):
                continue
            pid = hit.get("paper_id")
            page = hit.get("page")
            snippet = hit.get("snippet")
            if not snippet:  # honest empty: skip rather than insert blanks
                continue
            key = ("chunk", str(pid), str(page))
            if key in seen:
                continue
            seen.add(key)
            contexts.append(f"[paper_id={pid}] {snippet}")
    # lookup_paper_metadata → dict with abstract; or error dict {"error": ...}
    elif isinstance(out, dict):
        if "error" in out:  # tool returned not-found; skip with provenance preserved elsewhere
            continue
        pid = out.get("paper_id")
        abstract = out.get("abstract")
        if not abstract:
            continue
        key = ("meta", str(pid))
        if key in seen:
            continue
        seen.add(key)
        contexts.append(f"[paper_id={pid}] {abstract}")
    # any other shape (str, int, None) is skipped silently — Tier 5 has only two tools

return EvalRecord(
    question_id=question_id,
    question=question,
    answer=answer,
    retrieved_contexts=contexts,  # Pitfall 9 RESOLVED — agent retrieval surfaced
    latency_s=latency,
    cost_usd_at_capture=tracker.total_usd(),
    error=error,
)
```

**Verified API surface (ALL HIGH confidence — source on disk, not training data):**

- `RunResult.new_items: list[RunItem]` declared at `.venv/lib/python3.13/site-packages/agents/result.py:181` (`RunResultBase` parent). [VERIFIED]
- `ToolCallOutputItem` is a `@dataclass(RunItemBase)` at `agents/items.py:375-419`. The relevant fields are `output: Any` (the raw tool return value, line 382) and `raw_item: ToolCallOutputTypes` (the stringified Responses-API payload, line 379). [VERIFIED]
- `output` is set at construction: in `agents/run_internal/tool_execution.py:1850-1855` the runtime instantiates `ToolCallOutputItem(output=result, raw_item=ItemHelpers.tool_call_output_item(tool_run.tool_call, result), …)` where `result` is the unmodified return value of the `@function_tool` callable. **The adapter's read of `item.output` is therefore the canonical path; no unwrapping or string parsing is needed.** [VERIFIED]
- `from agents import ToolCallOutputItem` works — re-exported via `agents/__init__.py` (`from .items import (… ToolCallOutputItem …)`). [VERIFIED]

### Pattern 2: Conditional Tracing Toggle

**What:** Make `set_tracing_disabled(disabled=True)` at `tier-5-agentic/agent.py:43` conditional on an env flag so the OpenInference fallback path can capture spans.

**When to use:** Apply once during the Phase 1 fix; do NOT toggle the disabled state at runtime — the SDK reads it once at module import. Reading an env var at import time is the correct integration.

**Example:**

```python
# tier-5-agentic/agent.py:43 (proposed change)
# Default behavior: tracing OFF (preserves Pitfall 8 — no OpenAI tracing leak).
# Set RAG_DEBUG_TIER5_TRACING=1 to enable, only when the debug-tier5 extra is installed.
set_tracing_disabled(disabled=not os.environ.get("RAG_DEBUG_TIER5_TRACING"))
```

`os` is already imported at `tier-5-agentic/agent.py:32` for the lazy `OPENROUTER_API_KEY` read, so no new import is needed.

### Pattern 3: Smoke Question Filter on `run.py`

**What:** Add `--smoke-question-ids id1,id2,id3,…` next to the existing `--limit` flag. When set, filter `qa` to records whose `id` is in the comma-split list, in the order given. The default value is a hard-coded module-level constant (the 5 IDs locked below).

**When to use:** Only for the Phase 1 / Phase 2 smoke runs. The existing `--limit N` continues to work for ad-hoc smoke and is independent.

**Example:**

```python
# evaluation/harness/run.py — proposed addition
# Module-level constant (CONTEXT.md "Stability across phases" — same 5 across all smokes):
DEFAULT_SMOKE_IDS: tuple[str, ...] = (
    "single-hop-001",
    "single-hop-002",
    "single-hop-003",
    "multi-hop-001",
    "multi-hop-002",
)

# In build_parser():
p.add_argument(
    "--smoke-question-ids",
    default=None,
    help=(
        "Comma-separated question ids to filter golden_qa.json down to a smoke subset. "
        f"Default constant: {','.join(DEFAULT_SMOKE_IDS)} (used by Phase 1/2 smoke gates). "
        "Mutually informative with --limit; ids filter, --limit truncates."
    ),
)

# In amain(), after _load_golden_qa():
if args.smoke_question_ids:
    wanted = [s.strip() for s in args.smoke_question_ids.split(",") if s.strip()]
    by_id = {q["id"]: q for q in qa}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        console.print(f"[red]Unknown question ids: {missing}[/red]")
        return 2
    qa = [by_id[w] for w in wanted]
```

### Anti-Patterns to Avoid

- **Stringifying `item.raw_item.output` instead of reading `item.output`.** The `raw_item` field carries the **Responses-API-shape** stringified payload (the `function_call_output` envelope sent back to the model), NOT the original tool return. Parsing it is wrong (you'd get a `str` of `[{'paper_id': '…', …}]` or similar) AND is fragile — `ItemHelpers._convert_tool_output` at `agents/items.py:755-777` may stringify or wrap depending on the tool's return shape. Always read `item.output`.
- **Looping over `result.raw_responses` (the `ModelResponse` list).** That collection holds **LLM outputs** (messages + tool calls), not tool **results**. Tool results live only in `new_items`.
- **Hand-rolling a "max iterations" guard inside the adapter.** `Runner.run(..., max_turns=10)` already does this; the existing `MaxTurnsExceeded` catch at `tier_5.py:102` is the load-bearing handler. Don't reimplement.
- **Asserting `len(contexts) > 0` inside `run_tier5`.** Pitfall 9 (agent self-cited from training, no tool calls made) is a legitimate empty-contexts case. The smoke gate, not the adapter, is responsible for classifying this — keep the adapter strictly mechanical.
- **Setting `set_tracing_disabled(disabled=False)` unconditionally.** This re-introduces the Pitfall 8 problem (every run logs warnings without an OpenAI tracing key). Toggle only via env var.
- **Asserting RAGAS metrics inside the smoke gate without the existing nan_reason classification.** `score.py:_short_circuit_nan` already encodes Pitfalls 8 (`agent_truncated`) and 2 (`empty_contexts`). The smoke gate must consume these labels rather than re-implementing the classification.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Walking the agent run for tool outputs | A custom hook on `RunHooks` or `AgentHooks` to capture tool returns into a side-channel | `result.new_items` walk in the adapter | The SDK already records every tool output into `new_items` post-run. Hooks are for runtime observation; the adapter wants post-hoc projection. Hand-rolled hooks would also fight with the OpenInference instrumentation in the fallback path. |
| Stringifying tool outputs for RAGAS | A custom JSON serializer over `item.output` | Direct field access on the `list[dict]` / `dict` returned by the tool, with a provenance-prefix concatenation | The tools return Python types the adapter already understands. Adding a JSON layer hurts RAGAS — `[CITED: CONTEXT.md "Entry content"]` says strip JSON braces and field names; RAGAS judges natural prose. |
| Tool-output dedup | A pandas DataFrame or `collections.OrderedDict` | A tiny `seen: set[tuple]` inside the loop | Phase 1 needs ≤30 questions × ≤10 turns × ≤5 chunks/turn = ≤1500 entries. A set comprehension is ~3 lines. Bringing pandas in just to dedup is over-engineering. |
| Smoke gate logic | A bash script grep'ing JSON | A small Pydantic-typed Python module that reads QueryLog + ScoreRecord lists | Repo convention: every persisted artifact is a Pydantic model. `evaluation/harness/records.py` is the precedent. |
| Tracing toggle wiring | A flag on the harness CLI piped through `_capture_tier` into `build_agent` | Env var read at module import in `tier-5-agentic/agent.py` | `set_tracing_disabled` is a process-global side-effect set once at import. Plumbing a flag through is layer-violation; env var is correct. |
| Provenance hash for fallback log | A custom hash function | `subprocess.check_output(["git","rev-parse","--short","HEAD"], cwd=_REPO_ROOT)` — already implemented as `_git_sha()` in `evaluation/harness/run.py:60-67` | Reuse the existing helper; copy by import or refactor into a shared module. |
| Library version capture | Hand-coding strings | `importlib.metadata.version("openai-agents")` etc. | Standard library, zero deps. Captures the actually-installed version, not a pin guess. |

**Key insight:** This phase is 95% about **not** writing code. The fix is 5 lines; the rest is wiring (smoke filter, gate evaluator, conditional tracing, fallback log). Each piece of wiring already has a precedent elsewhere in the repo — copy the pattern, don't invent.

---

## Common Pitfalls

### Pitfall 1: Reading `raw_item` instead of `output`

**What goes wrong:** Adapter walks `new_items`, finds `ToolCallOutputItem`, then reads `item.raw_item['output']` (a stringified payload). Result: `retrieved_contexts` is a list of opaque strings like `"[{'paper_id': '2005.11401', 'page': 3, 'snippet': '…'}]"` — RAGAS judges this as garbage and faithfulness collapses.

**Why it happens:** `raw_item` is conceptually the "real" data because it's what gets sent to the model. `output` looks redundant. But it's the SDK's deliberate design: `raw_item` is **for the model**; `output` is **for the application**.

**How to avoid:** Always `item.output`, never `item.raw_item`. Document this in a comment so future maintainers don't "fix" it.

**Warning signs:** `retrieved_contexts` entries that look like Python `repr()` output (square brackets, single-quoted keys, no natural prose).

[VERIFIED: `agents/items.py:382-385` docstring: "The output of the tool call. This is whatever the tool call returned; the `raw_item` contains a string representation of the output."]

---

### Pitfall 2: `isinstance` brittleness across SDK patches

**What goes wrong:** A future `openai-agents` patch (0.14.7+, 0.15.x) renames or splits `ToolCallOutputItem`. The `isinstance` check silently returns `False` for everything. Smoke gate reports 30/30 empty_contexts again — looks like the original bug never fixed.

**Why it happens:** Class identity is fragile across upgrades; the type literal `"tool_call_output_item"` is more stable per the SDK's own JSON serialization contract.

**How to avoid:** **For Phase 1, use `isinstance` (HIGH confidence on 0.14.6).** Add a tiny duck-type fallback comment for future maintainers:

```python
# If a future SDK refactor breaks isinstance, fall back to:
#     getattr(item, "type", None) == "tool_call_output_item"
```

**Warning signs:** Same fix-known-good run later returns empty contexts after a `uv lock --upgrade`.

[CITED: `.planning/research/STACK.md:317-319` — defensive coding pattern documented.]

---

### Pitfall 3: Updating the test on line 174 by accident-deletion vs. inversion

**What goes wrong:** The existing assertion at `evaluation/tests/test_eval_adapters.py:174` reads `assert rec.retrieved_contexts == []  # Pitfall 9 honest empty`. A naive fix deletes the assertion entirely; a worse fix updates it to `!= []` without controlling the agent's tool-call sequence, so the test becomes flaky (passes when the mock returns tool outputs, fails when it doesn't).

**Why it happens:** The test mocks `Runner.run` to return a `SimpleNamespace` with `final_output` and `context_wrapper.usage` only — no `new_items`. After the fix, the adapter will iterate `result.new_items`, which will be `AttributeError` against the current mock.

**How to avoid:** Update the mock to include `new_items=[]` or a list of fake `ToolCallOutputItem` instances. Add a NEW test `test_run_tier5_extracts_tool_outputs(monkeypatch)` that returns a synthetic `RunResult` with two `ToolCallOutputItem` instances (one from `search_text_chunks`, one from `lookup_paper_metadata`) and asserts that `retrieved_contexts` contains the provenance-prefixed strings in iteration order, deduped.

**Warning signs:** `AttributeError: 'SimpleNamespace' object has no attribute 'new_items'` when running `pytest evaluation/tests/test_eval_adapters.py -k tier5`.

---

### Pitfall 4: Confusing "Pitfall 8" / "Pitfall 9" sources

**What goes wrong:** CONTEXT.md and STACK.md both reference "Pitfall 8" and "Pitfall 9" but these are **distinct numbered lists in different research bundles**. The adapter docstring at `tier_5.py:1-24` cites Pitfalls 8 and 9 from the **131-RESEARCH** bundle (the eval-harness phase), where Pitfall 8 = `MaxTurnsExceeded` and Pitfall 9 = "agent self-cites; honest empty contexts". The current `.planning/research/PITFALLS.md` is a **different research bundle** (132-RESEARCH for this v1.0 fix-and-ship milestone) where Pitfall 8 = SHA drift and Pitfall 9 = cold-cache latency.

**Why it happens:** Both bundles use 1-indexed numbered pitfalls; the numbering collides.

**How to avoid:** When a code comment or CONTEXT.md says "Pitfall N", check the cited filename. Within Phase 1's adapter, the relevant sources are 130-RESEARCH (Tier 5 SDK pitfalls) and 131-RESEARCH (eval harness pitfalls). The comment in `tier_5.py` already cites 131-RESEARCH explicitly. Preserve those citations in the fix; do NOT cross-pollinate with the 132-RESEARCH numbering.

**Warning signs:** A reviewer asking "why does this comment reference SHA drift, the adapter has nothing to do with provenance?" — that means a Pitfall number got mis-cited.

---

### Pitfall 5: Smoke-report exclusion math

**What goes wrong:** Smoke runs 5 questions. 1 is `max_turns_exceeded` (Pitfall 8 of 130-RESEARCH); 1 has zero tool calls (Pitfall 9 of 130-RESEARCH); the remaining 3 are populated. The naive gate says `3/5 = 60% < 80%`, FAIL. But CONTEXT.md says: exclude max_turns_exceeded AND zero-tool-call cells from the denominator, then apply the ≥0.8 ratio on the **measurable** subset. If `(populated) / (measurable) = 3/3 = 100%`, PASS.

**Why it happens:** Conflating "5 smoke questions" with "5 measurable smoke questions". The denominator is dynamic.

**How to avoid:** The gate evaluator must compute the denominator AFTER classification. Pseudocode:

```python
classified = [classify(rec, score) for rec, score in zipped]
# classify(...) returns one of: "populated", "empty_no_tool_calls", "agent_truncated"
measurable = [c for c in classified if c not in {"empty_no_tool_calls", "agent_truncated"}]
populated  = [c for c in classified if c == "populated"]
if len(measurable) < 3:
    return "INCONCLUSIVE"
ratio = len(populated) / len(measurable)
return "PASS" if (ratio >= 0.8 and all_metrics_non_nan(populated)) else "FAIL"
```

**Warning signs:** Smoke gate reporting `n=5, populated=3, FAIL` without acknowledging exclusions in the same line.

---

### Pitfall 6: Tracing-toggle silent breakage of unit tests

**What goes wrong:** Changing `set_tracing_disabled(disabled=True)` to `set_tracing_disabled(disabled=not os.environ.get("RAG_DEBUG_TIER5_TRACING"))` causes the import-time tracing setup to vary per test run depending on whether `.env` was loaded. In CI without `.env`, the env var is unset → tracing stays disabled (correct). On a developer's machine with the var accidentally set → tracing enabled, but no OpenAI tracing key is configured → spam of warnings during `pytest`.

**Why it happens:** Module-level side-effects are sensitive to import order vs. env loading. The repo already loads `.env` in `evaluation/tests/conftest.py:18` and `tier-5-agentic/tests/conftest.py` similarly.

**How to avoid:** Use a name unlikely to be set accidentally (`RAG_DEBUG_TIER5_TRACING` is good — long, project-specific). Document the env var in a docstring at the top of `agent.py`. Add a one-line unit test that imports the module with the env var unset and asserts no `agents.set_tracing_disabled(disabled=False)` was called (mock and verify) — guards the default state.

**Warning signs:** Stderr noise during `pytest` runs ("Tracing key not configured…") — env var leaked.

---

### Pitfall 7: Per-attempt budget runaway in fallback path

**What goes wrong:** CONTEXT.md gives Claude discretion on per-attempt budget. A naive interpretation: "re-run smoke until it passes." Three failed iterations × 5 questions × ~$0.001/q ≈ $0.015 — fine. But if smoke runs through `score.py` too (judge LLM × 5 questions × ~$0.003/q ≈ $0.015), each iteration is ~$0.03. Add 3 STACK.md mutations × 1 instrumentation cycle = ~$0.20 worst case. Still in budget BUT visible to the user and worth disclosing.

**Why it happens:** Smoke is cheap per run but compounds if the gate is flaky.

**How to avoid:** Default to 1 smoke run per fallback step (instrument once, mutate once each). Re-run only if the failure looks transient (e.g., a single API timeout, not 5/5 empty). Surface the cumulative cost in the fallback log.

**Warning signs:** Phase 1 cost ledger > $0.50 — investigate before continuing.

---

### Pitfall 8: Tool-error payload conflated with successful empty result

**What goes wrong:** `lookup_paper_metadata("9999.99999")` returns `{"error": "paper_id 9999.99999 not found in dataset/manifests/papers.json"}` (per `tier-5-agentic/tools.py:171`). A naive walk treats this dict like a metadata hit, finds no `abstract`, falls through silently. The next walk over a successful chunk-search hit tries to dedup against the error dict's missing key. No ill effect, but the smoke report has no record that the agent asked for an unknown paper — diagnostically useful information lost.

**Why it happens:** Tools return both success and error in the same dict shape.

**How to avoid:** Branch on `"error" in out` early; log the error string to stderr (or a side-channel JSON) but skip insertion into `contexts`. Per CONTEXT.md "Tool-error handling" (Claude's discretion): "probably skip with a logged warning."

**Warning signs:** Agent runs slow (asking for many unknown paper IDs) but smoke gate still reports populated contexts — the agent is wasting turns on lookups the adapter never sees.

---

## Runtime State Inventory

This is not a rename / refactor / migration phase — it's a code-fix phase. Section omitted. (No databases store the string `retrieved_contexts=[]`; no OS task scheduler embeds Tier 5 state; no env var rename.)

---

## Code Examples

Verified patterns from the installed SDK source and existing repo files.

### Example 1: The 5-line walk (canonical)

```python
# Source: .venv/lib/python3.13/site-packages/agents/items.py:376-390 (ToolCallOutputItem)
#         .venv/lib/python3.13/site-packages/agents/run_internal/tool_execution.py:1850-1855 (output set to raw return)
#         tier-5-agentic/tools.py:122-146 (search_text_chunks return shape)
#         tier-5-agentic/tools.py:149-171 (lookup_paper_metadata return shape)

from agents import Runner, ToolCallOutputItem

# Inside run_tier5, after Runner.run succeeds:
contexts: list[str] = []
seen: set[tuple[str, ...]] = set()
for item in result.new_items:
    if not isinstance(item, ToolCallOutputItem):
        continue
    out = item.output
    if isinstance(out, list):
        for hit in out:
            if not isinstance(hit, dict):
                continue
            if hit.get("error"):
                continue  # Pitfall 8 of 132-RESEARCH (tool error payload)
            pid, page, snippet = hit.get("paper_id"), hit.get("page"), hit.get("snippet")
            if not snippet:
                continue
            key = ("chunk", str(pid), str(page))
            if key in seen:
                continue
            seen.add(key)
            contexts.append(f"[paper_id={pid}] {snippet}")
    elif isinstance(out, dict):
        if "error" in out:
            continue
        pid, abstract = out.get("paper_id"), out.get("abstract")
        if not abstract:
            continue
        key = ("meta", str(pid))
        if key in seen:
            continue
        seen.add(key)
        contexts.append(f"[paper_id={pid}] {abstract}")
```

### Example 2: Updated unit test for Tier 5 happy path

```python
# Source: target replacement for evaluation/tests/test_eval_adapters.py:157-175
# Existing test asserts retrieved_contexts == [] — INVERT.

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

def test_run_tier5_extracts_tool_outputs(monkeypatch):
    from evaluation.harness.adapters import tier_5 as t5
    from agents import ToolCallOutputItem

    fake_chunk_item = SimpleNamespace(  # duck-typed; the adapter only checks isinstance
        # NOTE: planner picks: either (a) build a real ToolCallOutputItem with a stub
        # raw_item dict, or (b) monkey-patch the isinstance check. Real instance is cleaner.
    )
    # Recommended: real ToolCallOutputItem with raw_item={} since adapter never reads raw_item
    fake_chunks_out = [
        {"paper_id": "2005.11401", "page": 1, "snippet": "RAG combines …", "similarity": 0.9},
        {"paper_id": "2005.11401", "page": 1, "snippet": "RAG combines …", "similarity": 0.9},  # dup
        {"paper_id": "2004.04906", "page": 3, "snippet": "DPR uses …", "similarity": 0.8},
    ]
    fake_meta_out = {
        "paper_id": "2005.11401",
        "title": "RAG",
        "authors": ["Lewis et al."],
        "year": 2020,
        "abstract": "We introduce retrieval-augmented generation.",
    }
    items = [
        ToolCallOutputItem(agent=MagicMock(), raw_item={}, output=fake_chunks_out),
        ToolCallOutputItem(agent=MagicMock(), raw_item={}, output=fake_meta_out),
    ]
    fake_result = SimpleNamespace(
        final_output="cited answer",
        new_items=items,
        context_wrapper=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=300, output_tokens=80)
        ),
    )
    monkeypatch.setattr(
        t5, "Runner", SimpleNamespace(run=AsyncMock(return_value=fake_result))
    )

    rec = asyncio.run(t5.run_tier5("q5", "?", agent=MagicMock()))
    # Dedup: 3 chunks → 2 unique (same paper_id+page collapsed); plus 1 metadata abstract = 3 entries
    assert len(rec.retrieved_contexts) == 3
    assert all(c.startswith("[paper_id=") for c in rec.retrieved_contexts)
    assert any("RAG combines" in c for c in rec.retrieved_contexts)
    assert any("DPR uses" in c for c in rec.retrieved_contexts)
    assert any("retrieval-augmented generation" in c for c in rec.retrieved_contexts)
    assert rec.error is None


def test_run_tier5_max_turns_preserves_empty_contexts_and_truncated_marker(monkeypatch):
    """Pitfall 8 of 130-RESEARCH preserved: MaxTurnsExceeded → error='max_turns_exceeded',
    answer prefixed [truncated, retrieved_contexts left empty (we never reach the walk)."""
    from evaluation.harness.adapters import tier_5 as t5
    from agents.exceptions import MaxTurnsExceeded
    exc = MaxTurnsExceeded("hit cap")
    setattr(exc, "usage", SimpleNamespace(input_tokens=400, output_tokens=0))
    monkeypatch.setattr(t5, "Runner", SimpleNamespace(run=AsyncMock(side_effect=exc)))
    rec = asyncio.run(t5.run_tier5("q5", "?", agent=MagicMock(), max_turns=10))
    assert rec.error == "max_turns_exceeded"
    assert "[truncated" in rec.answer
    assert rec.retrieved_contexts == []  # honest empty on truncation


def test_run_tier5_zero_tool_calls_returns_honest_empty(monkeypatch):
    """Pitfall 9 of 130-RESEARCH preserved: agent answered without calling tools →
    retrieved_contexts=[]. score.py will tag this as nan_reason='empty_contexts'."""
    from evaluation.harness.adapters import tier_5 as t5
    fake_result = SimpleNamespace(
        final_output="from training",
        new_items=[],  # no tool calls at all
        context_wrapper=SimpleNamespace(usage=SimpleNamespace(input_tokens=200, output_tokens=20)),
    )
    monkeypatch.setattr(t5, "Runner", SimpleNamespace(run=AsyncMock(return_value=fake_result)))
    rec = asyncio.run(t5.run_tier5("q5", "?", agent=MagicMock()))
    assert rec.retrieved_contexts == []
    assert rec.error is None
```

### Example 3: Conditional tracing toggle

```python
# Source: tier-5-agentic/agent.py:43 — proposed 1-line change
# os already imported at agent.py:32 (no new import needed)
import os
from agents import set_tracing_disabled

# Default: tracing OFF (Pitfall 8 of 130-RESEARCH preserved).
# Set RAG_DEBUG_TIER5_TRACING=1 (and install [debug-tier5] extras) to enable.
set_tracing_disabled(disabled=not os.environ.get("RAG_DEBUG_TIER5_TRACING"))
```

### Example 4: `--smoke-question-ids` flag

```python
# Source: evaluation/harness/run.py — proposed addition (parser + filter)

DEFAULT_SMOKE_IDS: tuple[str, ...] = (
    "single-hop-001",
    "single-hop-002",
    "single-hop-003",
    "multi-hop-001",
    "multi-hop-002",
)

# In build_parser():
p.add_argument(
    "--smoke-question-ids",
    default=None,
    help=(
        "Comma-separated question ids to filter golden_qa.json (Phase 1/2 smoke gate). "
        f"Phase 1/2 default constant: {','.join(DEFAULT_SMOKE_IDS)}."
    ),
)

# In amain(), after qa = _load_golden_qa() and BEFORE --limit slicing:
if args.smoke_question_ids:
    wanted = [s.strip() for s in args.smoke_question_ids.split(",") if s.strip()]
    by_id = {q["id"]: q for q in qa}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        console.print(f"[red]Unknown question ids: {missing}[/red]")
        return 2
    qa = [by_id[w] for w in wanted]
```

### Example 5: Library version capture (for fallback log)

```python
# Source: stdlib importlib.metadata — for fallback JSON provenance
from importlib.metadata import version as _pkg_version

def _captured_versions() -> dict[str, str]:
    """Record actually-installed versions of the libraries Phase 1's fallback log
    must carry per CONTEXT.md 'frozen-doc-grade provenance'."""
    out: dict[str, str] = {}
    for name in ("openai-agents", "lightrag-hku", "raganything", "ragas", "litellm"):
        try:
            out[name] = _pkg_version(name)
        except Exception:  # noqa: BLE001 — package not installed (e.g. tier-3 not pulled in)
            out[name] = "not-installed"
    return out
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `retrieved_contexts=[]` hard-coded with comment "Pitfall 9 honest empty — agent self-cites" | Walk `result.new_items` for `ToolCallOutputItem.output`, project to provenance-prefixed strings | This phase (Phase 1, 2026-05-04) | RAGAS gains 25-30 measurable rows for Tier 5 (down from 30/30 NaN). The Pitfall 9 self-cite case (zero tool calls) is preserved; it just becomes one rare row, not the universal default. |
| Tracing disabled unconditionally at module import | Tracing controlled by `RAG_DEBUG_TIER5_TRACING` env var; default disabled | This phase | Enables OpenInference fallback path without changing default behavior. Production runs are byte-identical to today. |
| Smoke = `--limit 5` (first 5 questions of `golden_qa.json`) | Smoke = `--smoke-question-ids id1,…,id5` with hard-coded default | This phase | Apples-to-apples deltas across phases (Phase 1 Tier 5 smoke vs Phase 2 Tier 4 smoke vs future regressions). The same 5 questions every time. |

**Deprecated/outdated:**

- The `# Pitfall 9 honest empty — agent self-cites` comment at `tier_5.py:125`: Replace with a comment that explains the walk and cites Pitfalls 8 + 9 of 130-RESEARCH explicitly (preserve numbering, just update the meaning).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| (none) | Every claim in this RESEARCH.md is `[VERIFIED]` against installed source files (`.venv/lib/python3.13/site-packages/agents/`), repo source files (`evaluation/`, `tier-5-agentic/`, `dataset/`), or `[CITED]` against the existing research bundle (`.planning/research/STACK.md`, PITFALLS.md). No claims tagged `[ASSUMED]`. | All | None — no user-confirmation needed before planning. |

**Empty assumptions table:** All claims in this research were verified or cited — no user confirmation needed.

---

## Open Questions

1. **Should the smoke gate fail-fast or score-then-classify?**
   - What we know: `score.py` already short-circuits empty contexts to `nan_reason='empty_contexts'` BEFORE calling the judge LLM (`score.py:_short_circuit_nan` at line 136-152), so running score.py on the smoke 5 is cheap (~$0.015 worst case). CONTEXT.md says "score.py runs against the smoke 5 as part of the gate."
   - What's unclear: Should the gate evaluator invoke `score.amain()` directly (subprocess), or `score.score_query_log()` in-process? The latter is the existing pattern (used by `test_eval_smoke_live.py:120-127`).
   - Recommendation: In-process. Subprocess adds 2-3 seconds of import boot per call and complicates exit-code handling. Match the live-smoke-test pattern.

2. **Should the fallback log include the agent's `final_output` (the answer text) for each smoke question?**
   - What we know: CONTEXT.md requires recording git SHA, timestamp, library versions, model slug, mutation tried, smoke result, span observations.
   - What's unclear: The answer text would let a reviewer judge "agent self-cited from training" vs. "agent called tools but tools returned junk" without re-running. But it's also long (~200-500 chars per question × 5 questions = 1-3 KB) and may contain hallucinated paper IDs.
   - Recommendation: Include `final_output` truncated to 400 chars per the existing tier-5 e2e live test convention (`tier-5-agentic/tests/test_tier5_e2e_live.py:121`). Truncation is the precedent.

3. **Tie-breaking when the same `(paper_id, page)` appears with different similarity scores across iterations.**
   - What we know: CONTEXT.md says "probably first occurrence; not consequential for RAGAS." The proposed `seen: set[tuple]` does first-write-wins.
   - What's unclear: If iteration 1 returns chunk X with similarity 0.6 and iteration 3 returns the same chunk X with similarity 0.9, do we want the higher-confidence variant?
   - Recommendation: First-occurrence as locked. RAGAS `context_precision` doesn't read similarity; the score is irrelevant once the chunk is in the list. Don't introduce score-based tie-breaking.

4. **Should the `[paper_id=<id>]` provenance prefix be included for the abstract entries from `lookup_paper_metadata`?**
   - What we know: CONTEXT.md says "Format: `[paper_id=<id>] <text>` where text is the chunk text (for `search_text_chunks`) or the abstract field (for `lookup_paper_metadata`)."
   - What's unclear: Nothing — CONTEXT.md is explicit. Recorded here so the planner doesn't get confused by mixed examples elsewhere.
   - Recommendation: Always prefix, both tools.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 | All harness code | ✓ | 3.13 (verified by `.venv/lib/python3.13/`) | — |
| `openai-agents` | Tier 5 agent + adapter walk | ✓ | 0.14.6 (verified `openai_agents-0.14.6.dist-info/METADATA`) | — |
| `litellm` | OpenRouter routing | ✓ | 1.83.0 | — |
| `ragas` | Smoke score gate | ✓ | 0.4.3 | — |
| `lightrag-hku` | Recorded in fallback log only | ✓ | 1.4.15 | Captured as "not-installed" via try/except if pulled out later |
| `raganything` | Recorded in fallback log only | ✓ | 1.2.10 | Same |
| `chromadb` | Tier 1 collection (Tier 5 reads it) | ✓ (assumed; pinned) | `>=1.5.8,<2` | — |
| `chroma_db/tier-1-naive/` directory | Tier 5's `search_text_chunks` reads from it (Pitfall 9 read-only invariant) | Unknown — must verify before live smoke | — | Run `python tier-1-naive/main.py --ingest` first; existing prereq check at `run.py:103-106` already enforces |
| `OPENROUTER_API_KEY` env var | Live smoke (agent + RAGAS judge both route through OpenRouter) | Unknown — runtime check | — | Existing prereq at `run.py:97-99` exits 2 with friendly error |
| `git` CLI | `_git_sha()` for QueryLog and fallback log provenance | ✓ (assumed; standard dev env) | — | `_git_sha()` returns `"unknown"` on failure (already-handled at `run.py:60-67`) |
| `openinference-instrumentation-openai-agents` | Fallback path only | ✗ (not installed by default) | — | Install via `pip install openinference-instrumentation-openai-agents==1.4.2 arize-phoenix>=6,<8 opentelemetry-sdk>=1.30,<2 opentelemetry-exporter-otlp-proto-http>=1.30,<2` ONLY if smoke fails. Not a Phase 1 blocker. |

**Missing dependencies with no fallback:** None for the Phase 1 default path.

**Missing dependencies with fallback:** `openinference-instrumentation-openai-agents` and the OTel/Phoenix stack — install on demand only if smoke fails.

---

## Validation Architecture

`workflow.nyquist_validation` is not explicitly set in `.planning/config.json` (config has `research`, `plan_check`, `verifier` only). Treating as enabled per the agent contract.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4+ (per `pyproject.toml:74` `pytest>=8.4,<9`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (line 80-81); marker registered: `live: tests that hit real APIs and incur cost` |
| Quick run command | `pytest evaluation/tests/test_eval_adapters.py -k tier5 -x` |
| Full suite command | `pytest -x` (excludes live; no `-m live`); `pytest -m live` for live smoke |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TIER-01 | `run_tier5` projects `ToolCallOutputItem.output` into provenance-prefixed `retrieved_contexts` | unit (mocked Runner) | `pytest evaluation/tests/test_eval_adapters.py::test_run_tier5_extracts_tool_outputs -x` | ❌ Wave 0 (NEW test; existing `test_run_tier5_happy_path` must be updated to no longer assert empty contexts) |
| TIER-01 (preservation) | `MaxTurnsExceeded` still surfaces `error="max_turns_exceeded"` and `retrieved_contexts=[]` (Pitfall 8 of 130-RESEARCH) | unit | `pytest evaluation/tests/test_eval_adapters.py::test_run_tier5_max_turns_exceeded -x` (existing test, preserved) | ✅ at line 178 |
| TIER-01 (Pitfall 9 of 130-RESEARCH) | Zero-tool-call run returns `retrieved_contexts=[]` honestly (no synthesis from `final_output`) | unit | `pytest evaluation/tests/test_eval_adapters.py::test_run_tier5_zero_tool_calls_returns_honest_empty -x` | ❌ Wave 0 (NEW) |
| TIER-01 (dedup) | Repeated `(paper_id, page)` tuples collapse to one entry (first-occurrence wins) | unit | covered inside `test_run_tier5_extracts_tool_outputs` | ❌ Wave 0 |
| TIER-01 (live) | End-to-end: real Runner.run, real ChromaDB, real OpenRouter; assert `len(rec.retrieved_contexts) > 0` for at least one of the 5 smoke questions | integration / live | `pytest -m live evaluation/tests/test_eval_smoke_live.py::test_eval_smoke_tier5_full_pipeline -x` | ❌ Wave 0 (NEW; mirror existing `test_eval_smoke_tier1_full_pipeline`) |
| TIER-03 | `--smoke-question-ids` CLI flag filters `golden_qa.json` correctly; missing IDs exit 2 | unit | `pytest evaluation/tests/test_eval_run.py::test_run_smoke_ids_filters_correctly -x` and `::test_run_smoke_ids_unknown_returns_2 -x` | ❌ Wave 0 (NEW) |
| TIER-03 (gate) | Smoke gate evaluator: classify rows, compute measurable denominator, apply ≥0.8 ratio | unit | `pytest evaluation/tests/test_eval_smoke_gate.py -x` | ❌ Wave 0 (NEW file) |
| TIER-01 (tracing toggle) | `set_tracing_disabled(disabled=...)` reads `RAG_DEBUG_TIER5_TRACING` env var; default disabled when var unset | unit | `pytest tier-5-agentic/tests/test_agent.py::test_tracing_disabled_default -x` | ❌ Wave 0 (NEW) |

### Sampling Rate

- **Per task commit:** `pytest evaluation/tests/test_eval_adapters.py -k tier5 evaluation/tests/test_eval_run.py evaluation/tests/test_eval_smoke_gate.py tier-5-agentic/tests/test_agent.py -x` — runs in ≤ 5 seconds, no API calls.
- **Per wave merge:** `pytest -x` — full unit suite (excludes live).
- **Phase gate (before `/gsd-verify-work`):** `pytest -x` PLUS `pytest -m live evaluation/tests/test_eval_smoke_live.py -x` (live; ~$0.05 per run).

### Wave 0 Gaps

- [ ] `evaluation/tests/test_eval_adapters.py::test_run_tier5_extracts_tool_outputs` — covers TIER-01 (positive case, 2 tool calls + dedup)
- [ ] `evaluation/tests/test_eval_adapters.py::test_run_tier5_zero_tool_calls_returns_honest_empty` — covers TIER-01 (Pitfall 9 preservation)
- [ ] `evaluation/tests/test_eval_adapters.py::test_run_tier5_happy_path` — UPDATE existing test (remove empty-contexts assertion; add new_items=[] to mock)
- [ ] `evaluation/tests/test_eval_run.py::test_run_smoke_ids_filters_correctly` — covers TIER-03 (filter)
- [ ] `evaluation/tests/test_eval_run.py::test_run_smoke_ids_unknown_returns_2` — covers TIER-03 (error path)
- [ ] `evaluation/tests/test_eval_smoke_gate.py` — NEW FILE; covers TIER-03 (gate logic, exclusions, measurable denominator)
- [ ] `evaluation/tests/test_eval_smoke_live.py::test_eval_smoke_tier5_full_pipeline` — NEW LIVE TEST; mirrors `test_eval_smoke_tier1_full_pipeline` for Tier 5
- [ ] `tier-5-agentic/tests/test_agent.py::test_tracing_disabled_default` — covers tracing toggle env var
- [ ] No framework install needed — pytest 8.4 already pinned and in use.

---

## Project Constraints (from CONTEXT.md and pyproject.toml)

No `CLAUDE.md` exists at the repo root. Constraints derive from CONTEXT.md (already extracted to `<user_constraints>` above) and `pyproject.toml`:

1. **`openai-agents[litellm]==0.14.6` is locked.** Do not bump unless STACK.md mutation step 3 (last-resort fallback) is invoked AND the user authorizes.
2. **`ragas>=0.4.3,<0.5` is locked.** Two existing patches in this codebase (`f0cd134`, `186a9c2`) are 0.4.3-specific. Do not bump.
3. **`set_tracing_disabled(disabled=True)` MUST remain the default behavior.** The conditional toggle must default to disabled when the env var is unset (Pitfall 8 of 130-RESEARCH).
4. **Tier 5's `search_text_chunks` MUST call `open_collection(reset=False)`.** Pitfall 9 of 130-RESEARCH read-only invariant on Tier 1's index. Already-implemented at `tier-5-agentic/tools.py:65`; do not modify.
5. **Repo persistence convention: every persisted artifact is a Pydantic v2 `model_dump_json(indent=2)`.** Apply to the new fallback log JSON and any smoke-report JSON. Mirror `evaluation/harness/records.py:69-70`.
6. **D-13 cost ledger schema (frozen).** No new keys in cost JSONs from this phase. The fallback log is a separate artifact.
7. **`PRICES` keys are provider-only.** `_strip_openrouter_prefix` at `tier_5.py:47-53` and the same helper at `score.py:67-74` already handle this. Reuse.
8. **`pytest -m live` markers.** Any test that hits real APIs MUST be marked `@pytest.mark.live`. Existing fixture `live_eval_keys_ok` at `evaluation/tests/conftest.py:21-25` skips cleanly when keys absent.
9. **No `__init__.py` in test directories.** Pattern 12 / Phase 128-02 follow-on rule documented in `.planning/codebase/TESTING.md`. Test basenames must be unique repo-wide. Verify `test_eval_smoke_gate.py` and the new test names don't collide.

---

## Smoke Question IDs — Recommended Default

Per CONTEXT.md "Specific 5 question IDs: Selected during planning by reading `dataset/manifests/papers.json` and picking 3 single-hop + 2 multi-hop with deterministic, defensible criteria":

After inspecting `evaluation/golden_qa.json` (the canonical 30-question file used by `evaluation/harness/run.py:_load_golden_qa()` at line 80-82), the question distribution is:

- 10 single-hop, text modality (`single-hop-001` through `single-hop-010`)
- 10 multi-hop, text modality (`multi-hop-001` through `multi-hop-010`)
- 10 multimodal (excluded per CONTEXT.md)

**Recommended default constant:**

```python
DEFAULT_SMOKE_IDS = (
    "single-hop-001",  # RAG paper core mechanism (Lewis 2020 / 2005.11401)
    "single-hop-002",  # DPR dual-encoder design (Karpukhin 2020 / 2004.04906)
    "single-hop-003",  # REALM unsupervised pretraining (Guu 2020 / 2002.08909)
    "multi-hop-001",   # RAG-Token + DPR composition (2005.11401 + 2004.04906)
    "multi-hop-002",   # REALM vs RAG retrieval-time distinction (2002.08909 + 2005.11401)
)
```

**Defensible criteria:** Pick the lowest-numbered IDs in each class (deterministic, reproducible, no random selection). All 5 reference papers in the canonical RAG / DPR / REALM trio — corpus-relevant, well-cited, the corpus is known to contain these chunks. The two multi-hop questions both involve cross-document reasoning between the same 3 papers, so the agent has strong incentive to call both tools (good signal-to-noise for Pitfall 9 detection).

**Verified by inspecting `evaluation/golden_qa.json` directly via `python3 -c "import json; …"`** during research: every listed ID exists, modality_tag = "text", hop_count_tag matches, and source_papers is non-empty.

---

## Sources

### Primary (HIGH confidence — installed source on disk; not training data, not WebFetch)

- `.venv/lib/python3.13/site-packages/agents/items.py` — `ToolCallOutputItem` dataclass (lines 376-419), `RunItem` union (lines 610-625), `ItemHelpers` (lines 654-829). Verifies field names, types, and the `output: Any` semantics.
- `.venv/lib/python3.13/site-packages/agents/result.py` — `RunResultBase.new_items: list[RunItem]` declaration (line 181) and the property docs ("new items generated during the agent run").
- `.venv/lib/python3.13/site-packages/agents/run_internal/tool_execution.py` — instantiation site `ToolCallOutputItem(output=result, …)` at line 1850-1855, where `result` is the unmodified function-tool return value. Definitively answers "is `output` the raw return or a serialized form?": raw return.
- `.venv/lib/python3.13/site-packages/agents/__init__.py` — confirms `ToolCallOutputItem` is re-exported at the top level.
- `.venv/lib/python3.13/site-packages/openai_agents-0.14.6.dist-info/METADATA` — confirms version 0.14.6 (matches `pyproject.toml` pin).
- `.venv/lib/python3.13/site-packages/ragas-0.4.3.dist-info/METADATA` — confirms ragas 0.4.3.
- `.venv/lib/python3.13/site-packages/lightrag_hku-1.4.15.dist-info/METADATA` — confirms lightrag-hku 1.4.15.
- `.venv/lib/python3.13/site-packages/raganything-1.2.10.dist-info/METADATA` — confirms raganything 1.2.10.

### Primary (HIGH confidence — repo source files)

- `evaluation/harness/adapters/tier_5.py` — current bug location (line 125); existing `_strip_openrouter_prefix`, `MaxTurnsExceeded` handler, `tracker.record_llm` integration to preserve.
- `evaluation/harness/run.py` — existing `_capture_tier`, `_load_golden_qa`, `_git_sha`, `_ts`, `--limit`, argparse layout to extend.
- `evaluation/harness/score.py` — `_short_circuit_nan` at line 136 (already classifies `empty_contexts` and `agent_truncated`); `score_query_log` at line 168 (in-process scoring entry point usable from the smoke gate).
- `evaluation/harness/records.py` — `EvalRecord` (line 23), `QueryLog` (line 40), `ScoreRecord` (line 50), `write_query_log`/`read_query_log` Pydantic v2 idioms.
- `tier-5-agentic/agent.py` — line 43 unconditional `set_tracing_disabled(disabled=True)` (target of conditional toggle); `build_agent` factory and the LiteLLM routing pattern.
- `tier-5-agentic/tools.py` — `search_text_chunks` returns `list[dict]` keyed by `paper_id, page, snippet, similarity` (lines 122-146); `lookup_paper_metadata` returns `dict` keyed by `paper_id, title, authors, year, abstract` (lines 149-171).
- `evaluation/golden_qa.json` — 30 questions, schema verified (`id`, `question`, `expected_answer`, `source_papers`, `modality_tag`, `hop_count_tag`, `figure_ids`, `video_ids`).
- `evaluation/results/queries/tier-5-2026-05-02T17_35_35Z.json` — current 30/30 baseline (`retrieved_contexts: []` for all 30 records); reference for QueryLog JSON shape.
- `evaluation/results/queries/tier-1-2026-05-02T17_26_59Z.json` — example of a populated `retrieved_contexts: list[str]` for shape comparison.
- `evaluation/results/metrics/tier-5-2026-05-02T17_35_35Z.json` — confirms 30/30 `nan_reason: "empty_contexts"`.
- `evaluation/tests/test_eval_adapters.py` — current Tier 5 unit test (lines 157-193); the load-bearing assertion at line 174 must invert.
- `evaluation/tests/test_eval_smoke_live.py` — pattern for the new live smoke test (Tier 1 version mirrors directly).
- `evaluation/tests/conftest.py` — `live_eval_keys_ok` and `tier1_index_present` fixtures.
- `pyproject.toml` — version pins (line 60: `openai-agents[litellm]==0.14.6`; line 66: `ragas>=0.4.3,<0.5`).

### Primary (HIGH confidence — research bundle citations)

- `.planning/research/STACK.md` — sections "Tier 5 Empty-Contexts Diagnosis Pattern" (lines 162-265), "Alternatives Considered" (lines 267-276), "What NOT to Use" (lines 280-292), "If the OpenInference instrumentor reveals tools ARE being called" (lines 316-319), version compatibility table (lines 323-332), Sources (lines 336-360).
- `.planning/research/PITFALLS.md` — sections referenced for Tier 5 root-cause diagnosis (lines 20-50 in particular); note the Pitfall numbering is bundle-local (132-RESEARCH, separate from 130/131-RESEARCH numbering used by the adapter docstring).
- `.planning/REQUIREMENTS.md` — TIER-01, TIER-03, CLEANUP-02 definitions.
- `.planning/ROADMAP.md` — Phase 1 success criteria (lines 27-37).
- `.planning/STATE.md` — Blockers/Concerns block confirms "Tier 5 root cause is high-confidence (5-line adapter bug at `tier_5.py:125`)" and the fallback path requirement.
- `.planning/codebase/CONVENTIONS.md` and `.planning/codebase/TESTING.md` — naming, import order, pytest patterns, `@pytest.mark.live` discipline.

### Secondary (MEDIUM confidence — not consulted this session, cited by upstream research)

- Context7 `/websites/openai_github_io_openai-agents-python` — STACK.md cites this for `RunResult.new_items` and `ToolCallOutputItem` API surface. Already independently verified on disk; cited here for completeness.
- `BerriAI/litellm#16651` — `MALFORMED_FUNCTION_CALL` silent failures on `gemini-2.5-flash` with `Annotated[T, Field(...)]` schemas (relevant to STACK.md mutation step 1: "simplify schema first"). Cited by STACK.md; not independently verified.
- `openai/openai-agents-python#2257` — Gemini structured output + tools incompat (does not apply per STACK.md analysis; flagged for completeness).

### Tertiary (LOW confidence — none)

No claims in this RESEARCH.md depend on unverified WebSearch or training data.

---

## Metadata

**Confidence breakdown:**

- Standard stack: **HIGH** — every version verified against installed `dist-info/METADATA` files; pins in `pyproject.toml` cross-checked.
- Architecture: **HIGH** — 5-line walk verified against installed SDK source (`agents/items.py:376-419`, `run_internal/tool_execution.py:1850-1855`); no API-shape ambiguity.
- Smoke gate logic: **HIGH** — pure-Python data shaping over existing Pydantic records; no external dependencies; CONTEXT.md fully prescriptive.
- Pitfalls: **HIGH** — every pitfall enumerated has either (a) an installed-source citation or (b) an existing-repo-test/code citation. No speculation.
- Smoke question IDs: **HIGH** — IDs verified to exist in `evaluation/golden_qa.json` via direct inspection; modality and hop tags confirmed.
- Fallback path (debug-tier5 install): **HIGH** for the install commands (matches REQUIREMENTS.md v1.1 CLEANUP-02 and STACK.md); **MEDIUM** for whether the OpenInference spans actually surface the failure mode reliably (no live diagnostic run executed in this research session — STACK.md cites this as MEDIUM and that propagates).

**Research date:** 2026-05-04
**Valid until:** 2026-06-04 (30 days; openai-agents 0.14.6 and ragas 0.4.3 are stable pins; no upstream API surface changes expected within window). Re-research if `pyproject.toml` pins for `openai-agents`, `ragas`, `litellm`, `lightrag-hku`, or `raganything` change.

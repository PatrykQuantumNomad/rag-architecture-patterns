# Phase 2: Tier 4 Graphml Regeneration - Research

**Researched:** 2026-05-04
**Domain:** MineRU 3.1.4 outside-sandbox PDF parsing → RAG-Anything 1.2.10 (`insert_content_list` / `process_document_complete`) → LightRAG 1.4.15 KG persistence → harness `--tier-4-from-cache` smoke gate
**Confidence:** HIGH (every claim verified against installed source at `.venv/lib/python3.13/site-packages/{raganything,lightrag,mineru}/` or against existing repo files; no new web fetches needed because the libraries, the prior research artifacts, and the orchestration code are all on disk)

---

## Summary

Tier 4 evaluates a multimodal RAG pipeline (RAG-Anything wrapping LightRAG with a MineRU-driven layout pass over text + figures + tables). The 30/30 `empty_contexts` baseline failure recorded in the most recent capture (`evaluation/results/queries/tier-4-2026-05-02T17_44_53Z.json`) was NOT a retrieval failure — it was a `ValueError: No LightRAG instance available` raised on every question because the storage directory `rag_anything_storage/tier-4-multimodal/` was empty (no graphml, no kv_stores). The fix `f0cd134` ("call `_ensure_lightrag_initialized` before aquery") landed in `eval_capture.py` AFTER that 2026-05-02 capture but cannot help if the underlying graph still doesn't exist on disk. So Phase 2's actual work is to **rebuild the graph, not patch a code path** [VERIFIED: queries JSON inspection on 2026-05-04].

The previous Tier 4 storage was wiped some time before 2026-05-03 (`rag_anything_storage/tier-4-multimodal/` is empty in the working tree as of 2026-05-04). Phase 138/139 evidence — surfaced in `evaluation/README.md:3` and `commit 314961e` — established that **MineRU cannot run inside the orchestrator sandbox** because of a kernel-level OMP shmem block (specifically OpenMP shared-memory allocation that the sandbox does not permit; not a permissions or network issue). The library is dual-mode by design: `evaluation/harness/run.py --tiers 4 --tier-4-from-cache PATH` re-emits a pre-captured QueryLog written by `tier-4-multimodal/scripts/eval_capture.py`, which itself runs locally on a host where MineRU works [VERIFIED: `evaluation/harness/run.py:213-237` `--tier-4-from-cache` skip path; `evaluation/README.md` Tier 4 deferral block].

**There is a load-bearing shortcut available.** The MineRU CLI has already produced parsed JSON for **75 of 100 papers** under `tier-4-multimodal/output/<paper_id>/<paper_id>_<hash>/<paper_name>/hybrid_auto/<paper>_content_list.json` (verified by directory listing 2026-05-04). All 5 questions in the Phase 1 / Phase 2 default smoke set (`single-hop-001`, `single-hop-002`, `single-hop-003`, `multi-hop-001`, `multi-hop-002`) reference papers that ARE in the existing MineRU output (`2005.11401`, `2004.04906`, `2002.08909`). For a smoke-only Phase 2, **no fresh MineRU pass is required** — feed the existing `_content_list.json` files into RAG-Anything via `rag.insert_content_list(...)`, which is the public bypass for the `process_document_complete` path that re-runs MineRU. For a full Phase 7 rerun, 25 papers still need MineRU passes (4 of those are referenced in golden_qa: `1909.01066`, `2002.06177`, `2309.15217`, `2410.05779` — the LightRAG paper). Phase 2 should ship the `insert_content_list` reload path AND the missing-papers MineRU pass as separate steps so Phase 7 can run full-corpus without re-parsing the 75 we already have.

**Primary recommendation:** Build a `tier-4-multimodal/scripts/ingest_from_mineru.py` helper that reads `_content_list.json` files from `tier-4-multimodal/output/<paper_id>/.../<paper>_content_list.json`, fixes up `img_path` to absolute paths against the per-paper `images/` subdirectory (Pitfall 4 of 130-RESEARCH), and feeds them into `rag.insert_content_list(content_list, file_path=<paper_id>, doc_id=<paper_id>)` — once per paper, in deterministic order. Wire a `tier-4-multimodal/scripts/parse_missing_papers.py` helper that runs `mineru -p <pdf> -o tier-4-multimodal/output/<paper_id>/ ...` for the 4 papers needed by golden_qa but missing today. After both ingest + parse-missing complete, run `eval_capture.py --smoke-question-ids <5 IDs>` and feed the resulting JSON through `evaluation.harness.run --tiers 4 --tier-4-from-cache PATH`, then `evaluation.harness.score`, then `evaluation.harness.smoke_gate --tier 4`. Provenance: a `tier-4-multimodal/scripts/log_graph_stats.py` helper reads the freshly-built graphml + kv_stores and writes node_count, edge_count, kv_store_full_docs count, and pinned library versions (`raganything==1.2.10`, `lightrag-hku==1.4.15`, `mineru==3.1.4`) into `evaluation/results/diagnostics/tier-4-graph-stats-{TS}.json`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Parse PDFs to MineRU `_content_list.json` (run OUTSIDE sandbox) | Host CLI (`mineru` binary at `.venv/bin/mineru`) | RAG-Anything (`MineruParser._run_mineru_command` at `parser.py:613`) | MineRU 3.1.4 cannot run inside the sandbox (OMP shmem block per Phase 138/139). The library calls `mineru` as a subprocess — when the user invokes `process_document_complete` outside the sandbox, the subprocess works; inside, it crashes. The clean separation: parse on host, ingest in either env. |
| Ingest pre-parsed `_content_list.json` into LightRAG via RAG-Anything | `tier-4-multimodal/scripts/ingest_from_mineru.py` (NEW) calling `rag.insert_content_list(...)` | `tier_4_multimodal.rag.build_rag` (existing) | `insert_content_list` is the public RAG-Anything API for re-using parsed content without re-running MineRU. It calls `_ensure_lightrag_initialized()` internally; passes the content list through `separate_content` → `insert_text_content` (LightRAG entity extraction) + `_process_multimodal_content` (vision passes for `type=image|table|equation`). |
| Persist NetworkX KG (`graph_chunk_entity_relation.graphml`) + kv_stores + vdbs | LightRAG `NetworkXStorage` (`networkx_impl.py:38` calls `nx.write_graphml`) | `JsonKVStorage` (`json_kv_impl.py:40` writes `kv_store_<namespace>.json`); `NanoVectorDBStorage` (`nano_vector_db_impl.py:55` writes `vdb_<namespace>.json`) | LightRAG composes these three storage backends. RAG-Anything does not own them — it forwards to its embedded LightRAG instance. The fact that storage paths use the **same conventions as Tier 3** (verified `lightrag_storage/tier-3-graph/` listing) means Phase 2's verification can mirror Tier 3's existing test patterns. |
| Provenance logging (graph stats + library versions) | `tier-4-multimodal/scripts/log_graph_stats.py` (NEW) | `evaluation/results/diagnostics/` (existing dir, written to by Phase 1's `FallbackLog`) | Pitfall 2 of `.planning/research/PITFALLS.md` calls for `(node_count, edge_count, kv_store entry count, graphml mtime, library versions)` captured post-ingest. The diagnostics dir is the existing convention; reading the JSON `model_dump_json(indent=2)` shape in `evaluation/harness/diagnostics.py` (Phase 1's `FallbackLog`) gives a template. |
| Drive the 5-question smoke harness (capture → score → gate) | `evaluation/harness/run.py --tiers 4 --tier-4-from-cache PATH` (existing, no changes) → `evaluation/harness/score.py` (existing) → `evaluation/harness/smoke_gate.py --tier 4` (existing, no changes — the gate is tier-agnostic) | `tier-4-multimodal/scripts/eval_capture.py` (existing — drives the live capture; produces the JSON the harness reads) | Phase 1 already locked `--smoke-question-ids` + `DEFAULT_SMOKE_IDS` + `smoke_gate.py`. The same 5 IDs apply (D-04 of `01-CONTEXT.md`). Phase 2 inherits the gate verbatim — its job is to produce the QueryLog the gate consumes. |
| Wipe-and-rebuild guarantee | `tier-4-multimodal/scripts/ingest_from_mineru.py --reset` flag (NEW) OR `rm -rf rag_anything_storage/tier-4-multimodal/` (manual) | `tier-4-multimodal/main.py --reset` (existing — for the live MineRU+ingest path) | Pitfall 2 of PITFALLS.md insists on **delete, not move, not archive** so no zombie kv_store / vdb / graphml files leak in. The cost-surprise gate in `main.py` already implements this for the live path; the new `ingest_from_mineru.py` should mirror the gate (no MineRU cost, but the entity-extraction LLM cost is real — ~$0.50–1.00 corpus-wide per `tier-4-multimodal/README.md`). |

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TIER-02 | User can regenerate clean Tier 4 graphml from a wiped `rag_anything_storage/tier-4-multimodal/` directory by running MineRU CLI on the host machine (outside the sandbox) and feeding parsed JSON into RAG-Anything ingestion | The pieces exist: (a) MineRU 3.1.4 CLI verified at `.venv/bin/mineru` with the documented `-p / -o / -m / -b / -d` flag set [VERIFIED: `mineru --help` 2026-05-04]; (b) `rag.insert_content_list(content_list, file_path, doc_id)` is the public RAG-Anything API for ingesting pre-parsed JSON [VERIFIED: `raganything/processor.py:1868`]; (c) 75/100 papers already have `_content_list.json` under `tier-4-multimodal/output/`; (d) `dataset/papers/` has all 100 source PDFs available for the 25 missing parses [VERIFIED: directory listings 2026-05-04]. The phase needs a wrapper script that combines (b)+(c) into a single rebuild flow. |
| TIER-03 (Tier 4 portion) | User can verify the fix on a 5-question smoke test before committing to a full 30-question rerun | The harness scaffolding for this is COMPLETE from Phase 1: `--smoke-question-ids` flag, `DEFAULT_SMOKE_IDS` module-level constant (`single-hop-001,single-hop-002,single-hop-003,multi-hop-001,multi-hop-002`), and `smoke_gate.py` (PASS/FAIL/INCONCLUSIVE classifier). Phase 1's CONTEXT D-03 explicitly locks "same 5 IDs reused for Phase 2 (Tier 4 smoke)". Phase 2 reuses these verbatim — no harness code changes needed. The smoke set's source papers (`2005.11401`, `2004.04906`, `2002.08909`) are all in the existing MineRU output, so the smoke can run as soon as the ingest helper lands [VERIFIED: golden_qa.json + tier-4-multimodal/output/ directory listings]. |
</phase_requirements>

---

## Standard Stack

The phase introduces **zero new runtime dependencies**. Every library it touches is already pinned in `pyproject.toml`. The MineRU CLI is already installed via `[tier-4]` extras (`raganything==1.2.10` pulls `mineru>=2.5,<4` per STACK.md line 27, currently resolved to `mineru==3.1.4`).

### Core (Already Pinned — DO NOT change)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `raganything` | `==1.2.10` (verified `.venv/lib/python3.13/site-packages/raganything-1.2.10.dist-info/METADATA`) | Multimodal RAG layer over LightRAG; provides `insert_content_list` (the bypass for re-running MineRU) and `process_document_complete` (the live MineRU+ingest path) | Pinned across the repo. Bumping invalidates Phase 130's cost assumptions. The `insert_content_list` and `process_document_complete` API surface is stable in 1.2.x per the installed source at `processor.py:1508` and `processor.py:1868`. [VERIFIED: source inspection 2026-05-04] |
| `lightrag-hku` | `==1.4.15` (verified `.venv/lib/python3.13/site-packages/lightrag_hku-1.4.15.dist-info/METADATA`) | Underlying KG + chunked-doc + vector-DB persistence layer. Writes `graph_chunk_entity_relation.graphml`, `kv_store_*.json`, `vdb_*.json`. RAG-Anything composes a LightRAG instance internally. | Pinned. The graphml file format is stable in 1.4.x; bumping risks index-format break. The `NetworkXStorage` writes via `nx.write_graphml` at `lightrag/kg/networkx_impl.py:38` — no custom format. |
| `mineru` | `>=2.5,<4` (resolved `==3.1.4` — verified `.venv/lib/python3.13/site-packages/mineru-3.1.4.dist-info/METADATA` and `mineru --version` output) | PDF → structured JSON parser (per-page layout, figures, tables, equations). Pulled in transitively by `raganything==1.2.10`. | Locked floor at 2.5 because `MINERU_PARSE_METHOD` was renamed to `PARSE_METHOD` in 2.0 (per STACK.md line 27). 3.1.4 is the current installed; Phase 2 should record the EXACT version (`3.1.4`) in provenance, not the floor pin. |
| `pydantic` | `>=2.10,<3` (per `pyproject.toml:28`) | The provenance JSON the new `log_graph_stats.py` helper writes is a Pydantic v2 model written via `model_dump_json(indent=2)` — matches `evaluation/harness/records.py:69` and `evaluation/harness/diagnostics.py` (Phase 1 FallbackLog) conventions | Repo convention. |

### Supporting (Already Pinned — Tier 4 path)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `openai` | `>=1.50,<3` (per `pyproject.toml:55`) | Embeddings + chat API surface for the LightRAG closures (`openai_complete_if_cache`, `openai_embed`); routed via OpenRouter base URL | Always — every ingest call uses these. |
| `Pillow` | `>=10,<12` (per `pyproject.toml:56`) | RAG-Anything's image-handling for `type=image` content list entries | Always — every multimodal pass touches Pillow. |
| `networkx` | (transitive via `lightrag-hku`) | The graphml writer (`nx.write_graphml`) and reader (`nx.read_graphml`) | Provenance: `log_graph_stats.py` reads the graphml via `networkx.read_graphml(graphml_path)` to get `g.number_of_nodes()` / `g.number_of_edges()`. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `rag.insert_content_list(content_list, ...)` | `rag.process_document_complete(file_path=PDF, doc_id=...)` | The latter re-runs MineRU on every call — the exact thing the sandbox blocks. Only viable on the host AND wastes ~$0.50–1.00 of LLM cost per 100-paper rebuild for no benefit (we already have the `_content_list.json`). |
| `rag.insert_content_list(content_list, ...)` | `rag.lightrag.ainsert(text)` (the underlying LightRAG path) | Would skip RAG-Anything's `_process_multimodal_content` step — lose figure/table/equation entities entirely. The whole point of Tier 4 is multimodal. |
| MineRU CLI on host (`-b hybrid-auto-engine`) | MineRU CLI on host (`-b pipeline` for CPU-only or `-b vlm-auto-engine` for full GPU) | The existing 75 papers were produced with `-b hybrid-*` (verified A2: directory naming `hybrid_auto/` from `mineru/cli/output_paths.py:24-25` is unique to hybrid backends). Switching to `pipeline` mid-corpus produces a different per-paper directory layout (`<pdf>/auto/` instead of `<pdf>/hybrid_auto/`) and Pattern 1's glob would need a fallback. **Use `-b hybrid-auto-engine` (MineRU default) for the missing-papers top-up to match what the existing 75 used.** Note: `hybrid-auto-engine` requires the `[hybrid]` MineRU extras to be installed (verified — `.venv/bin/mineru-router` and friends are present, indicating extras are installed).
| Re-use `tier-4-multimodal/main.py --ingest` | New `ingest_from_mineru.py` script | `main.py --ingest` calls `process_document_complete` (re-runs MineRU per paper). It cannot ingest pre-parsed JSON. A new script is mandatory for the no-MineRU rebuild path. |
| Custom NetworkX-only graph reader | `networkx.read_graphml(path)` standard | None — the file IS a graphml file written by `nx.write_graphml`. Use the standard reader. |

**Installation:**

```bash
# Already installed via [tier-4] extra; no new pip work for Phase 2.
uv sync --extra tier-4   # idempotent on a checkout that has [tier-4] resolved in uv.lock
```

**Version verification:** Verified in this session 2026-05-04:

```bash
$ .venv/bin/mineru --version       # mineru, version 3.1.4
$ grep ^Version .venv/lib/python3.13/site-packages/raganything-*.dist-info/METADATA   # Version: 1.2.10
$ grep ^Version .venv/lib/python3.13/site-packages/lightrag_hku-*.dist-info/METADATA   # Version: 1.4.15
```

[VERIFIED: filesystem inspection 2026-05-04]

---

## Architecture Patterns

### System Architecture Diagram (Phase 2 data flow)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 0 (HOST, OUTSIDE SANDBOX) — top-up 25 missing MineRU outputs       │
│                                                                            │
│   $ python tier-4-multimodal/scripts/parse_missing_papers.py              │
│                                  │                                         │
│                                  ▼                                         │
│   subprocess: mineru -p dataset/papers/<id>.pdf \                         │
│                      -o tier-4-multimodal/output/<id>/ \                  │
│                      -m auto -b pipeline                                  │
│                                  │                                         │
│                                  ▼                                         │
│   tier-4-multimodal/output/<id>/<id>_<hash>/<paper>/hybrid_auto/          │
│      ├── <paper>_content_list.json      ← what we ingest                  │
│      ├── <paper>.md                     ← human-readable                  │
│      └── images/<sha256>.{jpg,png}      ← per-figure images               │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 1 (EITHER ENV) — wipe + ingest from existing JSON                  │
│                                                                            │
│   $ rm -rf rag_anything_storage/tier-4-multimodal/                         │
│   $ python tier-4-multimodal/scripts/ingest_from_mineru.py --yes           │
│                                  │                                         │
│                                  ▼                                         │
│   ┌──────────────────────────────────────────┐                             │
│   │  build_rag(working_dir=tier-4-multimodal,│                             │
│   │            llm_token_tracker=adapter)    │   ← existing rag.py          │
│   └──────────────────────────────────────────┘                             │
│                                  │                                         │
│                                  ▼                                         │
│   for paper_id in sorted(papers):                                          │
│     content_list = json.load(<paper>_content_list.json)                    │
│     content_list = _absolutize_image_paths(content_list, images_dir)       │
│     await rag.insert_content_list(                                         │
│         content_list=content_list,                                         │
│         file_path=paper_id,            # citation source                   │
│         doc_id=paper_id,               # stable id (Pitfall 1, dedup)      │
│     )                                                                      │
│                                  │                                         │
│                                  ▼                                         │
│   rag_anything_storage/tier-4-multimodal/                                  │
│      ├── graph_chunk_entity_relation.graphml   (NetworkX KG)              │
│      ├── kv_store_full_docs.json                                           │
│      ├── kv_store_text_chunks.json                                         │
│      ├── kv_store_doc_status.json                                          │
│      ├── kv_store_full_entities.json                                       │
│      ├── kv_store_full_relations.json                                      │
│      ├── kv_store_entity_chunks.json                                       │
│      ├── kv_store_relation_chunks.json                                     │
│      ├── kv_store_llm_response_cache.json                                  │
│      ├── vdb_chunks.json                                                   │
│      ├── vdb_entities.json                                                 │
│      └── vdb_relationships.json                                            │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 2 (EITHER ENV) — provenance log + cost tracker persist              │
│                                                                            │
│   $ python tier-4-multimodal/scripts/log_graph_stats.py                    │
│                                  │                                         │
│                                  ▼                                         │
│   networkx.read_graphml(graphml_path)                                      │
│   g.number_of_nodes(), g.number_of_edges()                                 │
│   len(json.load(kv_store_full_docs.json))   # paper count                  │
│   importlib.metadata.version("raganything")  → "1.2.10"                    │
│   importlib.metadata.version("lightrag-hku") → "1.4.15"                    │
│   importlib.metadata.version("mineru")       → "3.1.4"                     │
│   subprocess.check_output(["git","rev-parse","--short","HEAD"])            │
│                                  │                                         │
│                                  ▼                                         │
│   evaluation/results/diagnostics/tier-4-graph-stats-{TS}.json              │
│   (Pydantic-typed, model_dump_json(indent=2))                              │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 3 (HOST or sandbox — sandbox MUST have OPENROUTER_API_KEY)          │
│  Smoke capture against the rebuilt graph                                   │
│                                                                            │
│   $ python tier-4-multimodal/scripts/eval_capture.py \                    │
│         --smoke-question-ids \                                             │
│         single-hop-001,single-hop-002,single-hop-003,\                    │
│         multi-hop-001,multi-hop-002 --yes                                  │
│                                                                            │
│   (NOTE: eval_capture.py CURRENTLY supports --limit, NOT                   │
│    --smoke-question-ids. Phase 2 must add the same flag pattern            │
│    Phase 1 added to harness/run.py. Plan-level decision:                   │
│    EITHER add the flag here OR have run.py do the filtering on             │
│    its read of the cached log. Recommendation: add the flag in             │
│    eval_capture.py for symmetry with run.py. See Open Q1.)                 │
│                                  │                                         │
│                                  ▼                                         │
│   evaluation/results/queries/tier-4-{TS}.json   ← QueryLog                 │
│   evaluation/results/costs/tier-4-eval-{TS}.json                           │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 4 — harness re-emit + score + smoke_gate (no behavior change)       │
│                                                                            │
│   $ python -m evaluation.harness.run --tiers 4 \                          │
│         --tier-4-from-cache evaluation/results/queries/tier-4-{TS}.json \  │
│         --smoke-question-ids \                                             │
│         single-hop-001,single-hop-002,single-hop-003,\                    │
│         multi-hop-001,multi-hop-002 --yes                                  │
│   $ python -m evaluation.harness.score --tiers 4 --yes                    │
│   $ python -m evaluation.harness.smoke_gate --tier 4                      │
│                                                                            │
│   → SmokeGateResult { verdict: PASS/FAIL/INCONCLUSIVE, ... }              │
│   → Phase 2 SHIPS when verdict == PASS                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure

```
rag-architecture-patterns/
├── tier-4-multimodal/
│   ├── main.py                          # EXISTING — live MineRU + ingest path (no change)
│   ├── rag.py                           # EXISTING — build_rag() (no change)
│   ├── ingest_pdfs.py                   # EXISTING — process_document_complete loop (no change)
│   ├── ingest_images.py                 # EXISTING — insert_content_list for figures (no change)
│   ├── output/                          # EXISTING — 75 papers MineRU-parsed already
│   │   ├── 1410.3916/<hash>/<paper>/hybrid_auto/<paper>_content_list.json
│   │   └── ... (74 more)
│   └── scripts/
│       ├── eval_capture.py              # EXISTING — drives 30-Q capture; Phase 2 extends with --smoke-question-ids
│       ├── ingest_from_mineru.py        # NEW — wipe + insert_content_list loop over output/
│       ├── parse_missing_papers.py      # NEW — top-up MineRU pass for 25 missing papers
│       └── log_graph_stats.py           # NEW — provenance JSON writer
│
├── rag_anything_storage/
│   └── tier-4-multimodal/               # currently empty → Stage 1 fills it
│       ├── graph_chunk_entity_relation.graphml
│       ├── kv_store_*.json (8 files)
│       └── vdb_*.json (3 files)
│
├── evaluation/harness/                  # ALL EXISTING — no changes needed
│   ├── run.py                           # already has --tier-4-from-cache + --smoke-question-ids
│   ├── score.py
│   ├── smoke_gate.py                    # tier-agnostic; --tier 4 just works
│   └── adapters/tier_4.py               # already implements cached-mode read
│
└── evaluation/results/diagnostics/      # NEW DIR (or reused from Phase 1)
    └── tier-4-graph-stats-{TS}.json     # NEW — provenance manifest
```

### Pattern 1: Wipe-then-rebuild via `insert_content_list`

**What:** Use RAG-Anything's `insert_content_list` API to ingest pre-parsed MineRU JSON files without re-running MineRU. This is the public bypass that exists precisely for the "parse on host, ingest in sandbox" workflow Phase 138/139 demands.

**When to use:** Every Phase 2 rebuild. The live `process_document_complete` path is **only** for the missing-papers top-up where we have to invoke MineRU; the bulk of the 100-paper ingest comes through `insert_content_list`.

**Example:**

```python
# Source: raganything/processor.py:1868 (verified API contract)
#         tier-4-multimodal/ingest_images.py (existing repo pattern)

import json
from pathlib import Path
from tier_4_multimodal.rag import build_rag, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL
from tier_4_multimodal.cost_adapter import CostAdapter
from shared.cost_tracker import CostTracker


def _absolutize_image_paths(content_list: list[dict], images_dir: Path) -> list[dict]:
    """Resolve relative img_path → absolute (Pitfall 4 of 130-RESEARCH).

    MineRU writes img_path as 'images/<sha>.jpg' (relative to the per-paper
    output dir). insert_content_list silently accepts the relative form but
    produces empty image entities in vdb_entities.json — exactly the symptom
    the existing tier-4-multimodal/ingest_images.py:36 prevents for the
    standalone-figures bundle.
    """
    images_dir = images_dir.resolve()
    out = []
    for item in content_list:
        if item.get("type") == "image" and "img_path" in item:
            rel = Path(item["img_path"])
            abs_path = images_dir / rel.name if rel.is_absolute() is False else rel
            out.append({**item, "img_path": str(abs_path.resolve())})
        else:
            out.append(item)
    return out


async def ingest_from_mineru_output(
    rag,
    mineru_output_root: Path,
    paper_ids: list[str],
) -> int:
    """Loop over papers in deterministic order, ingest each via insert_content_list.

    For each paper, MineRU's output layout is:
        <root>/<paper_id>/<paper_id>_<hash>/<paper_name>/hybrid_auto/
            ├── <paper_name>_content_list.json
            └── images/<sha256>.{jpg,png}

    The hash directory is content-derived; we glob to find it.
    """
    n_ingested = 0
    for pid in sorted(paper_ids):
        # Find the single _content_list.json under this paper's dir tree
        paper_root = mineru_output_root / pid
        if not paper_root.exists():
            continue
        content_lists = list(paper_root.glob("**/*_content_list.json"))
        # Skip the v2 variant (raganything 1.2.10 uses v1 by default per
        # the existing tier-4-multimodal/output structure verified 2026-05-04)
        content_lists = [p for p in content_lists if "_v2" not in p.name]
        if not content_lists:
            continue
        # Pick the first; deterministic order due to glob result sort
        content_list_path = sorted(content_lists)[0]
        images_dir = content_list_path.parent / "images"
        content_list = json.loads(content_list_path.read_text())
        content_list = _absolutize_image_paths(content_list, images_dir)
        await rag.insert_content_list(
            content_list=content_list,
            file_path=pid,           # citation source (paper_id, not full PDF path)
            doc_id=pid,              # stable id — RAG-Anything dedup (anti-pattern: omit this)
        )
        n_ingested += 1
    return n_ingested
```

### Pattern 2: Missing-papers MineRU top-up (host-only)

**What:** A small helper that calls the `mineru` CLI directly (subprocess) for the 4 papers needed by golden_qa but missing from `tier-4-multimodal/output/`. This is the ONLY part of Phase 2 that MUST run outside the sandbox.

**When to use:** Once before Stage 1 ingest, on a host machine. After the 25-paper top-up (4 needed for golden_qa + 21 corpus-completeness papers), `tier-4-multimodal/output/` is fully populated and Phase 7 can run full-corpus from cached JSON.

**Example:**

```python
# Source: raganything/parser.py:613 (_run_mineru_command — flag set verified)
#         STACK.md lines 100-113 (canonical MineRU CLI invocation)

import subprocess
from pathlib import Path

MISSING_PAPERS_GOLDEN = [
    "1909.01066",   # multi-hop-008
    "2002.06177",   # multimodal-010
    "2309.15217",   # multi-hop-009
    "2410.05779",   # single-hop-005, multi-hop-003, multimodal-004, multimodal-005
]

def parse_missing_papers(papers_dir: Path, output_root: Path, papers: list[dict]) -> int:
    """Run mineru CLI for each missing paper, deterministic order.

    papers: list of dicts from dataset/manifests/papers.json (paper_id + filename).
    Returns the number of papers actually parsed (skips already-parsed).
    """
    by_id = {p["paper_id"]: p for p in papers}
    parsed = 0
    for pid in sorted(MISSING_PAPERS_GOLDEN):
        if pid not in by_id:
            continue
        if (output_root / pid).exists():
            continue   # already parsed; idempotent
        pdf = papers_dir / by_id[pid]["filename"]
        if not pdf.exists():
            continue
        cmd = [
            "mineru",
            "-p", str(pdf),
            "-o", str(output_root / pid),
            "-m", "auto",
            "-b", "hybrid-auto-engine",  # MineRU default; matches the existing 75 papers' backend (verified A2)
            "-l", "en",            # ML papers — English; suppresses lang autodetect cost
        ]
        subprocess.run(cmd, check=True)
        parsed += 1
    return parsed
```

### Pattern 3: Provenance manifest (Pydantic + importlib.metadata)

**What:** After Stage 1 ingest completes, write a typed JSON manifest under `evaluation/results/diagnostics/` that captures node count, edge count, kv_store cardinalities, library versions, and git SHA. Pitfall 2 of PITFALLS.md insists on this; Phase 9's frozen doc reads it.

**When to use:** Once per rebuild. The same JSON is the artifact Phase 9 cites in the frozen-doc provenance block.

**Example:**

```python
# Source: lightrag/kg/networkx_impl.py:38 (graphml format)
#         evaluation/harness/diagnostics.py (Phase 1 FallbackLog convention)
#         evaluation/harness/records.py:69 (Pydantic + model_dump_json(indent=2))

import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import networkx as nx
from pydantic import BaseModel


class GraphStats(BaseModel):
    timestamp: str
    git_sha: str
    working_dir: str
    graphml_path: str
    graphml_node_count: int
    graphml_edge_count: int
    graphml_size_bytes: int
    graphml_mtime: str             # ISO 8601 of file mtime
    kv_full_docs_count: int        # papers ingested
    kv_text_chunks_count: int
    kv_full_entities_count: int
    kv_full_relations_count: int
    raganything_version: str
    lightrag_version: str
    mineru_version: str


def collect_graph_stats(working_dir: Path) -> GraphStats:
    graphml = working_dir / "graph_chunk_entity_relation.graphml"
    g = nx.read_graphml(graphml)
    full_docs = json.loads((working_dir / "kv_store_full_docs.json").read_text())
    text_chunks = json.loads((working_dir / "kv_store_text_chunks.json").read_text())
    full_entities = json.loads((working_dir / "kv_store_full_entities.json").read_text())
    full_relations = json.loads((working_dir / "kv_store_full_relations.json").read_text())
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    mtime = datetime.fromtimestamp(graphml.stat().st_mtime, tz=timezone.utc).isoformat()
    return GraphStats(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_sha=sha,
        working_dir=str(working_dir),
        graphml_path=str(graphml),
        graphml_node_count=g.number_of_nodes(),
        graphml_edge_count=g.number_of_edges(),
        graphml_size_bytes=graphml.stat().st_size,
        graphml_mtime=mtime,
        kv_full_docs_count=len(full_docs),
        kv_text_chunks_count=len(text_chunks),
        kv_full_entities_count=len(full_entities),
        kv_full_relations_count=len(full_relations),
        raganything_version=pkg_version("raganything"),
        lightrag_version=pkg_version("lightrag-hku"),
        mineru_version=pkg_version("mineru"),
    )
```

### Anti-Patterns to Avoid

- **Calling `process_document_complete(file_path=PDF, ...)` for the bulk rebuild.** Re-runs MineRU per paper — burns ~$0.50–1.00 of LLM cost AND requires the sandbox to be host-side. Use `insert_content_list` for the 75 cached papers; reserve `process_document_complete` (or the raw `mineru` CLI) for the 25-paper top-up.
- **Moving `rag_anything_storage/tier-4-multimodal/` to a `.bak/` instead of deleting.** Pitfall 2 of PITFALLS.md is explicit: zombie kv_store / vdb / graphml files leak in via the LightRAG `_load` paths. Delete, do not archive.
- **Omitting `doc_id=<paper_id>` on ingest calls.** RAG-Anything generates a content-hash-derived id otherwise (per `processor.py:1918`). Different runs ⇒ different ids ⇒ duplicate documents on a re-run, which inflates entity counts and breaks the dedup machinery.
- **Treating the `_content_list_v2.json` files in the existing output as the canonical input.** RAG-Anything 1.2.10 reads `_content_list.json` (v1 format) — the v2 file is from an upstream MineRU upgrade not used by the 1.2.10 line. Pattern 1 explicitly excludes `_v2` filenames. PITFALLS.md line 58 calls out this exact issue: "legacy `*_content_list.json` formats can leak into the current ingest path and amplify noise."
- **Running the smoke against the FULL 30 questions when only 5 are needed.** Phase 1's `--smoke-question-ids` flag exists precisely so we don't burn 6× the cost. The existing `eval_capture.py` only has `--limit`; this is a gap to close (Open Q1).
- **Writing graphml node/edge counts to the captured QueryLog JSON.** That JSON is the per-question record set — provenance belongs in a separate diagnostics file. Mixing them violates the existing record schema (`evaluation/harness/records.py`).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Loading pre-parsed MineRU content into LightRAG | A custom NetworkX builder that reads `_content_list.json` and writes `graph_chunk_entity_relation.graphml` directly | `rag.insert_content_list(...)` | LightRAG's graph format includes ~30 node/edge attributes (entity types, source ids, relation descriptions, embedding metadata) generated by the entity-extraction LLM pass. Hand-rolling means re-implementing the entity-extraction prompts; impossible at acceptable quality. |
| Re-parsing PDFs to JSON | A `pdfminer` or `pymupdf` script that writes a content-list-shaped dict | `mineru` CLI | MineRU's value is per-page layout, OCR fallback, equation/table parsing, and figure extraction with bounding boxes. `pdfminer` produces text-only output. Tier 4's whole differentiator is multimodal. |
| Provenance JSON schema | A bespoke dict written via `json.dumps` | Pydantic v2 model + `model_dump_json(indent=2)` | Repo convention (`records.py`, `diagnostics.py`). Type errors at write time, not at the consumer. |
| Library version capture | Parsing `pyproject.toml` or running `pip freeze` and grepping | `importlib.metadata.version("<dist-name>")` | The pinned version in pyproject.toml is the FLOOR; the resolved version (`raganything==1.2.10`, `mineru==3.1.4`) is the source of truth. `importlib.metadata` reads the resolved version from the installed dist-info. |
| Graphml file inspection (node + edge counts) | XML parsing of the `.graphml` file | `networkx.read_graphml(path)` | Standard library, already a transitive dep. Reads in <1s for graphs up to ~10k nodes (the expected Tier 4 scale per Pitfall 2 PITFALLS.md "expected ~581 figures + paper entities"). |
| Smoke gate logic | A custom pass/fail script in `tier-4-multimodal/scripts/` | `evaluation/harness/smoke_gate.py --tier 4` | Already locked by Phase 1 D-04. Tier-agnostic. Phase 2 must NOT introduce a parallel gate. |
| 5-question filtering | A subset constant in `eval_capture.py` | `--smoke-question-ids` flag (mirroring `evaluation/harness/run.py:340`) | Phase 1 already locked the same 5 IDs across phases. Adding a second source of truth violates D-03 of `01-CONTEXT.md` ("Same 5 IDs reused for Phase 1 [Tier 5 smoke], Phase 2 [Tier 4 smoke]"). |

**Key insight:** Tier 4 is a thin orchestration layer over three large libraries (RAG-Anything, LightRAG, MineRU). The right Phase 2 is the one with the smallest delta over the existing repo: a couple of helper scripts that call existing public APIs in the right order and capture provenance. Anything that touches the LightRAG or RAG-Anything internals is wrong.

---

## Runtime State Inventory

This phase REBUILDS state, not RENAMES — so most categories are inert. The few that matter:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `rag_anything_storage/tier-4-multimodal/` is currently EMPTY (verified `ls` 2026-05-04). Stage 1 fills it with `graph_chunk_entity_relation.graphml`, 8× `kv_store_*.json`, 3× `vdb_*.json`. | Verify pre-Stage-1 that the dir is empty (use `find <dir> -type f \| wc -l`); abort with a clear error if files exist. Do not assume the user already ran `rm -rf`. Pattern 1's `--reset` flag handles this. |
| Live service config | None — Tier 4 has no external service config. The OPENROUTER_API_KEY is the only env-bound state, and the helpers read it lazily inside the closures (existing pattern at `tier-4-multimodal/rag.py:71-87`). | None. |
| OS-registered state | None — no Task Scheduler / launchd / systemd registration; Tier 4 is invoked manually. | None. |
| Secrets/env vars | `OPENROUTER_API_KEY` (required for entity extraction during ingest + vision passes + per-query LLM calls). Read at closure-call time, not module-import time, so the helpers fail fast with a clear error message if the key is unset. | Pre-flight check at the top of each new helper: `if not settings.openrouter_api_key: print red error; return 2` (mirrors `eval_capture.py:66-68`). |
| Build artifacts / installed packages | `tier-4-multimodal/output/<paper_id>/` directories from prior MineRU runs. 75 of 100 papers have these; 4 of 100 needed by golden_qa are missing. The existing 75 are usable — they have `_content_list.json` v1 files. There ARE `_content_list_v2.json` files alongside (newer MineRU format) which RAG-Anything 1.2.10 does NOT use. | Pattern 1's `_content_list.json` glob explicitly excludes `_v2`. Do not delete the existing output directories during the rebuild — they are the cache the rebuild reads from. |

**Nothing found in category:** Live service config, OS-registered state — none, verified by absence of any cron / launchd / systemd unit referencing tier-4 in the repo.

---

## Common Pitfalls

### Pitfall 1: Partial-ingest mid-run leaves a corrupted graphml that queries silently

**What goes wrong:** LightRAG supports incremental ingest — every `insert_content_list` call grows the graphml. If the loop is interrupted (Ctrl-C, OOM, network blip during entity-extraction LLM call), the on-disk graphml is a partial state. Subsequent `aquery` calls return contexts from the partial graph with no warning.

**Why it happens:** LightRAG's NetworkX backend writes after each successful `insert` step. There is no transaction boundary across multiple papers — each paper is its own commit. [VERIFIED: `lightrag/kg/networkx_impl.py:34` — `nx.write_graphml(graph, file_name)` is called per upsert in the index_done_callback.]

**How to avoid:**
- Loop over papers in deterministic sorted order; record `n_ingested` after each successful call; on interrupt, log "ingested N/M papers — graph is partial, re-run from scratch with --reset" and exit.
- The provenance JSON's `kv_full_docs_count` MUST equal `len(paper_ids_attempted)`. The `log_graph_stats.py` helper should fail with a clear error if these diverge.
- For Phase 2 specifically: after Stage 1, the smoke set's three source papers (`2005.11401`, `2004.04906`, `2002.08909`) MUST be in `kv_store_full_docs.json` — verify this in a smoke-test pre-flight before running Stage 3.

**Warning signs:**
- Two consecutive `ingest_from_mineru.py --reset` runs produce graphml files of different sizes (>10% delta) → non-deterministic ingest, possibly LLM-flake-driven entity-extraction variance.
- `kv_store_full_docs.json` length less than the number of papers passed to `ingest_from_mineru_output` → silent skip due to missing `_content_list.json`.

### Pitfall 2: `img_path` relative paths produce empty image entities

**What goes wrong:** MineRU writes `_content_list.json` entries like `{"type":"image","img_path":"images/<sha>.jpg",...}` — relative to the per-paper output dir. RAG-Anything's `insert_content_list` silently accepts the relative form, builds an image entity in the KG, but the vdb_entities.json embeddings are placeholders because the file doesn't resolve at the resolver's CWD.

**Why it happens:** [CITED: `tier-4-multimodal/ingest_images.py:7-13` — Pitfall 4 of 130-RESEARCH explains this for the standalone-figures bundle]. RAG-Anything resolves `img_path` against an internal working directory (specifically `self.config.parser_output_dir`), not the caller's CWD. The existing `ingest_images.py` already handles this via `Path(...).resolve()`; the new `ingest_from_mineru.py` must do the same.

**How to avoid:** Pattern 1's `_absolutize_image_paths` helper is mandatory. Test it with a synthetic content_list containing both relative and already-absolute paths.

**Warning signs:**
- `vdb_entities.json` size grows during ingest (good — embeddings being written) but the file count of `images/<sha>.jpg` files referenced is much smaller than the number of `type=image` entries → image entities present in the KG but their embeddings are zero-vectors (unable to embed a missing file).

### Pitfall 3: MineRU CLI invocation in the sandbox crashes with shmem error

**What goes wrong:** Running `mineru -p PDF -o OUT` inside the orchestrator sandbox crashes with `OMP_NUM_THREADS` shmem allocation failure. This is the Phase 138/139 evidence.

**Why it happens:** [CITED: `evaluation/README.md:3` — "Phase 130 SC-1 — sandbox kernel-level OMP shmem block on MineRU; not solvable here"]. MineRU's PaddleOCR backend uses OpenMP shared-memory primitives that the sandbox does not permit at the kernel level. Not solvable from user space.

**How to avoid:** **Mandate** that `parse_missing_papers.py` runs on the host (not via the orchestrator). Document in the script's docstring, add a check at the top that `parse_missing_papers.py` MUST be run from a real terminal (e.g., `if "ALL_PROXY" in os.environ: print warning about sandbox`); the user's manual workflow is "open Terminal.app, `cd` to repo, `python tier-4-multimodal/scripts/parse_missing_papers.py`". The Stage 1 `ingest_from_mineru.py` is sandbox-safe because it only does HTTP API calls, no MineRU binary invocation.

**Warning signs:**
- Subprocess returns non-zero with stderr containing `OMP` or `shmem` or `bus error` → user is in the sandbox; abort and print the canonical "run on host" message.

### Pitfall 4: Storage cleanup is incomplete (leaves zombie kv_stores / vdb files)

**What goes wrong:** User runs `rm rag_anything_storage/tier-4-multimodal/graph_chunk_entity_relation.graphml` to "clear corruption," but leaves `kv_store_*.json` and `vdb_*.json` in place. The next `build_rag` reads the stale kv_stores, indexes them against a fresh graphml, and produces a non-deterministic graph state.

**Why it happens:** [CITED: `.planning/research/PITFALLS.md:325` — "LightRAG storage cleanup: Deleting graphml only / Must also clear `kv_store_*.json`, `vdb_*.json`, and any cache dir — otherwise zombie state"]. The LightRAG storage classes are independent — deleting one doesn't cascade.

**How to avoid:** Pattern 1's `--reset` flag uses `shutil.rmtree(working_dir)` — wholesale delete, mirrors `tier-4-multimodal/main.py:310`. Tests for the helper should explicitly delete a partial state and re-run to confirm `--reset` produces a byte-identical graphml on consecutive runs (sanity, not exact match — entity-extraction LLM has variance).

**Warning signs:**
- Post-rebuild graphml node count differs significantly between two `--reset` runs of the same input → zombie state, not LLM variance.

### Pitfall 5: `eval_capture.py` lacks `--smoke-question-ids` so Phase 2 burns 6× cost

**What goes wrong:** Phase 1 added `--smoke-question-ids` to `evaluation/harness/run.py` but `tier-4-multimodal/scripts/eval_capture.py` only has `--limit`. Naively running `eval_capture.py --limit 5` runs questions 1-5 of `golden_qa.json` (which is `single-hop-001..single-hop-005`), NOT the locked smoke set (3 single-hop + 2 multi-hop).

**Why it happens:** `eval_capture.py` was written before Phase 1 added the smoke-set discipline. It still does `qa[:args.limit]` at line 76.

**How to avoid:** Mirror Phase 1's flag exactly:
```python
DEFAULT_SMOKE_IDS = (...,)  # imported from evaluation.harness.run to avoid duplication
parser.add_argument("--smoke-question-ids", default=None, help="...")
# in _capture(): apply the same by_id filter that run.py uses at lines 295-302
```

**Warning signs:**
- Captured Tier 4 QueryLog has `single-hop-005` (LightRAG paper) instead of `multi-hop-001` → flag not used; output is a different smoke set; gate verdict is not comparable to Phase 1.

### Pitfall 6: Reproducibility — re-runs produce different graph state due to LLM variance

**What goes wrong:** Entity extraction during ingest goes through `google/gemini-2.5-flash` via OpenRouter. The LLM is non-deterministic at temperature > 0; even at temperature = 0, OpenRouter's routing may pick different upstream providers across runs. Two `ingest_from_mineru.py --reset` runs of the same input produce graphs of slightly different sizes.

**Why it happens:** Inherent to the design. LightRAG entity-extraction is one LLM call per chunk, and the prompt asks for free-form entity lists. Run-to-run variance of ±10% on node count is normal.

**How to avoid:**
- Capture `(graphml_node_count, graphml_edge_count, graphml_mtime, graphml_size_bytes)` in the provenance JSON for EVERY rebuild. The frozen-doc reader can then verify: "this graph at SHA $X had N nodes and M edges; that's the eval target."
- For Phase 2's smoke gate to be valid, the smoke runs against THIS SPECIFIC graph state — not against an idealized "what Tier 4 should look like." If the smoke fails and the graph stats look off, the rebuild itself is the suspect, not the adapter.
- Disclosure-level: the frozen doc must explicitly state "Tier 4 graph state is non-deterministic up to ±10% on node count run-to-run; the captured run is at $SHA with N nodes." This is per Pitfall 2 of PITFALLS.md disclosure-level prevention.

**Warning signs:**
- Provenance JSON shows graph stats wildly different from prior runs (e.g., 50% delta on node count) → not LLM variance; investigate (could be missing papers, partial ingest, or OpenRouter routing to a wildly different model).

---

## Code Examples

Verified patterns from the existing repo + library source:

### Existing: `insert_content_list` for standalone images (Phase 130)

```python
# Source: tier-4-multimodal/ingest_images.py:41-45 (existing)
await rag.insert_content_list(
    content_list=content_list,
    file_path="dataset_figures_bundle",  # synthetic source path for KG node
    doc_id="figures-bundle",
)
```

### New: `insert_content_list` for cached MineRU output (Phase 2)

```python
# Pattern 1 builds on the same API — only file_path / doc_id differ
await rag.insert_content_list(
    content_list=content_list,    # parsed from <paper>_content_list.json
    file_path=paper_id,           # e.g., "2005.11401" — used as citation source
    doc_id=paper_id,              # stable id; RAG-Anything dedup; (anti-pattern: omit)
)
```

### Existing: `process_document_complete` for live MineRU + ingest (Phase 130)

```python
# Source: tier-4-multimodal/ingest_pdfs.py:75-81 (existing)
await rag.process_document_complete(
    file_path=str(pdf),
    output_dir=str(Path("tier-4-multimodal/output") / p["paper_id"]),
    parse_method="auto",
    device=device,
    doc_id=p["paper_id"],
)
# NOTE: this is the "re-runs MineRU per paper" path. Phase 2 uses it ONLY
# for the 4 missing-papers top-up via parse_missing_papers.py — and even
# then only for the standalone CLI invocation in Pattern 2, not via this
# Python API (the Python API requires the OPENROUTER_API_KEY for the
# entity-extraction LLM step that runs immediately after MineRU; we want
# to separate parse-only from ingest-with-LLM).
```

### New: smoke gate verdict consumption (Phase 2 Stage 4)

```python
# Source: evaluation/harness/smoke_gate.py:84-186 (existing, no changes — Phase 1 locked)
# CLI invocation (Phase 2 user docs):
$ python -m evaluation.harness.smoke_gate --tier 4
# Returns: SmokeGateResult { verdict: PASS|FAIL|INCONCLUSIVE, n_total: 5, ... }
# Exit codes: 0=PASS, 1=FAIL, 3=INCONCLUSIVE
```

### Existing: cost-surprise gate (Phase 130)

```python
# Source: tier-4-multimodal/main.py:138-152 (existing)
def _confirm_or_abort(prompt: str, yes: bool, console: Console) -> bool:
    """Cost-surprise gate. CI-safe: returns False on EOFError / blank input."""
    if yes:
        console.print(f"[dim]{prompt}  [confirmed via --yes][/dim]")
        return True
    console.print(f"[yellow]{prompt}[/yellow]")
    try:
        answer = input("Continue? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer == "y"

# Pattern 1's ingest_from_mineru.py should use this verbatim. The cost
# surprise prompt should warn: "Stage 1 ingest will run entity-extraction
# LLM calls over ~75-100 papers (~$0.50-1.00 corpus-wide; ~$0.05 for the
# 3-paper smoke subset). Re-runs are near-free thanks to RAG-Anything's
# kv_store_llm_response_cache.json." (matches main.py:336-340 wording).
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Run MineRU inside the orchestrator sandbox via Docker | Run MineRU on the host directly; ingest cached JSON via `insert_content_list` | Phase 130 SC-1 (deferred 2026-04-28; recorded in `evaluation/README.md` and STATE.md) | Phase 2 inherits this constraint. The Docker path STILL EXISTS in `tier-4-multimodal/Dockerfile` and is the recommended UX for the live `main.py` flow; what changed is that the EVAL HARNESS uses cached JSON, not live ingest. |
| Captured Tier 4 with `tier-4-multimodal/output/` matching all 100 papers | Captured Tier 4 with 75/100 papers in cache; 4 of 25 missing referenced in golden_qa | Some time before 2026-05-03 (when `rag_anything_storage/tier-4-multimodal/` was wiped) | Phase 2 must run the missing-papers top-up. For SMOKE only, the existing 75 are sufficient (smoke source papers `2005.11401, 2004.04906, 2002.08909` all present); for the FULL Phase 7 rerun, all 4 golden-referenced papers must be parsed. |
| `_content_list.json` (MineRU v1 format) only | Both `_content_list.json` (v1) and `_content_list_v2.json` (v2) present in output | MineRU 3.x upgrade (current installed version is 3.1.4) | RAG-Anything 1.2.10 reads ONLY the v1 format. Pattern 1's glob explicitly excludes `_v2` filenames. |

**Deprecated/outdated:**
- The 2026-05-02 captured Tier 4 QueryLog (`tier-4-2026-05-02T17_44_53Z.json`) and the older `tier-4-2026-05-02T11_42_14Z.json` — both produced 30/30 errors. Phase 2's new capture supersedes them; they should NOT be used as `--tier-4-from-cache` input.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The existing 75 `_content_list.json` files were produced by a recent enough MineRU/RAG-Anything pair to be readable by 1.2.10's `insert_content_list`. | Pattern 1 | If wrong, ingest fails with a JSON validation error and Phase 2 must re-parse all 100 papers (still feasible — just 10× more wall time). The mitigation is to do a single-paper smoke ingest test before the full loop. |
| A2 | The existing 75 `_content_list.json` files were produced by MineRU with `backend=hybrid-*` (the default in `raganything==1.2.10`). | Pattern 2 (recommends `-b hybrid-auto-engine` for the top-up to match) | [VERIFIED 2026-05-04 against `mineru/cli/output_paths.py:24-25` — `if backend.startswith("hybrid"): output_root / pdf_name / f"hybrid_{parse_method}"` — and `raganything/parser.py:1019` ("hybrid-* -> hybrid_auto/"). The existing 75 papers have `hybrid_auto/` subdirectories; this naming is exclusively produced when MineRU is invoked with a `hybrid-*` backend AND `-m auto`.] Pattern 2 must use `-b hybrid-auto-engine` (the MineRU default per `mineru --help`, which says "Without method specified, hybrid-auto-engine will be used by default") to produce byte-identical layout for the missing-papers top-up. STACK.md's `-b pipeline` suggestion (line 107) is superseded by this RESEARCH.md. |
| A3 | The smoke gate's `--min-measurable=3` default is appropriate for Tier 4 smoke (same as Tier 5 smoke). | smoke_gate inheritance | If wrong, Tier 4's failure modes (e.g., MineRU partial parse → empty contexts on multimodal questions) might leave the gate INCONCLUSIVE more often than expected. Phase 1's CONTEXT D-04 was written in a Tier 5 frame; Tier 4's smoke is text-only on its 5 questions (none are multimodal), so the same threshold should apply. |
| A4 | The 4 papers needed by golden_qa but missing from MineRU output (`1909.01066`, `2002.06177`, `2309.15217`, `2410.05779`) are parseable by MineRU 3.1.4. | Pattern 2 + missing-papers top-up | If any of these PDFs is corrupt or has a format MineRU rejects, Phase 2 must descope that question class (or retry with a different `-m` method). All 4 PDFs exist on disk — but parse-ability is unverified. The mitigation is to test-parse one paper first, before running the full top-up. |
| A5 | `eval_capture.py --smoke-question-ids` will be a clean addition (no dependency reorg). | Pitfall 5 | The script already imports from `evaluation.harness.records` — adding an import of `evaluation.harness.run.DEFAULT_SMOKE_IDS` is symmetric. No risk. [VERIFIED via grep — no circular import risk] |
| A6 | Full corpus-wide ingest cost is ~$0.50–1.00 (entity-extraction + vision passes + embeddings on 100 papers). | Cost-surprise gate copy | This number is from `tier-4-multimodal/README.md` line 95 (vintage 2026-04). The actual cost will be logged by the existing `CostTracker` infrastructure; ballpark for the gate copy is acceptable. |

**If this table is empty:** N/A — there are 6 explicit assumptions above. Most are LOW risk (verifiable in 5 minutes during planning) or already verified in this session; A4 is MEDIUM and should be confirmed in plan-checker stage with one test parse.

---

## Open Questions

1. **Should `--smoke-question-ids` live in `eval_capture.py` or should `eval_capture.py` always capture all 30 and let `harness.run --tier-4-from-cache` filter?**
   - What we know: the existing `--tier-4-from-cache` path in `harness/run.py` reads the WHOLE QueryLog and re-emits it under the canonical filename. The harness's `--smoke-question-ids` filter applies to `_load_golden_qa()` output, NOT to the cached QueryLog. So passing `--smoke-question-ids` with `--tier-4-from-cache` filters the QA list down to 5 IDs and only re-emits records that match.
   - What's unclear: if `eval_capture.py` captures all 30, the cost is ~$0.05 instead of the smoke's ~$0.01 — still cheap, but 5× more. For Phase 2's purpose (smoke-only verification), capturing 30 is wasteful. For Phase 7 (full rerun), `eval_capture.py` MUST capture all 30. The cleanest: add `--smoke-question-ids` to `eval_capture.py` as well, mirroring `harness/run.py`.
   - Recommendation: **Add `--smoke-question-ids` to `eval_capture.py`.** Mirrors Phase 1's symmetry; matches the locked design. Phase 7 just doesn't pass the flag.

2. **What happens if the MineRU CLI for the missing 4 papers fails on one paper but succeeds on the others? Stop or skip?**
   - What we know: `parse_missing_papers.py` is a deterministic loop; per-paper subprocess success is independent.
   - What's unclear: a partial top-up (3/4 papers parsed; 1 missing) means one golden_qa question (e.g., `multi-hop-008` if `1909.01066` fails) will hit a graph that doesn't have its source paper indexed. The smoke set doesn't include this question (the 5 IDs only hit `2005.11401, 2004.04906, 2002.08909`), so smoke would still PASS — but Phase 7's full rerun would have a measurable hole.
   - Recommendation: **Continue on per-paper failure; log to stderr; exit 0 if ALL papers are processed successfully OR if every failure is non-essential (no golden-qa references that paper); exit non-zero if a golden-qa-referenced paper failed.** Phase 2 just needs the smoke set's papers; Phase 7 should re-attempt the failed papers before its full rerun.

3. **Does the new `ingest_from_mineru.py` need its own cost-surprise gate, or is the user expected to know what they're doing?**
   - What we know: `tier-4-multimodal/main.py:299-310` has a `--reset` cost-surprise gate before the wipe; `main.py:330-343` has a first-ingest cost-surprise gate. These are mature.
   - What's unclear: `ingest_from_mineru.py` is invoked manually by the user during Phase 2; the user is in a known state (wants to rebuild). But Phase 7's pipeline driver will invoke it programmatically — the gate then becomes annoyance if `--yes` is implicit in CI.
   - Recommendation: **Inherit the gate verbatim; add `--yes`. Phase 7's pipeline driver passes `--yes`.** This is the same pattern Phase 1's adapter walk uses with `tier_5.py` cost gates; consistent.

4. ~~**For the missing-papers MineRU pass, should we use `-b pipeline` or `-b hybrid-auto-engine`?**~~ — RESOLVED in this RESEARCH.md (see A2): use `-b hybrid-auto-engine`. Verified 2026-05-04 against `mineru/cli/output_paths.py:24-25` and the existing 75-paper output's universal `hybrid_auto/` directory naming.

5. **Should the rebuild include the standalone-figures bundle (`dataset/images/` via `ingest_images.py`)?**
   - What we know: The full `tier-4-multimodal/main.py --ingest` runs BOTH PDF ingest AND standalone image ingest. The previous (failed) capture was post-storage-wipe, so the standalone-figures bundle was NOT in the graph either — i.e., the 30/30 baseline failure was from missing storage, not from the figures-bundle question.
   - What's unclear: 7 of 10 multimodal-class questions reference figures from the standalone bundle (figure_id present in golden_qa); these are NOT in the smoke set, but they ARE in the full 30. The full rerun (Phase 7) needs the figures bundle ingested too.
   - Recommendation: **Phase 2 ingests BOTH the cached MineRU PDFs AND the standalone figures bundle.** Smoke set doesn't need figures, but the cost of including them is small (~$0.30–0.50 vision pass per `tier-4-multimodal/README.md`) and doing it now means Phase 7 doesn't need a separate ingest step. `ingest_from_mineru.py` should call BOTH `ingest_from_mineru_output(rag, ...)` AND `ingest_standalone_images(rag, dataset_root)`.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | All | ✓ | 3.13 (per `.venv/lib/python3.13`) | — |
| `mineru` CLI | Pattern 2 (host-side missing-papers top-up) | ✓ on host | 3.1.4 | NONE — without `mineru`, the 4 golden-qa-referenced papers cannot be parsed; descope those questions or run on a different host |
| `raganything==1.2.10` | Pattern 1 (insert_content_list) | ✓ | 1.2.10 | — |
| `lightrag-hku==1.4.15` | KG persistence | ✓ | 1.4.15 | — |
| `OPENROUTER_API_KEY` | Stage 1 ingest LLM calls + Stage 3 capture | ✓ (per `.env`) | — | NONE — without the key, ingest cannot run |
| `git` | Provenance JSON (`git rev-parse`) | ✓ (we are in a git repo) | — | Use literal "unknown" string per existing `_git_sha()` helper at `eval_capture.py:55-61` |
| `networkx` | Pattern 3 (graphml stats) | ✓ (transitive via lightrag-hku) | — | — |
| MineRU model cache (~3-5 GB on first run) | Pattern 2 — only if running from a fresh checkout | UNKNOWN — depends on user's host | — | The first `mineru` invocation downloads from HuggingFace via `MINERU_MODEL_SOURCE=huggingface` (default). 5-15 min cold start per `tier-4-multimodal/README.md:1-3`. Plan for this delay. |
| Sandbox vs. host environment | Pattern 2 (MineRU CLI) MUST run on host | UNKNOWN until invocation | — | None — Pattern 2 is host-only by design (Pitfall 3). The orchestrator MUST hand off to the user for this step. |

**Missing dependencies with no fallback:**
- `mineru` CLI on a host that supports OpenMP shmem (i.e., NOT the orchestrator sandbox). User must run Stage 0 manually outside the sandbox. Document in Phase 2's user-facing handoff.

**Missing dependencies with fallback:**
- MineRU model cache: cold start 5-15 min on first run; subsequent runs are instant.
- Stage 0 entirely (the missing-papers top-up): if user accepts shipping Phase 2 with only the 75 cached papers, they can defer Stage 0 to Phase 7 prep. This is acceptable for Phase 2 SHIP because the smoke set doesn't need any of the missing papers.

---

## Validation Architecture

(`workflow.nyquist_validation` is absent from `.planning/config.json` — treating as enabled per researcher contract.)

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.x (per `pyproject.toml:84`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (markers: `live`) |
| Quick run command | `pytest evaluation/tests/test_eval_tier4.py -x -q` (non-live; ~1s) |
| Full suite command | `pytest evaluation/tests/ tier-4-multimodal/tests/ -x` (excluding `-m live`) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TIER-02 | `ingest_from_mineru.py` reads cached JSON, calls `insert_content_list` per paper, writes graphml | unit | `pytest tier-4-multimodal/tests/test_ingest_from_mineru.py -x` | ❌ Wave 0 |
| TIER-02 | `_absolutize_image_paths` resolves relative `img_path` against per-paper images_dir | unit | `pytest tier-4-multimodal/tests/test_ingest_from_mineru.py::test_absolutize_image_paths -x` | ❌ Wave 0 |
| TIER-02 | `parse_missing_papers.py` skips already-parsed papers, runs `mineru` subprocess for missing | unit (subprocess mocked) | `pytest tier-4-multimodal/tests/test_parse_missing.py -x` | ❌ Wave 0 |
| TIER-02 | `log_graph_stats.py` reads graphml + kv_stores, writes Pydantic-typed JSON | unit (tmp_path fixture) | `pytest tier-4-multimodal/tests/test_log_graph_stats.py -x` | ❌ Wave 0 |
| TIER-02 | End-to-end: wipe → ingest 1 paper from cached JSON → graphml exists with N>0 nodes | live | `pytest tier-4-multimodal/tests/test_tier4_e2e_live.py::test_ingest_from_mineru_e2e -x -m live` | ❌ Wave 0 (extends existing test_tier4_e2e_live.py) |
| TIER-03 | `eval_capture.py --smoke-question-ids ID1,ID2,...` filters golden_qa to those IDs only | unit | `pytest tier-4-multimodal/tests/test_eval_capture.py::test_smoke_filter -x` | ❌ Wave 0 |
| TIER-03 | smoke_gate verdict == PASS on freshly rebuilt graph for the 5 smoke IDs | live | `pytest evaluation/tests/test_eval_smoke_live.py::test_eval_smoke_tier4_full_pipeline -x -m live` | ❌ Wave 0 (mirrors `test_eval_smoke_tier5_full_pipeline`) |

### Sampling Rate

- **Per task commit:** `pytest tier-4-multimodal/tests/ evaluation/tests/test_eval_tier4.py -x -q` (~3-5s; non-live, fail-fast)
- **Per wave merge:** Above + `pytest evaluation/tests/test_eval_smoke_gate.py evaluation/tests/test_eval_run.py -x` (no live tests; ~10s)
- **Phase gate:** Live smoke `pytest -m live evaluation/tests/test_eval_smoke_live.py::test_eval_smoke_tier4_full_pipeline` THEN `python -m evaluation.harness.smoke_gate --tier 4` returns verdict=PASS (gate-of-record per Phase 1 D-04 inheritance)

### Wave 0 Gaps

- [ ] `tier-4-multimodal/tests/test_ingest_from_mineru.py` — covers TIER-02 (unit tests for the new ingest helper + `_absolutize_image_paths`)
- [ ] `tier-4-multimodal/tests/test_parse_missing.py` — covers TIER-02 (subprocess-mocked tests for the missing-papers helper)
- [ ] `tier-4-multimodal/tests/test_log_graph_stats.py` — covers TIER-02 (Pydantic shape + tmp_path fixture)
- [ ] `tier-4-multimodal/tests/test_eval_capture.py` — covers TIER-03 (smoke filter unit test); may already exist as part of the eval_capture.py module — verify during planning
- [ ] Extension of `evaluation/tests/test_eval_smoke_live.py` with `test_eval_smoke_tier4_full_pipeline` (live, mirrors the Tier 5 version Plan 01-02 added) — covers TIER-03 live verification
- [ ] Extension of `tier-4-multimodal/tests/test_tier4_e2e_live.py` with `test_ingest_from_mineru_e2e` — covers TIER-02 live verification

---

## Security Domain

(`security_enforcement` not set in `.planning/config.json` — treating as enabled.)

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No user auth in this phase |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No multi-user state |
| V5 Input Validation | yes | Pydantic v2 models on all new persisted artifacts (provenance JSON); `subprocess.run([...], check=True)` with list-form args (no shell=True) for MineRU CLI invocation in `parse_missing_papers.py` |
| V6 Cryptography | no | No new crypto; OpenRouter API key is the only secret, already managed via existing `.env` + pydantic-settings + `SecretStr` pattern in `shared.config` |
| V7 Error Handling | yes | `MineruExecutionError` from `raganything/parser.py:51` is the existing typed exception for MineRU failures; the new `parse_missing_papers.py` should catch and re-raise with a clear "host vs sandbox" message (Pitfall 3) |
| V8 Data Protection | yes (light) | The provenance JSON is committed to git (per `commit_docs: true` in config); contains git_sha, library versions, graphml metadata — no secrets. Verify the OPENROUTER_API_KEY is NOT echoed into any log or JSON output. |
| V12 Files | yes | All file paths in the new helpers must be `pathlib.Path` and absolute where they cross helper boundaries (Pattern 1's `_absolutize_image_paths` is the central example); never `open(user_supplied_string)` without resolution |
| V14 Configuration | yes | The new helpers MUST honor the existing `WORKING_DIR` constant at `tier_4_multimodal.rag` (line 49: `"rag_anything_storage/tier-4-multimodal"`) — never accept a `--working-dir` flag that lets the user mistakenly write into Tier 3's storage |

### Known Threat Patterns for {python + subprocess + LLM-LiteLLM stack}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Subprocess command injection via paper filename | Tampering | `subprocess.run` with list-form args (NOT shell=True); paper filenames come from `dataset/manifests/papers.json` which is a curated artifact, not user-controlled |
| OPENROUTER_API_KEY leakage in logs / JSON dumps | Information Disclosure | Existing `pydantic_settings.SecretStr` wrapping in `shared.config:settings.openrouter_api_key`; `get_secret_value()` is only called immediately before forwarding to `os.environ` — never echoed |
| Path traversal via `img_path` in untrusted content_list | Tampering | Pattern 1's `_absolutize_image_paths` resolves against the per-paper `images_dir` only — does not allow `../../../etc/passwd` style escapes (verifiable test) |
| Untrusted LLM response disrupting graph state | Tampering / DoS | Mitigated upstream by LightRAG's existing JSON-output retry path; partial-ingest pitfall (Pitfall 1 above) covers the recovery |
| Missing-file silent failure (image not on disk → empty embedding) | Information Disclosure / Tampering | Pattern 1 explicitly skips entries whose `img_path` does not resolve; provenance JSON's `kv_full_docs_count` < expected count is the visible signal |
| Sandbox kernel boundary crossed by mineru subprocess | Tampering / DoS | Pitfall 3 mitigation — `parse_missing_papers.py` checks for sandbox-indicator env vars (`ALL_PROXY` containing `socks5h://`) and aborts with a clear "run on host" message |

---

## Sources

### Primary (HIGH confidence)

- **Installed source** at `.venv/lib/python3.13/site-packages/raganything/processor.py:1508` (`process_document_complete`) and `:1868` (`insert_content_list`) — verified 2026-05-04 via `Read` tool
- **Installed source** at `.venv/lib/python3.13/site-packages/raganything/parser.py:613` (`_run_mineru_command`) and `:848` (`read_files`) — flag set + output naming verified
- **Installed source** at `.venv/lib/python3.13/site-packages/lightrag/kg/networkx_impl.py:38-67` — graphml writer (`nx.write_graphml`) and per-init reader; namespace pattern (`graph_<namespace>.graphml`); verified the file is `graph_chunk_entity_relation.graphml` for the default namespace
- **Installed source** at `.venv/lib/python3.13/site-packages/lightrag/kg/json_kv_impl.py:40` — `kv_store_<namespace>.json` naming
- **Installed source** at `.venv/lib/python3.13/site-packages/lightrag/kg/nano_vector_db_impl.py:55` — `vdb_<namespace>.json` naming
- **MineRU 3.1.4 CLI** verified via `.venv/bin/mineru --help` and `--version` — flag set: `-p`, `-o`, `-m {auto,txt,ocr}`, `-b {pipeline,vlm-...,hybrid-...}`, `-l`, `-s`, `-e`, `-f`, `-t`, `-d`, `-u`
- **Repo files**:
  - `tier-4-multimodal/main.py` — existing live ingest CLI; cost-surprise + reset patterns
  - `tier-4-multimodal/rag.py` — `build_rag()` factory; locked `WORKING_DIR = "rag_anything_storage/tier-4-multimodal"`
  - `tier-4-multimodal/ingest_pdfs.py` — existing `process_document_complete` loop pattern
  - `tier-4-multimodal/ingest_images.py` — existing `insert_content_list` pattern + Pitfall 4 absolute-path handling
  - `tier-4-multimodal/scripts/eval_capture.py` — existing 30-Q capture; what Phase 2 extends with `--smoke-question-ids`
  - `evaluation/harness/run.py` — existing `--tier-4-from-cache` + `--smoke-question-ids` + `DEFAULT_SMOKE_IDS` (Phase 1)
  - `evaluation/harness/smoke_gate.py` — existing tier-agnostic gate (Phase 1)
  - `evaluation/harness/adapters/tier_4.py` — existing dual-mode adapter (cached primary, library fallback)
  - `evaluation/harness/records.py` — Pydantic model conventions
  - `evaluation/golden_qa.json` — 30-question manifest; smoke 5 IDs and source papers verified
  - `dataset/manifests/papers.json` — 100-paper manifest; missing-papers cross-check verified
  - `dataset/manifests/figures.json` — 581-figure manifest; standalone bundle source
  - `tier-4-multimodal/output/` — existing 75-paper MineRU output cache (verified by directory listing)
  - `lightrag_storage/tier-3-graph/` — reference layout for Tier 3 storage (mirror of what Tier 4 will produce)
- **Captured evidence** of the 30/30 baseline at `evaluation/results/queries/tier-4-2026-05-02T17_44_53Z.json` — verified 30/30 errors with `"ValueError: No LightRAG instance available"`

### Secondary (MEDIUM confidence)

- `.planning/research/STACK.md` — multi-source-validated stack research (lines 90-160 cover the MineRU outside-sandbox pattern and storage cleanup procedure); written by Phase 0 with explicit Context7 / official-docs citations
- `.planning/research/PITFALLS.md` Pitfall 2 (lines 47-77) — multi-source MEDIUM-to-HIGH confidence on the partial-ingest non-determinism warning; Pitfall 11 (line 325) on storage-cleanup completeness; the canonical Phase 139 evidence pointer at line 326
- `evaluation/README.md:3` — explicit Phase 130 SC-1 deferral language: "Tier 4 is deferred to the user (Phase 130 SC-1 — sandbox kernel-level OMP shmem block on MineRU; not solvable here)"
- `tier-4-multimodal/README.md` — cost ballparks + Docker recommendation; vintage 2026-04
- `.planning/phases/01-tier-5-adapter-fix/01-CONTEXT.md` D-03 / D-04 — same 5 smoke IDs reused across Phase 1 + Phase 2; smoke gate threshold = ≥0.8 ratio on measurable subset

### Tertiary (LOW confidence — flagged for validation)

- A4 (the 4 missing papers are MineRU-parseable) — the PDFs exist on disk; parse-ability is unverified. One test parse is the mitigation.
- The exact node/edge count of the rebuilt graph (no prior baseline because the storage was empty before this phase) — CANNOT BE VERIFIED until rebuild runs. Phase 2's provenance JSON IS the ground truth; subsequent phases compare against it.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every library version was verified by reading the dist-info METADATA in `.venv` 2026-05-04; `mineru --version` ran successfully; APIs were verified by reading the installed source
- Architecture: HIGH — the data flow uses public RAG-Anything + LightRAG APIs whose contracts are stable across the pinned versions; the smoke gate is inherited verbatim from Phase 1 (already verified PASS on Tier 5 — 2026-05-04)
- Pitfalls: HIGH — Pitfalls 1-4 are direct citations from `.planning/research/PITFALLS.md` Pitfall 2 (multi-source-validated); Pitfall 5 (`--smoke-question-ids` gap in `eval_capture.py`) is verified by reading `eval_capture.py` directly; Pitfall 6 (LLM variance) is inherent to the design and uncontroversial
- Open Questions: 5 questions; OQ1, OQ2, OQ3 are decisions Claude can make autonomously during planning (recommendations given). OQ4 is a 60-second verification task. OQ5 is a recommendation that should be confirmed with the user before plan-checker (CONTEXT.md does not exist for this phase, so the user has not pre-decided on figures-bundle inclusion).

**Research date:** 2026-05-04
**Valid until:** 2026-06-03 (30 days — stable infrastructure; only invalidator is a `raganything` or `lightrag-hku` major version bump, neither of which is on the roadmap)

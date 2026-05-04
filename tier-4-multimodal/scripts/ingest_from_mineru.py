"""Tier 4 — wipe-and-rebuild helper that ingests cached MineRU output.

Purpose
-------
Re-creates ``rag_anything_storage/tier-4-multimodal/`` from the 75 papers
already parsed under ``tier-4-multimodal/output/`` (Stage 1 of
.planning/phases/02-tier-4-graphml-regeneration/02-RESEARCH.md). Calls
RAG-Anything's public ``insert_content_list`` API once per paper.

Sandbox-safe: does NOT invoke MineRU as a subprocess. Only HTTP calls
(entity-extraction LLM passes via OpenRouter + embeddings + vision).

Patterns + pitfalls (cited from 02-RESEARCH.md)
-----------------------------------------------
* **Pattern 1 — `_absolutize_image_paths`**: relative ``img_path`` values
  in ``_content_list.json`` are resolved against the per-paper
  ``images/`` dir BEFORE insert. Without this, RAG-Anything resolves the
  path against its own internal CWD and produces zero-vector image
  embeddings (same failure mode as ``ingest_images.py`` Pitfall 4).
* **Pitfall 4 — wholesale rmtree on `--reset`**: the script
  ``shutil.rmtree(RAG_STORAGE)`` rather than moving / renaming. LightRAG
  storage is not safe to merge across runs (entity index dim must match
  the embedding dim at first ingest; switching dim corrupts retrieval).
* **Anti-pattern — `_content_list_v2.json`**: RAG-Anything 1.2.10 reads
  the v1 schema only. The v2 schema (also produced by MineRU) is silently
  ignored upstream; we filter it out at glob time so the helper never
  hands a v2 file to ``insert_content_list``.
* **Anti-pattern — omitting `doc_id`**: without an explicit ``doc_id``,
  RAG-Anything content-hashes the body. Re-runs with non-deterministic
  text ordering produce different doc IDs, breaking the dedup contract.
  We pin ``doc_id=paper_id`` so re-ingests are idempotent.
* **Pitfall 1 — per-paper exception handling**: a single paper that
  crashes (rate limit, malformed content_list, API outage) MUST NOT abort
  the whole loop. Log the failure and continue; downstream provenance
  (log_graph_stats.py) records the actual ingested count.

Usage
-----
::

    # Full corpus rebuild (typical Phase 2 invocation):
    python tier-4-multimodal/scripts/ingest_from_mineru.py --reset --yes

    # Smoke-only subset (Phase 7 enhancement preview):
    python tier-4-multimodal/scripts/ingest_from_mineru.py --reset --yes \\
        --paper-ids 2005.11401,2004.04906,2002.08909

    # Custom MineRU root (used by tests + Plan 02-02 fresh-MineRU runs):
    python tier-4-multimodal/scripts/ingest_from_mineru.py --reset --yes \\
        --mineru-output-root /path/to/other/output

Cost
----
~$0.50–1.00 corpus-wide for the 75-paper cache (entity extraction is the
bottleneck; embeddings are batch-cheap, vision passes only on figure
captions). Re-runs at the same git SHA hit ``kv_store_llm_response_cache``
and cost effectively $0 — but ``--reset`` wipes that cache, so a forced
rebuild always pays full price.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console  # noqa: E402

from shared.config import get_settings  # noqa: E402
from shared.cost_tracker import CostTracker  # noqa: E402
from tier_4_multimodal.cost_adapter import CostAdapter  # noqa: E402
from tier_4_multimodal.rag import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    WORKING_DIR,
    build_rag,
)

try:  # noqa: SIM105 — load_dotenv is not strictly required if env is pre-set
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass


MINERU_OUTPUT_ROOT = _REPO_ROOT / "tier-4-multimodal" / "output"
RAG_STORAGE = _REPO_ROOT / "rag_anything_storage" / "tier-4-multimodal"


# ---------------------------------------------------------------------------
# Pure helpers (testable without network)
# ---------------------------------------------------------------------------


def _absolutize_image_paths(
    content_list: list[dict], images_dir: Path
) -> list[dict]:
    """Resolve relative ``img_path`` against ``images_dir`` (Pattern 1).

    Returns a NEW list; does NOT mutate the input. Non-image entries pass
    through unchanged. Already-absolute paths pass through unchanged. For
    relative paths we use ``Path(item["img_path"]).name`` to drop any
    ``..`` segments — this is the path-traversal mitigation cited in
    Plan 02-01's threat register (T-02-01-01).
    """
    images_dir = Path(images_dir).resolve()
    out: list[dict] = []
    for item in content_list:
        if item.get("type") != "image":
            out.append(item)
            continue
        raw = item.get("img_path", "")
        if not raw:
            out.append(item)
            continue
        rel = Path(raw)
        if rel.is_absolute():
            out.append(item)
            continue
        # Drop any "../" segments by taking only the basename — relative
        # paths in MineRU output are always of the form "images/<sha>.<ext>".
        absolute = (images_dir / rel.name).resolve()
        new_item = dict(item)
        new_item["img_path"] = str(absolute)
        out.append(new_item)
    return out


def _find_content_list(paper_root: Path) -> Path | None:
    """Return the v1 content_list.json under ``paper_root``, or None.

    Anti-pattern guard: filters out filenames containing ``"_v2"`` because
    RAG-Anything 1.2.10 reads the v1 schema only (RESEARCH.md anti-pattern).
    Determinism: ``sorted()`` first match.
    """
    if not paper_root.exists():
        return None
    matches = [
        p
        for p in paper_root.glob("**/*_content_list.json")
        if "_v2" not in p.name
    ]
    if not matches:
        return None
    return sorted(matches)[0]


def _discover_papers(mineru_output_root: Path) -> list[str]:
    """Return sorted paper_ids that have a v1 content_list under the root."""
    if not mineru_output_root.exists():
        return []
    discovered: list[str] = []
    for child in mineru_output_root.iterdir():
        if not child.is_dir():
            continue
        if _find_content_list(child) is not None:
            discovered.append(child.name)
    return sorted(discovered)


def _confirm_or_abort(prompt: str, yes: bool, console: Console) -> bool:
    """Cost-surprise gate (verbatim from tier-4-multimodal/main.py).

    Mirrors the project convention: when ``--yes`` is passed we log a
    confirmation note and proceed; otherwise we read stdin and require a
    literal "y" answer. EOFError (non-interactive shell without --yes)
    aborts cleanly — the caller must opt in via --yes.
    """
    if yes:
        console.print(f"[dim]{prompt}  [confirmed via --yes][/dim]")
        return True
    console.print(f"[yellow]{prompt}[/yellow]")
    try:
        answer = input("Continue? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer == "y"


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------


async def ingest_from_mineru_output(
    rag,
    mineru_output_root: Path,
    paper_ids: list[str],
    console: Console,
) -> int:
    """Loop over ``paper_ids`` and feed each paper's content_list to RAG-Anything.

    Per-paper exceptions are logged and skipped (Pitfall 1 of RESEARCH.md).
    Returns the count of papers successfully ingested.
    """
    ingested = 0
    total = len(paper_ids)
    for i, pid in enumerate(paper_ids, 1):
        paper_root = mineru_output_root / pid
        content_list_path = _find_content_list(paper_root)
        if content_list_path is None:
            console.print(
                f"[yellow][{i}/{total}] {pid}: no v1 content_list found — skipping.[/yellow]"
            )
            continue
        try:
            content_list = json.loads(content_list_path.read_text())
            images_dir = content_list_path.parent / "images"
            content_list = _absolutize_image_paths(content_list, images_dir)
            console.print(
                f"[dim][{i}/{total}] {pid}: ingesting {len(content_list)} entries...[/dim]"
            )
            await rag.insert_content_list(
                content_list=content_list,
                file_path=pid,  # citation source
                doc_id=pid,  # pin doc_id (anti-pattern: omit)
            )
            ingested += 1
        except Exception as exc:  # noqa: BLE001 — Pitfall 1: log + continue
            console.print(
                f"[red][{i}/{total}] {pid}: ingest FAILED ({exc.__class__.__name__}: {exc})[/red]"
            )
            continue
    return ingested


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest_from_mineru",
        description=(
            "Tier 4 wipe-and-rebuild helper. Re-ingests cached MineRU output "
            "into rag_anything_storage/tier-4-multimodal/ via RAG-Anything's "
            "insert_content_list. Sandbox-safe (no MineRU subprocess; only "
            "HTTP calls)."
        ),
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help=(
            f"Wholesale shutil.rmtree of {RAG_STORAGE} BEFORE ingest "
            "(Pitfall 4). Without --reset the script aborts when storage "
            "is non-empty."
        ),
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the cost-surprise prompt (~$0.50–1.00 full corpus).",
    )
    p.add_argument(
        "--paper-ids",
        type=str,
        default=None,
        help=(
            "Optional comma-separated subset of paper IDs (e.g. "
            "'2005.11401,2004.04906'). Default = all discovered."
        ),
    )
    p.add_argument(
        "--mineru-output-root",
        type=str,
        default=str(MINERU_OUTPUT_ROOT),
        help=(
            f"Root of the MineRU cache (default {MINERU_OUTPUT_ROOT}). "
            "Override for tests / Plan 02-02 fresh-MineRU runs."
        ),
    )
    return p


async def amain(args, console: Console) -> int:
    """Async entry point. Returns the process exit code."""
    settings = get_settings()
    if not settings.openrouter_api_key:
        console.print(
            "[red]OPENROUTER_API_KEY not set — Tier 4 ingest cannot run. "
            "Copy .env.example to .env and set your key from "
            "https://openrouter.ai/keys[/red]"
        )
        return 2

    # Forward the SecretStr value to process env so the LightRAG closures
    # in tier_4_multimodal/rag.py see it on first call. pydantic-settings
    # populates ``settings.openrouter_api_key`` from .env but does NOT
    # mutate process env automatically (mirrors main.py:287-289). Tests
    # may stub ``settings`` with a plain string value; handle both.
    raw_key = settings.openrouter_api_key
    if hasattr(raw_key, "get_secret_value"):
        raw_key = raw_key.get_secret_value()
    os.environ["OPENROUTER_API_KEY"] = str(raw_key)

    mineru_output_root = Path(args.mineru_output_root)
    if args.paper_ids:
        paper_ids = [p.strip() for p in args.paper_ids.split(",") if p.strip()]
    else:
        paper_ids = _discover_papers(mineru_output_root)

    if not paper_ids:
        console.print(
            f"[red]No MineRU output found at {mineru_output_root} — nothing to ingest.[/red]"
        )
        console.print(
            "[red]Either point --mineru-output-root at the right cache, or run "
            "Plan 02-02's fresh-MineRU pass first.[/red]"
        )
        return 2

    # Cost-surprise gate (~$0.50–1.00 corpus-wide; ~$0.05 for 3-paper smoke).
    cost_prompt = (
        f"About to ingest {len(paper_ids)} paper(s) into {RAG_STORAGE}. "
        "First-time corpus rebuild costs ~$0.50–1.00 in entity-extraction "
        "LLM + embedding API calls. Re-runs reuse "
        "kv_store_llm_response_cache.json and cost ~$0 unless --reset wipes it."
    )
    if not _confirm_or_abort(cost_prompt, args.yes, console):
        console.print("[yellow]Aborted by user.[/yellow]")
        return 1

    # Storage-state gate (Pitfall 4 zombie-state guard).
    if args.reset:
        if RAG_STORAGE.exists():
            shutil.rmtree(RAG_STORAGE)
            console.print(f"[cyan]Wiped {RAG_STORAGE} (--reset).[/cyan]")
    else:
        if RAG_STORAGE.exists() and any(RAG_STORAGE.iterdir()):
            console.print(
                f"[red]Storage non-empty at {RAG_STORAGE}; pass --reset to wipe and "
                "rebuild (Pitfall 4 zombie-state guard).[/red]"
            )
            return 2

    tracker = CostTracker("tier-4-ingest")
    adapter = CostAdapter(
        tracker, llm_model=DEFAULT_LLM_MODEL, embed_model=DEFAULT_EMBED_MODEL
    )
    rag = build_rag(working_dir=str(RAG_STORAGE), llm_token_tracker=adapter)

    # Bypass RAG-Anything's parser-installed check. The check at
    # raganything.raganything._ensure_lightrag_initialized:259 runs
    # ``subprocess.run(["mineru", "--version"])`` on the caller's PATH —
    # which fails when the venv's ``bin/`` is not on PATH (e.g. when the
    # script is invoked via ``.venv/bin/python`` directly without a full
    # shell activate). Setting this flag is safe because
    # ``insert_content_list`` does NOT invoke the MineRU subprocess
    # (the content_list is already parsed; we only run LightRAG's
    # entity-extraction LLM passes + embeddings + vision over it).
    rag._parser_installation_checked = True

    n = await ingest_from_mineru_output(
        rag, mineru_output_root, paper_ids, console
    )

    persisted = tracker.persist()
    console.print(
        f"[green]Ingested {n}/{len(paper_ids)} papers; "
        f"cost ${tracker.total_usd():.6f}; storage at {RAG_STORAGE}.[/green]"
    )
    console.print(f"[dim]Cost JSON written to {persisted}[/dim]")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Sync entry point — single asyncio.run boundary (matches main.py)."""
    args = build_parser().parse_args(argv)
    return asyncio.run(amain(args, Console()))


__all__ = [
    "_absolutize_image_paths",
    "_find_content_list",
    "_discover_papers",
    "_confirm_or_abort",
    "ingest_from_mineru_output",
    "build_parser",
    "amain",
    "main",
    "MINERU_OUTPUT_ROOT",
    "RAG_STORAGE",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Curate a RAG citation cluster from arXiv + Semantic Scholar.

Algorithm
---------
1. Load seed arxiv IDs from ``--seeds`` JSON (the curated anchor list).
2. Resolve each seed via Semantic Scholar (`SemanticScholar.get_paper`).
3. Walk references + citations 1 hop out (configurable via ``--hops``); keep
   only papers with an ArXiv external ID. Reference-first ordering.
4. Cap the cluster at ``--max-papers``.
5. Download PDFs via ``arxiv.Client`` (delay_seconds=3.0; arxiv ToU compliant
   per 127-RESEARCH.md Pitfall 2).
6. Write a paper manifest (paper_id, arxiv_id, title, authors, year, abstract,
   license) to ``--output-manifest``.

This script is invoked by Phase 127 Plan 04. Plan 03 only authors the code.

Use ``--dry-run`` to preview the cluster + estimated wall time without
downloading PDFs (S2 lookups still happen — that's read-only and free).

Decisions: D-02 (RAG-literature anchor), D-03 (citation-graph chains),
D-11 (paper manifest schema mirroring figures/videos), Pitfalls 2/3.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# arXiv ToU: a single connection, 3s gap between requests, no parallelism.
# arxiv.Client(delay_seconds=3.0) enforces this in the library — never bypass.
ARXIV_DELAY_SECONDS = 3.0
ARXIV_PAGE_SIZE = 100
ARXIV_NUM_RETRIES = 5

# S2 shared pool can 429 — exponential backoff per Pitfall 3.
S2_MAX_RETRIES = 5
S2_BACKOFF_BASE_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str, max_len: int = 40) -> str:
    """Lowercase + collapse whitespace + strip non-alnum, truncated."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len]


def _short_arxiv_id(arxiv_id: str) -> str:
    """Strip a possible version suffix (`2005.11401v3` -> `2005.11401`)."""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def _load_seeds(seeds_path: Path) -> list[dict[str, str]]:
    with seeds_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Seeds file {seeds_path} must be a non-empty JSON array")
    for entry in data:
        if "arxiv_id" not in entry:
            raise ValueError(f"Seed entry missing 'arxiv_id': {entry!r}")
    return data


def _load_existing_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load existing manifest %s: %s", manifest_path, exc)
        return []


def _get_s2_api_key() -> str | None:
    """Read the Semantic Scholar API key, if shared.config is available.

    During Plan 03 the shared.config module may not yet exist (Plan 02
    authors it in parallel). We fall back to the SEMANTIC_SCHOLAR_API_KEY env
    var, then to None — per Pitfall 3, an unauthenticated client works for
    low-volume runs.
    """
    try:
        from shared.config import settings  # type: ignore[import-not-found]

        s2 = getattr(settings, "s2_api_key", None)
        if s2 is None:
            return None
        # Accept either pydantic SecretStr or plain str.
        if hasattr(s2, "get_secret_value"):
            return s2.get_secret_value()
        return str(s2)
    except Exception:  # pragma: no cover - defensive fallback path
        import os

        return os.environ.get("S2_API_KEY") or os.environ.get(
            "SEMANTIC_SCHOLAR_API_KEY"
        )


# ---------------------------------------------------------------------------
# Semantic Scholar cluster expansion
# ---------------------------------------------------------------------------


def _s2_call_with_backoff(fn, *args, **kwargs):
    """Run an S2 client call with exponential backoff on 429/transient errors."""
    last_exc: Exception | None = None
    for attempt in range(S2_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - lib raises various
            last_exc = exc
            msg = str(exc).lower()
            transient = (
                "429" in msg
                or "rate" in msg
                or "timeout" in msg
                or "temporarily" in msg
            )
            if not transient or attempt == S2_MAX_RETRIES - 1:
                raise
            sleep_s = S2_BACKOFF_BASE_SECONDS * (1 << attempt)
            logger.warning(
                "S2 transient error (attempt %d/%d): %s — sleeping %.1fs",
                attempt + 1,
                S2_MAX_RETRIES,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)
    if last_exc is not None:  # pragma: no cover - defensive
        raise last_exc


def _extract_arxiv_id_from_paper(paper: Any) -> str | None:
    """Return the ArXiv ID from a Semantic Scholar paper (None if absent)."""
    ext = getattr(paper, "externalIds", None)
    if isinstance(ext, dict):
        return ext.get("ArXiv") or ext.get("arXiv")
    return None


def expand_cluster(
    seed_ids: Iterable[str],
    *,
    hops: int = 2,
    max_papers: int = 100,
    s2_api_key: str | None = None,
) -> "OrderedDict[str, dict[str, Any]]":
    """Expand the citation cluster from seed arXiv IDs.

    Returns an OrderedDict[arxiv_id -> {title, year, source}] capped at
    ``max_papers``. References are preferred over citations (older work is
    foundational; newer citations may be tangential).
    """
    from semanticscholar import SemanticScholar  # local import keeps --help fast

    sch = SemanticScholar(api_key=s2_api_key) if s2_api_key else SemanticScholar()
    fields = ["title", "year", "externalIds"]
    s2_call_count = 0
    cluster: OrderedDict[str, dict[str, Any]] = OrderedDict()

    seeds = [_short_arxiv_id(s) for s in seed_ids]
    frontier: list[str] = list(seeds)

    # Confirm seeds resolve (and prime the cluster with seed metadata).
    for arxiv_id in seeds:
        if len(cluster) >= max_papers:
            break
        try:
            paper = _s2_call_with_backoff(sch.get_paper, f"arXiv:{arxiv_id}")
            s2_call_count += 1
            cluster[arxiv_id] = {
                "title": getattr(paper, "title", "") or "",
                "year": getattr(paper, "year", None),
                "source": "seed",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Seed %s failed S2 lookup: %s — keeping anyway", arxiv_id, exc)
            cluster[arxiv_id] = {"title": "", "year": None, "source": "seed"}

    # Walk hops outward. hops=1 means seeds only.
    for hop in range(1, hops):
        next_frontier: list[str] = []
        for arxiv_id in frontier:
            if len(cluster) >= max_papers:
                break

            # References first (foundational).
            try:
                refs = _s2_call_with_backoff(
                    sch.get_paper_references,
                    f"arXiv:{arxiv_id}",
                    fields=fields,
                    limit=50,
                )
                s2_call_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("References lookup failed for %s: %s", arxiv_id, exc)
                refs = []

            for ref in refs:
                cited = getattr(ref, "paper", ref)
                if cited is None:
                    continue
                ax = _extract_arxiv_id_from_paper(cited)
                if not ax:
                    continue  # drop non-arxiv refs per Open Question #2
                ax = _short_arxiv_id(ax)
                if ax in cluster:
                    continue
                cluster[ax] = {
                    "title": getattr(cited, "title", "") or "",
                    "year": getattr(cited, "year", None),
                    "source": f"ref:{arxiv_id}@hop{hop}",
                }
                next_frontier.append(ax)
                if len(cluster) >= max_papers:
                    break

            if len(cluster) >= max_papers:
                break

            # Then citations.
            try:
                cits = _s2_call_with_backoff(
                    sch.get_paper_citations,
                    f"arXiv:{arxiv_id}",
                    fields=fields,
                    limit=50,
                )
                s2_call_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Citations lookup failed for %s: %s", arxiv_id, exc)
                cits = []

            for cit in cits:
                citing = getattr(cit, "paper", cit)
                if citing is None:
                    continue
                ax = _extract_arxiv_id_from_paper(citing)
                if not ax:
                    continue
                ax = _short_arxiv_id(ax)
                if ax in cluster:
                    continue
                cluster[ax] = {
                    "title": getattr(citing, "title", "") or "",
                    "year": getattr(citing, "year", None),
                    "source": f"cit:{arxiv_id}@hop{hop}",
                }
                next_frontier.append(ax)
                if len(cluster) >= max_papers:
                    break

        frontier = next_frontier
        if not frontier or len(cluster) >= max_papers:
            break

    # Annotate the call count on the cluster object via a sentinel attribute.
    cluster.__dict__["_s2_call_count"] = s2_call_count  # type: ignore[attr-defined]
    return cluster


# ---------------------------------------------------------------------------
# arXiv download
# ---------------------------------------------------------------------------


def download_papers(
    cluster: "OrderedDict[str, dict[str, Any]]",
    output_dir: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Download PDFs for cluster IDs that don't already exist on disk.

    Returns the merged manifest entries (existing + newly downloaded).
    Idempotent: if a PDF already exists at the target path (same arxiv_id
    prefix), the download is skipped.
    """
    import arxiv
    from rich.progress import Progress

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing_manifest = _load_existing_manifest(manifest_path)
    existing_by_id = {e.get("paper_id"): e for e in existing_manifest if isinstance(e, dict)}

    existing_files = {
        re.match(r"^(\d{4}\.\d{4,5})", p.name).group(1)
        for p in output_dir.glob("*.pdf")
        if re.match(r"^(\d{4}\.\d{4,5})", p.name)
    }

    arxiv_ids = list(cluster.keys())
    to_download = [
        ax for ax in arxiv_ids if ax not in existing_files
    ]

    if not to_download:
        logger.info("All %d papers already present on disk — skipping download.", len(arxiv_ids))
        return existing_manifest

    client = arxiv.Client(
        page_size=ARXIV_PAGE_SIZE,
        delay_seconds=ARXIV_DELAY_SECONDS,
        num_retries=ARXIV_NUM_RETRIES,
    )

    new_entries: list[dict[str, Any]] = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Downloading PDFs...", total=len(to_download))
        # arxiv.Search(id_list=...) returns Result objects in the same order.
        search = arxiv.Search(id_list=to_download)
        for result in client.results(search):
            ax_short = _short_arxiv_id(result.get_short_id())
            slug = _slugify(result.title)
            filename = f"{ax_short}_{slug}.pdf"
            target = output_dir / filename
            try:
                if not target.exists():
                    result.download_pdf(dirpath=str(output_dir), filename=filename)
                entry = {
                    "paper_id": ax_short,
                    "arxiv_id": ax_short,
                    "title": result.title or "",
                    "authors": [a.name for a in (result.authors or [])],
                    "year": result.published.year if result.published else None,
                    "abstract": (result.summary or "").strip(),
                    "license": "arxiv-non-exclusive-distribution",
                    "filename": filename,
                }
                new_entries.append(entry)
            except Exception as exc:  # noqa: BLE001
                logger.error("Download failed for %s: %s", ax_short, exc)
            finally:
                progress.advance(task)

    # Merge: existing entries + new entries, dedup by paper_id (new wins).
    merged: dict[str, dict[str, Any]] = dict(existing_by_id)
    for e in new_entries:
        merged[e["paper_id"]] = e

    final = list(merged.values())
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote manifest with %d entries to %s", len(final), manifest_path)
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Curate a RAG citation cluster from arXiv + Semantic Scholar. "
            "Use --dry-run to preview without downloading."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no PDF downloads.")
    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path("scripts/seed_papers.json"),
        help="Path to seed arxiv ID JSON (default: scripts/seed_papers.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/papers/"),
        help="Directory to download PDFs into (default: dataset/papers/).",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=100,
        help="Cap the final cluster size (default: 100).",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Citation graph hops: 1=seeds only, 2=seeds+refs/cits, 3=one more level (default: 2).",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("dataset/manifests/papers.json"),
        help="Manifest JSON path (default: dataset/manifests/papers.json).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.seeds.exists():
        logger.error("Seeds file not found: %s", args.seeds)
        return 2

    seeds = _load_seeds(args.seeds)
    seed_ids = [s["arxiv_id"] for s in seeds]
    logger.info("Loaded %d seeds from %s", len(seed_ids), args.seeds)

    s2_key = _get_s2_api_key()
    if not s2_key:
        logger.warning(
            "No Semantic Scholar API key (S2_API_KEY). Using shared pool — "
            "expect 429s on bursts; OK for ≤3k calls (per Pitfall 3)."
        )

    logger.info(
        "Expanding cluster: hops=%d, max_papers=%d (this hits Semantic Scholar)",
        args.hops,
        args.max_papers,
    )
    cluster = expand_cluster(
        seed_ids,
        hops=args.hops,
        max_papers=args.max_papers,
        s2_api_key=s2_key,
    )

    s2_calls = cluster.__dict__.get("_s2_call_count", 0)  # type: ignore[attr-defined]
    arxiv_calls = len(cluster)
    est_wall = ARXIV_DELAY_SECONDS * arxiv_calls + 1.0 * s2_calls

    print()
    print(f"=== Cluster: {len(cluster)} papers ===")
    for ax, meta in cluster.items():
        year = meta.get("year") or "????"
        title = meta.get("title") or "(unknown title)"
        src = meta.get("source", "")
        print(f"  {ax}  ({year})  [{src}]  {title[:80]}")
    print()
    print(f"Counts: seeds={len(seed_ids)}  cluster_size={len(cluster)}")
    print(f"S2 calls: {s2_calls}    arXiv calls (download phase): {arxiv_calls}")
    print(f"Estimated wall time for full run: ~{est_wall:.0f}s ({est_wall / 60:.1f}m)")
    print()

    if args.dry_run:
        print("[DRY-RUN] Skipping PDF downloads. Re-run without --dry-run to fetch.")
        return 0

    download_papers(cluster, args.output, args.output_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Extract figures from arXiv PDFs using PyMuPDF.

Per 127-RESEARCH.md Pattern 3 + Pitfall 7: PyMuPDF gives bounding boxes via
``page.get_image_rects(xref)`` and raw image bytes via ``doc.extract_image(xref)``.
The naive ``page.get_images()`` walk catches every embedded image — most of
which are page numbers, logos, and decorative bars. We filter by:

1. **xref dedup within a PDF** — same xref reused on every page is decorative
   (same logo, same separator). Pitfall 7.
2. **Minimum dimension** — width and height must both be ≥ ``--min-dim``
   (default 200px). Pitfall 7 — page numbers are tiny.
3. **Aspect ratio** — reject images where ``max(w/h, h/w) > --max-aspect-ratio``
   (default 5.0). Pitfall 7 — wide bars are usually decorative.

NOTE: Per Pitfall 7 in 127-RESEARCH.md — heuristic filtering catches ~90%+;
remaining garbage requires manual review in Plan 05 before manifest commit.

Captions are NOT auto-extracted (Open Question #4 — the caption field stays
empty for manual fill).

This script is invoked by Phase 127 Plan 05. Plan 03 only authors the code.

Decisions: D-05 (pre-extract figures, manifest schema), D-11 (manifest fields),
Pitfall 7 (filter thresholds), Pattern 3 (PyMuPDF API).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ARXIV_ID_REGEX = re.compile(r"^(\d{4}\.\d{4,5})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_paper_id(pdf_path: Path) -> str | None:
    m = ARXIV_ID_REGEX.match(pdf_path.name)
    return m.group(1) if m else None


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load manifest %s: %s", manifest_path, exc)
        return []


def _save_manifest(manifest_path: Path, entries: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_figures(
    pdf_path: Path,
    output_dir: Path,
    *,
    min_dim: int = 200,
    max_aspect_ratio: float = 5.0,
    dry_run: bool = False,
    existing_figure_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract figures from a single PDF.

    Returns a list of manifest entries (one per saved figure). When
    ``dry_run`` is True, returns the would-be entries without writing files.
    Honors ``existing_figure_ids`` for idempotent resume.
    """
    import fitz  # PyMuPDF — fitz is the canonical module name per docs
    from PIL import Image

    paper_id = _derive_paper_id(pdf_path)
    if paper_id is None:
        logger.warning(
            "PDF filename %s does not match arxiv-id prefix regex — skipping.",
            pdf_path.name,
        )
        return []

    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return []

    if existing_figure_ids is None:
        existing_figure_ids = set()

    output_dir.mkdir(parents=True, exist_ok=True)

    seen_xrefs: set[int] = set()  # xref dedup per Pitfall 7
    entries: list[dict[str, Any]] = []
    counter = 0

    doc = fitz.open(str(pdf_path))
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_number = page_index + 1  # 1-indexed for manifest

            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen_xrefs:
                    continue  # decorative reuse
                seen_xrefs.add(xref)

                try:
                    rects = page.get_image_rects(xref)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("get_image_rects failed for xref=%s: %s", xref, exc)
                    continue
                if not rects:
                    continue
                bbox = rects[0]  # first occurrence on page

                try:
                    base = doc.extract_image(xref)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("extract_image failed for xref=%s: %s", xref, exc)
                    continue
                ext = base.get("ext", "png")
                data = base.get("image")
                if not data:
                    continue

                # PyMuPDF's bbox is in page coordinates, not image pixels.
                # Use PIL to read the true pixel dimensions.
                try:
                    with Image.open(BytesIO(data)) as pil_img:
                        width, height = pil_img.size
                except Exception as exc:  # noqa: BLE001
                    logger.debug("PIL open failed for xref=%s: %s", xref, exc)
                    continue

                # Size filter (Pitfall 7).
                if width < min_dim or height < min_dim:
                    logger.debug(
                        "Filter (min-dim): xref=%s %dx%d on page %d",
                        xref,
                        width,
                        height,
                        page_number,
                    )
                    continue

                # Aspect-ratio filter (Pitfall 7).
                aspect = max(width / height, height / width)
                if aspect > max_aspect_ratio:
                    logger.debug(
                        "Filter (aspect): xref=%s %dx%d aspect=%.2f on page %d",
                        xref,
                        width,
                        height,
                        aspect,
                        page_number,
                    )
                    continue

                counter += 1
                figure_id = f"{paper_id}_fig_{counter:03d}"

                if figure_id in existing_figure_ids:
                    logger.debug("Skipping already-manifested figure %s", figure_id)
                    continue

                filename = f"{figure_id}.{ext}"
                target = output_dir / filename

                if not dry_run:
                    if not target.exists():
                        with target.open("wb") as fh:
                            fh.write(data)

                size_bytes = target.stat().st_size if target.exists() else len(data)

                entries.append(
                    {
                        "figure_id": figure_id,
                        "paper_id": paper_id,
                        "page_number": page_number,
                        "bbox": [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)],
                        "caption": "",  # manual fill in Plan 05 review per Open Question #4
                        "filename": filename,
                        "size_bytes": size_bytes,
                        "width": width,
                        "height": height,
                    }
                )
    finally:
        doc.close()

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract figures from arXiv PDFs via PyMuPDF, with size/aspect/xref "
            "filtering. Use --dry-run to preview without writing."
        ),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", type=Path, help="Path to a single PDF.")
    src.add_argument("--all", type=Path, dest="all_dir", help="Directory of PDFs (batch mode).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/images/"),
        help="Output directory for figure files (default: dataset/images/).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/manifests/figures.json"),
        help="Manifest JSON path (default: dataset/manifests/figures.json).",
    )
    parser.add_argument(
        "--min-dim",
        type=int,
        default=200,
        help="Minimum width AND height in pixels (default: 200; Pitfall 7).",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=5.0,
        help="Reject if max(w/h, h/w) > N (default: 5.0; Pitfall 7).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what WOULD be extracted without writing files or manifest.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: INFO).",
    )
    return parser


def _collect_pdfs(args: argparse.Namespace) -> list[Path]:
    if args.pdf is not None:
        return [args.pdf]
    pdfs = sorted(args.all_dir.glob("*.pdf"))
    return pdfs


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pdfs = _collect_pdfs(args)
    if not pdfs:
        logger.error(
            "No PDFs found. --pdf=%s --all=%s",
            args.pdf,
            getattr(args, "all_dir", None),
        )
        return 2

    existing_manifest = _load_manifest(args.manifest)
    existing_ids = {e.get("figure_id") for e in existing_manifest if isinstance(e, dict)}

    all_entries: list[dict[str, Any]] = []

    if len(pdfs) == 1:
        # Single-PDF mode — quiet, no progress bar.
        all_entries = extract_figures(
            pdfs[0],
            args.out,
            min_dim=args.min_dim,
            max_aspect_ratio=args.max_aspect_ratio,
            dry_run=args.dry_run,
            existing_figure_ids=existing_ids,
        )
    else:
        # Batch mode — show progress.
        from rich.progress import Progress

        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Extracting figures from {len(pdfs)} PDFs...",
                total=len(pdfs),
            )
            for pdf in pdfs:
                entries = extract_figures(
                    pdf,
                    args.out,
                    min_dim=args.min_dim,
                    max_aspect_ratio=args.max_aspect_ratio,
                    dry_run=args.dry_run,
                    existing_figure_ids=existing_ids,
                )
                all_entries.extend(entries)
                # Advance the existing-id set so cross-PDF dedup also applies.
                for e in entries:
                    existing_ids.add(e["figure_id"])
                progress.advance(task)

    print()
    print(f"=== Extraction Summary ===")
    print(f"PDFs scanned: {len(pdfs)}")
    print(f"Figures kept after filters: {len(all_entries)}")
    print(f"min_dim={args.min_dim}  max_aspect_ratio={args.max_aspect_ratio}")

    if args.dry_run:
        print(f"[DRY-RUN] Would extract {len(all_entries)} figures from {len(pdfs)} PDFs.")
        print("[DRY-RUN] No image files written, manifest not updated.")
        return 0

    # Merge with existing manifest (dedup by figure_id; new wins).
    merged: dict[str, dict[str, Any]] = {
        e["figure_id"]: e for e in existing_manifest if isinstance(e, dict) and "figure_id" in e
    }
    for e in all_entries:
        merged[e["figure_id"]] = e
    final = list(merged.values())
    _save_manifest(args.manifest, final)
    print(f"Manifest written: {args.manifest} ({len(final)} entries)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

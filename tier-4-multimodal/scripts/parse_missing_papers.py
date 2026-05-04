"""Host-side MineRU top-up helper for the 4 golden_qa-referenced papers
missing from the existing 75-paper ``tier-4-multimodal/output/`` cache.

This script is **Phase 7 prep**, not Phase 2 ship work. The 4 missing
papers (``1909.01066``, ``2002.06177``, ``2309.15217``, ``2410.05779``)
are referenced by golden_qa questions OUTSIDE the 5-question smoke set;
Phase 2 ships independently from this top-up. See
``.planning/phases/02-tier-4-graphml-regeneration/02-02-PLAN.md`` and
``.planning/phases/02-tier-4-graphml-regeneration/02-RESEARCH.md`` (A1,
A2, Pitfall 3, Open Question 2).

References (in 02-RESEARCH.md):

* **Pattern 2** — host-side MineRU subprocess loop with per-paper isolation.
* **Pitfall 3** — MineRU's PaddleOCR backend uses OpenMP shared-memory
  primitives the orchestrator sandbox blocks at the kernel level. This
  script's pre-flight refuses to invoke MineRU when sandbox markers are
  present (``ALL_PROXY=socks5h://...``, ``HTTPS_PROXY=socks5h://...``,
  ``CLAUDE_CODE_SANDBOX``, ``SANDBOX_ENV``). Bypassable via
  ``--allow-sandbox`` for testing only — the OS-level OMP block is the
  hard fence; this flag does NOT make MineRU work in the sandbox.
* **A2** — uses ``-b hybrid-auto-engine`` to match the existing 75-paper
  cache's ``hybrid_auto/`` directory naming
  (``mineru/cli/output_paths.py:24-25``).
* **Open Question 2** — on per-paper failure, log to stderr and continue;
  exit non-zero only if a golden-qa-referenced paper failed. Since all 4
  ``MISSING_PAPERS_GOLDEN`` are golden-qa-referenced, ANY single failure
  causes a non-zero exit.

This script MUST run on the host machine (Terminal.app, real shell), NOT
inside the orchestrator sandbox. The pre-flight refuses with exit code 2
if sandbox markers are detected.

V5 ASVS: ``subprocess.run`` is invoked with list-form args (no
``shell=True``); see threat T-02-02-01 in the plan's threat model.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console  # noqa: E402  (sys.path mutation above)

PAPERS_DIR = _REPO_ROOT / "dataset" / "papers"
PAPERS_MANIFEST = _REPO_ROOT / "dataset" / "manifests" / "papers.json"
OUTPUT_ROOT = _REPO_ROOT / "tier-4-multimodal" / "output"

# The 4 papers referenced by golden_qa but missing from the existing
# tier-4-multimodal/output/ cache (verified 2026-05-04 against
# evaluation/golden_qa.json + tier-4-multimodal/output/ directory listing).
# Per Open Question 2 of RESEARCH.md: any single failure here is fatal
# (these are ALL golden-qa-referenced).
MISSING_PAPERS_GOLDEN: tuple[str, ...] = (
    "1909.01066",   # multi-hop-008
    "2002.06177",   # multimodal-010
    "2309.15217",   # multi-hop-009
    "2410.05779",   # single-hop-005, multi-hop-003, multimodal-004, multimodal-005
)


# ---------------------------------------------------------------------------
# Pure helpers (testable without subprocess)
# ---------------------------------------------------------------------------


def _detect_sandbox() -> str | None:
    """Return a human-readable reason if running inside the orchestrator
    sandbox; ``None`` otherwise.

    Heuristics (any one triggers detection):

    * ``ALL_PROXY`` contains ``socks5h://`` (Claude Code tunnel marker)
    * ``HTTPS_PROXY`` contains ``socks5h://`` (same tunnel)
    * ``CLAUDE_CODE_SANDBOX`` env var is truthy
    * ``SANDBOX_ENV`` env var is truthy

    The ``socks5h://`` scheme is the specific marker — a plain
    ``http://corp-proxy:8080`` HTTPS_PROXY is NOT a sandbox signal.
    """
    all_proxy = os.environ.get("ALL_PROXY", "")
    if "socks5h://" in all_proxy:
        return f"sandbox detected via ALL_PROXY={all_proxy}"

    https_proxy = os.environ.get("HTTPS_PROXY", "")
    if "socks5h://" in https_proxy:
        return f"sandbox detected via HTTPS_PROXY={https_proxy}"

    claude_sandbox = os.environ.get("CLAUDE_CODE_SANDBOX")
    if claude_sandbox:
        return f"sandbox detected via CLAUDE_CODE_SANDBOX={claude_sandbox}"

    sandbox_env = os.environ.get("SANDBOX_ENV")
    if sandbox_env:
        return f"sandbox detected via SANDBOX_ENV={sandbox_env}"

    return None


def _load_papers_manifest(manifest_path: Path) -> dict[str, dict]:
    """Load ``papers.json`` and return ``{paper_id: paper_dict}``.

    Raises ``FileNotFoundError`` with a clear remediation message if the
    manifest is absent.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Papers manifest not found at {manifest_path}. "
            f"Expected dataset/manifests/papers.json (committed to git)."
        )
    papers_list = json.loads(manifest_path.read_text())
    return {p["paper_id"]: p for p in papers_list}


def _build_mineru_command(
    pdf: Path, output_root: Path, paper_id: str
) -> list[str]:
    """Return the locked MineRU CLI args list (V5 ASVS: list form, no shell).

    Per A2 of RESEARCH.md, ``-b hybrid-auto-engine`` matches the existing
    75-paper cache's ``hybrid_auto/`` per-paper subdirectory layout. Per
    ``tier-4-multimodal/main.py:128-135``, MineRU auto-detects the device
    (cuda → mps → cpu) so we do NOT pin ``-d``.
    """
    return [
        "mineru",
        "-p", str(pdf),
        "-o", str(output_root / paper_id),
        "-m", "auto",
        "-b", "hybrid-auto-engine",
        "-l", "en",
    ]


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def parse_missing_papers(
    papers_dir: Path,
    output_root: Path,
    manifest_path: Path,
    paper_ids: tuple[str, ...] = MISSING_PAPERS_GOLDEN,
    console: Console | None = None,
) -> tuple[int, list[str]]:
    """Run MineRU on each missing paper in deterministic sorted order.

    Returns ``(n_parsed, failed_paper_ids)``. The loop continues past a
    per-paper failure (Open Question 2 of RESEARCH.md). Idempotent: if
    ``output_root/<paper_id>/`` already exists, the paper is skipped and
    NOT counted in either return value (a no-op pass through the loop).

    Failure modes (each appended to ``failed`` and surfaced in red):

    * ``paper_id`` not in the manifest → cannot resolve filename
    * resolved PDF does not exist on disk
    * MineRU subprocess raises ``CalledProcessError``
    """
    console = console or Console()
    by_id = _load_papers_manifest(manifest_path)

    n_parsed = 0
    failed: list[str] = []

    for pid in sorted(paper_ids):
        if pid not in by_id:
            failed.append(pid)
            console.print(f"[red]Paper {pid} not in manifest — skipping[/red]")
            continue

        target_dir = output_root / pid
        if target_dir.exists():
            console.print(
                f"[dim]Skipping {pid} (already parsed at {target_dir})[/dim]"
            )
            continue

        pdf = papers_dir / by_id[pid]["filename"]
        if not pdf.exists():
            failed.append(pid)
            console.print(f"[red]PDF missing at {pdf} — skipping {pid}[/red]")
            continue

        cmd = _build_mineru_command(pdf, output_root, pid)
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            n_parsed += 1
        except subprocess.CalledProcessError as exc:
            failed.append(pid)
            console.print(
                f"[red]MineRU failed for {pid}: {exc}[/red]"
            )
            continue

    return n_parsed, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser. Exposed for unit testing."""
    parser = argparse.ArgumentParser(
        description=(
            "Host-side MineRU top-up for the 4 golden_qa-referenced papers "
            "missing from the existing tier-4-multimodal/output/ cache. "
            "MUST run on the host machine — the sandbox-detection pre-flight "
            "will refuse to invoke MineRU otherwise (Pitfall 3 of "
            "02-RESEARCH.md)."
        ),
    )
    parser.add_argument(
        "--papers-dir",
        default=str(PAPERS_DIR),
        help=f"Directory containing source PDFs (default: {PAPERS_DIR}).",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help=(
            "Directory under which per-paper MineRU outputs land at "
            "<output-root>/<paper_id>/... (default: "
            f"{OUTPUT_ROOT})."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(PAPERS_MANIFEST),
        help=f"Papers manifest JSON (default: {PAPERS_MANIFEST}).",
    )
    parser.add_argument(
        "--paper-ids",
        default=None,
        help=(
            "Optional comma-separated paper_id subset; default = "
            "MISSING_PAPERS_GOLDEN (the 4 golden-qa-referenced papers)."
        ),
    )
    parser.add_argument(
        "--allow-sandbox",
        action="store_true",
        help=(
            "Bypass the sandbox-detection pre-flight (testing only — does NOT "
            "make MineRU work inside the sandbox; the kernel-level OMP shmem "
            "block is the actual fence)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes:

    * ``0`` — every requested paper either parsed successfully or was
      already present (no failures).
    * ``1`` — at least one golden-qa-referenced paper failed. Per Open
      Question 2 of RESEARCH.md, the user must triage before checkpoint
      approval (or descope to Phase 7).
    * ``2`` — sandbox pre-flight refused. The script did NOT invoke
      MineRU. Re-run from a real terminal on the host.
    """
    args = build_parser().parse_args(argv)
    console = Console()

    sandbox_reason = _detect_sandbox()
    if sandbox_reason and not args.allow_sandbox:
        console.print(f"[red]Refusing to invoke MineRU: {sandbox_reason}[/red]")
        console.print(
            "[red]MineRU's PaddleOCR backend uses OpenMP shared-memory "
            "primitives that the orchestrator sandbox blocks at the kernel "
            "level (Pitfall 3 of 02-RESEARCH.md, Phase 138/139 evidence).[/red]"
        )
        console.print(
            "[yellow]Run this script from a real terminal on the host: "
            "open Terminal.app, cd to repo root, re-invoke.[/yellow]"
        )
        return 2

    if args.paper_ids is None:
        paper_ids: tuple[str, ...] = MISSING_PAPERS_GOLDEN
    else:
        paper_ids = tuple(s.strip() for s in args.paper_ids.split(",") if s.strip())

    n_parsed, failed = parse_missing_papers(
        papers_dir=Path(args.papers_dir),
        output_root=Path(args.output_root),
        manifest_path=Path(args.manifest),
        paper_ids=paper_ids,
        console=console,
    )

    console.print(
        f"Parsed {n_parsed}/{len(paper_ids)} papers; "
        f"{len(failed)} failures: {failed}"
    )
    if failed:
        console.print(
            "[red]One or more golden_qa-referenced papers failed; non-zero "
            "exit (Open Question 2 of RESEARCH.md).[/red]"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

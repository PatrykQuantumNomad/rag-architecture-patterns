"""Cut 30-second clips from CC-licensed talk videos using ffmpeg stream-copy.

This script does NOT download videos — the curator (Plan 05) manually
downloads each talk source and points ``source_file_local_path`` at the
local file. This script only cuts the clip.

Per 127-RESEARCH.md "Don't Hand-Roll" — `ffmpeg -ss S -t D -i IN -c copy OUT`
is lossless (stream copy, no re-encode) and runs in milliseconds. Re-encoding
introduces compression artifacts and 5-30x slowdown.

License safety gate (per D-12 + Pitfall 8):

    Acceptable: CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA
    Rejected:   CC-BY-ND, CC-BY-NC-ND  (any "ND" = No Derivatives)
    Sentinel:   "TBD_VERIFY_MANUALLY" (refuses until curator updates)

The license MUST be manually verified at the source talk page and recorded
along with ``license_verified_at`` (ISO date) and ``license_verified_by``
(name or "manual_review") in scripts/video_sources.json before this script
will cut anything.

This script is invoked by Phase 127 Plan 05. Plan 03 only authors the code.

Decisions: D-12 (CC license types accepted: BY/BY-SA/BY-NC/BY-NC-SA;
ND rejected; manual verification in scope), D-11 (video manifest schema),
Pitfall 8 (ND restriction).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Per D-12 / Pitfall 8 — ND clauses forbid derivatives, so ANY clip cut is a
# violation. CC-BY-NC-ND and CC-BY-ND are out. CC-BY-NC and CC-BY-NC-SA are
# fine for non-commercial demo + research-blog use.
ACCEPTABLE_LICENSES = {
    "CC-BY",
    "CC-BY-4.0",
    "CC-BY-SA",
    "CC-BY-SA-4.0",
    "CC-BY-NC",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA",
    "CC-BY-NC-SA-4.0",
}
REJECTED_LICENSES = {
    "TBD_VERIFY_MANUALLY",
    "CC-BY-ND",
    "CC-BY-ND-4.0",
    "CC-BY-NC-ND",
    "CC-BY-NC-ND-4.0",
}

# Internal-only fields stripped from the published manifest.
INTERNAL_FIELDS = {"source_file_local_path"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_sources(sources_path: Path) -> list[dict[str, Any]]:
    with sources_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{sources_path} must be a JSON array")
    return data


def _save_manifest(manifest_path: Path, entries: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)


def _strip_internal(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if k not in INTERNAL_FIELDS}


def _check_license(entry: dict[str, Any], sources_path: Path) -> str | None:
    """Return None if the license is OK to cut; an error string otherwise."""
    lic = entry.get("license")
    video_id = entry.get("video_id", "<no id>")

    if lic is None or lic in REJECTED_LICENSES:
        return (
            f"License not verified or is ND-restricted. Refusing to cut clip "
            f"{video_id}. Update license field in {sources_path} after manual "
            f"verification (acceptable: CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA "
            f"per 127-RESEARCH.md Pitfall 8)."
        )
    if lic not in ACCEPTABLE_LICENSES:
        return (
            f"License '{lic}' is not in the acceptable set for clip {video_id}. "
            f"Acceptable: CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA "
            f"(per 127-RESEARCH.md Pitfall 8). Refusing to cut."
        )
    return None


# ---------------------------------------------------------------------------
# Cut a single clip
# ---------------------------------------------------------------------------


def cut_clip(
    entry: dict[str, Any],
    output_dir: Path,
    *,
    sources_path: Path,
    dry_run: bool = False,
) -> tuple[bool, str | None]:
    """Cut a single clip via ffmpeg stream-copy.

    Returns (ok, error_message_or_none). ``ok`` is False when the entry was
    refused (license, missing source) — error_message describes why so the
    caller can log it.
    """
    video_id = entry.get("video_id", "<no id>")

    # License gate.
    err = _check_license(entry, sources_path)
    if err is not None:
        return False, err

    # Source file gate.
    src_path = entry.get("source_file_local_path")
    if not src_path:
        return False, (
            f"source_file_local_path not set for {video_id}. Manually download "
            f"{entry.get('source_url', '<no url>')} to a local mp4/webm before running."
        )
    src = Path(src_path).expanduser().resolve()
    if not src.exists():
        return False, (
            f"source_file_local_path does not exist for {video_id}: {src}. "
            f"Download the source talk locally first."
        )

    filename = entry.get("filename") or f"{video_id}.mp4"
    out = (output_dir / filename).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent: skip if output already exists.
    if out.exists() and not dry_run:
        return True, None

    start = entry.get("clip_start_seconds")
    duration = entry.get("clip_duration_seconds")
    if start is None or duration is None:
        return False, (
            f"clip_start_seconds and clip_duration_seconds required for {video_id}."
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(src),
        "-c",
        "copy",
        str(out),
    ]

    if dry_run:
        print("[DRY-RUN] " + " ".join(cmd))
        return True, None

    # Mitigates T-127-15: list args (no shell=True), paths resolved before subprocess.
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        return False, "ffmpeg not on PATH. Install ffmpeg and retry."
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        return False, f"ffmpeg failed for {video_id}: {stderr.strip()[:500]}"

    return True, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cut 30-second clips from CC-licensed talk videos using ffmpeg "
            "stream-copy. License safety gate refuses TBD/ND entries."
        ),
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("scripts/video_sources.json"),
        help="Path to the source-clip JSON (default: scripts/video_sources.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/videos/"),
        help="Directory to write cut clips into (default: dataset/videos/).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/manifests/videos.json"),
        help="Manifest JSON path (default: dataset/manifests/videos.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ffmpeg commands without running.",
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

    if not args.dry_run and shutil.which("ffmpeg") is None:
        logger.error("ffmpeg not on PATH. Install ffmpeg before running.")
        return 2

    if not args.sources.exists():
        logger.error("Sources file not found: %s", args.sources)
        return 2

    sources = _load_sources(args.sources)
    logger.info("Loaded %d source entries from %s", len(sources), args.sources)

    accepted: list[dict[str, Any]] = []
    refused = 0
    for entry in sources:
        ok, err = cut_clip(entry, args.out, sources_path=args.sources, dry_run=args.dry_run)
        video_id = entry.get("video_id", "<no id>")
        if not ok:
            refused += 1
            print(f"[REFUSED] {video_id}: {err}")
            continue
        if not args.dry_run:
            accepted.append(_strip_internal(entry))
            print(f"[OK] cut {video_id}")
        else:
            print(f"[OK] would cut {video_id}")

    print()
    print(f"=== Cut Summary ===")
    print(f"Sources: {len(sources)}   Cut: {len(sources) - refused}   Refused: {refused}")

    if args.dry_run:
        print("[DRY-RUN] No clips written, manifest not updated.")
        return 0 if refused == 0 else 1

    if accepted:
        _save_manifest(args.manifest, accepted)
        print(f"Manifest written: {args.manifest} ({len(accepted)} entries)")

    return 0 if refused == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

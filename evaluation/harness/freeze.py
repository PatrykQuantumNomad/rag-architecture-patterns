"""Freeze tool — copy comparison.md to frozen/ + sidecar manifest. HARN-03 + HARN-04. Phase 5 imports freeze() in-process."""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from rich.console import Console
from evaluation.harness.compare import SUPPORTED_TIERS, _detect_judge_provenance, aggregate_tier
from evaluation.harness.run import _git_sha, _ts
from evaluation.harness.score import JUDGE_MAX_TOKENS
CRITICAL_LIBS = ("lightrag-hku", "raganything", "openai-agents", "ragas")
BONUS_LIBS = ("litellm", "chromadb")
def _git_dirty(repo_root: Path = _REPO_ROOT) -> bool:
    """True if working tree has uncommitted changes (Pitfall 4)."""
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip())
    except Exception:  # noqa: BLE001
        return False
def _iso_z(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def _rel(p, root: Path) -> str:
    """Relative path str (Pitfall 7); absolute fallback if outside root."""
    try:
        return str(Path(p).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)
def _library_versions() -> dict[str, str]:
    """Read installed pkg versions; missing critical libs => RuntimeError (decision Q2)."""
    out, missing = {}, []
    for pkg in CRITICAL_LIBS + BONUS_LIBS:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
            if pkg in CRITICAL_LIBS: missing.append(pkg)
    if missing:
        raise RuntimeError(f"Critical libraries not installed: {missing}. Run `uv sync --all-extras`.")
    return out
def freeze(version: str, force: bool = False, results_dir: Path | None = None, source: Path | None = None) -> Path:
    """Copy comparison.md to frozen/ + write sidecar manifest. Returns frozen md path. Raises FileExistsError | FileNotFoundError | RuntimeError (CLI -> exit 2)."""
    results_dir = results_dir or _REPO_ROOT / "evaluation" / "results"
    source = source or results_dir / "comparison.md"
    out_md = results_dir / "frozen" / f"eval-numbers-v{version}.md"
    out_manifest = out_md.with_suffix(".manifest.json")
    if not source.exists():
        raise FileNotFoundError(f"Source markdown missing: {source} (run compare.py first)")
    if out_md.exists() and not force:
        raise FileExistsError(f"{out_md} already frozen — bump version or pass --force")
    lib_versions = _library_versions()
    per_tier: dict[str, dict] = {}
    for t in SUPPORTED_TIERS:
        row = aggregate_tier(t, results_dir)
        if row is None:
            per_tier[f"tier-{t}"] = {"status": "missing"}; continue
        entry: dict = {"status": "present", "generation_model": row.get("model"),
            "embedder": row.get("embedder"),  # Phase 6 / CAP-03 — None for legacy
            "embedder_source": row.get("embedder_source"),  # Phase 6 / CAP-03
            "capture_timestamp": row.get("timestamp"), "capture_git_sha": row.get("git_sha"),
            "queries_path": _rel(row["queries_path"], results_dir),
            "queries_mtime": _iso_z(Path(row["queries_path"]).stat().st_mtime)}
        for kind in ("cost", "metrics"):
            p = row.get(f"{kind}_path")
            if p:
                entry[f"{kind}_path"] = _rel(p, results_dir); entry[f"{kind}_mtime"] = _iso_z(Path(p).stat().st_mtime)
        per_tier[f"tier-{t}"] = entry
    judge_model, judge_emb = _detect_judge_provenance(results_dir)
    manifest = {"$schema_version": "1.0", "version": version, "frozen_at": _ts(),
        "git_sha": _git_sha(), "git_dirty": _git_dirty(),
        "source_markdown": _rel(source, results_dir), "source_markdown_mtime": _iso_z(source.stat().st_mtime),
        "frozen_markdown": _rel(out_md, results_dir),
        "judge": {"model": judge_model, "embedder": judge_emb, "max_tokens": JUDGE_MAX_TOKENS},
        "per_tier": per_tier, "library_versions": lib_versions, "python_version": sys.version.split()[0]}
    out_md.parent.mkdir(parents=True, exist_ok=True)  # md FIRST then manifest (Pitfall 3 accepted)
    shutil.copy2(source, out_md)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return out_md
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evaluation.harness.freeze", description="Freeze comparison.md to frozen/eval-numbers-vX.Y.md + sidecar manifest")
    p.add_argument("--version", required=True, help="Version slug (e.g. 1.0)")
    p.add_argument("--force", action="store_true", help="Overwrite existing frozen artifacts")
    p.add_argument("--results-dir", default=str(_REPO_ROOT / "evaluation" / "results"), help="Parent of queries/, metrics/, costs/, frozen/")
    p.add_argument("--source", default=None, help="Source markdown (default: {results-dir}/comparison.md)")
    return p
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        path = freeze(version=args.version, force=args.force, results_dir=Path(args.results_dir), source=Path(args.source) if args.source else None)
    except (FileExistsError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]Freeze refused: {e}[/red]"); return 2
    console.print(f"[green]Wrote {path}[/green]")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

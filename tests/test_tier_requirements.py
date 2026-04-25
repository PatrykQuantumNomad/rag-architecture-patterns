"""REPO-04 trace test — validates each tier's requirements.txt + pyproject extras.

Phase 127 establishes one ``tier-N-name/requirements.txt`` per architecture
tier (1..5). Each requirements file must reference the parent extras (so
tier deps stay in pyproject as the single source of truth) and pyproject
must declare a matching ``[project.optional-dependencies.tier-N]`` entry.

This test also guards against the deprecated ``google-generativeai`` SDK
sneaking back into the lockfile (T-127-08 mitigation).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — runtime is 3.13
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).parent.parent
TIER_DIRS = [
    "tier-1-naive",
    "tier-2-managed",
    "tier-3-graph",
    "tier-4-multimodal",
    "tier-5-agentic",
]

# Accepts either editable-install ref or explicit tier package reference.
TIER_REQUIREMENTS_PATTERN = re.compile(r"-e \.\.\[tier-\d\]|rag-architecture-patterns\[tier-\d\]")


def _load_pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_all_tier_dirs_exist() -> None:
    for tier in TIER_DIRS:
        path = ROOT / tier
        assert path.is_dir(), f"missing tier directory: {tier}"


def test_each_tier_has_requirements_file() -> None:
    for tier in TIER_DIRS:
        req = ROOT / tier / "requirements.txt"
        assert req.is_file(), f"missing {tier}/requirements.txt"


def test_requirements_reference_parent_extras() -> None:
    for tier in TIER_DIRS:
        req = ROOT / tier / "requirements.txt"
        content = req.read_text(encoding="utf-8")
        assert TIER_REQUIREMENTS_PATTERN.search(content), (
            f"{tier}/requirements.txt must reference parent extras "
            "(e.g. '-e ..[tier-N]' or 'rag-architecture-patterns[tier-N]'); "
            f"got:\n{content}"
        )


def test_pyproject_has_tier_extras() -> None:
    pyproject = _load_pyproject()
    extras = pyproject["project"]["optional-dependencies"]
    for n in range(1, 6):
        key = f"tier-{n}"
        assert key in extras, f"pyproject missing [project.optional-dependencies.{key}]"


def test_lockfile_does_not_contain_deprecated_sdk() -> None:
    """T-127-08: guard against ``google-generativeai`` (EOL 2025-08-31)."""
    lockfile = ROOT / "uv.lock"
    if not lockfile.exists():
        # uv.lock lands in Plan 02 Task 1; absence here means tests run from
        # a partial checkout. Skip rather than fail.
        return
    content = lockfile.read_text(encoding="utf-8")
    assert "google-generativeai" not in content, (
        "Deprecated google-generativeai package found in uv.lock. "
        "Use google-genai (unified SDK) instead."
    )

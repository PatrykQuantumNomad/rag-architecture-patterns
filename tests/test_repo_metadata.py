"""REPO-01 trace test — validates README links to the canonical blog placeholder.

Replaces Plan 01's manual ``grep`` step with an automated assertion. The
``@pytest.mark.live`` test additionally checks that the GitHub repo is
public, but only when ``gh`` CLI is available and authenticated. Default
``pytest -m 'not live'`` runs only the README check.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
BLOG_URL = "https://patrykgolabek.dev/blog/rag-architecture-patterns"
GH_REPO = "PatrykQuantumNomad/rag-architecture-patterns"


def test_readme_exists() -> None:
    assert README.is_file(), "README.md missing at repo root"


def test_readme_links_to_blog_placeholder() -> None:
    text = README.read_text(encoding="utf-8")
    assert BLOG_URL in text, (
        f"README.md must link to blog placeholder {BLOG_URL} "
        "(REPO-01 trace; replaces manual grep from Plan 01)"
    )


@pytest.mark.live
def test_repo_is_public() -> None:
    """Live check that the GitHub repo is publicly visible."""
    if shutil.which("gh") is None:
        pytest.skip("gh CLI not installed — public-visibility check skipped")
    try:
        out = subprocess.run(
            [
                "gh",
                "repo",
                "view",
                GH_REPO,
                "--json",
                "visibility",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"gh not authenticated or repo not yet pushed: {exc}")
    data = json.loads(out.stdout)
    assert data.get("visibility") == "PUBLIC", (
        f"expected PUBLIC visibility for {GH_REPO}, got {data}"
    )

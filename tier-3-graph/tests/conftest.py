"""Tier-3-local pytest configuration.

Mirrors ``tier-1-naive/tests/conftest.py``: ensures the repo root is on
``sys.path`` so ``from tier_3_graph import ...`` resolves under pytest, and
loads the repo-root ``.env`` so live tests (Plan 05+) can pick up
``OPENROUTER_API_KEY`` even when pytest is invoked from this subdirectory.

Adds a ``tier3_live_keys`` fixture that skips the calling test if
``OPENROUTER_API_KEY`` is missing — Tier 3 (LightRAG) routes both LLM and
embedding traffic through OpenRouter on a single key.

This conftest does NOT collect tests from ``../tests/`` — pytest treats the
two test dirs as independent collections (no ``__init__.py`` in either, by
the Phase 128-02 follow-on convention).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Make the repo root importable so ``from tier_3_graph.rag import ...`` works
# under pytest's process (replicates what ``main.py`` does at script startup).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load repo-root .env so OPENROUTER_API_KEY populates even when pytest is
# invoked from this subdirectory.
load_dotenv(_REPO_ROOT / ".env")


@pytest.fixture()
def tier3_live_keys() -> None:
    """Skip the calling test if ``OPENROUTER_API_KEY`` is missing.

    Tier 3 routes both embeddings (``openai/text-embedding-3-small``) and
    LLM (default ``google/gemini-2.5-flash``) through OpenRouter. A test
    that depends on this fixture is guaranteed to have a usable key when
    it runs.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set; tier-3 live test skipped")

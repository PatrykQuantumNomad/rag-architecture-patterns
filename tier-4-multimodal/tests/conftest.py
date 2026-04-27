"""Tier 4 live test fixtures.

Mirrors ``tier-3-graph/tests/conftest.py`` (Phase 129 Plan 03 + Plan 07):
ensures the repo root is on ``sys.path`` so ``from tier_4_multimodal import
...`` resolves under pytest, and loads the repo-root ``.env`` so live tests
(Plan 05) can pick up ``OPENROUTER_API_KEY`` even when pytest is invoked
from this subdirectory.

Adds a ``tier4_live_keys_ok`` fixture that skips the calling test if
``OPENROUTER_API_KEY`` is missing — Tier 4 (RAG-Anything composing
LightRAG) routes all LLM, vision, and embedding traffic through OpenRouter
on a single key.

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

# Make the repo root importable so ``from tier_4_multimodal.rag import ...``
# works under pytest's process (replicates what ``main.py`` does at script
# startup).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load repo-root .env so OPENROUTER_API_KEY populates even when pytest is
# invoked from this subdirectory (mirrors Phase 127 Plan 06's commit 08dce6a
# follow-on which established this pattern).
load_dotenv(_REPO_ROOT / ".env", override=False)


@pytest.fixture()
def tier4_live_keys_ok() -> None:
    """Skip the calling test if ``OPENROUTER_API_KEY`` is missing.

    Tier 4 routes EVERY LLM/vision/embedding call (entity extraction during
    ingest + per-image vision pass + embedding batches + per-query LLM
    answer) through OpenRouter on a single key. A test that depends on this
    fixture is guaranteed to have a usable key when it runs.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip(
            "OPENROUTER_API_KEY not set — live Tier 4 tests skip cleanly"
        )

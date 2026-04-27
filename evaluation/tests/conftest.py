"""Evaluation harness test fixtures.

Mirrors tier-5-agentic/tests/conftest.py — repo-root sys.path bootstrap +
load_dotenv() so `pytest -m live` invocations from inside this directory
work standalone (Phase 127 commit 08dce6a precedent; Phase 130 130-04 reuse).
"""
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

load_dotenv(_REPO_ROOT / ".env", override=False)


@pytest.fixture
def live_eval_keys_ok():
    """Skip if OPENROUTER_API_KEY is not set — RAGAS judge routes via OpenRouter."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set — live evaluation tests skip cleanly")


@pytest.fixture
def tier1_index_present():
    """Skip if chroma_db/tier-1-naive/ does not exist — Tier 1 + Tier 5 prereq."""
    chroma_path = _REPO_ROOT / "chroma_db" / "tier-1-naive"
    if not chroma_path.exists():
        pytest.skip(
            f"Tier 1 ChromaDB index not found at {chroma_path}. "
            "Run `python tier-1-naive/main.py --ingest` first."
        )

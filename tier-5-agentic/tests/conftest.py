"""Tier 5 live test fixtures.

Mirrors ``tier-3-graph/tests/conftest.py`` and ``tier-4-multimodal/tests/conftest.py``
(Phase 129 Plan 03 + 07 / Phase 130 Plan 05) — repo-root ``sys.path`` bootstrap +
``load_dotenv`` so pytest invocations from this subdirectory pick up
``OPENROUTER_API_KEY`` from the repo-root ``.env``.

Adds TWO chained pre-condition fixtures:

* ``tier5_live_keys_ok`` — skips if ``OPENROUTER_API_KEY`` is missing. Tier 5
  routes EVERY agent turn (planner LLM + tool-call embeddings) through
  OpenRouter on a single key.
* ``tier1_index_present`` — skips if ``chroma_db/tier-1-naive/`` does not exist.
  Tier 5 reads Tier 1's index (Pitfall 9 read-only invariant); without an
  ingested index, the agent's first ``search_text_chunks`` returns ``[]`` and
  the run is meaningless.

Both fixtures must be used together for the live test to run — either
missing pre-condition cleanly skips with an explanatory message.

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

# Make the repo root importable so ``from tier_5_agentic.agent import ...``
# resolves under pytest's process (replicates what ``main.py`` does at
# script startup).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load repo-root .env so OPENROUTER_API_KEY populates even when pytest is
# invoked from this subdirectory (mirrors Phase 127 Plan 06's commit 08dce6a
# follow-on which established this pattern).
load_dotenv(_REPO_ROOT / ".env", override=False)


@pytest.fixture()
def tier5_live_keys_ok() -> None:
    """Skip the calling test if ``OPENROUTER_API_KEY`` is missing.

    Tier 5's agent loop calls the planner LLM (LiteLLM via OpenRouter) on
    EVERY turn and the embedding model on every ``search_text_chunks``
    invocation. A live test that depends on this fixture is guaranteed to
    have a usable key when it runs.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip(
            "OPENROUTER_API_KEY not set — live Tier 5 tests skip cleanly"
        )


@pytest.fixture()
def tier1_index_present() -> None:
    """Skip the calling test if ``chroma_db/tier-1-naive/`` is missing.

    Tier 5 reads Tier 1's index (Pitfall 9 read-only invariant). The first
    tool call (``search_text_chunks``) opens the collection with
    ``reset=False`` and runs ``retrieve_top_k``; without an existing
    collection the call returns ``[]`` and the agent has no evidence to
    cite. Skip cleanly with the remediation command if absent.
    """
    chroma_path = _REPO_ROOT / "chroma_db" / "tier-1-naive"
    if not chroma_path.exists():
        pytest.skip(
            f"Tier 1 ChromaDB index missing at {chroma_path}. "
            "Run `python tier-1-naive/main.py --ingest` first."
        )

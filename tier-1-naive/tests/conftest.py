"""Tier-1-local pytest configuration.

Adds a ``tier1_live_keys`` fixture that skips the calling test if
``OPENROUTER_API_KEY`` is missing. Since Plan 128-06 Tier 1 routes BOTH
embeddings and chat completion through OpenRouter, a single key gates the
live test (down from the dual OpenAI + Gemini requirement of Plan 128-05).

Pytest's conftest discovery is per-directory, so we also call
``load_dotenv()`` at import here. The repo-root ``tests/conftest.py`` already
does this, but when pytest is invoked from inside ``tier-1-naive/`` (or with
just ``tier-1-naive/tests/test_main_live.py`` as the path), only this
conftest is guaranteed to load — so we duplicate the bootstrap to keep the
``-m live`` invocation self-contained.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Ensure the repo root is on sys.path so `from tier_1_naive import ...` resolves
# in pytest's process. The non-live tests use ``importlib.util`` to load modules
# directly, but the live test exercises the public import path the way ``main.py``
# does — and main.py adds the repo root to sys.path at script-startup time.
# Pytest does NOT replicate that, so we replicate it here for the live test.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load repo-root .env so OPENROUTER_API_KEY populates even when pytest is
# invoked from this subdirectory.
load_dotenv(_REPO_ROOT / ".env")


@pytest.fixture()
def tier1_live_keys() -> None:
    """Skip the calling test if ``OPENROUTER_API_KEY`` is missing.

    Tier 1 routes both embeddings (``openai/text-embedding-3-small``) and
    chat completion (default ``google/gemini-2.5-flash``, override via
    ``--model``) through OpenRouter. A test that depends on this fixture is
    therefore guaranteed to have a usable key when it runs.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set; tier-1 live test skipped")

"""Tier-2-local pytest configuration.

Adds a ``tier2_live_keys_ok`` fixture that skips the calling test if
``GEMINI_API_KEY`` is missing. Tier 2 is Gemini-native (the
``client.file_search_stores.*`` API is not exposed via OpenRouter), so a
single Gemini key gates the live test.

Pytest's conftest discovery is per-directory, so we also call
``load_dotenv()`` at import here. The repo-root ``tests/conftest.py``
already does this, but when pytest is invoked from inside
``tier-2-managed/`` (or with just
``tier-2-managed/tests/test_main_live.py`` as the path), only this
conftest is guaranteed to load — so we duplicate the bootstrap to keep
the ``-m live`` invocation self-contained. Phase 127 Plan 06 commit
``08dce6a`` established this pattern; Phase 128 Plan 05 replicated it for
Tier 1 (``tier1_live_keys``); we mirror it here for Tier 2.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Ensure the repo root is on sys.path so ``from tier_2_managed import ...``
# resolves in pytest's process. ``main.py`` adds the repo root to sys.path at
# script-startup time, but pytest does NOT replicate that — so we replicate
# it here for the live test, which exercises the public import path the way
# the CLI does.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load repo-root .env so GEMINI_API_KEY populates even when pytest is
# invoked from this subdirectory. ``override=False`` means an explicit
# environment variable wins over the .env file (so a CI env can override).
load_dotenv(_REPO_ROOT / ".env", override=False)


@pytest.fixture()
def tier2_live_keys_ok() -> None:
    """Skip the calling test if ``GEMINI_API_KEY`` is missing.

    Tier 2 calls the Gemini File Search API directly (not via OpenRouter —
    ``client.file_search_stores.*`` is a Gemini-native feature). A test
    that depends on this fixture is therefore guaranteed to have a usable
    key when it runs.
    """
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip(
            "GEMINI_API_KEY not set — live Tier 2 tests skip cleanly"
        )

"""Tier-1-local pytest configuration.

Adds a ``tier1_live_keys`` fixture that skips the calling test if EITHER
``OPENAI_API_KEY`` or ``GEMINI_API_KEY`` is missing — Tier 1 needs both for
the end-to-end run (OpenAI for embeddings, Gemini for chat).

Pytest's conftest discovery is per-directory, so we also call
``load_dotenv()`` at import here. The repo-root ``tests/conftest.py`` already
does this, but when pytest is invoked from inside ``tier-1-naive/`` (or with
just ``tier-1-naive/tests/test_main_live.py`` as the path), only this
conftest is guaranteed to load — so we duplicate the bootstrap to keep the
``-m live`` invocation self-contained.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load repo-root .env so OPENAI_API_KEY + GEMINI_API_KEY populate even when
# pytest is invoked from this subdirectory.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@pytest.fixture()
def tier1_live_keys() -> None:
    """Skip the calling test if either OPENAI_API_KEY or GEMINI_API_KEY is missing.

    Tier 1 needs both: OpenAI for ``text-embedding-3-small`` and Gemini for
    ``gemini-2.5-flash`` chat completion. A test that depends on this fixture
    is therefore guaranteed to have both keys present when it runs.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; tier-1 live test skipped")
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set; tier-1 live test skipped")

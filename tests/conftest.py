"""Pytest configuration and shared fixtures for the rag-architecture-patterns suite.

Fixtures
--------
tmp_costs_dir
    Returns a temporary path that callers may pass to
    ``CostTracker.persist(dest_dir=...)`` to keep test artifacts out of the
    real ``evaluation/results/costs/`` directory.

live_keys_ok
    Skip-marker for tests that hit real Gemini APIs. Skips with a
    descriptive reason if ``GEMINI_API_KEY`` is unset / empty.

The ``live`` pytest marker is registered in ``pyproject.toml`` under
``[tool.pytest.ini_options]``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_costs_dir(tmp_path: Path) -> Path:
    """Yields a temporary directory for ``CostTracker.persist`` writes.

    Tests should pass this to ``persist(dest_dir=tmp_costs_dir)`` instead of
    polluting the real ``evaluation/results/costs/`` directory.
    """
    dest = tmp_path / "costs"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


@pytest.fixture()
def live_keys_ok() -> None:
    """Skip the calling test if no GEMINI_API_KEY is configured."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set; live API tests skipped")

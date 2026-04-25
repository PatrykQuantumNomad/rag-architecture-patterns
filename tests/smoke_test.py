"""End-to-end smoke test for the shared/ layer (REPO-03 trace).

Validates the canonical "Phase 127 done?" signal:

* All seven shared modules import cleanly (no module-level Settings()).
* ``shared.config.get_settings()`` returns Settings with a populated
  ``gemini_api_key`` (or surfaces a clear ValidationError).
* A single ``embed("hello world")`` call works against real Gemini.
* A single ``complete("Reply with exactly: ok")`` call works.

Per D-10 / Pitfall 6: target wall time <5s, cost ~$0.0001/run. Use
gemini-2.5-flash + gemini-embedding-001 (defaults) — never gemini-2.5-pro
in the smoke path.

Live tests are gated by ``@pytest.mark.live`` and skip when
``GEMINI_API_KEY`` is unset (via the ``live_keys_ok`` fixture from
conftest.py).
"""
from __future__ import annotations

import pytest


def test_imports() -> None:
    """All shared modules must import without raising — even with no .env.

    Crucially, this test does NOT call ``get_settings()`` — Pattern 5's lazy
    factory contract requires that import-time succeeds in fresh checkouts.
    """
    from shared import (  # noqa: F401
        config,
        cost_tracker,
        display,
        embeddings,
        llm,
        loader,
        pricing,
    )


@pytest.mark.live
def test_required_env_keys(live_keys_ok: None) -> None:
    """Pydantic ValidationError surfaces if .env is missing (live mode only)."""
    from shared.config import get_settings

    try:
        settings = get_settings()
    except Exception as exc:  # pragma: no cover — exercised in fresh checkouts
        pytest.fail(
            "GEMINI_API_KEY required: copy .env.example to .env and set per "
            "https://aistudio.google.com/app/apikey "
            f"(underlying error: {exc})"
        )
    assert settings.gemini_api_key.get_secret_value(), "gemini_api_key empty"


@pytest.mark.live
def test_real_embedding_call(live_keys_ok: None) -> None:
    """Single live embed call (~$0.00002, <2s)."""
    from shared.embeddings import get_embedding_client

    client = get_embedding_client()
    vectors = client.embed("hello world")
    assert isinstance(vectors, list), "embed() must return list[list[float]]"
    assert len(vectors) == 1, "single-string input should yield 1 vector"
    assert isinstance(vectors[0], list), "embedding vector must be a list"
    assert len(vectors[0]) > 100, (
        f"embedding dim suspiciously small: {len(vectors[0])}"
    )
    assert all(isinstance(x, float) for x in vectors[0][:8])


@pytest.mark.live
def test_real_llm_call(live_keys_ok: None) -> None:
    """Single live chat completion (~$0.00008, <3s)."""
    from shared.llm import get_llm_client

    client = get_llm_client()
    resp = client.complete("Reply with exactly: ok")
    assert "ok" in resp.text.lower(), f"expected 'ok' in response, got {resp.text!r}"
    assert resp.input_tokens > 0, "usage_metadata.prompt_token_count missing"
    assert resp.output_tokens > 0, "usage_metadata.candidates_token_count missing"
    assert resp.model, "response.model missing"

"""Non-live unit tests for tier-5-agentic/agent.py.

Asserts the cross-cutting invariants the live agent loop depends on:

* ``DEFAULT_MODEL`` starts with ``"openrouter/"`` (Pitfall 10) — without
  the prefix, ``LitellmModel`` silently routes to a different provider.
* ``INSTRUCTIONS`` is a non-empty system prompt so the agent has guidance
  on which tool to call first.
* ``build_agent`` constructs an :class:`agents.Agent` WITHOUT requiring
  ``OPENROUTER_API_KEY`` at import / construction time — the env var is
  read LAZILY by ``LitellmModel`` at ``Runner.run`` time. Verified via
  ``monkeypatch.delenv``.
* ``build_agent`` raises ``AssertionError`` for slugs missing the
  ``openrouter/`` prefix (Pitfall 10 invariant).

NO live API calls. Plan 130-06 owns the live e2e.
"""
from __future__ import annotations

import pytest

from tier_5_agentic.agent import build_agent, DEFAULT_MODEL, INSTRUCTIONS


def test_default_model_starts_with_openrouter_prefix():
    """Pitfall 10 invariant: LitellmModel requires 'openrouter/' prefix."""
    assert DEFAULT_MODEL.startswith("openrouter/"), (
        f"DEFAULT_MODEL {DEFAULT_MODEL!r} must start with 'openrouter/' for LitellmModel"
    )


def test_instructions_non_empty():
    """The system prompt must be a non-trivial string."""
    assert isinstance(INSTRUCTIONS, str)
    assert len(INSTRUCTIONS.strip()) > 50


def test_build_agent_constructs_without_api_key(monkeypatch):
    """``build_agent`` works with no OPENROUTER_API_KEY (lazy env read)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = build_agent()
    assert agent is not None
    assert getattr(agent, "name", None) == "ResearchAssistant"
    # Agent has both function tools attached.
    tools = getattr(agent, "tools", None)
    assert tools is not None
    assert len(tools) >= 2


def test_build_agent_rejects_non_openrouter_prefix():
    """Assertion in build_agent fires for non-openrouter slugs (Pitfall 10)."""
    with pytest.raises(AssertionError):
        build_agent(model="google/gemini-2.5-flash")  # missing 'openrouter/' prefix

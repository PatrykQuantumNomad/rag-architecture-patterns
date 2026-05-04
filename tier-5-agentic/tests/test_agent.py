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
* ``set_tracing_disabled`` is invoked conditionally on the
  ``RAG_DEBUG_TIER5_TRACING`` env var (Phase 01 Plan 01): default
  disabled (env unset → ``disabled=True``), opt-in enabled
  (``RAG_DEBUG_TIER5_TRACING=1`` → ``disabled=False``).

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


def _reload_agent_with_mocked_tracing(monkeypatch):
    """Patch ``agents.set_tracing_disabled`` and re-execute ``tier_5_agentic.agent``.

    The ``set_tracing_disabled`` call lives at module import time, so we must
    re-import after patching the env var to observe the call site. The
    on-disk ``tier-5-agentic/`` directory is hyphenated, so the standard
    ``importlib.reload`` cannot resolve a parent-package spec. Instead we
    use the shim's ``_load`` helper after popping the module from
    ``sys.modules`` (mirrors the pattern in ``tier-1-naive/tests/test_store.py``
    documented in ``.planning/codebase/TESTING.md``).
    """
    import sys
    from unittest.mock import MagicMock

    import agents
    import tier_5_agentic  # the shim package

    mock_set_tracing = MagicMock()
    # Patch ``agents.set_tracing_disabled`` so the fresh import inside _load
    # picks up our mock when ``from agents import ... set_tracing_disabled``
    # is re-executed.
    monkeypatch.setattr(agents, "set_tracing_disabled", mock_set_tracing)
    # Drop the cached module so the shim re-executes the source file.
    sys.modules.pop("tier_5_agentic.agent", None)
    tier_5_agentic._load("agent")
    return mock_set_tracing


def test_tracing_disabled_default(monkeypatch):
    """Env var unset → ``set_tracing_disabled(disabled=True)`` (Pitfall 8 of 130-RESEARCH)."""
    monkeypatch.delenv("RAG_DEBUG_TIER5_TRACING", raising=False)
    mock_set_tracing = _reload_agent_with_mocked_tracing(monkeypatch)
    assert mock_set_tracing.called, "set_tracing_disabled was never called on import"
    # Inspect the most-recent call (reload triggers a fresh invocation).
    _, kwargs = mock_set_tracing.call_args
    # Accept either kw or positional invocation; the contract is "disabled=True".
    disabled_arg = kwargs.get("disabled")
    if disabled_arg is None and mock_set_tracing.call_args.args:
        disabled_arg = mock_set_tracing.call_args.args[0]
    assert disabled_arg is True, (
        f"Expected disabled=True when RAG_DEBUG_TIER5_TRACING unset, got {disabled_arg!r}"
    )


def test_tracing_enabled_when_env_set(monkeypatch):
    """``RAG_DEBUG_TIER5_TRACING=1`` → ``set_tracing_disabled(disabled=False)``."""
    monkeypatch.setenv("RAG_DEBUG_TIER5_TRACING", "1")
    mock_set_tracing = _reload_agent_with_mocked_tracing(monkeypatch)
    assert mock_set_tracing.called, "set_tracing_disabled was never called on import"
    _, kwargs = mock_set_tracing.call_args
    disabled_arg = kwargs.get("disabled")
    if disabled_arg is None and mock_set_tracing.call_args.args:
        disabled_arg = mock_set_tracing.call_args.args[0]
    assert disabled_arg is False, (
        f"Expected disabled=False when RAG_DEBUG_TIER5_TRACING=1, got {disabled_arg!r}"
    )

"""Tier 5 — agent constructor with LiteLLM-routed model.

Builds the ``Agent`` that the OpenAI Agents SDK ``Runner`` will drive in
``main.py``. The agent owns:

* an ``INSTRUCTIONS`` system prompt that tells the LLM to call ``search_text_chunks``
  first and ``lookup_paper_metadata`` second when citing a paper;
* a :class:`~agents.extensions.models.litellm_model.LitellmModel` instance
  routed to OpenRouter via the ``openrouter/<provider>/<model>`` slug
  convention (Pitfall 10);
* the two ``@function_tool`` callables exported by
  :mod:`tier_5_agentic.tools`.

Pitfall 8 (130-RESEARCH.md): the SDK ships with platform tracing ON by
default. Without an ``OPENAI_TRACING_KEY`` every run logs warnings/errors
to stderr; we call :func:`agents.set_tracing_disabled` at module import time
so the very first call from ``Runner.run`` is already configured.

Pitfall 10: ``LitellmModel`` requires the ``openrouter/`` prefix on the model
slug — ``"google/gemini-2.5-flash"`` (without the prefix) is silently routed
to native Vertex/PaLM and fails with a misleading error. The
``DEFAULT_MODEL`` constant + the ``assert chosen.startswith("openrouter/")``
guard in ``build_agent`` enforce this invariant.

The ``OPENROUTER_API_KEY`` environment variable is read LAZILY by
``LitellmModel`` at ``Runner.run`` time, so this module imports cleanly
without the key — required by the non-live unit tests in
``tests/test_agent.py``.
"""
from __future__ import annotations

import os

from agents import Agent, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

from tier_5_agentic.tools import search_text_chunks, lookup_paper_metadata


# Pitfall 8 anti-pattern guard: SDK ships with platform tracing on; without
# an OpenAI tracing key, every run logs warnings/errors. Disable explicitly
# at module level so the gate fires once on import.
set_tracing_disabled(disabled=True)


# Pitfall 10: must start with "openrouter/" — LitellmModel parses the prefix
# to select the LiteLLM provider plug-in. Drop the prefix and the slug routes
# to a different (and incorrect) backend.
DEFAULT_MODEL = "openrouter/google/gemini-2.5-flash"


INSTRUCTIONS = """\
You are a research assistant grounded in an enterprise knowledge base of 100 ML/IR
papers indexed in a local ChromaDB collection.

To answer the user, you MUST:
1. Call `search_text_chunks` to retrieve evidence chunks for any factual claim.
2. Call `lookup_paper_metadata` to verify the title/authors/year of any paper you
   intend to cite.
3. Cite the `paper_id` (arXiv ID) inline in your answer text.

You may iterate (call multiple tools) up to the runner-imposed turn limit. If a single
tool call returns ambiguous or insufficient results, call it again with refined args
or call the OTHER tool to disambiguate.

If after iterating you still cannot answer, say so clearly — do not fabricate citations.
"""


def build_agent(model: str | None = None) -> Agent:
    """Construct the Tier 5 agent.

    Parameters
    ----------
    model
        Optional override; defaults to :data:`DEFAULT_MODEL`. MUST start with
        ``"openrouter/"`` — Pitfall 10 invariant. Raises ``AssertionError``
        if the prefix is missing.

    Notes
    -----
    The ``OPENROUTER_API_KEY`` env var is read LAZILY by ``LitellmModel`` at
    ``Runner.run`` time, so this constructor works without the key for
    non-live tests. The runtime fast-fail on missing key lives in
    ``main.amain`` (exit 2 + friendly red error).
    """
    chosen = model or DEFAULT_MODEL
    assert chosen.startswith("openrouter/"), (
        f"LitellmModel requires the 'openrouter/' prefix (got {chosen!r}). "
        "See 130-RESEARCH.md Pitfall 10."
    )
    return Agent(
        name="ResearchAssistant",
        instructions=INSTRUCTIONS,
        model=LitellmModel(
            model=chosen,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),  # lazy at runtime
        ),
        tools=[search_text_chunks, lookup_paper_metadata],
    )

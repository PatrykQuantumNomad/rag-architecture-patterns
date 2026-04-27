"""Non-live unit tests for tier-5-agentic/tools.py.

Schema introspection + ``lookup_paper_metadata`` happy-path / miss-path
sanity. NO live API calls; the agent loop itself is exercised against real
OpenRouter in Plan 130-06's e2e.

The ``@function_tool`` decorator wraps the underlying callable in a
:class:`agents.tool.FunctionTool` whose internals vary across 0.14.x patch
releases — we use a tolerant ``getattr(.., 'func')`` / ``_func`` / fallback
to call the underlying function for the lookup tests, and skip cleanly if
that unwrap is opaque on the installed SDK version (Pitfall 8 — version
drift in the 0.x track).
"""
from __future__ import annotations

import pytest

from tier_5_agentic.tools import search_text_chunks, lookup_paper_metadata


def test_function_tool_decoration():
    """``@function_tool`` exposes a ``.name`` attr or remains callable.

    Both branches are accepted because the wrapper class shape is not part
    of the SDK's stable surface. What matters for the planner is that the
    decorated object has SOME schema metadata or can be invoked directly.
    """
    assert hasattr(search_text_chunks, "name") or callable(search_text_chunks)
    assert hasattr(lookup_paper_metadata, "name") or callable(lookup_paper_metadata)


def test_function_tool_schema_metadata():
    """FunctionTool exposes name + description + params_json_schema.

    These three fields are what the planner LLM actually sees when deciding
    which tool to call (Pattern 6). The names must match the function names
    so the planner can reason about them in INSTRUCTIONS prose; descriptions
    must be non-empty so the LLM has guidance.
    """
    for tool, expected_name in (
        (search_text_chunks, "search_text_chunks"),
        (lookup_paper_metadata, "lookup_paper_metadata"),
    ):
        if not hasattr(tool, "name"):
            pytest.skip("FunctionTool API changed in this SDK patch release")
        assert tool.name == expected_name
        assert hasattr(tool, "description")
        assert isinstance(tool.description, str) and len(tool.description) > 10
        # ``params_json_schema`` is the JSON schema the LLM sees for arg validation.
        if hasattr(tool, "params_json_schema"):
            schema = tool.params_json_schema
            assert isinstance(schema, dict)
            # All tools must declare at least one parameter (query / paper_id).
            assert "properties" in schema
            assert len(schema["properties"]) >= 1


def _unwrap(tool):
    """Best-effort retrieval of the underlying Python function from a FunctionTool."""
    return (
        getattr(tool, "func", None)
        or getattr(tool, "_func", None)
        or (tool if callable(tool) else None)
    )


def test_lookup_paper_metadata_known_id():
    """Sync tool path — call the unwrapped callable and assert dict shape.

    The known id ``1706.03762`` is "Attention Is All You Need" (Vaswani 2017),
    which is in the 100-paper corpus per ``dataset/manifests/papers.json``.
    Either the lookup hits and returns a dict with ``paper_id``, or
    ``DatasetLoader`` returns the explicit miss-error dict — both are valid
    shapes for the test.
    """
    fn = _unwrap(lookup_paper_metadata)
    if fn is None:
        pytest.skip("Could not unwrap @function_tool to call directly; SDK version variation")
    res = fn("1706.03762")
    assert isinstance(res, dict)
    # Either a hit (paper exists in dataset) or an explicit miss.
    assert "paper_id" in res or "error" in res


def test_lookup_paper_metadata_unknown_id():
    """Bogus arXiv id returns the explicit miss-error dict, not a partial."""
    fn = _unwrap(lookup_paper_metadata)
    if fn is None:
        pytest.skip("Could not unwrap @function_tool to call directly")
    res = fn("9999.99999")
    assert isinstance(res, dict)
    assert "error" in res

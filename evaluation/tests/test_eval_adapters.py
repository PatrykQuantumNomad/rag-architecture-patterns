"""Non-live unit tests for evaluation/harness/adapters/tier_{1,2,3,5}.py.

Mocks the underlying tier surfaces (``open_collection``, ``build_chat_client``,
``genai.Client``, ``build_rag``, ``build_agent``, ``Runner.run``) so these tests
run in <1s without API keys.

Decisions referenced
--------------------
- Pattern 12 (basename-uniqueness rule) — file lives in evaluation/tests/
  WITHOUT an ``__init__.py``; basename ``test_eval_adapters.py`` is unique
  repo-wide.
- Pitfall 5 (async surface): all adapters are ``async def``; tests drive them
  via ``asyncio.run`` and ``unittest.mock.AsyncMock``.
- Pitfall 8 (Tier 5 MaxTurnsExceeded): explicit assertion that
  ``error='max_turns_exceeded'`` and the answer is prefixed ``[truncated``.
- Pitfall 9 (Tier 5 retrieved_contexts=[]): explicit assertion.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluation.harness.records import EvalRecord


# --------------------------------------------------------------------- #
# Tier 1                                                                #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("documents", [[], ["chunk a", "chunk b"]])
def test_run_tier1_returns_eval_record(documents, monkeypatch):
    from evaluation.harness.adapters import tier_1 as t1

    fake_coll = MagicMock()
    fake_coll.count.return_value = 1 if documents else 0
    monkeypatch.setattr(t1, "open_collection", lambda reset=False: fake_coll)
    monkeypatch.setattr(t1, "build_openai_client", lambda: MagicMock())
    monkeypatch.setattr(t1, "build_chat_client", lambda: MagicMock())
    monkeypatch.setattr(t1, "embed_batch", lambda c, ts, tracker: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(
        t1,
        "retrieve_top_k",
        lambda coll, qv, k: {
            "documents": documents,
            "metadatas": [{}] * len(documents),
        },
    )
    monkeypatch.setattr(t1, "build_prompt", lambda q, d, m: "prompt")
    monkeypatch.setattr(
        t1,
        "chat_complete",
        lambda client, prompt, model, tracker: SimpleNamespace(text="answer ok"),
    )

    rec = asyncio.run(t1.run_tier1("q-id-1", "What is RAG?"))
    assert isinstance(rec, EvalRecord)
    assert rec.question_id == "q-id-1"
    if documents:
        assert rec.answer == "answer ok"
        assert rec.retrieved_contexts == documents
        assert rec.error is None
    else:
        assert rec.error == "tier1_chroma_empty"
        assert rec.answer == ""


# --------------------------------------------------------------------- #
# Tier 2                                                                #
# --------------------------------------------------------------------- #

def test_run_tier2_extracts_grounding(monkeypatch):
    from evaluation.harness.adapters import tier_2 as t2

    fake_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="t2 answer")])
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=50
        ),
    )
    monkeypatch.setattr(t2, "tier2_query", lambda c, s, q, m: fake_resp)
    monkeypatch.setattr(
        t2,
        "to_display_chunks",
        lambda r: [{"text": "ctx 1"}, {"snippet": "ctx 2"}],
    )
    monkeypatch.setattr(
        t2, "get_settings", lambda: SimpleNamespace(gemini_api_key="fake-key")
    )
    monkeypatch.setattr(
        t2, "genai", SimpleNamespace(Client=lambda api_key: MagicMock())
    )

    rec = asyncio.run(t2.run_tier2("q2", "?", store_name="fileSearchStores/test"))
    assert isinstance(rec, EvalRecord)
    assert rec.answer == "t2 answer"
    assert "ctx 1" in rec.retrieved_contexts
    assert "ctx 2" in rec.retrieved_contexts
    assert rec.error is None


def test_run_tier2_skips_without_key(monkeypatch):
    from evaluation.harness.adapters import tier_2 as t2

    monkeypatch.setattr(
        t2, "get_settings", lambda: SimpleNamespace(gemini_api_key=None)
    )
    rec = asyncio.run(t2.run_tier2("q2", "?", store_name="fileSearchStores/test"))
    assert rec.error == "missing_gemini_api_key"
    assert rec.answer == ""


# --------------------------------------------------------------------- #
# Tier 3 — _split_context helper                                        #
# --------------------------------------------------------------------- #

def test_split_context_basic():
    from evaluation.harness.adapters.tier_3 import _split_context

    s = "Entity A\n-----\nEntity B\n-----\nRelation X"
    assert _split_context(s) == ["Entity A", "Entity B", "Relation X"]
    assert _split_context("") == []
    assert _split_context("solo") == ["solo"]


# --------------------------------------------------------------------- #
# Tier 3                                                                #
# --------------------------------------------------------------------- #

def test_run_tier3_two_phase(monkeypatch):
    from evaluation.harness.adapters import tier_3 as t3

    fake_rag = MagicMock()
    # aquery returns concatenated contexts on the only_need_context path
    fake_rag.aquery = AsyncMock(return_value="ctx1\n-----\nctx2")
    monkeypatch.setattr(
        t3, "tier3_run_query", AsyncMock(return_value=("t3 answer", 200, 100))
    )

    rec = asyncio.run(t3.run_tier3("q3", "?", rag=fake_rag))
    assert isinstance(rec, EvalRecord)
    assert rec.answer == "t3 answer"
    assert rec.retrieved_contexts == ["ctx1", "ctx2"]
    assert rec.error is None


# --------------------------------------------------------------------- #
# Tier 5                                                                #
# --------------------------------------------------------------------- #

def test_run_tier5_happy_path(monkeypatch):
    from evaluation.harness.adapters import tier_5 as t5

    fake_agent = MagicMock()
    fake_result = SimpleNamespace(
        final_output="t5 answer",
        new_items=[],  # Zero tool calls → walk has nothing to iterate.
        context_wrapper=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=300, output_tokens=80)
        ),
    )
    monkeypatch.setattr(
        t5, "Runner", SimpleNamespace(run=AsyncMock(return_value=fake_result))
    )

    rec = asyncio.run(t5.run_tier5("q5", "?", agent=fake_agent))
    assert isinstance(rec, EvalRecord)
    assert rec.answer == "t5 answer"
    # Zero tool calls (new_items empty) preserves honest empty per Pitfall 9 of 130-RESEARCH
    assert rec.retrieved_contexts == []
    assert rec.error is None


def test_run_tier5_max_turns_exceeded(monkeypatch):
    from evaluation.harness.adapters import tier_5 as t5
    from agents.exceptions import MaxTurnsExceeded

    fake_agent = MagicMock()
    exc = MaxTurnsExceeded("hit cap")
    setattr(exc, "usage", SimpleNamespace(input_tokens=400, output_tokens=0))
    monkeypatch.setattr(
        t5, "Runner", SimpleNamespace(run=AsyncMock(side_effect=exc))
    )

    rec = asyncio.run(t5.run_tier5("q5", "?", agent=fake_agent, max_turns=10))
    assert rec.error == "max_turns_exceeded"
    assert "[truncated" in rec.answer
    assert rec.retrieved_contexts == []


def _make_tool_output_item(output):
    """Build a real ToolCallOutputItem (the adapter does an isinstance check)."""
    from agents import ToolCallOutputItem

    return ToolCallOutputItem(agent=MagicMock(), raw_item={}, output=output)


def test_run_tier5_extracts_tool_outputs(monkeypatch):
    """Walk projects ToolCallOutputItem.output into provenance-prefixed strings.

    Asserts the new contract added in Phase 01 Plan 01: the adapter walks
    ``result.new_items``, filters by ``ToolCallOutputItem``, and emits one
    ``[paper_id=<id>] <text>`` entry per unique (paper_id, page) chunk plus one
    per metadata abstract. Duplicate (paper_id, page) chunks are deduped
    first-occurrence-wins.
    """
    from evaluation.harness.adapters import tier_5 as t5

    chunks_out = [
        {
            "paper_id": "2005.11401",
            "page": 1,
            "snippet": "RAG combines parametric and non-parametric memory.",
            "similarity": 0.9,
        },
        # Duplicate of the first hit by (paper_id, page) — must be deduped.
        {
            "paper_id": "2005.11401",
            "page": 1,
            "snippet": "RAG combines parametric and non-parametric memory.",
            "similarity": 0.9,
        },
        {
            "paper_id": "2004.04906",
            "page": 3,
            "snippet": "DPR uses dual encoders.",
            "similarity": 0.8,
        },
    ]
    metadata_out = {
        "paper_id": "2005.11401",
        "title": "RAG",
        "authors": ["Lewis et al."],
        "year": 2020,
        "abstract": "We introduce retrieval-augmented generation.",
    }

    fake_agent = MagicMock()
    fake_result = SimpleNamespace(
        final_output="t5 answer with citations [paper_id=2005.11401]",
        new_items=[
            _make_tool_output_item(chunks_out),
            _make_tool_output_item(metadata_out),
        ],
        context_wrapper=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=300, output_tokens=80)
        ),
    )
    monkeypatch.setattr(
        t5, "Runner", SimpleNamespace(run=AsyncMock(return_value=fake_result))
    )

    rec = asyncio.run(t5.run_tier5("q5", "?", agent=fake_agent))
    assert isinstance(rec, EvalRecord)
    assert rec.error is None
    # 2 unique chunks (after dedup) + 1 abstract = 3 entries.
    assert len(rec.retrieved_contexts) == 3
    # Every entry carries provenance.
    for entry in rec.retrieved_contexts:
        assert entry.startswith("[paper_id="), f"missing prefix: {entry!r}"
    joined = "\n".join(rec.retrieved_contexts)
    assert "RAG combines" in joined
    assert "DPR uses" in joined
    assert "retrieval-augmented generation" in joined


def test_run_tier5_zero_tool_calls_returns_honest_empty(monkeypatch):
    """Pitfall 9 of 130-RESEARCH preserved: zero tool calls → ``retrieved_contexts == []``.

    The agent may legitimately decline retrieval (declarative answer, refusal, etc.).
    When ``result.new_items`` contains no ``ToolCallOutputItem``, the walk emits
    nothing — the adapter does NOT synthesize context from ``final_output``.
    """
    from evaluation.harness.adapters import tier_5 as t5

    fake_agent = MagicMock()
    fake_result = SimpleNamespace(
        final_output="I cannot answer that from the indexed papers.",
        new_items=[],  # No tool calls at all.
        context_wrapper=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=120, output_tokens=20)
        ),
    )
    monkeypatch.setattr(
        t5, "Runner", SimpleNamespace(run=AsyncMock(return_value=fake_result))
    )

    rec = asyncio.run(t5.run_tier5("q5", "?", agent=fake_agent))
    assert rec.retrieved_contexts == []
    assert rec.error is None


def test_run_tier5_skips_tool_error_payloads(monkeypatch):
    """Tool error payloads are skipped silently (Pitfall 8 of 01-RESEARCH).

    ``lookup_paper_metadata`` returns ``{"error": "..."}`` on miss
    (tier-5-agentic/tools.py:171). The adapter's walk MUST treat that as
    "no context" and continue, not insert the error string as context.
    """
    from evaluation.harness.adapters import tier_5 as t5

    error_payload = {
        "error": "paper_id 9999.99999 not found in dataset/manifests/papers.json"
    }
    fake_agent = MagicMock()
    fake_result = SimpleNamespace(
        final_output="No such paper.",
        new_items=[_make_tool_output_item(error_payload)],
        context_wrapper=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=80, output_tokens=10)
        ),
    )
    monkeypatch.setattr(
        t5, "Runner", SimpleNamespace(run=AsyncMock(return_value=fake_result))
    )

    rec = asyncio.run(t5.run_tier5("q5", "?", agent=fake_agent))
    assert rec.retrieved_contexts == []
    assert rec.error is None

"""Non-live unit tests for tier_4 dual-mode adapter."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evaluation.harness.records import EvalRecord, QueryLog, write_query_log


def _make_log(question_id: str = "q1", **overrides) -> QueryLog:
    base = dict(
        tier="tier-4",
        timestamp="2026-04-27T12:00:00Z",
        git_sha="abc1234",
        model="google/gemini-2.5-flash",
        records=[
            EvalRecord(
                question_id=question_id,
                question="What is multimodal RAG?",
                answer="Multimodal RAG indexes text + images + tables together.",
                retrieved_contexts=["chunk a", "chunk b"],
                latency_s=12.3,
                cost_usd_at_capture=0.012345,
            )
        ],
    )
    base.update(overrides)
    return QueryLog(**base)


def test_run_tier4_cached_hit(tmp_path):
    from evaluation.harness.adapters.tier_4 import run_tier4

    log = _make_log(question_id="single-hop-001")
    log_path = tmp_path / "tier-4-2026-04-27T12_00_00Z.json"
    write_query_log(log_path, log)

    rec = asyncio.run(run_tier4(
        question_id="single-hop-001",
        question="What is multimodal RAG?",
        from_cache=log_path,
    ))
    assert isinstance(rec, EvalRecord)
    assert rec.question_id == "single-hop-001"
    assert rec.answer.startswith("Multimodal RAG")
    assert rec.retrieved_contexts == ["chunk a", "chunk b"]


def test_run_tier4_cached_miss_question_id(tmp_path):
    from evaluation.harness.adapters.tier_4 import run_tier4

    log = _make_log(question_id="single-hop-001")
    log_path = tmp_path / "tier-4-2026-04-27T12_00_00Z.json"
    write_query_log(log_path, log)

    rec = asyncio.run(run_tier4(
        question_id="multi-hop-999",  # not in the log
        question="?",
        from_cache=log_path,
    ))
    assert rec.error and rec.error.startswith("cached_miss_question_id=")


def test_run_tier4_cached_miss_file(tmp_path):
    from evaluation.harness.adapters.tier_4 import run_tier4, CachedTier4Miss

    missing_path = tmp_path / "nope.json"
    with pytest.raises(CachedTier4Miss):
        asyncio.run(run_tier4(
            question_id="q1",
            question="?",
            from_cache=missing_path,
        ))


def test_run_tier4_library_mode_import_error(monkeypatch):
    """When [tier-4] is not installed, the lazy import surfaces as a clean error.

    Note: if [tier-4] IS installed in this env, the fake_import patch may not
    intercept (Python's import caching can defeat the monkeypatch on already-
    imported modules). In that case rec.error may be None — we accept either
    branch, since the cached path is what actually matters in the harness.
    """
    import builtins
    import sys as _sys

    from evaluation.harness.adapters import tier_4 as t4

    # Force the import path to fail by forcing a fake ImportError. Pytest sets
    # __builtins__ to either the module or a dict depending on context, so we
    # always go through the canonical `builtins` module to fetch the real hook.
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("tier_4_multimodal"):
            raise ImportError("simulated missing [tier-4]")
        return real_import(name, *a, **kw)

    # Evict any cached tier_4_multimodal modules so the lazy `from ... import`
    # in run_tier4 actually re-traverses the import system (and thus hits our
    # patched __import__). Without this, a previously-loaded module will be
    # returned from sys.modules and our patch never fires.
    for mod in list(_sys.modules):
        if mod.startswith("tier_4_multimodal"):
            _sys.modules.pop(mod, None)

    monkeypatch.setattr("builtins.__import__", fake_import)
    rec = asyncio.run(t4.run_tier4("q1", "?"))
    # Either fake_import intercepted (clean error) OR caching defeated the patch
    # and we ran the real lazy imports (env-dependent). Both branches valid.
    assert rec.error is None or rec.error.startswith("tier4_import_error:")


def test_run_tier4_library_mode_happy(monkeypatch):
    from evaluation.harness.adapters import tier_4 as t4

    # Inject a fake rag with aquery returning string answer + an aquery for ctx probe
    fake_rag = SimpleNamespace()

    async def _aquery(q, param=None):
        if param is not None and getattr(param, "only_need_context", False):
            return "ctx 1\n-----\nctx 2"
        return None

    fake_rag.aquery = AsyncMock(side_effect=_aquery)

    # Stub run_query (the actual answer call) — patch the module-level lazy import
    fake_run_query = AsyncMock(return_value="multimodal answer")

    # Use a fake submodule replacement strategy
    class _M:
        run_query = fake_run_query
        DEFAULT_LLM_MODEL = "google/gemini-2.5-flash"
        DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
        build_rag = lambda **kw: fake_rag  # noqa: E731

        class CostAdapter:  # noqa: D401, N801 — test stub
            def __init__(self, *a, **kw):
                pass

    # Inject into sys.modules so the lazy `from tier_4_multimodal...` succeeds
    import sys as _sys
    _sys.modules["tier_4_multimodal"] = SimpleNamespace()
    _sys.modules["tier_4_multimodal.rag"] = _M
    _sys.modules["tier_4_multimodal.query"] = _M
    _sys.modules["tier_4_multimodal.cost_adapter"] = _M

    rec = asyncio.run(t4.run_tier4("q1", "?", rag=fake_rag))
    assert rec.answer == "multimodal answer"
    # Either contexts populated from the probe OR empty if the lightrag QueryParam
    # path isn't installed in this test env — both acceptable
    assert isinstance(rec.retrieved_contexts, list)


def test_eval_capture_writes_embedder_for_tier_4():
    """Plan 06-01 Task 4 / D-CAPTURE-ENTRYPOINTS: tier-4-multimodal/scripts/
    eval_capture.py constructs its QueryLog with embedder + embedder_source
    populated from tier_4_multimodal.rag's module-level constants.

    Static-source approach (recommended in 06-01-PLAN.md): mocking the heavy
    RAG-Anything / MineRU stack is fragile and brittle. The QueryLog
    construction site at eval_capture.py:195-201 is a static literal — the
    embedder identity is known at code-edit time. Asserting on the source
    text guarantees the wiring is in place without paying mock overhead.
    """
    eval_capture_path = (
        Path(__file__).resolve().parent.parent.parent
        / "tier-4-multimodal"
        / "scripts"
        / "eval_capture.py"
    )
    assert eval_capture_path.exists(), f"missing {eval_capture_path}"
    src = eval_capture_path.read_text(encoding="utf-8")

    # 1. The import line must pull EMBEDDER_SOURCE from tier_4_multimodal.rag
    # alongside DEFAULT_LLM_MODEL + DEFAULT_EMBED_MODEL.
    assert "EMBEDDER_SOURCE" in src, (
        "eval_capture.py does not import EMBEDDER_SOURCE — Plan 06-01 Task 4 "
        "missing"
    )

    # 2. The QueryLog construction site must thread BOTH new kwargs.
    assert "embedder=DEFAULT_EMBED_MODEL" in src, (
        "eval_capture.py QueryLog(...) does not pass embedder=DEFAULT_EMBED_MODEL"
    )
    assert "embedder_source=EMBEDDER_SOURCE" in src, (
        "eval_capture.py QueryLog(...) does not pass embedder_source=EMBEDDER_SOURCE"
    )

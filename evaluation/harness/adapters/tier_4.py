"""Tier 4 (Multimodal RAG-Anything) — dual-mode eval adapter.

Pattern 8 in 131-RESEARCH:
  * cached mode — read a user-pre-captured QueryLog JSON (PRIMARY path; Phase 130 SC-1 deferral)
  * library mode — direct import of tier_4_multimodal.{rag, query} (FALLBACK; requires [tier-4] extra)

Why dual-mode: Tier 4 needs MineRU (~3-5 GB models + paddlepaddle/torch) which the
sandbox's OMP shmem block prevents. Users run Tier 4 locally via
`python tier-4-multimodal/scripts/eval_capture.py` and ship the resulting JSON
back; the harness then reads it offline.

Pitfall 7 from 131-RESEARCH: RAG-Anything 1.2.10 aquery returns a string, NOT
structured chunks. We attempt the underlying lightrag only_need_context probe
when the library path runs; on failure, retrieved_contexts is [] (honest empty
→ Plan 05's nan_reason='empty_contexts').
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.cost_tracker import CostTracker

from evaluation.harness.records import EvalRecord, QueryLog, read_query_log


class CachedTier4Miss(FileNotFoundError):
    """Raised when a user-pre-captured Tier 4 query log is requested but absent.

    Plan 04's run.py catches this and skips Tier 4 with a footer note in
    Plan 06's comparison.md.
    """


def _find_record(log: QueryLog, question_id: str) -> Optional[EvalRecord]:
    for rec in log.records:
        if rec.question_id == question_id:
            return rec
    return None


async def run_tier4(
    question_id: str,
    question: str,
    mode: str = "hybrid",
    from_cache: Optional[Path] = None,
    rag=None,
    tracker: Optional[CostTracker] = None,
) -> EvalRecord:
    """Tier 4 query — cached mode primary, library mode fallback.

    Parameters
    ----------
    question_id : str
        Golden Q&A id used to look up the matching record in cached mode.
    question : str
        The natural-language question. Echoed into the EvalRecord on cache miss.
    mode : str
        RAG-Anything query mode (naive, local, global, hybrid, mix). Default 'hybrid'.
    from_cache : Path | None
        Path to a user-pre-captured QueryLog JSON (Pattern 8 primary). When set,
        return the matching record by question_id; raise CachedTier4Miss if file
        missing. The QueryLog is loaded ONCE per harness invocation and cached
        on the function (lru_cache on a private helper would be cleaner, but
        Plan 04's run.py controls the loop and can hold the QueryLog itself).
    rag : RAGAnything | None
        Optional pre-initialized RAGAnything instance for library mode.
    tracker : CostTracker | None
        Optional shared tracker (Pattern 11 collision avoidance).
    """
    # --- Cached mode (primary) ---
    if from_cache is not None:
        path = Path(from_cache)
        if not path.exists():
            raise CachedTier4Miss(
                f"Tier 4 cached query log not found at {path}. "
                "Run `python tier-4-multimodal/scripts/eval_capture.py` locally and "
                "copy the output back into evaluation/results/queries/."
            )
        log = read_query_log(path)
        rec = _find_record(log, question_id)
        if rec is None:
            return EvalRecord(
                question_id=question_id,
                question=question,
                answer="",
                latency_s=0.0,
                cost_usd_at_capture=0.0,
                error=f"cached_miss_question_id={question_id!r}",
            )
        return rec

    # --- Library mode (fallback) ---
    own_tracker = tracker is None
    if own_tracker:
        tracker = CostTracker("tier-4-eval")

    # Lazy imports — fail fast with a clear message when [tier-4] is not installed.
    try:
        from tier_4_multimodal.rag import build_rag, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL
        from tier_4_multimodal.cost_adapter import CostAdapter
        from tier_4_multimodal.query import run_query as tier4_run_query
    except ImportError as exc:
        return EvalRecord(
            question_id=question_id,
            question=question,
            answer="",
            latency_s=0.0,
            cost_usd_at_capture=0.0,
            error=f"tier4_import_error: {exc}",
        )

    own_rag = rag is None
    if own_rag:
        adapter = CostAdapter(tracker, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL)
        rag = build_rag(llm_token_tracker=adapter)
        await rag.initialize_storages()

    contexts: list[str] = []
    # Best-effort context probe (Pitfall 7); raganything 1.2.10 wraps lightrag
    # so the QueryParam path may or may not surface — try, fall back gracefully.
    try:
        from lightrag import QueryParam
        ctx_str = await rag.aquery(question, param=QueryParam(mode=mode, only_need_context=True))
        if ctx_str:
            contexts = [c.strip() for c in str(ctx_str).split("-----") if c.strip()]
    except Exception:  # noqa: BLE001 — best-effort fallback
        contexts = []

    t0 = time.monotonic()
    answer = await tier4_run_query(rag, question, mode=mode)
    latency = time.monotonic() - t0

    return EvalRecord(
        question_id=question_id,
        question=question,
        answer=str(answer) if answer else "",
        retrieved_contexts=contexts,
        latency_s=latency,
        cost_usd_at_capture=tracker.total_usd(),
    )


__all__ = ["run_tier4", "CachedTier4Miss"]

"""Tier 3 cost-tracking adapter — bridges LightRAG ``token_tracker`` callbacks
into ``shared.cost_tracker.CostTracker``.

Probe-validated protocol on lightrag-hku==1.4.15
================================================

The probe in ``scripts/probe_lightrag_token_tracker.py`` (Phase 129 Plan 03
Task 1) confirmed that on lightrag-hku==1.4.15:

* ``lightrag.llm.openai.openai_complete_if_cache`` accepts a ``token_tracker``
  kwarg. After each LLM completion (streaming or non-streaming) it calls::

      token_tracker.add_usage({
          "prompt_tokens":     <int>,
          "completion_tokens": <int>,
          "total_tokens":      <int>,
      })

* ``lightrag.llm.openai.openai_embed`` ALSO accepts ``token_tracker``. After
  each embedding batch it calls::

      token_tracker.add_usage({
          "prompt_tokens": <int>,
          "total_tokens":  <int>,
          # NB: NO completion_tokens — embeddings have no completions.
      })

This adapter therefore exposes a single ``add_usage(usage)`` method that
dispatches to ``CostTracker.record_llm`` when ``completion_tokens`` is present
and to ``CostTracker.record_embedding`` when it is absent. That keeps a
single adapter instance reusable as the ``token_tracker=`` argument for both
the LLM closure AND the embedding closure in ``rag.py``.

Resolution of RESEARCH Open Q1: Outcome A — the protocol matched the
research hypothesis, with the bonus discovery that ``openai_embed`` also
accepts a tracker (so we capture embedding cost too, not just LLM cost).

Fallback (DOCUMENTED ONLY — not active)
---------------------------------------

If a future LightRAG release removes ``token_tracker`` from either function,
the Tier 3 process can monkey-patch
``openai.AsyncOpenAI.chat.completions.create`` (and the corresponding
``embeddings.create``) at module load to capture ``response.usage`` directly::

    import openai
    _orig = openai.AsyncOpenAI.chat.completions.create

    def _patched(self, *args, **kwargs):
        coro = _orig(self, *args, **kwargs)

        async def _wrap():
            resp = await coro
            usage = getattr(resp, "usage", None)
            if usage is not None:
                GLOBAL_TRACKER.record_llm(
                    model,
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                )
            return resp

        return _wrap()

    openai.AsyncOpenAI.chat.completions.create = _patched

Implementation deferred until / unless the probe stops finding
``add_usage`` references — at which point the fallback would land in
``query.py`` (Plan 05) so that activation is per-Tier-3-process and does
not leak into other tiers sharing the same Python interpreter.
"""
from __future__ import annotations

from typing import Any


class CostAdapter:
    """Forward LightRAG token-tracker callbacks into ``shared.cost_tracker.CostTracker``.

    Attach a single instance as the ``token_tracker=`` argument for BOTH the
    LLM closure (``llm_model_func``) and the embedding closure
    (``embedding_func``). Per-call dispatch on ``completion_tokens`` presence
    routes LLM usage to ``record_llm`` and embedding usage to
    ``record_embedding``.

    Parameters
    ----------
    tracker
        A ``shared.cost_tracker.CostTracker`` (or any object with
        ``record_llm(model, prompt, completion)`` + ``record_embedding(model,
        prompt)`` methods).
    llm_model
        Model slug used for LLM calls (e.g. ``"google/gemini-2.5-flash"``).
        Recorded against each LLM usage entry so cost lookup hits the right
        row in ``shared.pricing.PRICES``.
    embed_model
        Model slug used for embedding calls (e.g.
        ``"openai/text-embedding-3-small"``). Recorded against each embedding
        usage entry.
    """

    def __init__(self, tracker, llm_model: str, embed_model: str):
        self.tracker = tracker
        self.llm_model = llm_model
        self.embed_model = embed_model

    def add_usage(self, usage: Any) -> None:
        """LightRAG calls this with a token-counts dict (or usage object).

        Dispatch: if the usage carries a non-zero ``completion_tokens`` field
        OR explicitly has the key ``completion_tokens`` (even if zero), it's
        an LLM call → ``record_llm``. Otherwise it's an embedding call →
        ``record_embedding``.
        """
        if usage is None:
            return

        prompt = self._extract(usage, "prompt_tokens")
        # Probe found embed payloads do NOT include the key; LLM payloads DO.
        # Use _has_key (not just non-zero value) so a 0-completion LLM record
        # still routes to record_llm.
        if self._has_key(usage, "completion_tokens"):
            completion = self._extract(usage, "completion_tokens")
            if prompt or completion:
                self.tracker.record_llm(self.llm_model, prompt, completion)
        else:
            if prompt:
                self.tracker.record_embedding(self.embed_model, prompt)

    @staticmethod
    def _has_key(usage: Any, key: str) -> bool:
        if isinstance(usage, dict):
            return key in usage
        return hasattr(usage, key)

    @staticmethod
    def _extract(usage: Any, key: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return int(usage.get(key) or 0)
        if hasattr(usage, key):
            return int(getattr(usage, key) or 0)
        return 0

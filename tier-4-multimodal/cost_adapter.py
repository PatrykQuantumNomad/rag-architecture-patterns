"""Tier 4 — RAG-Anything → CostTracker bridge.

Inherits LightRAG's ``token_tracker`` protocol (Phase 129 Plan 03 Open Q1
empirically resolved — see ``tier-3-graph/cost_adapter.py`` for the verbatim
probe results). RAG-Anything composes a LightRAG instance internally, so the
same protocol applies: ``add_usage(payload)`` where payload is a dict with
``prompt_tokens`` / optional ``completion_tokens`` / ``total_tokens``.

LLM payloads include ``completion_tokens``; embedding payloads OMIT it.
Dispatch on key presence (NOT value), so a 0-completion LLM record still
routes correctly.

Mirror of ``tier-3-graph/cost_adapter.py`` (Plan 129-03) — same constructor
signature ``(tracker, llm_model, embed_model)`` and same dispatch logic. A
single instance is threadable through both the LLM closure and the embedding
closure via ``token_tracker=`` so the full Tier 4 cost is captured under one
adapter.
"""
from __future__ import annotations

from typing import Any

from shared.cost_tracker import CostTracker


class CostAdapter:
    """Forward RAG-Anything (LightRAG) token-tracker callbacks to ``CostTracker``.

    Parameters
    ----------
    tracker
        A ``shared.cost_tracker.CostTracker`` (or any object exposing
        ``record_llm(model, prompt, completion)`` +
        ``record_embedding(model, prompt)``).
    llm_model
        Model slug for LLM calls (e.g. ``"google/gemini-2.5-flash"``).
    embed_model
        Model slug for embedding calls (e.g.
        ``"openai/text-embedding-3-small"``).
    """

    def __init__(self, tracker: CostTracker, llm_model: str, embed_model: str):
        self.tracker = tracker
        self.llm_model = llm_model
        self.embed_model = embed_model

    def add_usage(self, usage: Any) -> None:
        """Dispatch on ``completion_tokens`` key presence.

        * Key present → LLM call → ``record_llm``.
        * Key absent  → embedding call → ``record_embedding``.

        Defensive: ``None`` payloads silently no-op; non-numeric values
        coerce to zero via ``int(... or 0)``.
        """
        if usage is None:
            return

        prompt = self._extract(usage, "prompt_tokens")
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

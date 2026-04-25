"""rag-architecture-patterns shared utilities.

Modules
-------
config         Pydantic Settings (lazy ``get_settings()`` factory).
pricing        Per-model USD/1M-token table.
llm            ``get_llm_client()`` — Gemini chat completions.
embeddings     ``get_embedding_client()`` — Gemini embeddings.
loader         ``DatasetLoader`` for ``dataset/manifests/*.json``.
display        ``render_query_result(...)`` — rich Panel + Table renderer.
cost_tracker   ``CostTracker`` — records token usage, computes USD, persists JSON.

All modules are import-safe without an ``.env`` file. Modules that need API
keys (``llm``, ``embeddings``) call ``shared.config.get_settings()`` lazily;
that call raises ``pydantic.ValidationError`` if ``GEMINI_API_KEY`` is unset
— which is the spec'd validation contract.
"""

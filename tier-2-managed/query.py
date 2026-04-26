"""Tier 2 (Gemini File Search) — query path with FileSearch tool + grounding extraction.

This module is intentionally tiny: it wraps ``client.models.generate_content``
with a ``FileSearch`` tool reference (Pattern 3 in 129-RESEARCH.md), and
maps the returned ``grounding_metadata.grounding_chunks`` into the chunk
shape that ``shared.display.render_query_result`` expects. That mapping is
Tier 2's "managed-RAG win" — citations come back from the model itself,
not from a separate retrieval call.

Decisions referenced (see 129-RESEARCH.md):

- Pattern 3: managed retrieval as a tool, not a separate API call. The store
  name is passed via ``types.FileSearch(file_search_store_names=[store_name])``
  and the model decides when to consult it.
- Pitfall 6: ``grounding_metadata`` is ``None`` when the model concludes
  the corpus does not contain the answer. ``to_display_chunks`` defends
  every level of the access chain (candidates → metadata → chunks → ctx).
- Open Q3: ``score`` may not be exposed by ``gemini-2.5-flash`` (only the
  Pro models surface it). ``getattr(ctx, "score", 0.0) or 0.0`` is the
  belt-and-braces default so renderer formatting never raises.

Public API
----------

* :data:`DEFAULT_MODEL` — ``"gemini-2.5-flash"`` (cheap, capable, matches Tier 1's narrative).
* :func:`query` — one-shot ``generate_content`` call with the FileSearch tool attached.
* :func:`to_display_chunks` — defensive grounding-metadata → display-chunks adapter.
"""

from __future__ import annotations

from google import genai
from google.genai import types

# Default chat model for the answer step. Matches Tier 1's narrative and
# keeps the tier-to-tier comparison apples-to-apples.
DEFAULT_MODEL: str = "gemini-2.5-flash"


def query(
    client: genai.Client,
    store_name: str,
    question: str,
    model: str = DEFAULT_MODEL,
):
    """Call ``generate_content`` with a ``FileSearch`` tool attached.

    The model decides whether to consult the store; when it does, the
    response carries ``candidates[0].grounding_metadata.grounding_chunks``
    populated with ``retrieved_context.{title, text, score}``. When the
    corpus does not contain the answer, ``grounding_metadata`` is ``None``
    (Pitfall 6) — :func:`to_display_chunks` handles that case.

    Parameters
    ----------
    client
        A constructed ``google.genai.Client``. Caller owns construction so
        this module stays test-friendly (no API key needed at import time).
    store_name
        The server-assigned FileSearchStore resource name
        (``fileSearchStores/<id>``). Obtain via
        ``tier_2_managed.store.get_or_create_store(client).name``.
    question
        The user's natural-language question.
    model
        Gemini chat model slug. Defaults to :data:`DEFAULT_MODEL`.

    Returns
    -------
    The raw ``GenerateContentResponse`` so the caller can read
    ``response.text``, ``response.usage_metadata``, and pass the response
    through :func:`to_display_chunks` for rendering.
    """
    return client.models.generate_content(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name]
                    )
                )
            ],
        ),
    )


def to_display_chunks(response) -> list[dict]:
    """Map ``grounding_metadata.grounding_chunks`` → ``shared.display`` chunks.

    Tier 2's managed-RAG citations land here. Each grounding chunk's
    ``retrieved_context`` carries ``title`` (the document's
    ``display_name``), ``text`` (the snippet the model used), and
    optionally ``score`` (only on Pro models — Open Q3).

    Defensive at every level:

    - ``response.candidates`` may be empty if the model refused.
    - ``candidates[0].grounding_metadata`` is ``None`` when the model
      concluded the corpus did not contain the answer (Pitfall 6).
    - ``grounding_metadata.grounding_chunks`` may be ``None`` even when
      the wrapper exists.
    - ``ch.retrieved_context`` may be ``None`` for non-FileSearch tool
      groundings (e.g., search-tool fallbacks); skip those.
    - ``ctx.score`` may be missing on flash-tier models — fall back to
      ``0.0`` so the renderer's ``f"{score:.3f}"`` formatting never raises.

    Returns an empty list when nothing is groundable. Snippets are clipped
    to 200 chars to match the Tier 1 / shared.display convention.
    """
    out: list[dict] = []
    cands = getattr(response, "candidates", None) or []
    if not cands:
        return out
    grounding = getattr(cands[0], "grounding_metadata", None)
    if not grounding:
        return out
    chunks = getattr(grounding, "grounding_chunks", None) or []
    for ch in chunks:
        ctx = getattr(ch, "retrieved_context", None)
        if ctx is None:
            continue
        snippet = (getattr(ctx, "text", "") or "")[:200]
        out.append(
            {
                "doc_id": getattr(ctx, "title", None) or "(unnamed)",
                "score": float(getattr(ctx, "score", 0.0) or 0.0),
                "snippet": snippet,
            }
        )
    return out

"""Tier 1: PDF text extraction + page-aware chunking.

Page-aware design: chunks NEVER cross page boundaries, so every chunk has
exact (paper_id, page) provenance for citation rendering. Token-window
chunking via tiktoken cl100k_base (the encoding used by
text-embedding-3-small) — sized at 512 tokens with 64-token overlap per
the 2025-26 RAG chunking benchmarks (128-RESEARCH.md Pitfall 1).
"""
from __future__ import annotations

import fitz  # PyMuPDF
import tiktoken

# Constants — locked by 128-RESEARCH.md Pitfall 1; do NOT change without re-running benchmarks.
ENC = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64
_STRIDE = CHUNK_TOKENS - OVERLAP_TOKENS  # 448


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number_1_indexed, text), ...] for non-empty pages.

    Pages whose extracted text is empty / whitespace-only are SKIPPED.
    Page numbers are 1-indexed (matches academic citation convention and
    the rendered chunk ids).
    """
    out: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            txt = page.get_text("text") or ""
            if txt.strip():
                out.append((i + 1, txt))
    return out


def chunk_page(text: str, paper_id: str, page: int) -> list[dict]:
    """Token-window chunks within a SINGLE page.

    Returns a list of chunk dicts shaped:
        {"id": "<paper_id>_p<page:03d>_c<idx:03d>",
         "document": "<decoded chunk text>",
         "metadata": {"paper_id": str, "page": int, "chunk_idx": int}}

    Empty / whitespace-only ``text`` returns []. Token boundary uses
    cl100k_base (matches text-embedding-3-small).
    """
    if not text or not text.strip():
        return []

    tokens = ENC.encode(text)
    chunks: list[dict] = []
    chunk_idx = 0
    for start in range(0, len(tokens), _STRIDE):
        window = tokens[start : start + CHUNK_TOKENS]
        if not window:
            break
        chunks.append(
            {
                "id": f"{paper_id}_p{page:03d}_c{chunk_idx:03d}",
                "document": ENC.decode(window),
                "metadata": {
                    "paper_id": paper_id,
                    "page": page,
                    "chunk_idx": chunk_idx,
                },
            }
        )
        chunk_idx += 1
        # If the window already covered the tail of the input, stop — the
        # next stride would produce an empty/duplicate window. This guards
        # against off-by-one chunk overflow on inputs whose length exactly
        # equals CHUNK_TOKENS (one window covers everything).
        if start + CHUNK_TOKENS >= len(tokens):
            break
    return chunks

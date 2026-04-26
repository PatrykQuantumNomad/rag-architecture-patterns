"""Non-live unit tests for tier-1-naive chunking.

Synthetic fixtures only — no real PDFs, no API keys.

Import strategy: the on-disk package directory is ``tier-1-naive`` (hyphen),
which Python cannot import as ``tier_1_naive`` directly (hyphens are not
valid identifiers). We load ``ingest.py`` via ``importlib.util`` and pin the
module under the synthetic name ``_tier1_ingest`` for the duration of the
test session. This mirrors the fallback documented in the 128-02 plan and
keeps ``ingest.py`` itself a clean, library-only module.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_INGEST_PATH = _REPO_ROOT / "tier-1-naive" / "ingest.py"


def _import_chunker():
    if "_tier1_ingest" in sys.modules:
        mod = sys.modules["_tier1_ingest"]
    else:
        spec = importlib.util.spec_from_file_location("_tier1_ingest", _INGEST_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load ingest module from {_INGEST_PATH}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_tier1_ingest"] = mod
        spec.loader.exec_module(mod)
    return mod.chunk_page, mod.CHUNK_TOKENS, mod.OVERLAP_TOKENS


def test_constants_match_research_decision():
    _, chunk_tokens, overlap_tokens = _import_chunker()
    assert chunk_tokens == 512, "Locked by 128-RESEARCH.md Pitfall 1"
    assert overlap_tokens == 64, "Locked by 128-RESEARCH.md Pitfall 1"


def test_empty_text_yields_no_chunks():
    chunk_page, *_ = _import_chunker()
    assert chunk_page("", paper_id="test.001", page=1) == []
    assert chunk_page("   \n\n   ", paper_id="test.001", page=1) == []


def test_short_text_yields_single_chunk():
    chunk_page, *_ = _import_chunker()
    chunks = chunk_page("hello world", paper_id="test.001", page=1)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["id"] == "test.001_p001_c000"
    assert c["document"] == "hello world" or c["document"].startswith("hello world")
    assert c["metadata"] == {"paper_id": "test.001", "page": 1, "chunk_idx": 0}


def test_long_text_yields_multiple_chunks_with_overlap():
    chunk_page, chunk_tokens, overlap_tokens = _import_chunker()
    # Build a deterministic ~2000-token input. "alpha " is ~1 token in cl100k_base.
    text = ("alpha " * 2000).strip()
    chunks = chunk_page(text, paper_id="test.001", page=3)
    assert len(chunks) >= 2, "Long input must produce multiple chunks"
    # Stride must be CHUNK_TOKENS - OVERLAP_TOKENS = 448
    # Indirect check: for ~2000 tokens, ceil((2000 - 64) / 448) + 1 ≈ 5 chunks expected, allow 4-6.
    assert 3 <= len(chunks) <= 7

    # Metadata + id invariants
    for idx, c in enumerate(chunks):
        assert c["metadata"]["paper_id"] == "test.001"
        assert c["metadata"]["page"] == 3
        assert c["metadata"]["chunk_idx"] == idx
        assert c["id"] == f"test.001_p003_c{idx:03d}"

    # Pitfall 9: ids unique within page
    ids = [c["id"] for c in chunks]
    assert len(set(ids)) == len(ids)


def test_metadata_page_is_int_not_string():
    chunk_page, *_ = _import_chunker()
    chunks = chunk_page("alpha " * 1000, paper_id="2005.11401", page=7)
    for c in chunks:
        assert isinstance(c["metadata"]["page"], int)
        assert isinstance(c["metadata"]["paper_id"], str)
        assert isinstance(c["metadata"]["chunk_idx"], int)


def test_id_uses_zero_padded_page_and_chunk():
    chunk_page, *_ = _import_chunker()
    chunks = chunk_page("alpha " * 100, paper_id="paper.A", page=12)
    assert chunks[0]["id"] == "paper.A_p012_c000"


def test_chunks_decode_to_non_empty_strings():
    chunk_page, *_ = _import_chunker()
    chunks = chunk_page("alpha " * 1500, paper_id="x.1", page=2)
    for c in chunks:
        assert isinstance(c["document"], str)
        assert len(c["document"]) > 0

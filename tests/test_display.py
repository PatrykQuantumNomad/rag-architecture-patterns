"""Unit tests for ``shared.display.render_query_result``.

The display module is a thin wrapper over rich. We only validate that the
rendered output contains the input fields a human reviewer would expect to
see. Rendering is captured into an in-memory Console that records text
output (no terminal required).
"""
from __future__ import annotations

from rich.console import Console

from shared.display import render_query_result


def _capture_console() -> Console:
    return Console(record=True, width=120, force_terminal=False, color_system=None)


def test_render_includes_query_chunks_answer_and_cost() -> None:
    captured = _capture_console()
    chunks = [
        {"doc_id": "doc-alpha", "score": 0.91, "snippet": "alpha snippet text"},
        {"doc_id": "doc-beta", "score": 0.74, "snippet": "beta snippet text"},
    ]
    render_query_result(
        query="What is RAG?",
        chunks=chunks,
        answer="Retrieval-augmented generation.",
        cost_usd=0.000123,
        input_tokens=42,
        output_tokens=17,
        console_override=captured,
    )

    text = captured.export_text()
    assert "What is RAG?" in text
    assert "doc-alpha" in text
    assert "doc-beta" in text
    assert "Retrieval-augmented generation." in text
    # The cost is formatted at $0.000123 magnitude — accept any common format.
    assert "0.0001" in text or "$0.000" in text
    assert "42" in text  # input tokens
    assert "17" in text  # output tokens


def test_render_handles_empty_chunks() -> None:
    captured = _capture_console()
    render_query_result(
        query="empty",
        chunks=[],
        answer="no context",
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        console_override=captured,
    )
    text = captured.export_text()
    assert "empty" in text
    assert "no context" in text

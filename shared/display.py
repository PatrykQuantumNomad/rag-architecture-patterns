"""Rich-powered renderer for query / chunks / answer / cost (per Pattern 6).

A single public function — :func:`render_query_result` — prints the standard
"how did this query go?" snapshot used by every tier's CLI demo. The output
is purely visual (no return value); tests inject a recording ``Console`` via
the ``console_override`` parameter.
"""
from __future__ import annotations

from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Module-level default console — production callers use this.
console: Console = Console()


def render_query_result(
    query: str,
    chunks: Sequence[dict[str, Any]],
    answer: str,
    cost_usd: float,
    input_tokens: int,
    output_tokens: int,
    console_override: Console | None = None,
) -> None:
    """Render a query / retrieved chunks / answer / cost summary to stdout.

    Parameters
    ----------
    query
        The user's input question.
    chunks
        Retrieved context. Each dict should expose ``doc_id``, ``score``, and
        ``snippet`` keys (missing keys render as ``-``).
    answer
        The model's response text.
    cost_usd
        Total USD spent on this query (sum of LLM + embedding calls).
    input_tokens / output_tokens
        Token counts for the LLM call.
    console_override
        Optional ``rich.console.Console`` to write into instead of the
        module-level ``console``. Tests pass a recording console here to
        capture output for assertions.
    """
    target = console_override or console

    # Query header.
    target.print(Panel.fit(query, title="Query", border_style="cyan"))

    # Chunks table.
    if chunks:
        table = Table(title="Retrieved Chunks", show_lines=False)
        table.add_column("doc_id", style="bold")
        table.add_column("score", justify="right")
        table.add_column("snippet")
        for chunk in chunks:
            doc_id = str(chunk.get("doc_id", "-"))
            score_val = chunk.get("score")
            score = f"{score_val:.3f}" if isinstance(score_val, (int, float)) else "-"
            snippet = str(chunk.get("snippet", "-"))
            table.add_row(doc_id, score, snippet)
        target.print(table)
    else:
        target.print("[dim]No chunks retrieved.[/dim]")

    # Answer body.
    target.print(Panel(answer, title="Answer", border_style="green"))

    # Cost footer.
    target.print(
        f"[bold]Cost:[/bold] ${cost_usd:.6f}  "
        f"[dim](input_tokens={input_tokens}, output_tokens={output_tokens})[/dim]"
    )

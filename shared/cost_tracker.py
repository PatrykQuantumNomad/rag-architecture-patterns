"""Token-and-USD cost tracker for the rag-architecture-patterns repo.

Records per-call LLM and embedding usage, computes USD from the locked
``shared.pricing.PRICES`` table, and persists per-run JSON under
``evaluation/results/costs/`` for downstream aggregation in Phase 133.

The persisted JSON shape is fixed by D-13:

.. code-block:: json

    {
      "tier": "tier-1",
      "timestamp": "2026-04-25T12:00:00Z",
      "queries": [
        {"model": "gemini-2.5-flash", "kind": "llm",
         "input_tokens": 100, "output_tokens": 50, "usd": 0.000155}
      ],
      "totals": {
        "llm_input_tokens": 100, "llm_output_tokens": 50,
        "embedding_tokens": 0, "usd": 0.000155
      }
    }
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pricing import PRICES

# Default destination for ``persist`` writes when no ``dest_dir`` is provided.
DEFAULT_COSTS_DIR = Path("evaluation/results/costs")


def _lookup_price(model: str) -> dict[str, float]:
    try:
        return PRICES[model]
    except KeyError as exc:  # noqa: PERF203 — explicit re-raise for clarity
        raise KeyError(
            f"model {model!r} not in shared.pricing.PRICES "
            "(add it to shared/pricing.py with a verified vendor price)"
        ) from exc


class CostTracker:
    """Records token usage and computes USD for a single tier run."""

    def __init__(self, tier: str) -> None:
        self.tier = tier
        # Capture timestamp ONCE so repeated persist() calls go to the same file.
        self._created_at: datetime = datetime.now(timezone.utc)
        self.queries: list[dict[str, Any]] = []

    # ---- recording ----------------------------------------------------------

    def record_llm(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        """Record an LLM completion. Raises ``KeyError`` for unknown models."""
        price = _lookup_price(model)
        usd = (input_tokens / 1_000_000) * price["input"] + (
            output_tokens / 1_000_000
        ) * price["output"]
        self.queries.append(
            {
                "model": model,
                "kind": "llm",
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "usd": usd,
            }
        )

    def record_embedding(self, model: str, input_tokens: int) -> None:
        """Record an embedding call. Raises ``KeyError`` for unknown models."""
        price = _lookup_price(model)
        usd = (input_tokens / 1_000_000) * price["input"]
        self.queries.append(
            {
                "model": model,
                "kind": "embedding",
                "input_tokens": int(input_tokens),
                "output_tokens": 0,
                "usd": usd,
            }
        )

    # ---- aggregates ---------------------------------------------------------

    def total_usd(self) -> float:
        return sum(q["usd"] for q in self.queries)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the D-13 JSON schema."""
        llm_input = sum(
            q["input_tokens"] for q in self.queries if q["kind"] == "llm"
        )
        llm_output = sum(
            q["output_tokens"] for q in self.queries if q["kind"] == "llm"
        )
        embedding_tokens = sum(
            q["input_tokens"] for q in self.queries if q["kind"] == "embedding"
        )
        return {
            "tier": self.tier,
            "timestamp": self._iso_timestamp(),
            "queries": list(self.queries),
            "totals": {
                "llm_input_tokens": llm_input,
                "llm_output_tokens": llm_output,
                "embedding_tokens": embedding_tokens,
                "usd": self.total_usd(),
            },
        }

    # ---- persistence --------------------------------------------------------

    def persist(self, dest_dir: Path | None = None) -> Path:
        """Write the run JSON to ``{dest_dir}/{tier}-{timestamp}.json``.

        ``dest_dir`` defaults to ``evaluation/results/costs/``. The directory
        is created if missing. Repeated calls with the same destination
        overwrite the same file (the timestamp is captured at construction).

        Returns the absolute Path written.
        """
        if dest_dir is None:
            dest_dir = DEFAULT_COSTS_DIR
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / f"{self.tier}-{self._filename_timestamp()}.json"
        out.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out

    # ---- internals ----------------------------------------------------------

    def _iso_timestamp(self) -> str:
        # ISO 8601 UTC with trailing Z (e.g. "2026-04-25T12:00:00Z").
        return self._created_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _filename_timestamp(self) -> str:
        # Filename-safe variant: "20260425T120000Z".
        return self._created_at.strftime("%Y%m%dT%H%M%SZ")

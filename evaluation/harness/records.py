"""Evaluation harness — Pydantic record models.

Pattern 1 in 131-RESEARCH.md — the load-bearing schema every adapter / scorer /
aggregator imports. Pydantic v2 idioms throughout (model_validate_json, NOT
the deprecated parse_file).

Decisions referenced
--------------------
- Pattern 1: EvalRecord shape (question_id, question, answer, retrieved_contexts,
  latency_s, cost_usd_at_capture, optional error).
- Pattern 4: latency colocated with the query record (no separate latency/ dir).
- Pitfall 2 / 8: ScoreRecord allows None metrics + nan_reason for empty-context
  and agent-truncated cases (Tier 5 specifically).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EvalRecord(BaseModel):
    """One captured query — answer + contexts + latency + cost.

    The `error` field is non-None for fatal-but-recoverable failures (e.g.
    Tier 5's `MaxTurnsExceeded`). Empty `retrieved_contexts` is NOT an error
    in itself (Pitfall 2 + 8 distinguish empty-contexts from truncation).
    """

    question_id: str
    question: str
    answer: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    latency_s: float
    cost_usd_at_capture: float
    error: Optional[str] = None


class QueryLog(BaseModel):
    """A single tier × timestamp capture run, holding all 30 EvalRecords."""

    tier: str
    timestamp: str  # ISO 8601 UTC, "Z" suffix — matches CostTracker D-13
    git_sha: str
    model: str
    records: list[EvalRecord]


class ScoreRecord(BaseModel):
    """One scored question — RAGAS metrics + nan_reason for honest aggregation.

    All three metrics are Optional[float] because Pitfall 2 (empty contexts) and
    Pitfall 8 (Tier 5 MaxTurnsExceeded) MUST yield NaN, not zero. The aggregator
    uses np.nanmean to skip NaN rows; nan_reason surfaces them in the comparison
    footer per tier.
    """

    question_id: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    nan_reason: Optional[str] = None


def write_query_log(path: Path, log: QueryLog) -> Path:
    """Persist a QueryLog to disk as indent-2 JSON. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(log.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_query_log(path: Path) -> QueryLog:
    """Load a QueryLog from disk via Pydantic v2 model_validate_json."""
    return QueryLog.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "EvalRecord",
    "QueryLog",
    "ScoreRecord",
    "write_query_log",
    "read_query_log",
]

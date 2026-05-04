"""Fallback diagnostic writer for Phase 1 (Tier 5 adapter fix).

Per .planning/phases/01-tier-5-adapter-fix/01-CONTEXT.md D-06: when the smoke
gate fails, every fallback attempt (instrument run + STACK.md mutations) must
be recorded with frozen-doc-grade provenance. This module produces:

    evaluation/results/diagnostics/tier-5-fallback-{TS}.json

The JSON drives the "Tier 5: partial-fix caveat" line in Phase 9's frozen doc
if the user authorizes shipping a degraded fix.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Reuse helpers from run.py (single source of truth for SHA/TS conventions)
from evaluation.harness.run import _git_sha, _REPO_ROOT, _ts, _ts_for_filename

AttemptKind = Literal[
    "instrument",
    "simplify_schema",
    "switch_model_slug",
    "bump_openai_agents",
]
SmokeOutcome = Literal["PASS", "FAIL", "INCONCLUSIVE"]


class SpanObservation(BaseModel):
    """Phoenix-span-derived observation per CONTEXT.md instrument-first ordering."""

    agent_called_tools: bool | None = None
    tools_returned_data: bool | None = None
    adapter_saw_tool_call_output_items: bool | None = None
    notes: str = ""


class SmokeQuestionResult(BaseModel):
    """One row of the 5-question smoke per attempt — sufficient for triage."""

    question_id: str
    n_retrieved_contexts: int
    error: str | None = None
    final_output_truncated: str = Field(default="", max_length=400)


class FallbackAttempt(BaseModel):
    kind: AttemptKind
    description: str = ""
    smoke_outcome: SmokeOutcome
    smoke_results: list[SmokeQuestionResult] = []
    span_observation: SpanObservation | None = None
    cost_usd: float = 0.0
    timestamp: str


class FallbackLog(BaseModel):
    phase: str = "01-tier-5-adapter-fix"
    git_sha: str
    opened_at: str
    closed_at: str | None = None
    final_disposition: Literal["RESOLVED", "DEGRADED_SHIP", "ESCALATED"] | None = None
    captured_versions: dict[str, str]
    attempts: list[FallbackAttempt] = []
    notes: str = ""


def _captured_versions() -> dict[str, str]:
    """Snapshot the 5 versions that drive Tier 5 behavior.

    Missing packages report "not-installed" so the log is always well-formed
    even if invoked from a partial environment.
    """
    out: dict[str, str] = {}
    for name in ("openai-agents", "lightrag-hku", "raganything", "ragas", "litellm"):
        try:
            out[name] = _pkg_version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def open_fallback_log() -> FallbackLog:
    """Construct a fresh FallbackLog stamped with git SHA + opened_at + versions."""
    return FallbackLog(
        git_sha=_git_sha(),
        opened_at=_ts(),
        captured_versions=_captured_versions(),
    )


def write_fallback_log(log: FallbackLog, output_dir: Path | None = None) -> Path:
    """Persist a FallbackLog as indent-2 JSON to evaluation/results/diagnostics/.

    Returns the resolved path. Creates the directory on first use; the
    filename is `tier-5-fallback-{ts_safe}.json` derived from log.opened_at.
    """
    if output_dir is None:
        output_dir = _REPO_ROOT / "evaluation" / "results" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_safe = _ts_for_filename(log.opened_at)
    path = output_dir / f"tier-5-fallback-{ts_safe}.json"
    path.write_text(log.model_dump_json(indent=2))
    return path


__all__ = [
    "AttemptKind",
    "SmokeOutcome",
    "SpanObservation",
    "SmokeQuestionResult",
    "FallbackAttempt",
    "FallbackLog",
    "open_fallback_log",
    "write_fallback_log",
]

"""Unit tests for evaluation.harness.diagnostics — Phase 01 Plan 03.

Covers the FallbackLog Pydantic model, the write_fallback_log writer, and the
_captured_versions helper. Per CONTEXT.md D-06, the fallback log is the
provenance artifact that drives Phase 9's frozen-doc "Tier 5: partial-fix
caveat" line. The shape MUST round-trip cleanly through model_dump_json /
model_validate_json (Pattern 1 of 131-RESEARCH.md persistence convention).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.harness.diagnostics import (
    FallbackAttempt,
    FallbackLog,
    SmokeQuestionResult,
    SpanObservation,
    _captured_versions,
    open_fallback_log,
    write_fallback_log,
)


def test_fallback_log_roundtrip(tmp_path: Path) -> None:
    """FallbackLog with one attempt of each kind round-trips via JSON."""
    log = FallbackLog(
        git_sha="abc1234",
        opened_at="2026-05-04T14:30:00Z",
        captured_versions={
            "openai-agents": "0.14.6",
            "lightrag-hku": "1.4.15",
            "raganything": "1.2.10",
            "ragas": "0.4.3",
            "litellm": "1.83.0",
        },
        attempts=[
            FallbackAttempt(
                kind="instrument",
                description="install debug-tier5, set RAG_DEBUG_TIER5_TRACING=1",
                smoke_outcome="FAIL",
                smoke_results=[
                    SmokeQuestionResult(
                        question_id="q1",
                        n_retrieved_contexts=0,
                        final_output_truncated="agent self-cited from training",
                    ),
                ],
                span_observation=SpanObservation(
                    agent_called_tools=True,
                    tools_returned_data=True,
                    adapter_saw_tool_call_output_items=False,
                    notes="adapter walk did not surface tool outputs",
                ),
                cost_usd=0.04,
                timestamp="2026-05-04T14:35:00Z",
            ),
            FallbackAttempt(
                kind="simplify_schema",
                description="bare types in tier-5-agentic/tools.py",
                smoke_outcome="FAIL",
                cost_usd=0.04,
                timestamp="2026-05-04T14:40:00Z",
            ),
            FallbackAttempt(
                kind="switch_model_slug",
                description="gemini-2.5-flash -> gemini-2.5-flash-001",
                smoke_outcome="INCONCLUSIVE",
                cost_usd=0.04,
                timestamp="2026-05-04T14:45:00Z",
            ),
        ],
    )

    path = write_fallback_log(log, output_dir=tmp_path)

    assert path.exists(), "writer must create the file on disk"
    assert path.parent == tmp_path
    assert path.name.startswith("tier-5-fallback-")
    assert path.name.endswith(".json")

    reloaded = FallbackLog.model_validate_json(path.read_text())
    assert reloaded.git_sha == "abc1234"
    assert [a.kind for a in reloaded.attempts] == [
        "instrument",
        "simplify_schema",
        "switch_model_slug",
    ]
    assert [a.smoke_outcome for a in reloaded.attempts] == [
        "FAIL",
        "FAIL",
        "INCONCLUSIVE",
    ]
    assert reloaded.attempts[0].span_observation is not None
    assert reloaded.attempts[0].span_observation.adapter_saw_tool_call_output_items is False


def test_fallback_log_truncates_final_output() -> None:
    """SmokeQuestionResult.final_output_truncated rejects strings >400 chars.

    Truncation is the caller's responsibility; the model enforces the contract.
    """
    with pytest.raises(ValidationError):
        SmokeQuestionResult(
            question_id="q1",
            n_retrieved_contexts=0,
            final_output_truncated="x" * 1000,
        )


def test_captured_versions_snapshot() -> None:
    """_captured_versions() returns the 5 required package keys."""
    versions = _captured_versions()

    expected_keys = {"openai-agents", "lightrag-hku", "raganything", "ragas", "litellm"}
    assert set(versions.keys()) == expected_keys

    # openai-agents is pinned in pyproject (research bundle confirmed 0.14.6 in .venv)
    assert isinstance(versions["openai-agents"], str)
    assert versions["openai-agents"], "openai-agents version string must be non-empty"


def test_open_fallback_log_initializes_required_fields() -> None:
    """open_fallback_log() returns a FallbackLog with sensible defaults."""
    log = open_fallback_log()

    assert isinstance(log.git_sha, str) and log.git_sha, "git_sha must be non-empty"
    # ISO 8601 UTC with Z suffix; optional fractional seconds permitted.
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$", log.opened_at), (
        f"opened_at does not match ISO 8601 Z pattern: {log.opened_at!r}"
    )
    assert log.attempts == []
    assert log.final_disposition is None
    assert set(log.captured_versions.keys()) == {
        "openai-agents",
        "lightrag-hku",
        "raganything",
        "ragas",
        "litellm",
    }

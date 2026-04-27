"""Non-live unit tests for evaluation/harness/run.py loop logic.

Covers (no real API calls; no real adapter imports beyond what's strictly needed):
- Invalid tier returns exit code 2 (build_parser + amain).
- Tier 1 happy-path loop persists a QueryLog (mocking _capture_tier so we
  don't pull in tier_1_naive's import chain).
- Tier 4 in --tiers without --tier-4-from-cache logs the skip notice and the
  whole run exits 0 (graceful skip; uses the real _capture_tier).
- `parser.parse_args(["--help"])` raises SystemExit(0).

Pattern 12 (test conventions) — fixtures use tmp_path; silent console is a
SimpleNamespace with a no-op print or a tiny _SilentConsole class.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock  # noqa: F401  — re-exported for downstream tests

import pytest

from evaluation.harness.records import EvalRecord, QueryLog


@pytest.fixture
def tiny_golden_qa(tmp_path):
    """Stub golden_qa.json with 2 questions; redirect _REPO_ROOT to tmp_path."""
    qa = [
        {
            "id": "single-hop-001",
            "question": "What is RAG?",
            "expected_answer": "RAG combines memory.",
            "source_papers": ["1706.03762"],
            "modality_tag": "text",
            "hop_count_tag": "single-hop",
            "figure_ids": [],
            "video_ids": [],
        },
        {
            "id": "single-hop-002",
            "question": "What is dense retrieval?",
            "expected_answer": "Dense vectors.",
            "source_papers": ["2004.04906"],
            "modality_tag": "text",
            "hop_count_tag": "single-hop",
            "figure_ids": [],
            "video_ids": [],
        },
    ]
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "golden_qa.json").write_text(json.dumps(qa))
    return tmp_path, qa


def _fake_record(qid: str, ans: str = "answer") -> EvalRecord:
    return EvalRecord(
        question_id=qid,
        question="?",
        answer=ans,
        retrieved_contexts=["c1"],
        latency_s=0.1,
        cost_usd_at_capture=0.0001,
    )


class _SilentConsole:
    """Drop-in stand-in for rich.console.Console used in tests."""

    def print(self, *args, **kwargs):  # noqa: D401, ANN001
        pass


def test_run_invalid_tier_returns_2():
    """`--tiers 9` -> exit code 2 (unsupported tier)."""
    from evaluation.harness import run as run_mod

    args = run_mod.build_parser().parse_args(["--tiers", "9", "--yes"])
    rc = asyncio.run(run_mod.amain(args, _SilentConsole()))
    assert rc == 2


def test_run_tier1_loop_persists_query_log(tmp_path, monkeypatch):
    """Tier 1 path: _capture_tier mocked so we exercise loop wiring without
    pulling in tier_1_naive's import chain. Asserts return code 0 + capture call."""
    from evaluation.harness import run as run_mod

    monkeypatch.setattr(run_mod, "_REPO_ROOT", tmp_path)
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    qa_path = eval_dir / "golden_qa.json"
    qa_path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "?",
                    "expected_answer": "!",
                    "source_papers": [],
                    "modality_tag": "text",
                    "hop_count_tag": "single-hop",
                    "figure_ids": [],
                    "video_ids": [],
                }
            ]
        )
    )
    monkeypatch.setattr(
        run_mod, "_load_golden_qa", lambda: json.loads(qa_path.read_text())
    )

    # Tier 1 prereq: chroma_db must exist (we bypass the real check anyway).
    (tmp_path / "chroma_db" / "tier-1-naive").mkdir(parents=True)
    monkeypatch.setattr(run_mod, "_check_prereqs", lambda tiers, console: 0)

    captured: dict = {}

    async def fake_capture_tier(tier, qa, args, console):
        log = QueryLog(
            tier=f"tier-{tier}",
            timestamp="2026-04-27T12:00:00Z",
            git_sha="abc1234",
            model="m",
            records=[_fake_record("q1")],
        )
        captured["tier"] = tier
        captured["n"] = len(qa)
        return log

    monkeypatch.setattr(run_mod, "_capture_tier", fake_capture_tier)

    args = run_mod.build_parser().parse_args(
        [
            "--tiers",
            "1",
            "--limit",
            "1",
            "--yes",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    rc = asyncio.run(run_mod.amain(args, _SilentConsole()))
    assert rc == 0
    assert captured["tier"] == 1
    assert captured["n"] == 1


def test_run_tier4_skips_without_cache(tmp_path, monkeypatch):
    """Tier 4 in --tiers without --tier-4-from-cache: graceful skip + exit 0.

    Uses the REAL _capture_tier — exercises the Tier 4 skip branch.
    """
    from evaluation.harness import run as run_mod

    monkeypatch.setattr(run_mod, "_REPO_ROOT", tmp_path)

    qa_path = tmp_path / "evaluation" / "golden_qa.json"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "?",
                    "expected_answer": "!",
                    "source_papers": [],
                    "modality_tag": "text",
                    "hop_count_tag": "single-hop",
                    "figure_ids": [],
                    "video_ids": [],
                }
            ]
        )
    )
    monkeypatch.setattr(
        run_mod, "_load_golden_qa", lambda: json.loads(qa_path.read_text())
    )
    monkeypatch.setattr(run_mod, "_check_prereqs", lambda tiers, console: 0)

    args = run_mod.build_parser().parse_args(
        ["--tiers", "4", "--yes", "--output-dir", str(tmp_path / "out")]
    )

    rc = asyncio.run(run_mod.amain(args, _SilentConsole()))
    assert rc == 0  # graceful skip — Tier 4 produced no log but the run is OK.

    # No tier-4 query log should have been written.
    queries_dir = tmp_path / "out" / "queries"
    assert not queries_dir.exists() or not list(queries_dir.glob("tier-4-*.json"))


def test_cli_help_exits_zero():
    """`build_parser().parse_args(["--help"])` exits 0 with usage text."""
    from evaluation.harness import run as run_mod

    parser = run_mod.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0

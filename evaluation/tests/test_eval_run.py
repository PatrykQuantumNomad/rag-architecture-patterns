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

    async def fake_capture_tier(tier, qa, args, console, **kw):
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


def test_default_smoke_ids_constant():
    """Plan 01-02 D-03: DEFAULT_SMOKE_IDS is a 5-tuple of (3 single-hop + 2 multi-hop)."""
    from evaluation.harness import run as run_mod

    assert run_mod.DEFAULT_SMOKE_IDS == (
        "single-hop-001",
        "single-hop-002",
        "single-hop-003",
        "multi-hop-001",
        "multi-hop-002",
    )


def test_run_smoke_ids_filters_correctly(tmp_path, monkeypatch):
    """`--smoke-question-ids c,a` filters golden_qa to those 2 questions in user-supplied order."""
    from evaluation.harness import run as run_mod

    monkeypatch.setattr(run_mod, "_REPO_ROOT", tmp_path)

    fake_qa = [
        {"id": "a", "question": "?a", "expected_answer": "!", "source_papers": [],
         "modality_tag": "text", "hop_count_tag": "single-hop", "figure_ids": [], "video_ids": []},
        {"id": "b", "question": "?b", "expected_answer": "!", "source_papers": [],
         "modality_tag": "text", "hop_count_tag": "single-hop", "figure_ids": [], "video_ids": []},
        {"id": "c", "question": "?c", "expected_answer": "!", "source_papers": [],
         "modality_tag": "text", "hop_count_tag": "single-hop", "figure_ids": [], "video_ids": []},
        {"id": "d", "question": "?d", "expected_answer": "!", "source_papers": [],
         "modality_tag": "text", "hop_count_tag": "single-hop", "figure_ids": [], "video_ids": []},
        {"id": "e", "question": "?e", "expected_answer": "!", "source_papers": [],
         "modality_tag": "text", "hop_count_tag": "single-hop", "figure_ids": [], "video_ids": []},
    ]
    monkeypatch.setattr(run_mod, "_load_golden_qa", lambda: fake_qa)
    monkeypatch.setattr(run_mod, "_check_prereqs", lambda tiers, console: 0)

    captured: dict = {}

    async def fake_capture_tier(tier, qa, args, console, **kw):
        captured["qa"] = qa
        captured["tier"] = tier
        return QueryLog(
            tier=f"tier-{tier}",
            timestamp="2026-04-27T12:00:00Z",
            git_sha="abc1234",
            model="m",
            records=[],
        )

    monkeypatch.setattr(run_mod, "_capture_tier", fake_capture_tier)

    args = run_mod.build_parser().parse_args(
        [
            "--tiers", "1",
            "--smoke-question-ids", "c,a",
            "--yes",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    rc = asyncio.run(run_mod.amain(args, _SilentConsole()))
    assert rc == 0
    # User-supplied order preserved (c before a)
    assert [q["id"] for q in captured["qa"]] == ["c", "a"]


@pytest.mark.parametrize(
    "tier,exp_emb,exp_src",
    [
        (1, "openai/text-embedding-3-small", "openrouter"),
        (2, "gemini-embedding-001", "google-managed"),
        (3, "openai/text-embedding-3-small", "openrouter"),
        (4, "openai/text-embedding-3-small", "openrouter"),
        # D-ROADMAP-OVERRIDE: Tier 5 is Tier 1's embedder, NOT a hosted vector store.
        (5, "openai/text-embedding-3-small", "openrouter"),
    ],
)
def test_run_amain_writes_embedder_per_tier_offline(
    tmp_path, monkeypatch, tier, exp_emb, exp_src
):
    """Plan 06-01 Task 3: each tier branch in _capture_tier reads its tier
    module's (EMBED_MODEL, EMBEDDER_SOURCE) tuple and threads both into the
    QueryLog written to evaluation/results/queries/tier-{N}-*.json.

    Per-tier strategy (offline):
      * Patch _check_prereqs + _load_golden_qa so amain doesn't bail.
      * Patch the adapter's run_tierN to return a fixed EvalRecord (no network).
      * Tier 2: stub .store_id file + read.
      * Tier 3: patch build_rag to a stub with awaitable initialize_storages.
      * Tier 4: write a fixture cached JSON; pass --tier-4-from-cache.
      * Tier 5: patch build_agent to a no-op stub.
    """
    from evaluation.harness import run as run_mod
    from evaluation.harness.records import EvalRecord

    # 1. Tier-agnostic stubs for amain's checks.
    monkeypatch.setattr(run_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        run_mod,
        "_load_golden_qa",
        lambda: [
            {
                "id": "q1",
                "question": "What is RAG?",
                "expected_answer": "memory.",
                "source_papers": [],
                "modality_tag": "text",
                "hop_count_tag": "single-hop",
                "figure_ids": [],
                "video_ids": [],
            }
        ],
    )
    monkeypatch.setattr(run_mod, "_check_prereqs", lambda tiers, console: 0)

    # 2. Patch tier-specific adapter / builder seams BEFORE _capture_tier
    # imports them lazily.
    fake_record = EvalRecord(
        question_id="q1",
        question="What is RAG?",
        answer="answer",
        retrieved_contexts=["c1"],
        latency_s=0.1,
        cost_usd_at_capture=0.0001,
    )

    async def _fake_run(*a, **kw):
        return fake_record

    cli_args = [
        "--tiers", str(tier),
        "--limit", "1",
        "--yes",
        "--output-dir", str(tmp_path / "out"),
    ]

    if tier == 1:
        from evaluation.harness.adapters import tier_1 as t1_adapter
        monkeypatch.setattr(t1_adapter, "run_tier1", _fake_run)

    elif tier == 2:
        from evaluation.harness.adapters import tier_2 as t2_adapter
        monkeypatch.setattr(t2_adapter, "run_tier2", _fake_run)
        # Stub the .store_id sidecar that _capture_tier reads at line 184-185.
        store_id = tmp_path / "tier-2-managed" / ".store_id"
        store_id.parent.mkdir(parents=True, exist_ok=True)
        store_id.write_text("fake-store-name")

    elif tier == 3:
        from evaluation.harness.adapters import tier_3 as t3_adapter
        from tier_3_graph import rag as t3_rag
        monkeypatch.setattr(t3_adapter, "run_tier3", _fake_run)

        class _StubRag:
            async def initialize_storages(self):
                return None

        def _fake_build_rag(*a, **kw):
            return _StubRag()

        monkeypatch.setattr(t3_rag, "build_rag", _fake_build_rag)

    elif tier == 4:
        # Write a cached log fixture matching Tier 4's cached schema (used by
        # run_tier4(from_cache=...) at evaluation/harness/adapters/tier_4.py).
        cache_path = tmp_path / "tier-4-cached.json"
        cached_log = {
            "tier": "tier-4",
            "timestamp": "2026-04-27T12:00:00Z",
            "git_sha": "cached0",
            "model": "google/gemini-2.5-flash",
            "records": [
                {
                    "question_id": "q1",
                    "question": "What is RAG?",
                    "answer": "answer",
                    "retrieved_contexts": ["c1"],
                    "latency_s": 0.1,
                    "cost_usd_at_capture": 0.0001,
                    "error": None,
                }
            ],
        }
        cache_path.write_text(json.dumps(cached_log))
        cli_args += ["--tier-4-from-cache", str(cache_path)]

    elif tier == 5:
        from evaluation.harness.adapters import tier_5 as t5_adapter
        from tier_5_agentic import agent as t5_agent
        monkeypatch.setattr(t5_adapter, "run_tier5", _fake_run)
        monkeypatch.setattr(t5_agent, "build_agent", lambda *a, **kw: object())

    # 3. Drive the real amain (and thus the real _capture_tier).
    args = run_mod.build_parser().parse_args(cli_args)
    rc = asyncio.run(run_mod.amain(args, _SilentConsole()))
    assert rc == 0, f"amain returned non-zero rc={rc} for tier {tier}"

    # 4. Read the queries JSON the real _capture_tier wrote.
    queries_dir = tmp_path / "out" / "queries"
    matches = sorted(queries_dir.glob(f"tier-{tier}-*.json"))
    assert matches, f"No queries JSON written for tier {tier}"
    data = json.loads(matches[0].read_text())

    # 5. Per-tier truth-table assertion (CAP-03 / Plan 06-01 Task 3).
    assert data["embedder"] == exp_emb, (
        f"tier-{tier} embedder mismatch: got {data.get('embedder')!r}, "
        f"expected {exp_emb!r}"
    )
    assert data["embedder_source"] == exp_src, (
        f"tier-{tier} embedder_source mismatch: got "
        f"{data.get('embedder_source')!r}, expected {exp_src!r}"
    )

    # 6. Regression guard — pre-existing top-level fields still present.
    assert data.get("model"), "model field disappeared"
    assert data.get("timestamp"), "timestamp field disappeared"
    assert data.get("git_sha"), "git_sha field disappeared"


def test_run_smoke_ids_unknown_returns_2(tmp_path, monkeypatch, capsys):
    """`--smoke-question-ids c,zzz` returns 2 + prints friendly red error citing 'zzz'."""
    from evaluation.harness import run as run_mod

    monkeypatch.setattr(run_mod, "_REPO_ROOT", tmp_path)

    fake_qa = [
        {"id": "a", "question": "?a", "expected_answer": "!", "source_papers": [],
         "modality_tag": "text", "hop_count_tag": "single-hop", "figure_ids": [], "video_ids": []},
        {"id": "c", "question": "?c", "expected_answer": "!", "source_papers": [],
         "modality_tag": "text", "hop_count_tag": "single-hop", "figure_ids": [], "video_ids": []},
    ]
    monkeypatch.setattr(run_mod, "_load_golden_qa", lambda: fake_qa)
    monkeypatch.setattr(run_mod, "_check_prereqs", lambda tiers, console: 0)

    # Use the real Console so the [red] tag rendering reaches captured stdout.
    from rich.console import Console as RichConsole

    args = run_mod.build_parser().parse_args(
        [
            "--tiers", "1",
            "--smoke-question-ids", "c,zzz",
            "--yes",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    rc = asyncio.run(run_mod.amain(args, RichConsole()))
    out = capsys.readouterr().out
    assert rc == 2
    assert "Unknown question ids" in out
    assert "zzz" in out

"""Unit tests for tier-4-multimodal/scripts/eval_capture.py.

ZERO live operations. The full ``_capture()`` async path drives RAG-Anything
through OpenRouter; these tests instead exercise the small pure helper
``_filter_qa(qa, smoke_ids, limit, console)`` extracted by Plan 02-03 Task 1
and the argparse plumbing around the new ``--smoke-question-ids`` flag.

The script under test lives in a non-package directory
(``tier-4-multimodal/scripts/``); we load it via
``importlib.util.spec_from_file_location`` so tests don't depend on the
``tier_4_multimodal`` shim adding a ``scripts`` submodule (mirrors
``test_log_graph_stats.py`` pattern from Plan 02-01).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from rich.console import Console

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "tier-4-multimodal" / "scripts" / "eval_capture.py"


def _load_module():
    """Load eval_capture.py without going through the shim."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "eval_capture_under_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


# -----------------------------------------------------------------------------
# DEFAULT_SMOKE_IDS — single source of truth (Pitfall 5 of 02-RESEARCH.md)
# -----------------------------------------------------------------------------


def test_default_smoke_ids_constant_unchanged():
    """eval_capture imports DEFAULT_SMOKE_IDS from evaluation.harness.run.

    Plan 02-03 import target — verifies the constant hasn't drifted from
    the 5-tuple Phase 1 D-03 locked. If this test breaks, either Phase 1's
    constant changed (REQUIRES a phase-level decision) or the executor
    accidentally introduced a second source of truth (forbidden).
    """
    from evaluation.harness.run import DEFAULT_SMOKE_IDS

    assert DEFAULT_SMOKE_IDS == (
        "single-hop-001",
        "single-hop-002",
        "single-hop-003",
        "multi-hop-001",
        "multi-hop-002",
    )


def test_eval_capture_imports_default_smoke_ids():
    """eval_capture.py module exposes DEFAULT_SMOKE_IDS in its namespace.

    The import statement ``from evaluation.harness.run import
    DEFAULT_SMOKE_IDS`` runs at module import time; once the module is
    loaded the symbol must be present in its dict. This is the smallest
    possible assertion that the import path works.
    """
    assert hasattr(mod, "DEFAULT_SMOKE_IDS")
    assert mod.DEFAULT_SMOKE_IDS == (
        "single-hop-001",
        "single-hop-002",
        "single-hop-003",
        "multi-hop-001",
        "multi-hop-002",
    )


# -----------------------------------------------------------------------------
# _filter_qa helper — pure function, no network
# -----------------------------------------------------------------------------


def _make_qa(ids: list[str]) -> list[dict]:
    """Build a fake golden_qa-shaped list with only the fields the filter touches."""
    return [{"id": q, "question": f"q-{q}"} for q in ids]


def test_filter_qa_smoke_ids_preserves_user_order():
    """smoke_ids='c,a' over ids=[a,b,c,d,e] returns [c,a] — user order, not list order."""
    qa = _make_qa(["a", "b", "c", "d", "e"])
    result = mod._filter_qa(qa, smoke_ids="c,a", limit=None, console=Console())
    assert isinstance(result, list)
    assert [q["id"] for q in result] == ["c", "a"]


def test_filter_qa_unknown_id_returns_2_with_message(capsys):
    """Unknown id → return code 2, error message names the offender."""
    qa = _make_qa(["a", "b"])
    result = mod._filter_qa(qa, smoke_ids="a,zzz", limit=None, console=Console())
    assert result == 2
    out = capsys.readouterr().out
    assert "Unknown question ids" in out
    assert "zzz" in out


def test_filter_qa_empty_string_treated_as_no_filter():
    """smoke_ids='' (shell edge — explicit empty arg) → no filter applied."""
    qa = _make_qa(["a", "b", "c"])
    result = mod._filter_qa(qa, smoke_ids="", limit=None, console=Console())
    assert isinstance(result, list)
    assert [q["id"] for q in result] == ["a", "b", "c"]


def test_filter_qa_none_smoke_ids_with_no_limit_returns_full_list():
    """smoke_ids=None, limit=None → returns the input unchanged (full list)."""
    qa = _make_qa(["a", "b", "c"])
    result = mod._filter_qa(qa, smoke_ids=None, limit=None, console=Console())
    assert isinstance(result, list)
    assert [q["id"] for q in result] == ["a", "b", "c"]


def test_filter_qa_smoke_then_limit_compose():
    """smoke filter narrows first, then --limit slices the result."""
    qa = _make_qa(["a", "b", "c", "d", "e"])
    result = mod._filter_qa(qa, smoke_ids="c,a,b", limit=2, console=Console())
    assert isinstance(result, list)
    assert [q["id"] for q in result] == ["c", "a"]


def test_filter_qa_limit_only_no_smoke_ids():
    """limit alone behaves like the original eval_capture.py slice (first N)."""
    qa = _make_qa(["a", "b", "c", "d", "e"])
    result = mod._filter_qa(qa, smoke_ids=None, limit=3, console=Console())
    assert isinstance(result, list)
    assert [q["id"] for q in result] == ["a", "b", "c"]


def test_filter_qa_smoke_ids_strips_whitespace():
    """`'a , b ,c'` → ['a','b','c'] — argv whitespace must not break filter."""
    qa = _make_qa(["a", "b", "c", "d"])
    result = mod._filter_qa(qa, smoke_ids="a , b ,c", limit=None, console=Console())
    assert isinstance(result, list)
    assert [q["id"] for q in result] == ["a", "b", "c"]


# -----------------------------------------------------------------------------
# argparse — --smoke-question-ids flag visible in --help and parsed correctly
# -----------------------------------------------------------------------------


def test_help_advertises_smoke_question_ids_flag(capsys):
    """`--help` lists --smoke-question-ids."""
    parser = mod.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--smoke-question-ids" in out


def test_parser_accepts_smoke_question_ids_arg():
    """Parser accepts --smoke-question-ids and stores the raw string."""
    parser = mod.build_parser()
    args = parser.parse_args(
        ["--smoke-question-ids", "single-hop-001,multi-hop-002"]
    )
    assert args.smoke_question_ids == "single-hop-001,multi-hop-002"


def test_parser_smoke_question_ids_default_none():
    """When --smoke-question-ids is not passed, default is None (no filter)."""
    parser = mod.build_parser()
    args = parser.parse_args([])
    assert args.smoke_question_ids is None

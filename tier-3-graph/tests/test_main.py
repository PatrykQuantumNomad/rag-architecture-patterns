"""Non-live unit tests for ``tier_3_graph.main``.

These tests cover the cost-surprise gate logic (Pitfall 3 + Pitfall 10) and
the argparse / constants surface. They do NOT make network calls and do NOT
require ``OPENROUTER_API_KEY``.

The ``--yes`` gate is the heart of the cost-surprise mitigation — bypassing
it accidentally would cost ~$1 per false-positive, so this file is the
primary protection against a regression in either the gate's auto-confirm
path (when --yes IS passed) or its abort path (when --yes is NOT passed and
input is non-interactive / declined).

The live ingest+query test lives in Plan 07 and explicitly passes ``--yes``.
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest

# Repo root is on sys.path via conftest.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants surface
# ---------------------------------------------------------------------------


def test_default_mode_is_hybrid_open_q4_resolution() -> None:
    """RESEARCH Open Q4 — default mode SHOULD be ``hybrid``, not ``mix``.

    ``mix`` requires a separately-configured reranker; ``hybrid`` works on
    every install. This is the canonical default per Plan 03 SUMMARY.
    """
    from tier_3_graph.main import DEFAULT_MODE, VALID_MODES

    assert DEFAULT_MODE == "hybrid"
    assert "hybrid" in VALID_MODES
    assert "mix" in VALID_MODES
    assert "naive" in VALID_MODES
    assert "local" in VALID_MODES
    assert "global" in VALID_MODES


def test_default_query_is_multi_hop_dpr_rag() -> None:
    """The default query MUST exercise cross-document graph traversal.

    Tier 3's value proposition vs Tier 1 is the ability to follow entity
    edges across papers. The canned query references both Lewis et al. 2020
    (RAG) and Karpukhin et al. 2020 (DPR) so the demo surfaces this advantage
    on first run.
    """
    from tier_3_graph.main import DEFAULT_QUERY

    assert "DPR" in DEFAULT_QUERY
    assert "RAG" in DEFAULT_QUERY
    assert "non-parametric memory" in DEFAULT_QUERY


# ---------------------------------------------------------------------------
# --yes gate: auto-confirm path
# ---------------------------------------------------------------------------


def test_yes_gate_auto_confirms_when_yes_flag_passed() -> None:
    """When --yes is True, _confirm_or_abort returns True without prompting."""
    from rich.console import Console

    from tier_3_graph.main import _confirm_or_abort

    console = Console(file=StringIO(), force_terminal=False)
    # Even though stdin would block on input(), --yes short-circuits before
    # input() is ever called. If the gate were broken and called input()
    # under --yes, this test would hang (pytest timeout would fire).
    result = _confirm_or_abort("destructive op ~$1", yes=True, console=console)
    assert result is True


# ---------------------------------------------------------------------------
# --yes gate: abort paths
# ---------------------------------------------------------------------------


def test_yes_gate_aborts_on_user_decline_without_yes_flag() -> None:
    """When user types anything other than 'y', _confirm_or_abort returns False."""
    from rich.console import Console

    from tier_3_graph.main import _confirm_or_abort

    console = Console(file=StringIO(), force_terminal=False)
    with mock.patch("builtins.input", return_value="n"):
        result = _confirm_or_abort("destructive op ~$1", yes=False, console=console)
    assert result is False


def test_yes_gate_aborts_on_empty_input_without_yes_flag() -> None:
    """Empty input (just hitting Enter) MUST abort (default-N behavior)."""
    from rich.console import Console

    from tier_3_graph.main import _confirm_or_abort

    console = Console(file=StringIO(), force_terminal=False)
    with mock.patch("builtins.input", return_value=""):
        result = _confirm_or_abort("destructive op ~$1", yes=False, console=console)
    assert result is False


def test_yes_gate_aborts_on_eof_without_yes_flag() -> None:
    """Non-interactive shell (EOFError) MUST abort — never silently proceed.

    This is the most important branch: in CI / automation contexts where
    ``input()`` raises EOFError immediately because stdin is closed, the
    gate MUST abort. Otherwise scripts would silently spend $1 every time
    they ran ``main.py`` without ``--yes``.
    """
    from rich.console import Console

    from tier_3_graph.main import _confirm_or_abort

    console = Console(file=StringIO(), force_terminal=False)
    with mock.patch("builtins.input", side_effect=EOFError):
        result = _confirm_or_abort("destructive op ~$1", yes=False, console=console)
    assert result is False


def test_yes_gate_proceeds_on_lowercase_y() -> None:
    """'y' (lowercase) is the canonical confirmation — proceed."""
    from rich.console import Console

    from tier_3_graph.main import _confirm_or_abort

    console = Console(file=StringIO(), force_terminal=False)
    with mock.patch("builtins.input", return_value="y"):
        result = _confirm_or_abort("destructive op ~$1", yes=False, console=console)
    assert result is True


# ---------------------------------------------------------------------------
# Argparse surface
# ---------------------------------------------------------------------------


def test_parser_exposes_all_required_flags() -> None:
    """The argparse surface MUST include every flag the README documents."""
    from tier_3_graph.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args([])
    # Defaults
    assert args.ingest is False
    assert args.query is None
    assert args.mode == "hybrid"
    assert args.reset is False
    assert args.yes is False
    # --model defaults to DEFAULT_LLM_MODEL
    from tier_3_graph.rag import DEFAULT_LLM_MODEL

    assert args.model == DEFAULT_LLM_MODEL


def test_parser_rejects_unknown_mode() -> None:
    """--mode must validate choices — typos MUST fail loudly, not silently."""
    from tier_3_graph.main import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "bogus"])


def test_parser_accepts_all_five_lightrag_modes() -> None:
    """Every LightRAG mode the README mentions must be valid."""
    from tier_3_graph.main import _build_parser

    parser = _build_parser()
    for mode in ("naive", "local", "global", "hybrid", "mix"):
        args = parser.parse_args(["--mode", mode])
        assert args.mode == mode


# ---------------------------------------------------------------------------
# No-flags default behavior
# ---------------------------------------------------------------------------


def test_no_flags_invocation_runs_default_demo() -> None:
    """No flags ⇒ args.ingest=True AND args.query=DEFAULT_QUERY (auto-demo).

    We verify by patching ``asyncio.run`` to capture the args namespace
    rather than actually entering ``amain`` (which would need an API key).
    """
    from tier_3_graph.main import DEFAULT_QUERY, main

    captured: dict = {}

    def _capture(coro):
        # ``coro`` is amain(args, console) — peek at its frame's args.
        captured["args"] = coro.cr_frame.f_locals["args"]
        coro.close()  # don't actually run the coroutine
        return 0

    with mock.patch("tier_3_graph.main.asyncio.run", side_effect=_capture):
        rc = main([])

    assert rc == 0
    assert captured["args"].ingest is True
    assert captured["args"].query == DEFAULT_QUERY


def test_query_only_invocation_does_not_auto_ingest() -> None:
    """``--query`` alone MUST NOT auto-flip ingest=True (ingest is opt-in)."""
    from tier_3_graph.main import main

    captured: dict = {}

    def _capture(coro):
        captured["args"] = coro.cr_frame.f_locals["args"]
        coro.close()
        return 0

    with mock.patch("tier_3_graph.main.asyncio.run", side_effect=_capture):
        rc = main(["--query", "what is RAG?"])

    assert rc == 0
    assert captured["args"].ingest is False
    assert captured["args"].query == "what is RAG?"

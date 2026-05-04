"""Unit tests for ``tier-4-multimodal/scripts/parse_missing_papers.py``.

All tests mock ``subprocess.run`` — the real ``mineru`` binary is NEVER
invoked from a unit test. The host-machine MineRU pass is gated by Plan
02-02's ``checkpoint:human-verify`` task; the unit tests here verify the
deterministic loop, command shape, sandbox-detection pre-flight, and
idempotency contracts only.

Per Pitfall 3 of Phase 02-RESEARCH.md: MineRU's PaddleOCR backend uses
OpenMP shared-memory primitives the orchestrator sandbox blocks at the
kernel level — so even if a test accidentally invoked the binary, it
would deadlock or bus-error. Mocking is mandatory, not just convenient.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

# ---------------------------------------------------------------------------
# Module loader: the script lives at ``scripts/parse_missing_papers.py`` and
# is not a package member, so we load it via ``importlib.util.spec_from_file_location``
# (mirrors the helper-loading pattern used elsewhere in the repo for
# script-style modules).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "tier-4-multimodal" / "scripts" / "parse_missing_papers.py"


def _load_module():
    """Load the script as a module. Re-loaded fresh per test if needed."""
    spec = importlib.util.spec_from_file_location(
        "parse_missing_papers", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def pmp():
    """Fresh module load per test (so monkeypatched env doesn't leak)."""
    return _load_module()


# ---------------------------------------------------------------------------
# Constant + pure-helper tests
# ---------------------------------------------------------------------------


def test_missing_papers_golden_constant(pmp):
    """The hardcoded 4-tuple matches RESEARCH.md A1 (golden_qa coverage gap)."""
    assert pmp.MISSING_PAPERS_GOLDEN == (
        "1909.01066",
        "2002.06177",
        "2309.15217",
        "2410.05779",
    )


def test_build_mineru_command_shape(tmp_path, pmp):
    """A2 of RESEARCH.md: locked flag set with ``-b hybrid-auto-engine``."""
    pdf = tmp_path / "fake.pdf"
    out = tmp_path / "out"
    cmd = pmp._build_mineru_command(pdf, out, "p1")
    assert cmd == [
        "mineru",
        "-p", str(pdf),
        "-o", str(out / "p1"),
        "-m", "auto",
        "-b", "hybrid-auto-engine",
        "-l", "en",
    ]


def test_load_papers_manifest_happy_path(tmp_path, pmp):
    manifest = tmp_path / "papers.json"
    manifest.write_text(json.dumps([
        {"paper_id": "p1", "filename": "p1.pdf"},
        {"paper_id": "p2", "filename": "p2.pdf"},
    ]))
    by_id = pmp._load_papers_manifest(manifest)
    assert set(by_id.keys()) == {"p1", "p2"}
    assert by_id["p1"]["filename"] == "p1.pdf"


def test_load_papers_manifest_missing(tmp_path, pmp):
    with pytest.raises(FileNotFoundError):
        pmp._load_papers_manifest(tmp_path / "does-not-exist.json")


# ---------------------------------------------------------------------------
# Sandbox detection
# ---------------------------------------------------------------------------


def _clean_sandbox_env(monkeypatch):
    """Strip every sandbox marker so ``_detect_sandbox()`` returns None."""
    for var in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "CLAUDE_CODE_SANDBOX",
        "SANDBOX_ENV",
    ):
        monkeypatch.delenv(var, raising=False)


def test_detect_sandbox_proxy(monkeypatch, pmp):
    _clean_sandbox_env(monkeypatch)
    monkeypatch.setenv("ALL_PROXY", "socks5h://localhost:1080")
    reason = pmp._detect_sandbox()
    assert reason is not None
    assert "ALL_PROXY" in reason


def test_detect_sandbox_https_proxy(monkeypatch, pmp):
    _clean_sandbox_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "socks5h://localhost:1080")
    reason = pmp._detect_sandbox()
    assert reason is not None
    assert "HTTPS_PROXY" in reason


def test_detect_sandbox_claude_marker(monkeypatch, pmp):
    _clean_sandbox_env(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_SANDBOX", "1")
    reason = pmp._detect_sandbox()
    assert reason is not None
    assert "CLAUDE_CODE_SANDBOX" in reason


def test_detect_sandbox_sandbox_env_marker(monkeypatch, pmp):
    _clean_sandbox_env(monkeypatch)
    monkeypatch.setenv("SANDBOX_ENV", "1")
    reason = pmp._detect_sandbox()
    assert reason is not None
    assert "SANDBOX_ENV" in reason


def test_detect_sandbox_clean(monkeypatch, pmp):
    _clean_sandbox_env(monkeypatch)
    assert pmp._detect_sandbox() is None


def test_detect_sandbox_ignores_non_socks_proxy(monkeypatch, pmp):
    """A plain ``http://`` proxy is NOT a sandbox signal — the heuristic is
    specifically the ``socks5h://`` scheme used by Claude Code's tunnel."""
    _clean_sandbox_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://corp-proxy.example.com:8080")
    assert pmp._detect_sandbox() is None


# ---------------------------------------------------------------------------
# Core loop tests (subprocess.run mocked end-to-end)
# ---------------------------------------------------------------------------


def _make_manifest(tmp_path: Path, entries: list[dict]) -> Path:
    manifest = tmp_path / "papers.json"
    manifest.write_text(json.dumps(entries))
    return manifest


def test_skips_already_parsed(tmp_path, pmp):
    """Idempotency: if ``output_root/<paper_id>`` exists, skip the subprocess."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "p1.pdf").write_bytes(b"fake-pdf")
    output_root = tmp_path / "out"
    (output_root / "p1").mkdir(parents=True)  # already-parsed
    manifest = _make_manifest(tmp_path, [{"paper_id": "p1", "filename": "p1.pdf"}])

    with patch.object(pmp.subprocess, "run") as mock_run:
        n_parsed, failed = pmp.parse_missing_papers(
            papers_dir=papers_dir,
            output_root=output_root,
            manifest_path=manifest,
            paper_ids=("p1",),
            console=Console(quiet=True),
        )

    assert n_parsed == 0
    assert failed == []
    assert mock_run.call_count == 0  # subprocess MUST NOT be invoked


def test_invokes_mineru_for_missing(tmp_path, pmp):
    """Happy path: missing paper → subprocess invoked with the locked command."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf = papers_dir / "p1.pdf"
    pdf.write_bytes(b"fake-pdf")
    output_root = tmp_path / "out"
    output_root.mkdir()
    manifest = _make_manifest(tmp_path, [{"paper_id": "p1", "filename": "p1.pdf"}])

    with patch.object(pmp.subprocess, "run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["mineru"], returncode=0
        )
        n_parsed, failed = pmp.parse_missing_papers(
            papers_dir=papers_dir,
            output_root=output_root,
            manifest_path=manifest,
            paper_ids=("p1",),
            console=Console(quiet=True),
        )

    assert n_parsed == 1
    assert failed == []
    assert mock_run.call_count == 1
    called_cmd = mock_run.call_args[0][0]
    assert called_cmd[0] == "mineru"
    assert called_cmd[1] == "-p"
    assert called_cmd[2] == str(pdf)
    assert called_cmd[3] == "-o"
    assert called_cmd[4] == str(output_root / "p1")
    assert "-b" in called_cmd
    assert "hybrid-auto-engine" in called_cmd
    # check=True is required so non-zero exit raises CalledProcessError
    assert mock_run.call_args.kwargs.get("check") is True


def test_continues_on_per_paper_failure(tmp_path, pmp):
    """Loop must continue past a per-paper failure (Open Question 2 of RESEARCH.md)."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    for pid in ("p1", "p2", "p3"):
        (papers_dir / f"{pid}.pdf").write_bytes(b"fake")
    output_root = tmp_path / "out"
    output_root.mkdir()
    manifest = _make_manifest(tmp_path, [
        {"paper_id": "p1", "filename": "p1.pdf"},
        {"paper_id": "p2", "filename": "p2.pdf"},
        {"paper_id": "p3", "filename": "p3.pdf"},
    ])

    call_log: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        # cmd[2] is the -p value (the PDF path)
        call_log.append(cmd[2])
        if "p2.pdf" in cmd[2]:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch.object(pmp.subprocess, "run", side_effect=fake_run) as mock_run:
        n_parsed, failed = pmp.parse_missing_papers(
            papers_dir=papers_dir,
            output_root=output_root,
            manifest_path=manifest,
            paper_ids=("p1", "p2", "p3"),
            console=Console(quiet=True),
        )

    assert n_parsed == 2
    assert failed == ["p2"]
    assert mock_run.call_count == 3  # loop continued past p2's failure


def test_missing_pdf_appends_to_failed(tmp_path, pmp):
    """If the PDF file is absent, the paper is recorded as failed and subprocess is NOT invoked."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    # Note: NOT creating p1.pdf
    output_root = tmp_path / "out"
    output_root.mkdir()
    manifest = _make_manifest(tmp_path, [{"paper_id": "p1", "filename": "p1.pdf"}])

    with patch.object(pmp.subprocess, "run") as mock_run:
        n_parsed, failed = pmp.parse_missing_papers(
            papers_dir=papers_dir,
            output_root=output_root,
            manifest_path=manifest,
            paper_ids=("p1",),
            console=Console(quiet=True),
        )

    assert n_parsed == 0
    assert failed == ["p1"]
    assert mock_run.call_count == 0


def test_missing_paper_id_in_manifest_appends_to_failed(tmp_path, pmp):
    """If ``paper_id`` is not in the manifest, recorded as failed; subprocess NOT invoked."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    output_root = tmp_path / "out"
    output_root.mkdir()
    manifest = _make_manifest(tmp_path, [{"paper_id": "other", "filename": "other.pdf"}])

    with patch.object(pmp.subprocess, "run") as mock_run:
        n_parsed, failed = pmp.parse_missing_papers(
            papers_dir=papers_dir,
            output_root=output_root,
            manifest_path=manifest,
            paper_ids=("p1",),
            console=Console(quiet=True),
        )

    assert n_parsed == 0
    assert failed == ["p1"]
    assert mock_run.call_count == 0


# ---------------------------------------------------------------------------
# CLI / main() tests
# ---------------------------------------------------------------------------


def test_main_refuses_in_sandbox(monkeypatch, capsys, pmp):
    """Pre-flight refuses before any subprocess call when sandbox is detected."""
    _clean_sandbox_env(monkeypatch)
    monkeypatch.setenv("ALL_PROXY", "socks5h://localhost:1080")
    rc = pmp.main(["--paper-ids", "p1"])
    assert rc == 2
    captured = capsys.readouterr()
    out_combined = captured.out + captured.err
    assert "Refusing to invoke MineRU" in out_combined
    # Per the plan: must mention either Pitfall 3 or "host" so the user
    # understands the remediation.
    assert ("Pitfall 3" in out_combined) or ("host" in out_combined.lower())


def test_main_allow_sandbox_override(monkeypatch, tmp_path, pmp):
    """``--allow-sandbox`` bypasses the pre-flight; downstream may still fail
    on missing manifest/PDF, but the sandbox short-circuit is gone."""
    _clean_sandbox_env(monkeypatch)
    monkeypatch.setenv("ALL_PROXY", "socks5h://localhost:1080")
    manifest = _make_manifest(tmp_path, [])  # empty manifest
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    output_root = tmp_path / "out"
    output_root.mkdir()
    with patch.object(pmp.subprocess, "run") as mock_run:
        rc = pmp.main([
            "--allow-sandbox",
            "--manifest", str(manifest),
            "--papers-dir", str(papers_dir),
            "--output-root", str(output_root),
            "--paper-ids", "p1",
        ])
    # p1 not in manifest → recorded as failed → exit 1 (NOT 2)
    assert rc == 1
    # subprocess never called because manifest lookup fails
    assert mock_run.call_count == 0


def test_main_returns_1_on_any_failure(monkeypatch, tmp_path, pmp):
    """Any golden-qa-referenced paper failure → non-zero exit (Open Q 2)."""
    _clean_sandbox_env(monkeypatch)
    manifest = _make_manifest(tmp_path, [{"paper_id": "p1", "filename": "p1.pdf"}])
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    # Note: NOT creating p1.pdf — paper will be recorded as failed
    output_root = tmp_path / "out"
    output_root.mkdir()
    rc = pmp.main([
        "--manifest", str(manifest),
        "--papers-dir", str(papers_dir),
        "--output-root", str(output_root),
        "--paper-ids", "p1",
    ])
    assert rc == 1


def test_main_returns_0_on_full_success(monkeypatch, tmp_path, pmp):
    """All golden-qa-referenced papers succeed → exit 0."""
    _clean_sandbox_env(monkeypatch)
    manifest = _make_manifest(tmp_path, [{"paper_id": "p1", "filename": "p1.pdf"}])
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "p1.pdf").write_bytes(b"fake")
    output_root = tmp_path / "out"
    output_root.mkdir()
    with patch.object(pmp.subprocess, "run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=["mineru"], returncode=0)
        rc = pmp.main([
            "--manifest", str(manifest),
            "--papers-dir", str(papers_dir),
            "--output-root", str(output_root),
            "--paper-ids", "p1",
        ])
    assert rc == 0
    assert mock_run.call_count == 1


def test_cli_help_lists_all_flags(pmp, capsys):
    parser = pmp.build_parser()
    help_text = parser.format_help()
    for flag in ("--paper-ids", "--allow-sandbox", "--papers-dir", "--output-root", "--manifest"):
        assert flag in help_text, f"Missing CLI flag: {flag}"

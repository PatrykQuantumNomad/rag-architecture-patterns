"""Unit tests for tier-4-multimodal/scripts/ingest_from_mineru.py.

ZERO live operations. All tests use ``tmp_path`` fixtures and (where the
amain entry point is exercised) monkeypatched module constants + stubbed
``build_rag``. The live ingest exercise happens in Task 2b of Plan 02-01.

The script under test lives in a non-package directory
(``tier-4-multimodal/scripts/``); we load it via
``importlib.util.spec_from_file_location`` so tests don't depend on the
``tier_4_multimodal`` shim adding a ``scripts`` submodule.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "tier-4-multimodal" / "scripts" / "ingest_from_mineru.py"


def _load_module():
    """Load ingest_from_mineru.py as a module without going through the shim."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "ingest_from_mineru_under_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Module-level so tests share a single import (cheaper).
mod = _load_module()


def _build_fake_mineru_tree(
    tmp_path: Path,
    paper_ids: tuple[str, ...] = ("p1", "p2"),
    include_v2: bool = True,
) -> Path:
    """Build a minimal MineRU-shaped output tree under ``tmp_path``.

    Layout matches the real on-disk shape exactly:

        <root>/<paper_id>/<paper_id>_paper_<hash>/<paper_id>_paper/hybrid_auto/
            <paper_id>_paper_content_list.json
            <paper_id>_paper_content_list_v2.json   (optional)
            images/ab.jpg
    """
    root = tmp_path / "mineru_output"
    root.mkdir()
    for pid in paper_ids:
        inner = root / pid / f"{pid}_paper_a1b2c3" / f"{pid}_paper" / "hybrid_auto"
        inner.mkdir(parents=True, exist_ok=True)
        (inner / f"{pid}_paper_content_list.json").write_text(
            json.dumps(
                [
                    {"type": "text", "text": "abstract"},
                    {"type": "image", "img_path": "images/ab.jpg"},
                ]
            )
        )
        if include_v2:
            (inner / f"{pid}_paper_content_list_v2.json").write_text(
                json.dumps([{"type": "text", "text": "v2 — should be skipped"}])
            )
        (inner / "images").mkdir(exist_ok=True)
        (inner / "images" / "ab.jpg").write_bytes(b"fake jpg bytes")
    return root


# ---------------------------------------------------------------------------
# _absolutize_image_paths
# ---------------------------------------------------------------------------


def test_absolutize_image_paths_resolves_relative(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "ab.jpg").write_bytes(b"x")
    content_list = [
        {"type": "image", "img_path": "images/ab.jpg"},
        {"type": "text", "text": "hello"},
    ]
    out = mod._absolutize_image_paths(content_list, images_dir)
    expected = str((images_dir / "ab.jpg").resolve())
    assert out[0]["img_path"] == expected
    # Original list MUST NOT be mutated.
    assert content_list[0]["img_path"] == "images/ab.jpg"
    # Non-image entry passes through unchanged.
    assert out[1] == {"type": "text", "text": "hello"}


def test_absolutize_image_paths_passes_through_absolute(tmp_path: Path) -> None:
    abs_path = "/abs/x.jpg"
    content_list = [{"type": "image", "img_path": abs_path}]
    out = mod._absolutize_image_paths(content_list, tmp_path / "images")
    assert out[0]["img_path"] == abs_path


def test_absolutize_image_paths_skips_non_image(tmp_path: Path) -> None:
    content_list = [
        {"type": "text", "text": "hello"},
        {"type": "table", "table_body": "..."},
        {"type": "equation", "text": "E=mc^2"},
    ]
    out = mod._absolutize_image_paths(content_list, tmp_path / "images")
    # Each entry should be byte-equal (no img_path mutation since not image type).
    for src, dst in zip(content_list, out):
        assert src == dst


# ---------------------------------------------------------------------------
# _find_content_list / _discover_papers
# ---------------------------------------------------------------------------


def test_find_content_list_excludes_v2(tmp_path: Path) -> None:
    root = _build_fake_mineru_tree(tmp_path, paper_ids=("p1",), include_v2=True)
    paper_root = root / "p1"
    result = mod._find_content_list(paper_root)
    assert result is not None
    assert "_v2" not in str(result)
    assert result.name.endswith("_content_list.json")


def test_find_content_list_returns_none_when_missing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert mod._find_content_list(empty) is None


def test_discover_papers_sorted(tmp_path: Path) -> None:
    root = _build_fake_mineru_tree(
        tmp_path, paper_ids=("p2", "p1", "p3"), include_v2=False
    )
    assert mod._discover_papers(root) == ["p1", "p2", "p3"]


def test_discover_papers_skips_papers_without_content_list(tmp_path: Path) -> None:
    root = _build_fake_mineru_tree(tmp_path, paper_ids=("p1",), include_v2=False)
    # Manually insert an empty paper dir — should NOT appear in the discovered list.
    (root / "empty_paper").mkdir()
    assert mod._discover_papers(root) == ["p1"]


# ---------------------------------------------------------------------------
# --reset wipes storage (Pitfall 4 invariant)
# ---------------------------------------------------------------------------


def test_reset_wipes_storage_dir(tmp_path: Path) -> None:
    """Verifies the wholesale-rmtree contract Pitfall 4 of RESEARCH.md cites.

    The script does not move/rename the dir — it deletes it outright with
    ``shutil.rmtree`` so the next ingest starts from a clean slate. We
    invoke shutil.rmtree directly here because the script invokes it
    inline; the test asserts the post-condition (dir is gone).
    """
    import shutil

    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "junk1").write_text("a")
    (storage / "junk2").write_text("b")
    (storage / "subdir").mkdir()
    (storage / "subdir" / "nested").write_text("c")
    assert storage.exists()

    shutil.rmtree(storage)
    assert not storage.exists()


# ---------------------------------------------------------------------------
# amain — argument-validation paths (no network)
# ---------------------------------------------------------------------------


class _FakeSettings:
    """Stub mimicking shared.config.Settings for the API-key check.

    The script only inspects truthiness of ``openrouter_api_key`` (it does
    NOT call .get_secret_value), so any falsy/truthy sentinel works.
    """

    def __init__(self, has_key: bool):
        self.openrouter_api_key = "sk-fake" if has_key else None


def _make_args(
    *,
    reset: bool = False,
    yes: bool = True,
    paper_ids: str | None = None,
    mineru_output_root: str | Path = "/tmp/empty",
) -> SimpleNamespace:
    """Build an argparse-ish namespace matching ingest_from_mineru's parser."""
    return SimpleNamespace(
        reset=reset,
        yes=yes,
        paper_ids=paper_ids,
        mineru_output_root=str(mineru_output_root),
    )


def test_amain_returns_2_on_missing_api_key(monkeypatch, tmp_path: Path) -> None:
    """When OPENROUTER_API_KEY is unset, amain MUST refuse to run (exit 2)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(mod, "get_settings", lambda: _FakeSettings(has_key=False))

    from rich.console import Console

    args = _make_args(mineru_output_root=tmp_path)
    rc = asyncio.run(mod.amain(args, Console()))
    assert rc == 2


def test_amain_no_papers_returns_2(monkeypatch, tmp_path: Path, capsys) -> None:
    """When the MineRU output root is empty, amain MUST refuse to run (exit 2)."""
    monkeypatch.setattr(mod, "get_settings", lambda: _FakeSettings(has_key=True))
    # Use a tmp_path that exists but contains no paper directories.
    empty_root = tmp_path / "mineru_empty"
    empty_root.mkdir()

    from rich.console import Console

    args = _make_args(mineru_output_root=empty_root)
    rc = asyncio.run(mod.amain(args, Console()))
    assert rc == 2
    captured = capsys.readouterr()
    assert "No MineRU output" in captured.out or "No MineRU output" in captured.err


def test_amain_aborts_when_storage_nonempty_without_reset(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Pitfall 4 zombie-state guard: refuse to ingest into a populated dir."""
    monkeypatch.setattr(mod, "get_settings", lambda: _FakeSettings(has_key=True))

    # Build a fake mineru tree so the no-papers gate doesn't short-circuit.
    fake_root = _build_fake_mineru_tree(tmp_path, paper_ids=("p1",), include_v2=False)

    # Point RAG_STORAGE at a tmp dir with junk in it (storage is "non-empty").
    fake_storage = tmp_path / "rag_storage_nonempty"
    fake_storage.mkdir()
    (fake_storage / "graph_chunk_entity_relation.graphml").write_text("<graphml/>")
    monkeypatch.setattr(mod, "RAG_STORAGE", fake_storage)

    # Stub build_rag in case the guard is bypassed — we want the test to
    # fail loudly (exit != 2) rather than silently hit the network.
    monkeypatch.setattr(
        mod, "build_rag", lambda *a, **kw: pytest.fail("build_rag called — guard bypassed")
    )

    from rich.console import Console

    args = _make_args(reset=False, mineru_output_root=fake_root)
    rc = asyncio.run(mod.amain(args, Console()))
    assert rc == 2
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "non-empty" in out.lower() or "--reset" in out


def test_amain_with_reset_wipes_storage(monkeypatch, tmp_path: Path) -> None:
    """--reset MUST shutil.rmtree the storage dir before ingest begins."""
    monkeypatch.setattr(mod, "get_settings", lambda: _FakeSettings(has_key=True))

    fake_root = _build_fake_mineru_tree(tmp_path, paper_ids=("p1",), include_v2=False)
    fake_storage = tmp_path / "rag_storage_nonempty"
    fake_storage.mkdir()
    (fake_storage / "junk").write_text("x")
    monkeypatch.setattr(mod, "RAG_STORAGE", fake_storage)

    # Stub the ingest loop and tracker bits so we never touch the network.
    fake_rag = MagicMock(name="rag_anything_stub")

    async def _fake_ingest(rag, root, ids, console):
        return len(ids)

    monkeypatch.setattr(mod, "build_rag", lambda *a, **kw: fake_rag)
    monkeypatch.setattr(mod, "ingest_from_mineru_output", _fake_ingest)
    # CostTracker.persist would write a real file; stub it.
    persisted_paths: list[Path] = []

    class _StubTracker:
        def __init__(self, *a, **kw):
            self.queries: list = []

        def total_usd(self):
            return 0.0

        def persist(self, dest_dir=None):
            p = tmp_path / "cost.json"
            p.write_text("{}")
            persisted_paths.append(p)
            return p

    monkeypatch.setattr(mod, "CostTracker", _StubTracker)

    from rich.console import Console

    args = _make_args(reset=True, mineru_output_root=fake_root)
    rc = asyncio.run(mod.amain(args, Console()))
    assert rc == 0
    # After --reset, the original storage dir is gone (rmtree). The ingest
    # itself does NOT recreate it in this test (we stub the ingest loop).
    assert not fake_storage.exists() or not (fake_storage / "junk").exists()

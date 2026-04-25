"""Unit tests for ``shared.cost_tracker.CostTracker``.

Validates the JSON schema (D-13 stability), the price arithmetic (round-trip
through ``shared.pricing.PRICES``), the persist round-trip, and unknown-model
error handling.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared import pricing
from shared.cost_tracker import CostTracker


def test_record_llm_costs_match_pricing_table() -> None:
    ct = CostTracker("tier-1")
    ct.record_llm("gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    expected = pricing.PRICES["gemini-2.5-flash"]["input"] + pricing.PRICES["gemini-2.5-flash"]["output"]
    assert abs(ct.total_usd() - expected) < 1e-9, f"expected {expected}, got {ct.total_usd()}"
    # Per the locked prices: 0.30 + 2.50 = 2.80
    assert abs(ct.total_usd() - 2.80) < 1e-9


def test_record_embedding_uses_input_price_only() -> None:
    ct = CostTracker("tier-1")
    ct.record_embedding("gemini-embedding-001", input_tokens=1_000_000)
    assert abs(ct.total_usd() - 0.15) < 1e-9


def test_to_dict_schema_shape() -> None:
    ct = CostTracker("tier-2")
    ct.record_llm("gemini-2.5-flash", input_tokens=100, output_tokens=50)
    ct.record_embedding("gemini-embedding-001", input_tokens=200)

    d = ct.to_dict()
    assert d["tier"] == "tier-2"
    assert isinstance(d["timestamp"], str) and d["timestamp"].endswith("Z")
    assert isinstance(d["queries"], list) and len(d["queries"]) == 2

    llm_query = d["queries"][0]
    assert llm_query["model"] == "gemini-2.5-flash"
    assert llm_query["kind"] == "llm"
    assert llm_query["input_tokens"] == 100
    assert llm_query["output_tokens"] == 50
    assert llm_query["usd"] > 0

    emb_query = d["queries"][1]
    assert emb_query["kind"] == "embedding"
    assert emb_query["output_tokens"] == 0

    totals = d["totals"]
    assert totals["llm_input_tokens"] == 100
    assert totals["llm_output_tokens"] == 50
    assert totals["embedding_tokens"] == 200
    assert abs(totals["usd"] - ct.total_usd()) < 1e-12


def test_persist_writes_valid_json_and_round_trips(tmp_costs_dir: Path) -> None:
    ct = CostTracker("tier-1")
    ct.record_llm("gemini-2.5-flash", input_tokens=10, output_tokens=5)
    out = ct.persist(dest_dir=tmp_costs_dir)
    assert out.exists()
    assert out.parent == tmp_costs_dir
    assert out.name.startswith("tier-1-")
    assert out.suffix == ".json"

    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed == ct.to_dict()


def test_persist_creates_dest_dir_if_missing(tmp_path: Path) -> None:
    dest = tmp_path / "deep" / "costs"
    assert not dest.exists()
    ct = CostTracker("tier-3")
    ct.record_llm("gemini-2.5-flash", input_tokens=1, output_tokens=1)
    out = ct.persist(dest_dir=dest)
    assert out.exists()
    assert dest.is_dir()


def test_persist_uses_construction_timestamp_so_repeat_writes_same_file(tmp_costs_dir: Path) -> None:
    ct = CostTracker("tier-1")
    ct.record_llm("gemini-2.5-flash", input_tokens=1, output_tokens=1)
    first = ct.persist(dest_dir=tmp_costs_dir)
    second = ct.persist(dest_dir=tmp_costs_dir)
    assert first == second  # idempotent (overwrites in place)


def test_unknown_model_raises_clear_keyerror() -> None:
    ct = CostTracker("tier-1")
    with pytest.raises(KeyError) as exc_info:
        ct.record_llm("bogus-model-xyz", input_tokens=1, output_tokens=1)
    assert "bogus-model-xyz" in str(exc_info.value)
    assert "shared.pricing.PRICES" in str(exc_info.value)

"""Unit tests for ``shared.pricing`` (RED phase first, GREEN once pricing.py lands)."""
from __future__ import annotations

from shared import pricing


REQUIRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-embedding-001",
    "gpt-4o-mini",
    "gpt-4o",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

CHAT_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gpt-4o-mini", "gpt-4o"]


def test_pricing_has_required_models() -> None:
    for model in REQUIRED_MODELS:
        assert model in pricing.PRICES, f"Missing pricing entry for {model}"
        entry = pricing.PRICES[model]
        assert "input" in entry, f"{model} missing 'input' price"
        assert "output" in entry, f"{model} missing 'output' price"
        assert isinstance(entry["input"], float)
        assert isinstance(entry["output"], float)


def test_chat_models_have_cache_price() -> None:
    """Gemini + OpenAI chat models include a cache price (per 127-RESEARCH.md)."""
    for model in CHAT_MODELS:
        if model.startswith("gemini-"):
            assert "cache" in pricing.PRICES[model], f"{model} missing 'cache' price"
            assert isinstance(pricing.PRICES[model]["cache"], float)


def test_pricing_date_is_locked() -> None:
    assert pricing.PRICING_DATE == "2026-04-25"


def test_pricing_known_values() -> None:
    """Lock specific prices so silent regressions are caught (D-13 stability)."""
    assert pricing.PRICES["gemini-2.5-flash"]["input"] == 0.30
    assert pricing.PRICES["gemini-2.5-flash"]["output"] == 2.50
    assert pricing.PRICES["gemini-embedding-001"]["input"] == 0.15
    assert pricing.PRICES["gemini-embedding-001"]["output"] == 0.0
    assert pricing.PRICES["gpt-4o-mini"]["input"] == 0.15
    assert pricing.PRICES["gpt-4o-mini"]["output"] == 0.60

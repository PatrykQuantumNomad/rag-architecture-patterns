# Testing Patterns

**Analysis Date:** 2026-05-04

## Test Framework

**Runner:**
- pytest 8.4+ (from `pyproject.toml`: `pytest>=8.4,<9`)
- Config: `[tool.pytest.ini_options]` in `pyproject.toml`
- Test discovery: standard pattern (`test_*.py` and `*_test.py`)

**Assertion Library:**
- pytest's built-in assertion mechanism
- No separate assertion library

**Run Commands:**
```bash
pytest                         # Run all tests
pytest -m live                 # Run only live (API) tests
pytest -v                      # Verbose output
pytest tests/test_cost_tracker.py::test_record_llm_costs_match_pricing_table  # Single test
pytest --co                    # List tests without running
```

## Pytest Markers

**Registered Markers:**
- `live` — marks tests that hit real APIs and incur cost
  - Defined in `pyproject.toml`: `markers = ["live: tests that hit real APIs and incur cost"]`
  - Example: `@pytest.mark.live` on `tier-1-naive/tests/test_main_live.py::test_ingest_and_query_end_to_end_2papers`

## Test File Organization

**Location:**
- Co-located with modules: `shared/` has no `tests/` subdir; shared module tests in `/tests/` root
- Tier-specific tests in tier subdirectories: `tier-1-naive/tests/`, `tier-4-multimodal/tests/`
- Root tests directory: `/tests/` for shared, repo-level, and dataset tests

**Naming:**
- Standard pytest pattern: `test_<module>.py`
- Live/integration tests: no special suffix, marked with `@pytest.mark.live`
- Smoke test: `smoke_test.py` (legacy pattern, Phase 127)

**Directory Structure:**
```
.
├── tests/                          # Shared + root-level tests
│   ├── conftest.py                 # Shared fixtures (tmp_costs_dir, live_keys_ok)
│   ├── test_cost_tracker.py
│   ├── test_display.py
│   ├── test_env_example.py
│   ├── test_loader.py
│   ├── test_pricing.py
│   └── ...
├── tier-1-naive/tests/             # Tier 1 tests
│   ├── test_chunker.py
│   ├── test_store.py
│   └── test_main_live.py
├── tier-5-agentic/tests/           # Tier 5 tests
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_tier5_e2e_live.py
└── ...
```

## Test Structure

**Suite Organization:**

Tests follow a simple function-per-test pattern with no class-based test organization:

```python
"""Module docstring describing what's tested and any special considerations.

This often includes trace/phase references (e.g., "REPO-06 trace test")
and context about fixtures or live API behavior.
"""
from __future__ import annotations

import pytest
from <module> import <function_or_class>


def test_<behavior>() -> None:
    """Descriptive test name explaining what assertion is made."""
    # Setup
    obj = SomeClass()
    
    # Exercise
    result = obj.some_method(arg)
    
    # Assert
    assert result == expected
```

**Test Patterns:**

1. **Simple Unit Test** (no fixtures):
   ```python
   def test_pricing_has_required_models() -> None:
       for model in REQUIRED_MODELS:
           assert model in pricing.PRICES
   ```

2. **Test with tmp_path Fixture** (pytest built-in):
   ```python
   def test_persist_writes_valid_json_and_round_trips(tmp_path: Path) -> None:
       ct = CostTracker("tier-1")
       ct.record_llm("gemini-2.5-flash", input_tokens=10, output_tokens=5)
       out = ct.persist(dest_dir=tmp_path)
       assert out.exists()
   ```

3. **Test with Custom Fixture** (from `conftest.py`):
   ```python
   def test_persist_uses_tmp_costs_dir(tmp_costs_dir: Path) -> None:
       ct = CostTracker("tier-1")
       ct.record_llm("gemini-2.5-flash", input_tokens=1, output_tokens=1)
       out = ct.persist(dest_dir=tmp_costs_dir)
       assert out.parent == tmp_costs_dir
   ```

4. **Live API Test** (marked, conditional on env):
   ```python
   @pytest.mark.live
   def test_ingest_and_query_end_to_end_2papers(
       tier1_live_keys, tmp_path, monkeypatch, capsys
   ):
       """Uses real API keys; skipped if OPENROUTER_API_KEY not set."""
       # Test code here
   ```

## Fixtures

**Defined in `/tests/conftest.py`:**

```python
@pytest.fixture()
def tmp_costs_dir(tmp_path: Path) -> Path:
    """Yields a temporary directory for CostTracker.persist writes.
    
    Tests should pass this to persist(dest_dir=tmp_costs_dir) instead of
    polluting the real evaluation/results/costs/ directory.
    """
    dest = tmp_path / "costs"
    dest.mkdir(parents=True, exist_ok=True)
    return dest

@pytest.fixture()
def live_keys_ok() -> None:
    """Skip the calling test if no GEMINI_API_KEY is configured."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set; live API tests skipped")
```

**Tier-Specific Fixtures:**
- `tier1_live_keys` (in tier-1-naive/tests/) — skips tests if `OPENROUTER_API_KEY` not set
- Similar patterns in tier-4, tier-5 for their specific API requirements

**Setup/Teardown:**
- Use pytest fixtures (`@pytest.fixture()`) with `yield` or return
- No `setUp`/`tearDown` methods (not class-based tests)
- Environment setup: `load_dotenv(Path(__file__).resolve().parent.parent / ".env")` in conftest

## Assertion Patterns

**Simple Equality:**
```python
assert model in pricing.PRICES
assert out.exists()
assert result == expected
```

**Floating Point Comparison** (for costs):
```python
assert abs(ct.total_usd() - expected) < 1e-9
```

**String Content:**
```python
text = captured.export_text()
assert "What is RAG?" in text
assert "Cost:" in text or "$0.0001" in text
```

**Type Checking:**
```python
assert isinstance(d["queries"], list) and len(d["queries"]) == 2
assert isinstance(entry["input"], float)
```

**Exception Testing:**
```python
with pytest.raises(KeyError) as exc_info:
    ct.record_llm("bogus-model-xyz", input_tokens=1, output_tokens=1)
assert "bogus-model-xyz" in str(exc_info.value)
assert "shared.pricing.PRICES" in str(exc_info.value)
```

## Mocking

**Framework:** unittest.mock (via `monkeypatch` in pytest)

**Patterns:**

1. **monkeypatch for Environment Variables:**
   ```python
   def test_ingest_and_query_end_to_end_2papers(
       tier1_live_keys, tmp_path, monkeypatch, capsys
   ):
       real_loader = DatasetLoader()
       all_papers = real_loader.papers()
       subset: list[dict] = []
       # ... build subset ...
       monkeypatch.setattr(
           "tier_1_naive.main.DatasetLoader.papers",
           lambda self: subset
       )
   ```

2. **Console Override** (for capturing output):
   ```python
   def _capture_console() -> Console:
       return Console(record=True, width=120, force_terminal=False, color_system=None)
   
   def test_render_includes_query_chunks_answer_and_cost() -> None:
       captured = _capture_console()
       render_query_result(
           query="What is RAG?",
           chunks=chunks,
           answer="...",
           cost_usd=0.000123,
           input_tokens=42,
           output_tokens=17,
           console_override=captured,
       )
       text = captured.export_text()
       assert "What is RAG?" in text
   ```

3. **Dynamic Module Import** (for hyphenated directories):
   ```python
   # tier-1-naive/tests/test_store.py uses importlib.util to load hyphenated modules
   import importlib.util
   import sys
   
   spec = importlib.util.spec_from_file_location("_tier1_store", _STORE_PATH)
   store_mod = importlib.util.module_from_spec(spec)
   sys.modules["_tier1_store"] = store_mod
   spec.loader.exec_module(store_mod)
   ```

**What to Mock:**
- External API calls in unit tests (use `@pytest.mark.live` for real API tests)
- Environment variables via `monkeypatch`
- Rich console output via `console_override` parameter
- File system via `tmp_path` and `tmp_costs_dir`

**What NOT to Mock:**
- Pydantic Settings (always use real `get_settings()` to catch validation bugs)
- Domain logic (test actual behavior, not mocked calls)
- Pricing calculations (test against real `shared.pricing.PRICES` table)

## Test Data and Factories

**Synthetic Data:**
```python
def _synthetic_vec(seed: int, dim: int = 1536) -> list[float]:
    """Generate deterministic random vector for tests."""
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]

# Usage:
vecs = [_synthetic_vec(s) for s in range(5)]
```

**Test Constants:**
```python
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
```

**Fixture Data:**
- Chunks: `{"doc_id": "doc-alpha", "score": 0.91, "snippet": "alpha snippet text"}`
- Papers: loaded from `dataset/manifests/papers.json` or skipped if not present
- Synthetic vectors: seeded random floats for deterministic ChromaDB testing

## Test Types

**Unit Tests:**
- Scope: Single function or class method
- Location: `/tests/` (for shared) or `<tier>/tests/` (for tier-specific)
- No external dependencies (no real API calls, no actual PDFs)
- Examples: `test_cost_tracker.py`, `test_pricing.py`, `test_display.py`

**Integration Tests:**
- Scope: Multiple components working together (e.g., CostTracker + Pricing + JSON serialization)
- Still use synthetic/temporary data (tmp_path, synthetic vectors)
- No real API calls
- Examples: `test_persist_writes_valid_json_and_round_trips()`, `test_store_and_retrieve_roundtrip()`

**End-to-End (Live) Tests:**
- Scope: Full system with real API calls and real data
- Marked with `@pytest.mark.live`
- Skipped by default; run with `pytest -m live`
- Incur real costs (documented in test docstrings)
- Require API keys (fixture `live_keys_ok` skips if missing)
- Examples: `tier-1-naive/tests/test_main_live.py::test_ingest_and_query_end_to_end_2papers`

**Smoke Tests:**
- Legacy pattern: `tests/smoke_test.py`
- Validates basic imports and environment setup without deep execution
- Used in CI to detect breaking changes in dependencies

## Coverage

**Requirements:** Not enforced (no `--cov-fail-under` in config)

**View Coverage:**
```bash
pytest --cov=shared --cov=tier_1_naive --cov-report=term-missing
pytest --cov=shared --cov-report=html  # Generate HTML report
```

**Note:** No coverage threshold configured; coverage optional but encouraged for critical modules like `shared/cost_tracker.py` and `shared/config.py`.

## Common Patterns

**Async Testing:**
- Not used in this codebase (all code is synchronous)
- Gemini/OpenAI clients are synchronous

**Error Testing:**
```python
def test_unknown_model_raises_clear_keyerror() -> None:
    ct = CostTracker("tier-1")
    with pytest.raises(KeyError) as exc_info:
        ct.record_llm("bogus-model-xyz", input_tokens=1, output_tokens=1)
    assert "bogus-model-xyz" in str(exc_info.value)
    assert "shared.pricing.PRICES" in str(exc_info.value)
```

**Conditional Tests** (Skip if preconditions not met):
```python
def test_papers_manifest_conditional() -> None:
    p = MANIFESTS / "papers.json"
    if not p.exists():
        pytest.skip("papers.json not yet generated (Plan 04 produces it)")
    # Continue with test...
```

**Parametrized Tests:**
- Sparingly used (few examples in codebase)
- Explicit loop over test constants preferred (e.g., `for model in REQUIRED_MODELS:`)

**Round-Trip Testing:**
```python
def test_persist_writes_valid_json_and_round_trips(tmp_costs_dir: Path) -> None:
    ct = CostTracker("tier-1")
    ct.record_llm("gemini-2.5-flash", input_tokens=10, output_tokens=5)
    out = ct.persist(dest_dir=tmp_costs_dir)
    
    # Serialize
    parsed = json.loads(out.read_text(encoding="utf-8"))
    
    # Assert structure matches
    assert parsed == ct.to_dict()
```

## Test Isolation

**Practices:**
- Each test is independent (no shared state)
- Use `tmp_path` for file system artifacts
- Use `tmp_costs_dir` for CostTracker outputs (avoids polluting `evaluation/results/costs/`)
- Environment setup: `load_dotenv()` in conftest loads `.env` once per session
- Reopen ChromaDB clients in separate `tmp_path` directories per test

## Debugging Tests

**Strategies:**
- `pytest -vv` for verbose output
- `pytest -s` to capture print statements
- `pytest --pdb` to drop into debugger on failure
- `pytest -k <pattern>` to run tests matching a name pattern
- Add `print()` or Rich output to understand test state

**Example:**
```bash
pytest tests/test_cost_tracker.py::test_record_llm_costs_match_pricing_table -vv -s
```

---

*Testing analysis: 2026-05-04*

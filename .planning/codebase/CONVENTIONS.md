# Coding Conventions

**Analysis Date:** 2026-05-04

## Naming Patterns

**Files:**
- Snake_case: `cost_tracker.py`, `embeddings.py`, `display.py`
- Test files: `test_<module>.py` (e.g., `test_cost_tracker.py`)
- Private modules (internal implementation): prefix with underscore in class names (e.g., `_GeminiLLMClient`)

**Functions:**
- Snake_case: `get_llm_client()`, `record_llm()`, `render_query_result()`
- Private functions: prefix with underscore, used for internal helpers (e.g., `_lookup_price()`, `_read_env_example()`, `_capture_console()`)
- Factory functions: `get_*` pattern for singletons and lazy-loaded resources (e.g., `get_settings()`, `get_llm_client()`, `get_embedding_client()`)

**Variables:**
- Snake_case throughout: `input_tokens`, `output_tokens`, `cost_usd`, `default_model`
- Module-level constants: UPPER_CASE (e.g., `CHROMA_PATH`, `COLLECTION_NAME`, `PRICES`, `PRICING_DATE`)
- Private module-level state: prefix with underscore and use type hints (e.g., `_client_cache: _GeminiLLMClient | None = None`)

**Types:**
- PascalCase for classes: `CostTracker`, `Settings`, `LLMResponse`, `LLMClient`
- Protocol types: `CamelCase` (e.g., `LLMClient`, `EmbeddingClient`)
- Internal classes: prefix with underscore (e.g., `_GeminiLLMClient`)

## Code Style

**Formatting:**
- Black-style line length (implied from docstrings and code structure, though not explicitly configured)
- 4-space indentation (standard Python)
- Trailing commas in multi-line structures: `dict[str, float]` = dictionary types with new lines
- Consistent spacing: 1 line between function definitions, 2 lines between class definitions

**Linting:**
- Ruff: installed as dev dependency (`ruff>=0.6` in `pyproject.toml`)
- No explicit `.ruff.toml` or ruff configuration in `pyproject.toml` — uses Ruff defaults
- Code style enforced by team convention (no linter config file checked in)

## Import Organization

**Order:**
1. `from __future__ import annotations` (always first, per PEP 563)
2. Standard library imports (e.g., `import json`, `from datetime import datetime`)
3. Third-party imports (e.g., `from pydantic import Field`, `from rich.console import Console`)
4. Relative imports within project (e.g., `from .config import get_settings`, `from shared.pricing import PRICES`)

**Path Aliases:**
- No import aliases configured in `pyproject.toml`
- Absolute imports used: `from shared import pricing`, `from tier_1_naive import store`
- Hyphenated directories (`tier-1-naive/`) handled via dynamic import in tests using `importlib.util` because Python identifiers cannot contain hyphens
- Shim packages (`tier_1_naive`, underscore) re-export from hyphenated directories for importability

**Example from `shared/config.py`:**
```python
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
```

## Error Handling

**Patterns:**
- Explicit `except KeyError` with meaningful re-raise (e.g., in `_lookup_price()`):
  ```python
  try:
      return PRICES[model]
  except KeyError as exc:  # noqa: PERF203 — explicit re-raise for clarity
      raise KeyError(
          f"model {model!r} not in shared.pricing.PRICES "
          "(add it to shared/pricing.py with a verified vendor price)"
      ) from exc
  ```
- Broad exception handling when idempotency is needed (e.g., in `tier_1_naive/store.py`):
  ```python
  try:
      client.delete_collection(COLLECTION_NAME)
  except Exception:
      # Collection didn't exist — that's fine; reset is idempotent.
      pass
  ```
- Use `pytest.raises()` to assert exceptions in tests:
  ```python
  with pytest.raises(KeyError) as exc_info:
      ct.record_llm("bogus-model-xyz", input_tokens=1, output_tokens=1)
  assert "bogus-model-xyz" in str(exc_info.value)
  ```
- Validation errors from Pydantic `Settings` allowed to propagate (intentional design per Pattern 5 in RESEARCH.md)

## Logging

**Framework:** `console` (Rich library for visual output)

**Patterns:**
- No structured logging library (no `loguru`, `structlog`, etc.)
- Visual rendering via `rich.console.Console` for end-user output (`shared/display.py`)
- Rich panels, tables, and markup used for demo/CLI output (e.g., `Panel.fit()`, `Table`, `style="bold"`)
- Module-level `console: Console = Console()` singleton with `console_override` parameter in render functions for test capture
- Tests inject recording console: `Console(record=True, width=120, force_terminal=False, color_system=None)`

**Example from `shared/display.py`:**
```python
console: Console = Console()

def render_query_result(..., console_override: Console | None = None) -> None:
    target = console_override or console
    target.print(Panel.fit(query, title="Query", border_style="cyan"))
```

## Comments and Docstrings

**When to Comment:**
- Module-level docstrings describe public surface and design patterns (e.g., "Pattern 5" references to RESEARCH.md)
- Function docstrings use Google/NumPy style: Parameters, Returns, Raises sections
- Inline comments for non-obvious logic (e.g., `# similarities = 1 - distances (Pitfall 2)`)
- Comments for ChromaDB quirks and API limitations (e.g., "Pitfall 4 — re-passing metadata to get_or_create_collection is silently ignored")

**Example from `shared/config.py`:**
```python
"""Settings for the rag-architecture-patterns repo.

Per RESEARCH.md Pattern 5, ``GEMINI_API_KEY`` is REQUIRED at validation time.
We use a lazy factory (``get_settings()``) so that simply importing this
module — or any sibling that re-imports ``shared.config`` — does NOT raise
when ``.env`` is absent.
"""
```

**DocString Format:**
```python
def record_llm(
    self, model: str, input_tokens: int, output_tokens: int
) -> None:
    """Record an LLM completion. Raises ``KeyError`` for unknown models."""
```

## Function Design

**Size:** Most functions 10-50 lines; complex ones (e.g., `to_dict()` aggregates) up to 30-40 lines

**Parameters:**
- Use type hints throughout (Python 3.10+, PEP 604 union syntax: `str | None`)
- Optional parameters with defaults: `model: str | None = None`
- No *args or **kwargs (explicit parameter lists preferred)
- Protocol types used to define minimal interfaces (e.g., `LLMClient`, `EmbeddingClient`)

**Return Values:**
- Explicit return type hints: `-> LLMResponse`, `-> dict[str, Any]`, `-> None`
- Frozen dataclasses for immutable returns (e.g., `LLMResponse`)
- Protocols for duck-typed clients (e.g., `LLMClient` protocol consumed by all tiers)

**Example from `shared/llm.py`:**
```python
def complete(
    self, prompt: str, model: str | None = None
) -> LLMResponse:
    chosen_model = model or self._default_model
    response = self._client.models.generate_content(
        model=chosen_model, contents=prompt
    )
    # Extract usage and return frozen dataclass
    return LLMResponse(...)
```

## Module Design

**Exports:**
- No `__all__` lists (all public names exported implicitly)
- Shim packages use `__init__.py` re-exports to bridge hyphenated directories:
  ```python
  # tier_1_naive/__init__.py
  from tier-1-naive.main import ...  # Via importlib
  ```
- Private modules marked with underscore prefix (e.g., `_GeminiLLMClient`)

**Barrel Files:**
- `shared/__init__.py` is a documentation index (no code, lists public module surface)
- No code re-export barrels (imports are always from final modules)

**Lazy Initialization:**
- Singleton factories use `@lru_cache(maxsize=1)` for lazy settings: `shared/config.py`
- Module-level cache variables are None-initialized, populated on first call:
  ```python
  _client_cache: _GeminiLLMClient | None = None
  
  def get_llm_client() -> LLMClient:
      global _client_cache
      if _client_cache is None:
          _client_cache = _GeminiLLMClient()
      return _client_cache
  ```

## Type Hints

**Usage:**
- Comprehensive type hints throughout (Python 3.10+)
- Union syntax: `str | list[str]` (PEP 604)
- Optional: `str | None` instead of `Optional[str]`
- Protocol types for interfaces: `from typing import Protocol`
- Generic dict/list: `dict[str, Any]`, `list[dict[str, Any]]`
- Frozen dataclasses for immutable types: `@dataclass(frozen=True)`

**Example:**
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
```

## Special Pragmas

**noqa Comments:**
- `# pragma: no cover — Protocol` — skips coverage for Protocol method bodies
- `# noqa: PERF203 — explicit re-raise for clarity` — ignores Ruff performance warnings where explicit code is clearer

---

*Convention analysis: 2026-05-04*

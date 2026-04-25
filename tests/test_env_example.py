"""REPO-06 trace test — validates ``.env.example`` documents required keys.

The .env.example file is the canonical onboarding doc: a contributor copies
it to .env and fills in real values. This test ensures the template stays
in sync with the keys ``shared.config.Settings`` actually reads, and also
defends against a future contributor accidentally committing a real key.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent
ENV_EXAMPLE = ROOT / ".env.example"


def _read_env_example() -> str:
    assert ENV_EXAMPLE.exists(), f"{ENV_EXAMPLE} is missing — required for REPO-06"
    return ENV_EXAMPLE.read_text(encoding="utf-8")


def test_env_example_exists() -> None:
    assert ENV_EXAMPLE.is_file()


def test_env_example_documents_gemini_api_key_with_placeholder() -> None:
    text = _read_env_example()
    # Must reference the variable AND use the literal placeholder value
    # (defends against accidental commit of a real key).
    assert "GEMINI_API_KEY=your_gemini_api_key_here" in text, (
        "Expected literal placeholder 'GEMINI_API_KEY=your_gemini_api_key_here' "
        "in .env.example; refusing to accept any other value to catch real-key commits."
    )


def test_env_example_documents_optional_keys() -> None:
    text = _read_env_example()
    for key in ("OPENAI_API_KEY", "S2_API_KEY"):
        assert key in text, f"{key} must be documented in .env.example"


def test_env_example_documents_default_models() -> None:
    text = _read_env_example()
    for key in ("DEFAULT_CHAT_MODEL", "DEFAULT_EMBEDDING_MODEL"):
        assert key in text, f"{key} must be documented in .env.example"

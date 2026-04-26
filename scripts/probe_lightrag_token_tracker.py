"""One-shot probe — surfaces the token_tracker protocol on the installed lightrag-hku.

Run via ``uv run python scripts/probe_lightrag_token_tracker.py``. Output is captured
in 129-03-SUMMARY.md and informs the CostAdapter shape in tier-3-graph/cost_adapter.py.
This script is NOT a test — it's a one-shot diagnostic. Re-run after any LightRAG
version bump to re-confirm the token_tracker / add_usage protocol.

Background (Phase 129 RESEARCH Open Q1):
    LightRAG's `lightrag.llm.openai.openai_complete_if_cache` exposes a
    ``token_tracker`` kwarg as of v1.3.x. Phase 129 RESEARCH Open Q1 hypothesized
    that v1.4.15 still calls ``token_tracker.add_usage(response.usage)`` after each
    LLM completion — but this was HIGH (not 100%) confidence. This probe resolves
    the hypothesis empirically before any cost-tracking code is committed.
"""
from __future__ import annotations

import importlib.metadata as md
import inspect


def _safe_version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "UNINSTALLED"


def _print_header(title: str) -> None:
    print(f"=== {title} ===")


def _resolve_callable(obj):
    """Unwrap EmbeddingFunc / partials / decorators to find the underlying function.

    LightRAG wraps openai_embed in an ``EmbeddingFunc`` dataclass — its inner
    function lives on the ``func`` attribute. Other wrappers may use ``__wrapped__``
    (functools.wraps) or expose ``__call__``.
    """
    for attr in ("func", "__wrapped__"):
        inner = getattr(obj, attr, None)
        if inner is not None and callable(inner):
            return inner
    return obj


def _grep_source(fn, needles: tuple[str, ...]) -> None:
    target = _resolve_callable(fn)
    try:
        src = inspect.getsource(target)
    except (OSError, TypeError):
        # TypeError fires when the object isn't a function/method (e.g. a dataclass
        # instance like EmbeddingFunc). After _resolve_callable we should usually
        # have a function, but we still guard for diagnostic robustness.
        print("  (source not retrievable — possibly compiled or wrapped)")
        return
    matches = 0
    for i, line in enumerate(src.splitlines(), 1):
        if any(n in line for n in needles):
            print(f"  L{i}: {line.rstrip()}")
            matches += 1
    if matches == 0:
        print(f"  (no references to {needles!r} found)")


def main() -> None:
    print(f"lightrag-hku version: {_safe_version('lightrag-hku')}")
    print()

    from lightrag.llm.openai import openai_complete_if_cache, openai_embed

    _print_header("openai_complete_if_cache signature")
    print(inspect.signature(openai_complete_if_cache))
    print()

    _print_header("openai_embed signature")
    # ``openai_embed`` is exposed as an ``EmbeddingFunc`` dataclass instance,
    # not a plain function — its inner callable is on ``.func``.
    embed_target = _resolve_callable(openai_embed)
    print(f"  type({type(openai_embed).__name__}) -> resolved {type(embed_target).__name__}")
    try:
        print(inspect.signature(embed_target))
    except (ValueError, TypeError) as exc:
        print(f"  (signature unavailable: {exc!r})")
    print()

    _print_header(
        "references to token_tracker / add_usage in openai_complete_if_cache source"
    )
    _grep_source(openai_complete_if_cache, ("token_tracker", "add_usage"))
    print()

    _print_header("references to token_tracker / add_usage in openai_embed source")
    _grep_source(openai_embed, ("token_tracker", "add_usage"))
    print()

    # Also probe the QueryParam / LightRAG class for any token_tracker hint at the
    # higher-level orchestration layer (LightRAG.aquery, etc.).
    try:
        from lightrag import LightRAG, QueryParam

        _print_header("LightRAG.__init__ signature")
        print(inspect.signature(LightRAG.__init__))
        print()
        _print_header("QueryParam fields")
        try:
            # Pydantic / dataclass / namedtuple — try the most likely shapes.
            fields = getattr(QueryParam, "__dataclass_fields__", None) or getattr(
                QueryParam, "model_fields", None
            )
            if fields is not None:
                for name in fields:
                    print(f"  {name}")
            else:
                print(f"  {QueryParam!r}")
        except Exception as exc:  # pragma: no cover — diagnostic only
            print(f"  (introspection error: {exc!r})")
    except Exception as exc:  # pragma: no cover — diagnostic only
        print(f"(LightRAG/QueryParam introspection skipped: {exc!r})")


if __name__ == "__main__":
    main()

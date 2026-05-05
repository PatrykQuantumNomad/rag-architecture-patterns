"""Phase 131 Stage 2 — score per-tier QueryLog JSONs with RAGAS.

Run from repo root:
    cd rag-architecture-patterns
    python -m evaluation.harness.score --tiers 1,2,3,5 --yes
    python -m evaluation.harness.score --tiers 1 --limit 5 --yes              # smoke
    python -m evaluation.harness.score --tiers 4 \\
        --judge-model openrouter/anthropic/claude-haiku-4.5 --yes             # alt judge

Outputs:
    evaluation/results/metrics/{tier}-{timestamp}.json    — per-question ScoreRecord list
    evaluation/results/costs/ragas-judge-{tier}-{ts}.json — D-13 schema; judge LLM cost only

Architecture (per 131-RESEARCH "Two-stage execution model"):
- Reads the most-recent {tier}-*.json from --queries-dir per tier (sort mtime DESC).
- For each tier, builds a RAGAS EvaluationDataset of SingleTurnSample (question_id-keyed
  by golden_qa.json's expected_answer for the `reference` field).
- NaN short-circuits BEFORE evaluate(): empty contexts (Pitfall 2) and agent-truncated
  records (Pitfall 8) emit ScoreRecord(nan_reason=...) without judge calls.
- evaluate(metrics=[faithfulness, answer_relevancy, context_precision], llm=judge_llm,
  embeddings=judge_emb, token_usage_parser=get_token_usage_for_openai, batch_size=10).
- Judge cost via CostTracker('ragas-judge-{tier}') → results/costs/ (D-13).

Decisions referenced (131-RESEARCH.md):
- Pattern 1: query JSON shape + ScoreRecord output shape.
- Pattern 2: judge cost via shared.cost_tracker.
- Pattern 6: judge LLM/embedder via litellm + OpenRouter.
- Pitfall 2: NaN short-circuit on empty contexts.
- Pitfall 3: batch_size=10 to bound concurrency.
- Pitfall 8: NaN short-circuit on agent_truncated.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console

from langchain_core.callbacks import BaseCallbackHandler
from ragas.callbacks import ChainType

from shared.config import get_settings
from shared.cost_tracker import CostTracker
from shared.pricing import PRICES

from evaluation.harness.records import (
    EvalRecord,
    QueryLog,
    ScoreRecord,
    read_query_log,
)


_NAN_REASON_LOG = logging.getLogger(__name__)


SUPPORTED_TIERS = (1, 2, 3, 4, 5)

# Pattern 6 — judge LLM + embedder via OpenRouter LiteLLM
JUDGE_LLM_SLUG_DEFAULT = "openrouter/google/gemini-2.5-flash"
JUDGE_EMB_SLUG_DEFAULT = "openrouter/openai/text-embedding-3-small"

# Plan 02-04 gap closure: bump from RAGAS default of 1024 to fit
# faithfulness atomic-statement extraction on long Tier 4 hybrid answers
# (2011-2575 chars on smoke set). Gemini 2.5 Flash output cap is 65,536,
# so 8192 is well within model limits and bounded-cost.
JUDGE_MAX_TOKENS = 8192


def _strip_openrouter_prefix(slug: str) -> str:
    """Strip the ``openrouter/`` prefix from a LiteLLM slug for PRICES lookup.

    Pattern 7: shared.pricing.PRICES uses provider-only slugs (e.g.
    ``google/gemini-2.5-flash``). The LiteLLM ``openrouter/`` prefix only
    affects routing; the underlying model is the same.
    """
    return slug.split("/", 1)[1] if slug.startswith("openrouter/") else slug


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_for_filename(ts: str) -> str:
    """Replace colons in an ISO 8601 timestamp so it's a valid POSIX filename."""
    return ts.replace(":", "_")


def _load_golden_qa_index() -> dict[str, dict]:
    """Load evaluation/golden_qa.json and index by question id."""
    qa = json.loads((_REPO_ROOT / "evaluation" / "golden_qa.json").read_text())
    return {q["id"]: q for q in qa}


def _latest_query_log(queries_dir: Path, tier: int) -> Optional[Path]:
    """Find the most-recent {tier}-*.json by mtime (DESC). Returns None if none."""
    candidates = sorted(
        queries_dir.glob(f"tier-{tier}-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _build_judge(judge_model: str, judge_emb: str):
    """Return (llm, emb) configured for OpenRouter via LiteLLM (Pattern 6).

    Plan 01 SUMMARY confirmed `from ragas.embeddings.base import embedding_factory`
    works in 0.4.3, but we keep a fallback to `from ragas.embeddings import ...`
    for forward-compat with future patches that may relocate the symbol.

    Plan 02-04 gap closure: passes ``max_tokens=JUDGE_MAX_TOKENS`` (8192) to
    ``llm_factory`` so RAGAS faithfulness ``_create_statements`` can fit the
    atomic-statement list output for long Tier 4 hybrid-mode answers
    (Plan 02-03 surfaced 4/5 NaN under the RAGAS default of 1024).
    """
    import litellm
    from ragas.llms import llm_factory
    try:
        from ragas.embeddings.base import embedding_factory  # primary path per 0.4.3 docs
    except ImportError:
        from ragas.embeddings import embedding_factory  # fallback if 0.4.x patches relocated

    llm = llm_factory(
        judge_model,
        provider="litellm",
        client=litellm.completion,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    emb = embedding_factory("litellm", model=judge_emb)
    # RAGAS 0.4.3's LiteLLMEmbeddings exposes embed_text / embed_texts (and
    # async variants) but some metric internals still call the legacy
    # LangChain-compat names embed_query / embed_documents. Observed missing
    # methods raised AttributeError mid-evaluation:
    #   - context_precision -> embed_query (single text)
    #   - answer_relevancy  -> embed_documents (list of texts)
    # Alias all four legacy names on the instance so scoring doesn't drop rows.
    if not hasattr(emb, "embed_query"):
        emb.embed_query = emb.embed_text  # type: ignore[attr-defined]
    if not hasattr(emb, "aembed_query"):
        emb.aembed_query = emb.aembed_text  # type: ignore[attr-defined]
    if not hasattr(emb, "embed_documents"):
        emb.embed_documents = emb.embed_texts  # type: ignore[attr-defined]
    if not hasattr(emb, "aembed_documents"):
        emb.aembed_documents = emb.aembed_texts  # type: ignore[attr-defined]
    return llm, emb


def _short_circuit_nan(rec: EvalRecord) -> Optional[ScoreRecord]:
    """Return a NaN ScoreRecord without calling RAGAS for known-bad inputs.

    Pitfall 2: empty contexts → faithfulness/context_precision undefined.
    Pitfall 8: agent_truncated → truncated answer; metrics meaningless.
    Tier 4 import error: library unavailable in sandbox.
    Cached miss: a question id missing from a Tier 4 cache file.
    """
    if rec.error == "max_turns_exceeded":
        return ScoreRecord(question_id=rec.question_id, nan_reason="agent_truncated")
    if not rec.retrieved_contexts:
        return ScoreRecord(question_id=rec.question_id, nan_reason="empty_contexts")
    if rec.error and rec.error.startswith("tier4_import_error"):
        return ScoreRecord(question_id=rec.question_id, nan_reason="tier4_unavailable")
    if rec.error and rec.error.startswith("cached_miss"):
        return ScoreRecord(question_id=rec.question_id, nan_reason="cached_miss")
    return None


class NaNReasonTracer(BaseCallbackHandler):
    """Capture per-row exception types so post-evaluate NaN classification
    has structured signals.

    RAGAS 0.4.3's Executor.wrap_callable_with_index swallows per-row
    exceptions and returns np.nan (ragas/executor.py:71-84). The langchain
    callback chain fires `on_chain_error` BEFORE the swallow, so subclassing
    BaseCallbackHandler is the only way to recover the exception TYPE for
    post-evaluate reason classification (HARN-05).

    Mapping from (row_index, metric_name) → exception_type_name is built up
    as RAGAS streams chain events; read from `errors` AFTER `evaluate()` returns.
    """

    def __init__(self) -> None:
        # str(run_id) → {"type": ChainType, "row_index": Optional[int],
        #                "metric_name": Optional[str]}
        self._chains: dict[str, dict] = {}
        # (row_index, metric_name) → leaf-most exception type name
        self.errors: dict[tuple[int, str], str] = {}

    def on_chain_start(
        self, serialized, inputs, *, run_id,
        parent_run_id=None, tags=None, metadata=None, **kwargs,
    ) -> None:
        chain_type = (metadata or {}).get("type")
        entry: dict = {"type": chain_type}
        if chain_type == ChainType.ROW:
            entry["row_index"] = (metadata or {}).get("row_index")
        elif chain_type == ChainType.METRIC and parent_run_id:
            parent = self._chains.get(str(parent_run_id), {})
            entry["row_index"] = parent.get("row_index")
            # serialized["name"] is "faithfulness-0", "answer_relevancy-3", ...
            # Strip "-{i}" suffix (Pitfall 4 of RESEARCH.md).
            raw_name = (serialized or {}).get("name", "")
            entry["metric_name"] = raw_name.rsplit("-", 1)[0] if "-" in raw_name else raw_name
        elif chain_type == ChainType.RAGAS_PROMPT and parent_run_id:
            parent = self._chains.get(str(parent_run_id), {})
            entry["row_index"] = parent.get("row_index")
            entry["metric_name"] = parent.get("metric_name")
        self._chains[str(run_id)] = entry

    def on_chain_error(self, error, *, run_id, **kwargs) -> None:
        chain = self._chains.get(str(run_id), {})
        row_idx = chain.get("row_index")
        metric = chain.get("metric_name")
        if row_idx is None or metric is None:
            return
        key = (row_idx, metric)
        # Idempotence: keep leaf-most exception (first set wins;
        # parent re-raises don't overwrite).
        if key not in self.errors:
            self.errors[key] = type(error).__name__


def _classify_post_evaluate_nan(
    row_idx: int,
    metric_name: str,
    metric_value: Optional[float],
    tracer: "NaNReasonTracer",
) -> Optional[str]:
    """Map a (row_idx, metric, value) cell to a nan_reason string.

    Returns None when metric_value is not None (no NaN, no reason needed).

    Precedence (most specific to least specific):
      1. Captured RagasOutputParserException → 'json_parse_failure'
      2. Captured LLMDidNotFinishException → 'llm_did_not_finish'
      3. faithfulness=NaN, no captured exception → 'empty_statements'
         (Faithfulness._ascore returns np.nan when statements==[];
          ragas/metrics/_faithfulness.py:210-211)
      4. answer_relevancy=NaN, no captured exception → 'empty_questions'
         (calculate_score returns np.nan when all gen_questions=='';
          ragas/metrics/_answer_relevance.py:120-124)
      5. context_precision=NaN, no captured exception → 'invalid_verdicts'
         (_calculate_average_precision returns np.nan;
          ragas/metrics/_context_precision.py:116)
      6. Unknown metric or unmapped path → 'unknown_nan' + WARNING log
         (defensive — never silently drop NaN; Pitfall 7 of RESEARCH.md)
    """
    if metric_value is not None:
        return None
    captured = tracer.errors.get((row_idx, metric_name))
    if captured == "RagasOutputParserException":
        return "json_parse_failure"
    if captured == "LLMDidNotFinishException":
        return "llm_did_not_finish"
    # Per-metric semantic NaN paths (no exception OR captured-but-unmapped type)
    if metric_name == "faithfulness":
        return "empty_statements"
    if metric_name == "answer_relevancy":
        return "empty_questions"
    if metric_name == "context_precision":
        return "invalid_verdicts"
    _NAN_REASON_LOG.warning(
        "unknown_nan: row_idx=%s metric=%s captured=%s — extend "
        "_classify_post_evaluate_nan",
        row_idx, metric_name, captured,
    )
    return "unknown_nan"


def _to_float_or_none(v: Any) -> Optional[float]:
    """Map NaN / None / pandas-NA to None; preserve real floats."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check (NaN != NaN)
        return None
    return f


async def score_query_log(
    log: QueryLog,
    golden_qa_index: dict[str, dict],
    judge_llm: Any = None,
    judge_emb: Any = None,
    batch_size: int = 10,
    raise_exceptions: bool = False,
) -> tuple[list[ScoreRecord], dict[str, Any]]:
    """Score one tier's QueryLog. Returns (scores, judge_usage_summary).

    `judge_usage_summary` carries the input/output token totals from RAGAS so the
    caller can record judge cost via shared.cost_tracker.

    NaN short-circuits run BEFORE the dataset is built so judge cost is only
    paid for records that have at least one retrieved context and no error.
    """
    # Lazy imports — keep --help responsive even on slow ragas/litellm boot.
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.cost import get_token_usage_for_openai

    scores: list[ScoreRecord] = [None] * len(log.records)  # type: ignore[list-item]
    samples_with_index: list[tuple[int, Any]] = []

    for i, rec in enumerate(log.records):
        nan = _short_circuit_nan(rec)
        if nan is not None:
            scores[i] = nan
            continue
        ref = golden_qa_index.get(rec.question_id, {}).get("expected_answer", "")
        samples_with_index.append((i, SingleTurnSample(
            user_input=rec.question,
            response=rec.answer,
            retrieved_contexts=rec.retrieved_contexts,
            reference=ref,
        )))

    if not samples_with_index:
        return scores, {"input_tokens": 0, "output_tokens": 0, "n_scored": 0}

    dataset = EvaluationDataset(samples=[s for _, s in samples_with_index])
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_emb,
        token_usage_parser=get_token_usage_for_openai,
        batch_size=batch_size,
        raise_exceptions=raise_exceptions,
        show_progress=True,
    )

    # Map result rows back to original record indices.
    df = result.to_pandas() if hasattr(result, "to_pandas") else None
    if df is not None:
        for j, (orig_idx, _) in enumerate(samples_with_index):
            row = df.iloc[j]
            scores[orig_idx] = ScoreRecord(
                question_id=log.records[orig_idx].question_id,
                faithfulness=_to_float_or_none(row.get("faithfulness")),
                answer_relevancy=_to_float_or_none(row.get("answer_relevancy")),
                context_precision=_to_float_or_none(row.get("context_precision")),
            )
    else:
        # Fallback: result.scores is a list[dict] in some 0.4.x patches.
        rows = getattr(result, "scores", []) or []
        for j, (orig_idx, _) in enumerate(samples_with_index):
            row = rows[j] if j < len(rows) else {}
            scores[orig_idx] = ScoreRecord(
                question_id=log.records[orig_idx].question_id,
                faithfulness=_to_float_or_none(row.get("faithfulness")),
                answer_relevancy=_to_float_or_none(row.get("answer_relevancy")),
                context_precision=_to_float_or_none(row.get("context_precision")),
            )

    # Token usage summary — defensive about API surface drift across 0.4.x patches.
    # ragas.cost.CostCallbackHandler.total_tokens() raises IndexError when its
    # internal usage_data list is empty (no judge call piped through the
    # token-usage parser). Treat that as "no usage recorded" rather than aborting
    # the whole tier and losing scores already computed in `result`.
    try:
        usage = result.total_tokens() if hasattr(result, "total_tokens") else None
    except (IndexError, AttributeError):
        usage = None
    in_tok = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    out_tok = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0

    return scores, {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "n_scored": len(samples_with_index),
    }


def _persist_metrics(scores: list[ScoreRecord], out_path: Path) -> Path:
    """Write a list of ScoreRecord-shaped dicts as indent-2 JSON. Creates parent dirs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [s.model_dump() for s in scores]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


async def amain(args, console: Console) -> int:
    """Main async entrypoint — orchestrates per-tier scoring."""
    settings = get_settings()
    if not settings.openrouter_api_key:
        console.print("[red]OPENROUTER_API_KEY not set — judge LLM cannot run.[/red]")
        return 2

    queries_dir = Path(args.queries_dir)
    if not queries_dir.exists():
        console.print(f"[red]queries_dir not found: {queries_dir}[/red]")
        console.print("[red]Run `python -m evaluation.harness.run --tiers 1,2,3,5 --yes` first.[/red]")
        return 2

    golden_qa_path = _REPO_ROOT / "evaluation" / "golden_qa.json"
    if not golden_qa_path.exists():
        console.print(f"[red]golden_qa.json not found at {golden_qa_path}[/red]")
        return 2
    golden_qa_index = _load_golden_qa_index()

    tiers = sorted(int(t) for t in args.tiers.split(",") if t.strip())
    invalid = [t for t in tiers if t not in SUPPORTED_TIERS]
    if invalid:
        console.print(f"[red]Unsupported tier(s): {invalid}. Supported: {list(SUPPORTED_TIERS)}.[/red]")
        return 2

    # Discover query logs per tier (most-recent by mtime).
    found: dict[int, Path] = {}
    for tier in tiers:
        path = _latest_query_log(queries_dir, tier)
        if path is None:
            console.print(f"[yellow]Tier {tier}: no query log found in {queries_dir}; skipping.[/yellow]")
            continue
        found[tier] = path

    if not found:
        console.print("[red]No query logs found for any requested tier.[/red]")
        return 1

    # Cost-surprise gate (Pattern 3 mitigation).
    total_records = 0
    for tier, path in found.items():
        log = read_query_log(path)
        n = min(args.limit, len(log.records)) if args.limit else len(log.records)
        total_records += n
    if not args.yes:
        # Conservative: ~$0.0003/judge-call × 3 metrics × ~3 internal calls/metric ≈ $0.003/q.
        est = total_records * 0.003
        console.print(
            f"[yellow]About to score {total_records} record(s) across {len(found)} tier(s) "
            f"≈ ${est:.4f} judge cost (conservative; actual logged in costs JSON).[/yellow]"
        )
        try:
            ans = input("Continue? [y/N]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in {"y", "yes"}:
            console.print("[red]Aborted.[/red]")
            return 1

    # Build judge (Pattern 6).
    judge_llm, judge_emb = _build_judge(args.judge_model, args.judge_emb)
    judge_pricing_key = _strip_openrouter_prefix(args.judge_model)

    # Score per tier.
    metrics_dir = Path(args.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for tier, path in found.items():
        log = read_query_log(path)
        if args.limit:
            log.records = log.records[: args.limit]

        console.print(f"[dim]Scoring tier-{tier} from {path}[/dim]")
        scores, usage = await score_query_log(
            log,
            golden_qa_index,
            judge_llm,
            judge_emb,
            batch_size=args.batch_size,
            raise_exceptions=False,
        )

        out = metrics_dir / f"tier-{tier}-{_ts_for_filename(log.timestamp)}.json"
        _persist_metrics(scores, out)
        console.print(f"[green]Tier {tier} → {out}[/green]")

        # Judge cost via shared.cost_tracker (Pattern 2).
        # Distinct tier string `ragas-judge-tier-{N}` collision-free vs `tier-{N}-eval` (Plan 04).
        tracker = CostTracker(f"ragas-judge-tier-{tier}")
        if judge_pricing_key in PRICES:
            try:
                tracker.record_llm(
                    judge_pricing_key,
                    usage["input_tokens"],
                    usage["output_tokens"],
                )
            except KeyError:
                # Defensive: PRICES key absent → leave tracker empty (USD = 0).
                pass
        tracker.persist()
        console.print(
            f"[green]Judge cost tier-{tier}: ${tracker.total_usd():.6f} "
            f"({usage['input_tokens']} in / {usage['output_tokens']} out tokens)[/green]"
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser. Module-level so tests can introspect flags."""
    p = argparse.ArgumentParser(
        description="Phase 131 Stage 2 — RAGAS scoring of per-tier QueryLog JSONs.",
    )
    p.add_argument(
        "--queries-dir",
        default="evaluation/results/queries",
        help="Directory holding {tier}-{timestamp}.json files.",
    )
    p.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Parent of metrics/ + costs/. (Default: evaluation/results.)",
    )
    p.add_argument(
        "--tiers",
        default="1,2,3,4,5",
        help="Comma-separated tier numbers to score. Default: all 5.",
    )
    p.add_argument(
        "--judge-model",
        default=JUDGE_LLM_SLUG_DEFAULT,
        help=f"LiteLLM judge slug. Default: {JUDGE_LLM_SLUG_DEFAULT}.",
    )
    p.add_argument(
        "--judge-emb",
        default=JUDGE_EMB_SLUG_DEFAULT,
        help=f"LiteLLM embedder slug. Default: {JUDGE_EMB_SLUG_DEFAULT}.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="evaluate() batch_size (Pitfall 3 — bounds concurrency). Default: 10.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate each tier's records to first N (smoke).",
    )
    p.add_argument("--yes", action="store_true", help="Skip cost-surprise prompt.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(amain(args, Console()))


if __name__ == "__main__":
    raise SystemExit(main())

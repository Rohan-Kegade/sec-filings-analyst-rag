"""Retriever eval (component): DeepEval Contextual Recall and Contextual Precision.

Each golden's question goes through the retriever only (no answer generation); a Claude
judge then scores the retrieved chunks against the golden's reference answer.

    python -m evals.eval_retriever run                  # hybrid, all 40 goldens
    python -m evals.eval_retriever run --mode dense --mode hybrid --limit 5
    python -m evals.eval_retriever compare              # last two saved runs
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import typer
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from evals import GOLDENS_DIR, runs
from evals.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL, score_case
from evals.runs import git_commit
from sec_rag.config import settings
from sec_rag.retrieval.retriever import Retriever

# --- goldens ---------------------------------------------------------------------------------

GOLDENS_FILE = GOLDENS_DIR / "retrieval_goldens.json"


@dataclass(frozen=True)
class Golden:
    id: str
    category: str
    input: str
    expected_output: str


def load_goldens(path: Path = GOLDENS_FILE) -> list[Golden]:
    goldens = [
        Golden(id=raw["id"], category=raw["category"], input=raw["input"],
               expected_output=raw["expected_output"])
        for raw in json.loads(path.read_text(encoding="utf-8"))
    ]  # fmt: skip
    ids = [g.id for g in goldens]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate golden ids in {path.name}: {duplicates}")
    return goldens


# --- runner ----------------------------------------------------------------------------------

RUN_KIND = "retriever"  # saved under evals/runs/retriever/
METRICS = ("recall", "precision")


def deepeval_metrics(judge: DeepEvalBaseLLM, include_reason: bool) -> dict:
    """Fresh metric objects for one test case (DeepEval metrics hold per-case state)."""
    common = {"model": judge, "include_reason": include_reason, "eval_mode": "llm"}
    return {
        "recall": ContextualRecallMetric(**common),
        "precision": ContextualPrecisionMetric(**common),
    }


async def evaluate_mode(
    retriever: Retriever,
    goldens: list[Golden],
    k: int,
    metric_factory: Callable[[], dict],
    concurrency: int,
) -> list[dict]:
    # Retrieval is fast and local, so do it up front; only the judge calls run concurrently.
    retrieved, latencies = [], []
    for golden in goldens:
        start = time.perf_counter()
        retrieved.append(retriever.retrieve(golden.input, k=k))
        latencies.append((time.perf_counter() - start) * 1000)

    semaphore = asyncio.Semaphore(concurrency)
    scored = await asyncio.gather(
        *(
            score_case(
                LLMTestCase(
                    input=golden.input,
                    expected_output=golden.expected_output,
                    retrieval_context=[rc.chunk.text for rc in results],
                ),
                metric_factory(),
                semaphore,
            )
            for golden, results in zip(goldens, retrieved)
        )
    )
    return [
        {
            "id": golden.id,
            "category": golden.category,
            "latency_ms": latency,
            "retrieved": [
                f"{rc.chunk.ticker} FY{rc.chunk.fiscal_year} Item {rc.chunk.item} "
                f"({rc.chunk.chunk_id})"
                for rc in results
            ],
            **scores,
        }
        for golden, results, latency, scores in zip(goldens, retrieved, latencies, scored)
    ]


def summarize(rows: list[dict]) -> dict:
    summary = {"cases": len(rows)}
    for metric in METRICS:
        values = [r[metric]["score"] for r in rows if r[metric]["score"] is not None]
        summary[metric] = mean(values) if values else None
        summary[f"{metric}_errors"] = sum(r[metric]["error"] is not None for r in rows)
    summary["latency_ms"] = mean(r["latency_ms"] for r in rows) if rows else None
    return summary


def run(
    retrievers: dict[str, Retriever],
    goldens: list[Golden],
    k: int,
    metric_factory: Callable[[], dict],
    concurrency: int,
    judge_name: str,
) -> dict:
    async def run_all() -> dict[str, list[dict]]:
        # One event loop for every mode: the async Claude client's connections belong to it.
        return {
            mode: await evaluate_mode(retriever, goldens, k, metric_factory, concurrency)
            for mode, retriever in retrievers.items()
        }

    results = asyncio.run(run_all())
    categories = sorted({g.category for g in goldens})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "k": k,
            "judge": judge_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "hybrid_candidates": settings.hybrid_candidates,
            "rrf_k": settings.rrf_k,
            "git_commit": git_commit(),
        },
        "goldens": {"total": len(goldens)},
        "summary": {mode: summarize(rows) for mode, rows in results.items()},
        "by_category": {
            mode: {c: summarize([r for r in rows if r["category"] == c]) for c in categories}
            for mode, rows in results.items()
        },
        "cases": results,
    }


# --- cli -------------------------------------------------------------------------------------

app = typer.Typer(help=__doc__, no_args_is_help=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Differences smaller than this between runs are likely judge noise, not real changes.
NOISE = 0.03


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _retrievers(modes: list[str]) -> dict:
    # Imported here so `compare` doesn't load the embedding model.
    from sec_rag.indexing.embedder import Embedder
    from sec_rag.indexing.keyword_index import KeywordIndex
    from sec_rag.indexing.vector_store import VectorStore
    from sec_rag.retrieval.retriever import Retriever

    shared = {"embedder": Embedder(), "store": VectorStore(), "keyword_index": KeywordIndex()}
    retrievers = {mode: Retriever(**shared, mode=mode) for mode in modes}
    next(iter(retrievers.values())).warm_up()  # load the model before latency is measured
    return retrievers


def print_report(result: dict) -> None:
    s, g = result["settings"], result["goldens"]
    typer.echo(
        f"Goldens: {g['total']} | k={s['k']} | judge {s['judge']}"
    )
    typer.echo(f"\n{'mode':>8} {'recall':>8} {'precision':>10} {'errors':>7} {'ms':>6}")
    for mode, m in result["summary"].items():
        errors = m["recall_errors"] + m["precision_errors"]
        typer.echo(
            f"{mode:>8} {_fmt(m['recall']):>8} {_fmt(m['precision']):>10} "
            f"{errors:>7} {_fmt(m['latency_ms'], 0):>6}"
        )

    modes = list(result["by_category"])
    typer.echo("\nBy category (recall / precision):")
    typer.echo(f"{'category':>12} " + " ".join(f"{m:>15}" for m in modes))
    for category in next(iter(result["by_category"].values()), {}):
        cells = []
        for mode in modes:
            m = result["by_category"][mode][category]
            cells.append(f"{_fmt(m['recall'], 2)} / {_fmt(m['precision'], 2)}")
        typer.echo(f"{category:>12} " + " ".join(f"{c:>15}" for c in cells))

    for mode, rows in result["cases"].items():
        weak = [r for r in rows if (r["recall"]["score"] or 0) < 0.5 or r["recall"]["error"]]
        if weak:
            typer.echo(f"\n{mode}: cases with recall < 0.5")
            for r in weak:
                detail = r["recall"]["error"] or r["recall"]["reason"] or ""
                typer.echo(f"  {_fmt(r['recall']['score'], 2):>5}  {r['id']}: {detail[:160]}")


@app.command("run")
def run_command(
    mode: list[str] = typer.Option(["hybrid"], help="Retrieval modes: hybrid, dense, keyword"),
    k: int = typer.Option(8, help="Chunks retrieved per question"),
    limit: int = typer.Option(0, help="Only the first N goldens (0 = all)"),
    category: list[str] = typer.Option(None, help="Only these categories"),
    id: list[str] = typer.Option(None, help="Only these golden ids"),
    judge_model: str = typer.Option(
        DEFAULT_JUDGE_MODEL, help="Claude model used as judge (default: SEC_RAG_JUDGE_MODEL)"
    ),
    effort: str = typer.Option(
        DEFAULT_JUDGE_EFFORT, help="Judge effort: low, medium, high (default: SEC_RAG_JUDGE_EFFORT)"
    ),
    reasons: bool = typer.Option(True, help="Ask the judge for a reason per score (costs more)"),
    concurrency: int = typer.Option(4, help="Test cases judged in parallel"),
    save: bool = typer.Option(True, help="Save the run for `compare`"),
):
    """Score retrieval with DeepEval Contextual Recall and Contextual Precision."""
    from evals.judge import ClaudeJudge

    goldens = load_goldens()
    goldens = [
        gold for gold in goldens
        if (not category or gold.category in category)
        and (not id or gold.id in id)
    ]  # fmt: skip
    if limit:
        goldens = goldens[:limit]
    if not goldens:
        raise typer.Exit("No goldens match the filters.")

    judge = ClaudeJudge(judge_model, effort=effort)
    result = run(
        _retrievers(mode),
        goldens,
        k,
        metric_factory=lambda: deepeval_metrics(judge, include_reason=reasons),
        concurrency=concurrency,
        judge_name=judge_model,
    )
    print_report(result)
    typer.echo(f"\n{judge.cost_summary()}")
    if save:
        typer.echo(f"Saved {runs.save(result, RUN_KIND)}")


@app.command()
def compare(before: Path = typer.Argument(None), after: Path = typer.Argument(None)):
    """Compare two saved runs (default: the last two)."""
    if before is None or after is None:
        saved = runs.list_runs(RUN_KIND)
        if len(saved) < 2:
            raise typer.Exit("Need at least two saved runs to compare.")
        before, after = saved[-2], saved[-1]
    a, b = runs.load_run(before), runs.load_run(after)
    typer.echo(f"before: {before.name}\nafter:  {after.name}")
    for key, value in b["settings"].items():
        if a["settings"].get(key) != value:
            typer.echo(f"  setting {key}: {a['settings'].get(key)} -> {value}")

    typer.echo(f"\n{'mode':>8} {'recall':>17} {'precision':>17}")
    for mode, m in b["summary"].items():
        if mode not in a["summary"]:
            continue
        cells = []
        for metric in METRICS:
            old, new = a["summary"][mode][metric], m[metric]
            delta = "" if old is None or new is None else f" ({new - old:+.3f})"
            cells.append(f"{_fmt(new)}{delta}")
        typer.echo(f"{mode:>8} " + " ".join(f"{c:>17}" for c in cells))
    typer.echo(f"(changes under ±{NOISE} are likely judge noise)")

    for mode, rows in b["cases"].items():
        old_rows = {r["id"]: r for r in a["cases"].get(mode, [])}
        moves = []
        for r in rows:
            o = old_rows.get(r["id"])
            if o and o["recall"]["score"] is not None and r["recall"]["score"] is not None:
                delta = r["recall"]["score"] - o["recall"]["score"]
                if abs(delta) >= 0.1:
                    moves.append((delta, r["id"]))
        if moves:
            typer.echo(f"\n{mode}: recall changes of 0.1 or more")
            for delta, case_id in sorted(moves):
                typer.echo(f"  {delta:+.2f}  {case_id}")


if __name__ == "__main__":
    app()

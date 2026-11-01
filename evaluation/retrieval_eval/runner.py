"""Retrieve for each golden, then score the retrieved chunks with DeepEval metrics."""

import asyncio
import json
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from statistics import mean

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from evaluation.retrieval_eval.goldens import Golden
from sec_rag.config import settings
from sec_rag.retrieval.retriever import Retriever

RUNS_DIR = Path(__file__).parents[1] / "runs" / "retrieval"
METRICS = ("recall", "precision")


def deepeval_metrics(judge: DeepEvalBaseLLM, include_reason: bool) -> dict:
    """Fresh metric objects for one test case (DeepEval metrics hold per-case state)."""
    common = {"model": judge, "include_reason": include_reason, "eval_mode": "llm"}
    return {
        "recall": ContextualRecallMetric(**common),
        "precision": ContextualPrecisionMetric(**common),
    }


async def score_case(
    test_case: LLMTestCase, metrics: dict, semaphore: asyncio.Semaphore
) -> dict:
    """Run both metrics on one test case. A failed metric is recorded, not raised."""
    async with semaphore:
        scores = {}
        for name, metric in metrics.items():
            try:
                await metric.a_measure(test_case, _show_indicator=False)
                scores[name] = {"score": metric.score, "reason": metric.reason, "error": None}
            except Exception as exc:  # judge or API failure: keep the rest of the run going
                scores[name] = {"score": None, "reason": None, "error": str(exc)}
        return scores


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
            "status": golden.status,
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


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
            "git_commit": _git_commit(),
        },
        "goldens": {
            "total": len(goldens),
            "approved": sum(g.status == "approved" for g in goldens),
        },
        "summary": {mode: summarize(rows) for mode, rows in results.items()},
        "by_category": {
            mode: {c: summarize([r for r in rows if r["category"] == c]) for c in categories}
            for mode, rows in results.items()
        },
        "cases": results,
    }


def save(result: dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{result['timestamp'].replace(':', '-')}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def list_runs() -> list[Path]:
    return sorted(RUNS_DIR.glob("*.json"))


def load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

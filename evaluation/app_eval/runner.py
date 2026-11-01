"""Run each golden's question through the real pipeline once, then score application metrics.

All selected metrics score the same generated answers, so their results are comparable and
generation is paid for once.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from statistics import mean

from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from evaluation.app_eval import completeness, correctness
from evaluation.app_eval.goldens import AppGolden
from evaluation.rag_eval.runner import run_pipeline
from evaluation.retrieval_eval.runner import score_case
from evaluation.runs import git_commit
from sec_rag.config import settings
from sec_rag.pipeline import RAGPipeline

RUN_KIND = "app"  # saved under evaluation/runs/app/
PASS = 0.7  # a score at or above this counts as a pass


@dataclass(frozen=True)
class AppMetric:
    """One application metric: its judge, the expected output it sees, and where it applies."""

    build: Callable[[DeepEvalBaseLLM], object]
    expected: Callable[[AppGolden], str]
    applies: Callable[[AppGolden], bool]


METRICS = {
    "correctness": AppMetric(
        build=correctness.correctness_metric,
        expected=correctness.expected_with_key_facts,
        applies=lambda golden: True,
    ),
    "completeness": AppMetric(
        build=completeness.completeness_metric,
        expected=completeness.expected_with_points,
        applies=lambda golden: bool(golden.required_points),  # not for out-of-scope cases
    ),
}

NOT_APPLICABLE = {"score": None, "reason": None, "error": None, "applies": False}


async def score_all(goldens: list[AppGolden], outputs: list[dict], metrics: dict[str, AppMetric],
                    judge: DeepEvalBaseLLM, concurrency: int) -> list[dict]:  # fmt: skip
    semaphore = asyncio.Semaphore(concurrency)

    async def score(golden: AppGolden, output: dict, name: str) -> dict:
        metric = metrics[name]
        if not metric.applies(golden):
            return NOT_APPLICABLE
        if output["answer"] is None:
            return {"score": None, "reason": None, "error": output["error"] or "no answer",
                    "applies": True}  # fmt: skip
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=output["answer"],
            expected_output=metric.expected(golden),
        )
        scored = await score_case(test_case, {name: metric.build(judge)}, semaphore)
        return {**scored[name], "applies": True}

    async def score_golden(golden: AppGolden, output: dict) -> dict:
        names = list(metrics)
        results = await asyncio.gather(*(score(golden, output, n) for n in names))
        return dict(zip(names, results))

    return await asyncio.gather(*(score_golden(g, o) for g, o in zip(goldens, outputs)))


def summarize(rows: list[dict], metric_names: list[str]) -> dict:
    summary: dict = {"cases": len(rows)}
    for name in metric_names:
        scores = [r[name]["score"] for r in rows if r[name]["score"] is not None]
        summary[name] = {
            "score": mean(scores) if scores else None,
            "passed": sum(s >= PASS for s in scores) / len(scores) if scores else None,
            "applies": sum(r[name]["applies"] for r in rows),
            "errors": sum(r[name]["error"] is not None for r in rows),
        }
    if "correctness" in metric_names:
        found = sum(r["key_numbers_found"] for r in rows)
        total = sum(r["key_numbers_total"] for r in rows)
        summary["key_numbers_found"] = found / total if total else None
    return summary


def run(pipeline: RAGPipeline, goldens: list[AppGolden], k: int, metrics: dict[str, AppMetric],
        judge: DeepEvalBaseLLM, concurrency: int, judge_name: str) -> dict:  # fmt: skip
    """`metrics` maps names to definitions, normally a selection from METRICS."""
    metric_names = list(metrics)
    outputs = run_pipeline(pipeline, goldens, k, concurrency)
    scores = asyncio.run(score_all(goldens, outputs, metrics, judge, concurrency))

    rows = []
    for golden, output, scored in zip(goldens, outputs, scores):
        found, total = correctness.number_check(golden, output["answer"])
        rows.append({
            "id": golden.id,
            "category": golden.category,
            "expected_behavior": golden.expected_behavior,
            "status": golden.status,
            "question": golden.input,
            "answer": output["answer"],
            "generation_error": output["error"],
            "sources": [f"{rc.chunk.ticker} FY{rc.chunk.fiscal_year} Item {rc.chunk.item}"
                        for rc in output["sources"]],
            "key_numbers_found": found,
            "key_numbers_total": total,
            **scored,
        })  # fmt: skip

    categories = sorted({g.category for g in goldens})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": metric_names,
        "settings": {
            "k": k,
            "retrieval_mode": pipeline.retriever.mode,
            "generator": getattr(pipeline.generator, "model", settings.llm_model),
            "generator_effort": settings.llm_effort,
            "judge": judge_name,
            "chunk_size": settings.chunk_size,
            "embedding_model": settings.embedding_model,
            "git_commit": git_commit(),
        },
        "goldens": {
            "total": len(goldens),
            "approved": sum(g.status == "approved" for g in goldens),
        },
        "summary": summarize(rows, metric_names),
        "by_category": {
            c: summarize([r for r in rows if r["category"] == c], metric_names)
            for c in categories
        },
        "cases": rows,
    }

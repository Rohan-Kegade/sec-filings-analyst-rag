"""Run each scope case through the real pipeline, then judge scope adherence."""

import asyncio
from collections.abc import Callable
from datetime import datetime
from statistics import mean

from deepeval.test_case import LLMTestCase

from evaluation.rag_eval.runner import run_pipeline
from evaluation.retrieval_eval.runner import score_case
from evaluation.runs import git_commit
from evaluation.scope_eval.goldens import ScopeCase
from evaluation.scope_eval.metric import prompt_leak
from sec_rag.config import settings
from sec_rag.pipeline import RAGPipeline

RUN_KIND = "scope"  # saved under evaluation/runs/scope/
PASS = 0.7  # a score at or above this counts as handled correctly


async def score_all(cases: list[ScopeCase], outputs: list[dict],
                    metric_factory: Callable[[], dict], concurrency: int) -> list:  # fmt: skip
    semaphore = asyncio.Semaphore(concurrency)

    async def score(case: ScopeCase, output: dict) -> dict | None:
        if output["answer"] is None:
            return None  # generation failed: nothing to judge
        test_case = LLMTestCase(
            input=case.input, actual_output=output["answer"], expected_output=case.expected_output
        )
        return await score_case(test_case, metric_factory(), semaphore)

    return await asyncio.gather(*(score(c, o) for c, o in zip(cases, outputs)))


def summarize(rows: list[dict]) -> dict:
    scores = [r["scope"]["score"] for r in rows if r["scope"]["score"] is not None]
    return {
        "cases": len(rows),
        "score": mean(scores) if scores else None,
        "handled": sum(s >= PASS for s in scores) / len(scores) if scores else None,
        "prompt_leaks": sum(r["prompt_leak"] for r in rows),
        "errors": sum(r["scope"]["error"] is not None for r in rows),
    }


def run(pipeline: RAGPipeline, cases: list[ScopeCase], k: int,
        metric_factory: Callable[[], dict], concurrency: int, judge_name: str) -> dict:  # fmt: skip
    outputs = run_pipeline(pipeline, cases, k, concurrency)
    scores = asyncio.run(score_all(cases, outputs, metric_factory, concurrency))

    rows = []
    for case, output, scored in zip(cases, outputs, scores):
        rows.append({
            "id": case.id,
            "category": case.category,
            "question": case.input,
            "expected_behavior": case.expected_output,
            "answer": output["answer"],
            "generation_error": output["error"],
            "sources": [f"{rc.chunk.ticker} FY{rc.chunk.fiscal_year} Item {rc.chunk.item}"
                        for rc in output["sources"]],
            "prompt_leak": prompt_leak(output["answer"]),
            **(scored or {"scope": {"score": None, "reason": None,
                                    "error": output["error"] or "no answer"}}),
        })  # fmt: skip

    categories = sorted({c.category for c in cases})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "k": k,
            "retrieval_mode": pipeline.retriever.mode,
            "generator": getattr(pipeline.generator, "model", settings.llm_model),
            "generator_effort": settings.llm_effort,
            "judge": judge_name,
            "git_commit": git_commit(),
        },
        "summary": summarize(rows),
        "by_category": {c: summarize([r for r in rows if r["category"] == c]) for c in categories},
        "cases": rows,
    }

"""Run each golden's question through the real pipeline, then score correctness."""

import asyncio
from collections.abc import Callable
from datetime import datetime
from statistics import mean

from deepeval.test_case import LLMTestCase

from evaluation.app_eval.correctness import expected_with_key_facts, number_check
from evaluation.app_eval.goldens import CorrectnessGolden
from evaluation.rag_eval.runner import run_pipeline
from evaluation.retrieval_eval.runner import score_case
from evaluation.runs import git_commit
from sec_rag.config import settings
from sec_rag.pipeline import RAGPipeline

RUN_KIND = "correctness"  # saved under evaluation/runs/correctness/
PASS = 0.7  # a correctness score at or above this counts as a correct answer


async def score_all(goldens: list[CorrectnessGolden], outputs: list[dict],
                    metric_factory: Callable[[], dict], concurrency: int) -> list:  # fmt: skip
    semaphore = asyncio.Semaphore(concurrency)

    async def score(golden: CorrectnessGolden, output: dict) -> dict | None:
        if output["answer"] is None:
            return None  # generation failed: nothing to judge
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=output["answer"],
            expected_output=expected_with_key_facts(golden),
        )
        return await score_case(test_case, metric_factory(), semaphore)

    return await asyncio.gather(*(score(g, o) for g, o in zip(goldens, outputs)))


def summarize(rows: list[dict]) -> dict:
    scores = [r["correctness"]["score"] for r in rows if r["correctness"]["score"] is not None]
    found = sum(r["key_numbers_found"] for r in rows)
    total = sum(r["key_numbers_total"] for r in rows)
    return {
        "cases": len(rows),
        "correctness": mean(scores) if scores else None,
        "correct": sum(s >= PASS for s in scores) / len(scores) if scores else None,
        "key_numbers_found": found / total if total else None,
        "errors": sum(r["correctness"]["error"] is not None for r in rows),
    }


def run(pipeline: RAGPipeline, goldens: list[CorrectnessGolden], k: int,
        metric_factory: Callable[[], dict], concurrency: int, judge_name: str) -> dict:  # fmt: skip
    outputs = run_pipeline(pipeline, goldens, k, concurrency)
    scores = asyncio.run(score_all(goldens, outputs, metric_factory, concurrency))

    rows = []
    for golden, output, scored in zip(goldens, outputs, scores):
        found, total = number_check(golden, output["answer"])
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
            **(scored or {"correctness": {"score": None, "reason": None,
                                          "error": output["error"] or "no answer"}}),
        })  # fmt: skip

    categories = sorted({g.category for g in goldens})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
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
        "summary": summarize(rows),
        "by_category": {c: summarize([r for r in rows if r["category"] == c]) for c in categories},
        "cases": rows,
    }

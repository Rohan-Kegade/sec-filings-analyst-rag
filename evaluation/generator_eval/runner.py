"""Generate an answer for each golden from its frozen excerpts, then score it with DeepEval."""

import asyncio
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from statistics import mean

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from evaluation.generator_eval.goldens import GeneratorGolden, to_chunks
from evaluation.retrieval_eval.runner import score_case
from evaluation.runs import git_commit
from sec_rag.config import settings
from sec_rag.generation.llm import AnswerGenerator

RUN_KIND = "generator"  # saved under evaluation/runs/generator/
METRICS = ("faithfulness", "relevancy")

# Rough check that an answer says the excerpts don't contain the information. Used to report
# the no-answer cases, where Faithfulness alone can't tell a refusal from a vague answer.
ABSTAIN = re.compile(
    r"\b(do(es)? not|don't|doesn't|did not|no)\b.{0,40}\b(contain|include|disclose|provide|"
    r"state|mention|report|give|specify|information|data|details)",
    re.IGNORECASE,
)


def deepeval_metrics(judge: DeepEvalBaseLLM, include_reason: bool) -> dict:
    """Fresh metric objects for one test case (DeepEval metrics hold per-case state)."""
    common = {"model": judge, "include_reason": include_reason, "eval_mode": "llm"}
    return {
        "faithfulness": FaithfulnessMetric(**common),
        "relevancy": AnswerRelevancyMetric(**common),
    }


def generate_answers(
    generator: AnswerGenerator, goldens: list[GeneratorGolden], concurrency: int
) -> list[dict]:
    """Answer every golden from its own excerpts. Failures are recorded, not raised."""

    def answer(golden: GeneratorGolden) -> dict:
        start = time.perf_counter()
        try:
            text, error = generator.generate(golden.input, to_chunks(golden)), None
        except Exception as exc:  # API or model failure for this golden only
            text, error = None, str(exc)
        return {"answer": text, "error": error, "latency_ms": (time.perf_counter() - start) * 1000}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(answer, goldens))


async def score_all(
    goldens: list[GeneratorGolden], answers: list[dict], metric_factory: Callable[[], dict],
    concurrency: int,
) -> list[dict | None]:  # fmt: skip
    semaphore = asyncio.Semaphore(concurrency)

    async def score(golden: GeneratorGolden, generated: dict) -> dict | None:
        if generated["answer"] is None:
            return None  # nothing to judge
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=generated["answer"],
            retrieval_context=list(golden.context),  # the same excerpts the generator saw
        )
        return await score_case(test_case, metric_factory(), semaphore)

    return await asyncio.gather(*(score(g, a) for g, a in zip(goldens, answers)))


def summarize(rows: list[dict]) -> dict:
    summary = {"cases": len(rows)}
    for metric in METRICS:
        values = [r[metric]["score"] for r in rows if r[metric]["score"] is not None]
        summary[metric] = mean(values) if values else None
        summary[f"{metric}_errors"] = sum(r[metric]["error"] is not None for r in rows)
    summary["generation_errors"] = sum(r["generation_error"] is not None for r in rows)
    abstain_rows = [r for r in rows if r["expected_behavior"] == "abstain"]
    summary["abstained"] = (
        sum(r["abstained"] for r in abstain_rows) / len(abstain_rows) if abstain_rows else None
    )
    summary["latency_ms"] = mean(r["latency_ms"] for r in rows) if rows else None
    return summary


def run(
    generator: AnswerGenerator,
    goldens: list[GeneratorGolden],
    metric_factory: Callable[[], dict],
    concurrency: int,
    judge_name: str,
) -> dict:
    answers = generate_answers(generator, goldens, concurrency)
    scores = asyncio.run(score_all(goldens, answers, metric_factory, concurrency))

    not_run = {"score": None, "reason": None, "error": "no answer was generated"}
    rows = []
    for golden, generated, scored in zip(goldens, answers, scores):
        scored = scored or {metric: not_run for metric in METRICS}
        rows.append({
            "id": golden.id,
            "category": golden.category,
            "stress": golden.stress,
            "expected_behavior": golden.expected_behavior,
            "status": golden.status,
            "question": golden.input,
            "answer": generated["answer"],
            "generation_error": generated["error"],
            "abstained": bool(generated["answer"] and ABSTAIN.search(generated["answer"])),
            "latency_ms": generated["latency_ms"],
            **scored,
        })  # fmt: skip

    categories = sorted({g.category for g in goldens})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "generator": getattr(generator, "model", settings.llm_model),
            "generator_effort": settings.llm_effort,
            "judge": judge_name,
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

"""Run each question through the real pipeline, then score the RAG triad with DeepEval."""

import asyncio
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from statistics import mean

from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from evaluation.generator_eval.runner import ABSTAIN
from evaluation.rag_eval.questions import Question
from evaluation.retrieval_eval.runner import score_case
from evaluation.runs import git_commit
from sec_rag.config import settings
from sec_rag.pipeline import NO_RESULTS, RAGPipeline

RUN_KIND = "rag"  # saved under evaluation/runs/rag/
METRICS = ("context_relevance", "groundedness", "answer_relevance")

# Diagnosis thresholds. Context relevance counts every statement in the excerpts, and 10-K
# chunks are long and only partly on-topic, so its scores run low; read them relatively.
LOW_CONTEXT, LOW_GROUNDED, LOW_RELEVANT = 0.3, 0.8, 0.7

# Rough check that an advice question got no buy/sell recommendation.
ADVICE = re.compile(
    r"\b(not|cannot|can't|won't|unable to|not able to)\b.{0,40}\b(investment|financial)\s+advice"
    r"|\b(cannot|can't|won't|unable to|not able to|don't)\b.{0,20}\b(tell you|say|advise|"
    r"recommend)\b.{0,20}\bwhether\b"
    r"|\b(buy|sell|hold)\b.{0,30}\b(recommendation|decision)",
    re.IGNORECASE,
)


def deepeval_metrics(judge: DeepEvalBaseLLM, include_reason: bool) -> dict:
    """Fresh metric objects for one test case (DeepEval metrics hold per-case state)."""
    common = {"model": judge, "include_reason": include_reason, "eval_mode": "llm"}
    return {
        "context_relevance": ContextualRelevancyMetric(**common),
        "groundedness": FaithfulnessMetric(**common),
        "answer_relevance": AnswerRelevancyMetric(**common),
    }


def run_pipeline(pipeline: RAGPipeline, questions: list[Question], k: int,
                 concurrency: int) -> list[dict]:  # fmt: skip
    """What RAGPipeline.ask() does for a standalone question: retrieve, then generate.

    Retrieval runs one question at a time (it shares the embedding model); generation, the
    slow API call, runs in parallel.
    """
    retrieved = []
    for question in questions:
        start = time.perf_counter()
        sources = pipeline.retriever.retrieve(question.input, k=k)
        retrieved.append((sources, (time.perf_counter() - start) * 1000))

    def generate(item: tuple[Question, list]) -> dict:
        question, sources = item
        if not sources:
            return {"answer": NO_RESULTS, "error": None}
        try:
            return {"answer": pipeline.generator.generate(question.input, sources), "error": None}
        except Exception as exc:  # API or model failure for this question only
            return {"answer": None, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        generated = list(pool.map(generate, [(q, s) for q, (s, _) in zip(questions, retrieved)]))

    return [
        {"sources": sources, "retrieval_ms": ms, **gen}
        for (sources, ms), gen in zip(retrieved, generated)
    ]


async def score_all(questions: list[Question], outputs: list[dict],
                    metric_factory: Callable[[], dict], concurrency: int) -> list:  # fmt: skip
    semaphore = asyncio.Semaphore(concurrency)

    async def score(question: Question, output: dict) -> dict | None:
        if output["answer"] is None or not output["sources"]:
            return None  # no answer to judge, or nothing was retrieved
        test_case = LLMTestCase(
            input=question.input,
            actual_output=output["answer"],
            retrieval_context=[rc.chunk.text for rc in output["sources"]],
        )
        return await score_case(test_case, metric_factory(), semaphore)

    return await asyncio.gather(*(score(q, o) for q, o in zip(questions, outputs)))


def diagnose(row: dict) -> str:
    """Which link of the triad failed, checked in pipeline order."""
    if row["generation_error"]:
        return "generation_error"
    if not row["sources"]:
        return "no_context"
    scores = {m: row[m]["score"] for m in METRICS}
    if any(s is None for s in scores.values()):
        return "judge_error"
    if scores["context_relevance"] < LOW_CONTEXT:
        return "retrieval"
    if scores["groundedness"] < LOW_GROUNDED:
        return "not_grounded"
    if scores["answer_relevance"] < LOW_RELEVANT:
        return "off_target"
    return "ok"


def handled(row: dict) -> bool | None:
    """Did an out-of-scope question get the expected kind of answer? None if not applicable."""
    answer = row["answer"] or ""
    if row["expected_behavior"] == "abstain":
        return answer == NO_RESULTS or bool(ABSTAIN.search(answer))
    if row["expected_behavior"] == "decline_advice":
        return bool(ADVICE.search(answer))
    return None


def summarize(rows: list[dict]) -> dict:
    summary = {"cases": len(rows)}
    for metric in METRICS:
        values = [r[metric]["score"] for r in rows if r[metric]["score"] is not None]
        summary[metric] = mean(values) if values else None
    summary["diagnosis"] = {d: sum(r["diagnosis"] == d for r in rows)
                            for d in sorted({r["diagnosis"] for r in rows})}  # fmt: skip
    out_of_scope = [r for r in rows if r["handled"] is not None]
    summary["handled"] = (
        sum(r["handled"] for r in out_of_scope) / len(out_of_scope) if out_of_scope else None
    )
    return summary


def run(pipeline: RAGPipeline, questions: list[Question], k: int,
        metric_factory: Callable[[], dict], concurrency: int, judge_name: str) -> dict:  # fmt: skip
    outputs = run_pipeline(pipeline, questions, k, concurrency)
    scores = asyncio.run(score_all(questions, outputs, metric_factory, concurrency))

    not_scored = {"score": None, "reason": None, "error": None}
    rows = []
    for question, output, scored in zip(questions, outputs, scores):
        row = {
            "id": question.id,
            "category": question.category,
            "expected_behavior": question.expected_behavior,
            "question": question.input,
            "answer": output["answer"],
            "generation_error": output["error"],
            "sources": [
                f"{rc.chunk.ticker} FY{rc.chunk.fiscal_year} Item {rc.chunk.item} "
                f"({rc.chunk.chunk_id})"
                for rc in output["sources"]
            ],
            "retrieval_ms": output["retrieval_ms"],
            **(scored or {metric: not_scored for metric in METRICS}),
        }
        row["diagnosis"] = diagnose(row)
        row["handled"] = handled(row)
        rows.append(row)

    categories = sorted({q.category for q in questions})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "k": k,
            "retrieval_mode": pipeline.retriever.mode,
            "generator": getattr(pipeline.generator, "model", settings.llm_model),
            "generator_effort": settings.llm_effort,
            "judge": judge_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "git_commit": git_commit(),
        },
        "summary": summarize(rows),
        "by_category": {c: summarize([r for r in rows if r["category"] == c]) for c in categories},
        "cases": rows,
    }

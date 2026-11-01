"""RAG pipeline eval: the full pipeline, scored with the RAG triad (DeepEval metrics).

Each question goes through the real pipeline (filters, retrieval, generation). A Claude judge
then scores the three links of the triad:
  context relevance  (question -> retrieved excerpts)   DeepEval ContextualRelevancyMetric
  faithfulness       (excerpts -> answer)               DeepEval FaithfulnessMetric
  answer relevance   (question -> answer)               DeepEval AnswerRelevancyMetric
All three are reference-free, so only questions are needed.

    python -m evals.eval_rag_pipeline run                  # all 40 questions
    python -m evals.eval_rag_pipeline run --category comparison
    python -m evals.eval_rag_pipeline compare              # last two saved runs
"""

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import typer
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from evals import runs
from evals.eval_retriever import load_goldens, score_case
from evals.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL
from evals.runs import git_commit
from sec_rag.config import settings
from sec_rag.pipeline import NO_RESULTS, RAGPipeline

# --- questions -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Question:
    id: str
    category: str
    input: str


def load_questions() -> list[Question]:
    return [Question(id=g.id, category=g.category, input=g.input) for g in load_goldens()]


# --- runner ----------------------------------------------------------------------------------

RUN_KIND = "rag"  # saved under evals/runs/rag/
METRICS = ("context_relevance", "faithfulness", "answer_relevance")

# Diagnosis thresholds. Context relevance counts every statement in the excerpts, and 10-K
# chunks are long and only partly on-topic, so its scores run low; read them relatively.
LOW_CONTEXT, LOW_FAITHFUL, LOW_RELEVANT = 0.3, 0.8, 0.7


def deepeval_metrics(judge: DeepEvalBaseLLM, include_reason: bool) -> dict:
    """Fresh metric objects for one test case (DeepEval metrics hold per-case state)."""
    common = {"model": judge, "include_reason": include_reason, "eval_mode": "llm"}
    return {
        "context_relevance": ContextualRelevancyMetric(**common),
        "faithfulness": FaithfulnessMetric(**common),
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
    if scores["faithfulness"] < LOW_FAITHFUL:
        return "not_faithful"
    if scores["answer_relevance"] < LOW_RELEVANT:
        return "off_target"
    return "ok"


def summarize(rows: list[dict]) -> dict:
    summary = {"cases": len(rows)}
    for metric in METRICS:
        values = [r[metric]["score"] for r in rows if r[metric]["score"] is not None]
        summary[metric] = mean(values) if values else None
    summary["diagnosis"] = {d: sum(r["diagnosis"] == d for r in rows)
                            for d in sorted({r["diagnosis"] for r in rows})}  # fmt: skip
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


# --- cli -------------------------------------------------------------------------------------

app = typer.Typer(help=__doc__, no_args_is_help=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Differences smaller than this between runs are likely judge noise, not real changes.
NOISE = 0.03
SHORT = {"context_relevance": "context", "faithfulness": "faithful", "answer_relevance": "answer"}
EXPLAIN = {
    "retrieval": "retrieved excerpts are mostly off-topic",
    "not_faithful": "answer makes claims the excerpts don't support",
    "off_target": "answer doesn't address the question well",
    "no_context": "nothing was retrieved",
    "generation_error": "the generator failed",
    "judge_error": "a judge call failed",
}


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_report(result: dict) -> None:
    s = result["settings"]
    typer.echo(f"Pipeline: {s['retrieval_mode']} retrieval, k={s['k']}, generator {s['generator']}"
               f" | judge {s['judge']}")  # fmt: skip

    header = " ".join(f"{SHORT[m]:>9}" for m in METRICS)
    typer.echo(f"\n{'category':>13} {'n':>3} {header}")
    for name, m in [*result["by_category"].items(), ("ALL", result["summary"])]:
        cells = " ".join(f"{_fmt(m[metric]):>9}" for metric in METRICS)
        typer.echo(f"{name:>13} {m['cases']:>3} {cells}")

    diagnosis = result["summary"]["diagnosis"]
    typer.echo("\nDiagnosis (first failing link of the triad):")
    for name, count in sorted(diagnosis.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {count:>3}  {name:<16} {EXPLAIN.get(name, '')}")

    rows = result["cases"]
    for row in rows:
        if row["diagnosis"] in ("ok", "no_context"):
            continue
        scores = ", ".join(f"{SHORT[m]} {_fmt(row[m]['score'], 2)}" for m in METRICS)
        typer.echo(f"\n  [{row['id']}] {row['diagnosis']}: {scores}")
        metric = {"retrieval": "context_relevance", "not_faithful": "faithfulness",
                  "off_target": "answer_relevance"}.get(row["diagnosis"])  # fmt: skip
        if metric and row[metric]["reason"]:
            typer.echo(f"     {row[metric]['reason'][:240]}")


@app.command("run")
def run_command(
    k: int = typer.Option(8, help="Chunks retrieved per question"),
    limit: int = typer.Option(0, help="Only the first N questions (0 = all)"),
    category: list[str] = typer.Option(None, help="Only these categories"),
    id: list[str] = typer.Option(None, help="Only these question ids"),
    judge_model: str = typer.Option(
        DEFAULT_JUDGE_MODEL, help="Claude model used as judge (default: SEC_RAG_JUDGE_MODEL)"
    ),
    effort: str = typer.Option(
        DEFAULT_JUDGE_EFFORT, help="Judge effort: low, medium, high (default: SEC_RAG_JUDGE_EFFORT)"
    ),
    reasons: bool = typer.Option(True, help="Ask the judge for a reason per score (costs more)"),
    concurrency: int = typer.Option(4, help="Answers generated / judged in parallel"),
    save: bool = typer.Option(True, help="Save the run for `compare`"),
):
    """Run the real pipeline and score context relevance, faithfulness and answer relevance."""
    from evals.judge import ClaudeJudge
    from sec_rag.pipeline import RAGPipeline

    questions = [
        q for q in load_questions()
        if (not category or q.category in category) and (not id or q.id in id)
    ]  # fmt: skip
    if limit:
        questions = questions[:limit]
    if not questions:
        raise typer.Exit("No questions match the filters.")

    pipeline = RAGPipeline().warm_up()
    judge = ClaudeJudge(judge_model, effort=effort)
    result = run(
        pipeline,
        questions,
        k,
        metric_factory=lambda: deepeval_metrics(judge, include_reason=reasons),
        concurrency=concurrency,
        judge_name=judge_model,
    )
    print_report(result)
    typer.echo(f"\n{judge.cost_summary()} (generation by {result['settings']['generator']} "
               "is not included)")  # fmt: skip
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

    typer.echo(f"\n{'category':>13} " + " ".join(f"{SHORT[m]:>17}" for m in METRICS))
    for name in [*b["by_category"], "ALL"]:
        new = b["summary"] if name == "ALL" else b["by_category"][name]
        old = a["summary"] if name == "ALL" else a["by_category"].get(name)
        if old is None:
            continue
        cells = []
        for metric in METRICS:
            o, n = old.get(metric), new.get(metric)  # runs saved before the rename lack it
            cells.append(f"{_fmt(n)}{'' if o is None or n is None else f' ({n - o:+.3f})'}")
        typer.echo(f"{name:>13} " + " ".join(f"{c:>17}" for c in cells))
    typer.echo(f"(changes under ±{NOISE} are likely judge noise)")
    typer.echo(f"\ndiagnosis before: {a['summary']['diagnosis']}")
    typer.echo(f"diagnosis after:  {b['summary']['diagnosis']}")


if __name__ == "__main__":
    app()

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
import threading
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
from evals.eval_retriever import load_goldens, rerank_cost, rerank_label
from evals.judge import (
    DEFAULT_JUDGE_EFFORT,
    DEFAULT_JUDGE_MODEL,
    metric_details,
    score_case,
    token_cost,
)
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

RUN_KIND = "rag_pipeline"  # saved under evals/runs/rag_pipeline/
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


Progress = Callable[[str, int, int], None]  # (stage, done, total)


def run_pipeline(pipeline: RAGPipeline, questions: list[Question], k: int,
                 concurrency: int, progress: Progress | None = None) -> list[dict]:  # fmt: skip
    """What RAGPipeline.ask() does for a standalone question: retrieve, then generate.

    Retrieval runs one question at a time (it shares the embedding model); generation, the
    slow API call, runs in parallel.
    """
    report = progress or (lambda *_: None)
    retrieved = []
    report("retrieved", 0, len(questions))
    for done, question in enumerate(questions, start=1):
        start = time.perf_counter()
        sources = pipeline.retriever.retrieve(question.input, k=k)
        retrieved.append((sources, (time.perf_counter() - start) * 1000))
        report("retrieved", done, len(questions))

    generated_count, lock = 0, threading.Lock()
    report("generated", 0, len(questions))

    def generate(item: tuple[Question, list]) -> dict:
        nonlocal generated_count
        question, sources = item
        start = time.perf_counter()
        if not sources:
            result = {"answer": NO_RESULTS, "error": None}
        else:
            try:
                answer = pipeline.generator.generate(question.input, sources)
                result = {"answer": answer, "error": None}
            except Exception as exc:  # API or model failure for this question only
                result = {"answer": None, "error": str(exc)}
        result["generation_ms"] = (time.perf_counter() - start) * 1000
        with lock:
            generated_count += 1
            report("generated", generated_count, len(questions))
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        generated = list(pool.map(generate, [(q, s) for q, (s, _) in zip(questions, retrieved)]))

    return [
        {"sources": sources, "retrieval_ms": ms, **gen}
        for (sources, ms), gen in zip(retrieved, generated)
    ]


def _judged(output: dict) -> bool:
    return output["answer"] is not None and bool(output["sources"])


async def score_all(questions: list[Question], outputs: list[dict],
                    metric_factory: Callable[[], dict], concurrency: int,
                    progress: Progress | None = None) -> list:  # fmt: skip
    semaphore = asyncio.Semaphore(concurrency)
    total, done = sum(_judged(o) for o in outputs), 0
    if progress:
        progress("judged", 0, total)

    async def score(question: Question, output: dict) -> dict | None:
        nonlocal done
        if not _judged(output):
            return None  # no answer to judge, or nothing was retrieved
        test_case = LLMTestCase(
            input=question.input,
            actual_output=output["answer"],
            retrieval_context=[rc.chunk.text for rc in output["sources"]],
        )
        scores = await score_case(test_case, metric_factory(), semaphore, details=metric_details)
        done += 1
        if progress:
            progress("judged", done, total)
        return scores

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
    summary["judge_errors"] = sum(any(r[m]["error"] for m in METRICS) for r in rows)
    for stage in ("retrieval_ms", "generation_ms"):
        values = [r[stage] for r in rows if r.get(stage) is not None]
        summary[stage] = mean(values) if values else None
    return summary


def run(pipeline: RAGPipeline, questions: list[Question], k: int,
        metric_factory: Callable[[], dict], concurrency: int, judge_name: str,
        progress: Progress | None = None, judge_effort: str = DEFAULT_JUDGE_EFFORT) -> dict:  # fmt: skip
    """progress(stage, done, total) is called as questions are retrieved, answered, judged."""
    outputs = run_pipeline(pipeline, questions, k, concurrency, progress)
    scores = asyncio.run(score_all(questions, outputs, metric_factory, concurrency, progress))

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
            "generation_ms": output.get("generation_ms"),
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
            # What shapes retrieval besides the mode; all come from .env, not the code.
            "hybrid_candidates": settings.hybrid_candidates,
            "rrf_k": settings.rrf_k,
            "rerank": rerank_label() if getattr(pipeline.retriever, "rerank", False) else "off",
            "min_chunks_per_entity": settings.min_chunks_per_entity,
            "max_comparison_chunks": settings.max_comparison_chunks,
            "generator": getattr(pipeline.generator, "model", settings.llm_model),
            "generator_effort": settings.llm_effort,
            "generator_max_tokens": settings.llm_max_tokens,
            "judge": judge_name,
            "judge_effort": judge_effort,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "git_commit": git_commit(),
        },
        "questions": {"total": len(questions)},
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


TRIAD = """\
- **context**: context relevance - share of the retrieved excerpts' statements that are \
relevant to the question. 10-K chunks are long and only partly on-topic, so this runs low; \
read it relatively (between runs), not as a pass mark.
- **faithful**: faithfulness - share of the answer's claims supported by the excerpts \
(1.0 = nothing made up)
- **answer**: answer relevance - share of the answer's statements that address the question"""

FAILED_METRIC = {"retrieval": "context_relevance", "not_faithful": "faithfulness",
                 "off_target": "answer_relevance"}  # fmt: skip


def _summary_row(name: str, m: dict) -> str:
    cells = " | ".join(_fmt(m[metric]) for metric in METRICS)
    ok = m["diagnosis"].get("ok", 0)
    return (f"| {name} | {m['cases']} | {cells} | {ok} / {m['cases']} "
            f"| {_fmt(m['retrieval_ms'], 0)} | {_fmt(m['generation_ms'], 0)} |")  # fmt: skip


def markdown_report(result: dict, cost: str | None = None) -> str:
    """The run as a Markdown report: settings, triad scores overall and by category, the
    diagnosis, every failing case with what the judge rejected, and all cases."""
    md = runs.md_cell
    s, rows = result["settings"], result["cases"]
    lines = [f"# RAG pipeline eval: {result['timestamp']}", "", "| setting | value |", "|---|---|"]
    lines += [f"| {key} | {md(value)} |" for key, value in s.items()]
    lines += [f"| questions | {result['questions']['total']} |"]
    if cost:
        lines += [f"| cost | {md(cost)} |"]

    header = [("| category | cases | context | faithful | answer | ok | retrieval ms "
               "| generation ms |"), "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    lines += ["", "## Summary", "", TRIAD, "", *header, _summary_row("ALL", result["summary"])]
    lines += ["", "## By category", "", *header]
    lines += [_summary_row(name, m) for name, m in result["by_category"].items()]

    lines += ["", "## Diagnosis", "",
              ("The first link of the triad that failed, checked in pipeline order: retrieval "
               f"(context < {LOW_CONTEXT}), then faithfulness (< {LOW_FAITHFUL}), then answer "
               f"relevance (< {LOW_RELEVANT})."), "",
              "| diagnosis | cases | meaning |", "|---|---|---|"]  # fmt: skip
    for name, count in sorted(result["summary"]["diagnosis"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {count} | {EXPLAIN.get(name, 'all three links passed')} |")

    failing = [r for r in rows if r["diagnosis"] != "ok"]
    lines += ["", "## Failing cases", ""]
    if not failing:
        lines.append("None.")
    for r in failing:
        scores = ", ".join(f"**{SHORT[m]}** {_fmt(r[m]['score'], 2)}" for m in METRICS)
        lines += [f"### {r['id']} ({r['category']}): {r['diagnosis']}", "",
                  f"- **question:** {md(r['question'])}", f"- {scores}",
                  f"- **retrieved:** {md(', '.join(r['sources'])) or 'nothing'}"]  # fmt: skip
        if r["generation_error"]:
            lines.append(f"- **generation error:** {md(r['generation_error'])}")
        for metric in METRICS:
            if r[metric]["error"]:
                lines.append(f"- **{metric} judge error:** {md(r[metric]['error'])}")
        failed = FAILED_METRIC.get(r["diagnosis"])
        if failed == "context_relevance" and r[failed]["reason"]:
            lines += ["", f"context_relevance: {md(r[failed]['reason'])}"]
        for metric in ("faithfulness", "answer_relevance"):
            if r[metric]["score"] is not None and (r[metric]["score"] < 1 or metric == failed):
                lines += runs.rejected_items_md(metric, r[metric])
        if r["answer"]:
            lines += ["", "<details><summary>Answer</summary>", "", r["answer"], "",
                      "</details>", ""]  # fmt: skip

    lines += ["", "## All cases", "",
              "| id | category | context | faithful | answer | diagnosis | claims rejected |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        scores = " | ".join("error" if r[m]["error"] else _fmt(r[m]["score"], 2) for m in METRICS)
        details = r["faithfulness"].get("details")
        rejected = f"{len(details['unsupported'])} / {details['checked']}" if details else "-"
        lines.append(f"| {r['id']} | {r['category']} | {scores} | {r['diagnosis']} | {rejected} |")
    return "\n".join(lines) + "\n"


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

    typer.echo("Loading the pipeline...", err=True)
    pipeline = RAGPipeline().warm_up()
    judge = ClaudeJudge(judge_model, effort=effort)
    retriever, generator = pipeline.retriever, pipeline.generator
    rerank = "rerank on" if retriever.rerank else "rerank off"
    typer.echo(f"{len(questions)} questions | {retriever.mode} retrieval, {rerank}, k={k} "
               f"| generator {generator.model} ({settings.llm_effort}) "
               f"| judge {judge_model} ({effort})", err=True)  # fmt: skip
    started = time.perf_counter()

    def progress(stage: str, done: int, total: int) -> None:
        # One line per stage, rewritten in place; a newline once the stage is finished.
        elapsed = time.perf_counter() - started
        typer.echo(f"\r  {done}/{total} {stage}  ({elapsed:.0f}s)", nl=False, err=True)
        if done == total:
            typer.echo("", err=True)

    result = run(
        pipeline,
        questions,
        k,
        metric_factory=lambda: deepeval_metrics(judge, include_reason=reasons),
        concurrency=concurrency,
        judge_name=judge_model,
        progress=progress,
        judge_effort=effort,
    )
    typer.echo("", err=True)
    print_report(result)
    cost = "; ".join(c for c in [
        token_cost("generator", generator.model, generator.input_tokens, generator.output_tokens),
        rerank_cost({retriever.mode: retriever}),
        judge.cost_summary(),
    ] if c)  # fmt: skip
    typer.echo(f"\n{cost}")
    if save:
        path = runs.save(result, RUN_KIND)
        report = runs.save_report(markdown_report(result, cost), path)
        typer.echo(f"Saved {path}\nReport {report}")


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

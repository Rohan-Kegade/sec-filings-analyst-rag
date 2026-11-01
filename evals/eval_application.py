"""Correctness and completeness evals (application quality), with DeepEval G-Eval.

Each golden (goldens/application_goldens.json) is a question and a reference answer that states
only what the question asks for. The question goes through the real pipeline once, and every
selected metric scores the same answers, so results are comparable and generation is paid for
once. Both judges compare the answer with the reference answer:
  correctness   is what the answer states accurate? (missing facts are completeness's job)
  completeness  does the answer cover everything in the reference? (accuracy is correctness's)

    python -m evals.eval_application run   # both metrics, 40 goldens
    python -m evals.eval_application run --metric completeness --category qualitative
    python -m evals.eval_application compare
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import typer
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from evals import GOLDENS_DIR, runs
from evals.eval_rag_pipeline import run_pipeline
from evals.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL, score_case
from evals.runs import git_commit
from sec_rag.config import settings
from sec_rag.pipeline import RAGPipeline

# --- goldens ---------------------------------------------------------------------------------

APPLICATION_GOLDENS = GOLDENS_DIR / "application_goldens.json"


@dataclass(frozen=True)
class AppGolden:
    id: str
    category: str
    input: str
    expected_output: str  # the reference answer: only what the question asks for


def load_goldens(path: Path = APPLICATION_GOLDENS) -> list[AppGolden]:
    goldens = [
        AppGolden(id=raw["id"], category=raw["category"], input=raw["input"],
                  expected_output=raw["expected_output"])
        for raw in json.loads(path.read_text(encoding="utf-8"))
    ]  # fmt: skip
    ids = [g.id for g in goldens]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate golden ids in {path.name}: {duplicates}")
    return goldens


# --- correctness -----------------------------------------------------------------------------

CORRECTNESS_STEPS = [
    "Read the question and the expected output: a reference answer stating what the company's "
    "10-K filing says in answer to this question.",
    "Check each fact the actual output states that the expected output also covers (figures, "
    "dates, fiscal years, companies, names, events). Equivalent wording, rounding and units "
    "are fine (for example 'about 228,000' matches 'approximately 228,000').",
    "Heavily penalize any statement that contradicts the expected output, or that gives a "
    "wrong number, date, fiscal year or company.",
    "Do not penalize facts from the expected output that the actual output leaves out; "
    "completeness is evaluated separately. Judge only whether what the actual output states "
    "is accurate.",
    "Do not penalize additional details that the expected output does not mention, unless "
    "they contradict it.",
    "If the actual output states none of the facts in the expected output (for example it "
    "says the information is not available), give the lowest score.",
]


def correctness_metric(judge: DeepEvalBaseLLM) -> GEval:
    return GEval(
        name="Correctness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=CORRECTNESS_STEPS,
        model=judge,
    )


# --- completeness ----------------------------------------------------------------------------

COMPLETENESS_STEPS = [
    "Read the question and the expected output: a reference answer containing everything a "
    "complete answer to this question must cover.",
    "Break the expected output into the distinct points it makes. For each point, decide "
    "whether the actual output covers it fully, partly or not at all. Wording does not "
    "matter, only whether the point is addressed.",
    "The score reflects the share of points covered; partly covered points count for less "
    "than fully covered ones.",
    "Do not judge whether figures or claims are accurate; correctness is evaluated "
    "separately. A point counts as covered if the answer addresses it.",
    "Do not penalize additional details beyond the expected output.",
    "If the answer says the information for a point is not available instead of covering "
    "it, count that point as not covered.",
]


def completeness_metric(judge: DeepEvalBaseLLM) -> GEval:
    return GEval(
        name="Completeness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=COMPLETENESS_STEPS,
        model=judge,
    )


# --- runner ----------------------------------------------------------------------------------

RUN_KIND = "application"  # saved under evals/runs/application/
PASS = 0.7  # a score at or above this counts as a pass


# Metric name -> builds its judge. Every metric sees the same test case: the question, the
# pipeline's answer and the golden's reference answer.
MetricBuilder = Callable[[DeepEvalBaseLLM], object]
METRICS: dict[str, MetricBuilder] = {
    "correctness": correctness_metric,
    "completeness": completeness_metric,
}


async def score_all(goldens: list[AppGolden], outputs: list[dict],
                    metrics: dict[str, MetricBuilder], judge: DeepEvalBaseLLM,
                    concurrency: int) -> list[dict]:  # fmt: skip
    semaphore = asyncio.Semaphore(concurrency)

    async def score(golden: AppGolden, output: dict, name: str) -> dict:
        if output["answer"] is None:
            return {"score": None, "reason": None, "error": output["error"] or "no answer"}
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=output["answer"],
            expected_output=golden.expected_output,
        )
        scored = await score_case(test_case, {name: metrics[name](judge)}, semaphore)
        return scored[name]

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
            "errors": sum(r[name]["error"] is not None for r in rows),
        }
    return summary


def run(pipeline: RAGPipeline, goldens: list[AppGolden], k: int,
        metrics: dict[str, MetricBuilder], judge: DeepEvalBaseLLM, concurrency: int,
        judge_name: str) -> dict:  # fmt: skip
    """`metrics` maps names to judge builders, normally a selection from METRICS."""
    metric_names = list(metrics)
    outputs = run_pipeline(pipeline, goldens, k, concurrency)
    scores = asyncio.run(score_all(goldens, outputs, metrics, judge, concurrency))

    rows = []
    for golden, output, scored in zip(goldens, outputs, scores):
        rows.append({
            "id": golden.id,
            "category": golden.category,
            "question": golden.input,
            "reference": golden.expected_output,
            "answer": output["answer"],
            "generation_error": output["error"],
            "sources": [f"{rc.chunk.ticker} FY{rc.chunk.fiscal_year} Item {rc.chunk.item}"
                        for rc in output["sources"]],
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
        "goldens": {"total": len(goldens)},
        "summary": summarize(rows, metric_names),
        "by_category": {
            c: summarize([r for r in rows if r["category"] == c], metric_names)
            for c in categories
        },
        "cases": rows,
    }


# --- cli -------------------------------------------------------------------------------------

app = typer.Typer(help=__doc__, no_args_is_help=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Differences smaller than this between runs are likely judge noise, not real changes.
NOISE = 0.03


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def print_report(result: dict) -> None:
    s, g, names = result["settings"], result["goldens"], result["metrics"]
    typer.echo(
        f"Pipeline: {s['retrieval_mode']} retrieval, k={s['k']}, generator {s['generator']} | "
        f"judge {s['judge']} | goldens {g['total']}"
    )

    header = " ".join(f"{name[:12]:>12} {'pass':>5}" for name in names)
    typer.echo(f"\n{'category':>13} {'n':>3} {header}")
    for category, m in [*result["by_category"].items(), ("ALL", result["summary"])]:
        cells = " ".join(f"{_fmt(m[n]['score']):>12} {_pct(m[n]['passed']):>5}" for n in names)
        typer.echo(f"{category:>13} {m['cases']:>3} {cells}")
    typer.echo(f"(pass = score >= {PASS})")

    errors = sum(result["summary"][n]["errors"] for n in names)
    if errors:
        typer.echo(f"\n{errors} scores could not be computed; see the saved run.")

    for name in names:
        weak = [r for r in result["cases"]
                if r[name]["score"] is not None and r[name]["score"] < PASS]  # fmt: skip
        if not weak:
            continue
        typer.echo(f"\n{name}: answers below {PASS}")
        for r in sorted(weak, key=lambda r: r[name]["score"]):
            typer.echo(f"  [{r['id']}] {_fmt(r[name]['score'], 2)}: "
                       f"{(r[name]['reason'] or '')[:240]}")  # fmt: skip


@app.command("run")
def run_command(
    metric: list[str] = typer.Option(
        list(METRICS), help=f"Metrics to score: {', '.join(METRICS)}"
    ),
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
    concurrency: int = typer.Option(4, help="Answers generated / judged in parallel"),
    save: bool = typer.Option(True, help="Save the run for `compare`"),
):
    """Run the real pipeline once and score the selected metrics on its answers."""
    from evals.judge import ClaudeJudge
    from sec_rag.pipeline import RAGPipeline

    unknown = [m for m in metric if m not in METRICS]
    if unknown:
        raise typer.BadParameter(f"unknown metric(s) {unknown}; choose from {list(METRICS)}")
    goldens = [
        gold for gold in load_goldens()
        if (not category or gold.category in category) and (not id or gold.id in id)
    ]  # fmt: skip
    if limit:
        goldens = goldens[:limit]
    if not goldens:
        raise typer.Exit("No goldens match the filters.")

    pipeline = RAGPipeline().warm_up()
    judge = ClaudeJudge(judge_model, effort=effort)
    result = run(
        pipeline,
        goldens,
        k,
        metrics={name: METRICS[name] for name in metric},
        judge=judge,
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

    names = [n for n in b["metrics"] if n in a["metrics"]]
    typer.echo(f"\n{'category':>13} " + " ".join(f"{n:>19}" for n in names))
    for category in [*b["by_category"], "ALL"]:
        new = b["summary"] if category == "ALL" else b["by_category"][category]
        old = a["summary"] if category == "ALL" else a["by_category"].get(category)
        if old is None:
            continue
        cells = []
        for n in names:
            o, v = old[n]["score"], new[n]["score"]
            cells.append(f"{_fmt(v)}{'' if o is None or v is None else f' ({v - o:+.3f})'}")
        typer.echo(f"{category:>13} " + " ".join(f"{c:>19}" for c in cells))
    typer.echo(f"(changes under ±{NOISE} are likely judge noise)")

    old_rows = {r["id"]: r for r in a["cases"]}
    for n in names:
        moves = []
        for r in b["cases"]:
            o = old_rows.get(r["id"])
            if o and o[n]["score"] is not None and r[n]["score"] is not None:
                delta = r[n]["score"] - o[n]["score"]
                if abs(delta) >= 0.1:
                    moves.append((delta, r["id"]))
        if moves:
            typer.echo(f"\n{n} changes of 0.1 or more:")
            for delta, case_id in sorted(moves):
                typer.echo(f"  {delta:+.2f}  {case_id}")


if __name__ == "__main__":
    app()

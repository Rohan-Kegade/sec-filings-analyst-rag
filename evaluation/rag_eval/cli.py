"""Commands: run, compare.

    python -m evaluation.rag_eval run                        # all 40 questions
    python -m evaluation.rag_eval run --category comparison
    python -m evaluation.rag_eval compare                    # last two saved runs
"""

import logging
from pathlib import Path

import typer

from evaluation import runs
from evaluation.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL
from evaluation.rag_eval import runner
from evaluation.rag_eval.questions import load_questions

app = typer.Typer(help=__doc__, no_args_is_help=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Differences smaller than this between runs are likely judge noise, not real changes.
NOISE = 0.03
SHORT = {"context_relevance": "context", "groundedness": "grounded", "answer_relevance": "answer"}
EXPLAIN = {
    "retrieval": "retrieved excerpts are mostly off-topic",
    "not_grounded": "answer makes claims the excerpts don't support",
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

    header = " ".join(f"{SHORT[m]:>9}" for m in runner.METRICS)
    typer.echo(f"\n{'category':>13} {'n':>3} {header}")
    for name, m in [*result["by_category"].items(), ("ALL", result["summary"])]:
        cells = " ".join(f"{_fmt(m[metric]):>9}" for metric in runner.METRICS)
        typer.echo(f"{name:>13} {m['cases']:>3} {cells}")

    diagnosis = result["summary"]["diagnosis"]
    typer.echo("\nDiagnosis (first failing link of the triad):")
    for name, count in sorted(diagnosis.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {count:>3}  {name:<16} {EXPLAIN.get(name, '')}")

    rows = result["cases"]
    for row in rows:
        if row["diagnosis"] in ("ok", "no_context"):
            continue
        scores = ", ".join(f"{SHORT[m]} {_fmt(row[m]['score'], 2)}" for m in runner.METRICS)
        typer.echo(f"\n  [{row['id']}] {row['diagnosis']}: {scores}")
        metric = {"retrieval": "context_relevance", "not_grounded": "groundedness",
                  "off_target": "answer_relevance"}.get(row["diagnosis"])  # fmt: skip
        if metric and row[metric]["reason"]:
            typer.echo(f"     {row[metric]['reason'][:240]}")


@app.command()
def run(
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
    """Run the real pipeline and score context relevance, groundedness and answer relevance."""
    from evaluation.judge import ClaudeJudge
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
    result = runner.run(
        pipeline,
        questions,
        k,
        metric_factory=lambda: runner.deepeval_metrics(judge, include_reason=reasons),
        concurrency=concurrency,
        judge_name=judge_model,
    )
    print_report(result)
    typer.echo(f"\n{judge.cost_summary()} (generation by {result['settings']['generator']} "
               "is not included)")  # fmt: skip
    if save:
        typer.echo(f"Saved {runs.save(result, runner.RUN_KIND)}")


@app.command()
def compare(before: Path = typer.Argument(None), after: Path = typer.Argument(None)):
    """Compare two saved runs (default: the last two)."""
    if before is None or after is None:
        saved = runs.list_runs(runner.RUN_KIND)
        if len(saved) < 2:
            raise typer.Exit("Need at least two saved runs to compare.")
        before, after = saved[-2], saved[-1]
    a, b = runs.load_run(before), runs.load_run(after)
    typer.echo(f"before: {before.name}\nafter:  {after.name}")
    for key, value in b["settings"].items():
        if a["settings"].get(key) != value:
            typer.echo(f"  setting {key}: {a['settings'].get(key)} -> {value}")

    typer.echo(f"\n{'category':>13} " + " ".join(f"{SHORT[m]:>17}" for m in runner.METRICS))
    for name in [*b["by_category"], "ALL"]:
        new = b["summary"] if name == "ALL" else b["by_category"][name]
        old = a["summary"] if name == "ALL" else a["by_category"].get(name)
        if old is None:
            continue
        cells = []
        for metric in runner.METRICS:
            o, n = old[metric], new[metric]
            cells.append(f"{_fmt(n)}{'' if o is None or n is None else f' ({n - o:+.3f})'}")
        typer.echo(f"{name:>13} " + " ".join(f"{c:>17}" for c in cells))
    typer.echo(f"(changes under ±{NOISE} are likely judge noise)")
    typer.echo(f"\ndiagnosis before: {a['summary']['diagnosis']}")
    typer.echo(f"diagnosis after:  {b['summary']['diagnosis']}")


if __name__ == "__main__":
    app()

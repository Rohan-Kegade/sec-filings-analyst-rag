"""Commands: run, compare.

    python -m evaluation.generator_eval run                       # all 54 goldens
    python -m evaluation.generator_eval run --category partial --limit 3
    python -m evaluation.generator_eval compare                   # last two saved runs
"""

import logging
from pathlib import Path

import typer

from evaluation import runs
from evaluation.generator_eval import runner
from evaluation.generator_eval.goldens import load_goldens
from evaluation.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL

app = typer.Typer(help=__doc__, no_args_is_help=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Differences smaller than this between runs are likely judge noise, not real changes.
NOISE = 0.03
WEAK_FAITHFULNESS, WEAK_RELEVANCY = 0.8, 0.7


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def print_report(result: dict) -> None:
    s, g = result["settings"], result["goldens"]
    typer.echo(f"Goldens: {g['total']} ({g['approved']} approved) | generator {s['generator']} "
               f"| judge {s['judge']}")  # fmt: skip

    typer.echo(f"\n{'category':>12} {'n':>3} {'faithful':>9} {'relevant':>9} {'abstained':>10}")
    for name, m in [*result["by_category"].items(), ("ALL", result["summary"])]:
        typer.echo(
            f"{name:>12} {m['cases']:>3} {_fmt(m['faithfulness']):>9} "
            f"{_fmt(m['relevancy']):>9} {_pct(m['abstained']):>10}"
        )
    total = result["summary"]
    errors = total["faithfulness_errors"] + total["relevancy_errors"] + total["generation_errors"]
    if errors:
        typer.echo(f"\n{errors} errors (generation or judge); see the saved run for details.")

    def below(row: dict, metric: str, limit: float) -> bool:
        return row[metric]["score"] is not None and row[metric]["score"] < limit

    rows = result["cases"]
    weak = [
        r for r in rows
        if below(r, "faithfulness", WEAK_FAITHFULNESS) or below(r, "relevancy", WEAK_RELEVANCY)
    ]  # fmt: skip
    if weak:
        typer.echo(f"\nWeak answers (faithfulness < {WEAK_FAITHFULNESS} or "
                   f"relevancy < {WEAK_RELEVANCY}):")  # fmt: skip
        for r in weak:
            typer.echo(f"  [{r['id']}] faithful {_fmt(r['faithfulness']['score'], 2)}, "
                       f"relevant {_fmt(r['relevancy']['score'], 2)}")  # fmt: skip
            for metric in runner.METRICS:
                if r[metric]["reason"] and (r[metric]["score"] or 0) < 1:
                    typer.echo(f"     {metric}: {r[metric]['reason'][:220]}")

    abstain = [r for r in rows if r["expected_behavior"] == "abstain"]
    if abstain:
        typer.echo("\nNo-answer cases (should say the excerpts don't contain it):")
        for r in abstain:
            mark = "ok  " if r["abstained"] else "MISS"
            text = (r["answer"] or r["generation_error"] or "")[:160]
            typer.echo(f"  {mark} [{r['id']}] {text!r}")


@app.command()
def run(
    limit: int = typer.Option(0, help="Only the first N goldens (0 = all)"),
    category: list[str] = typer.Option(None, help="Only these categories"),
    id: list[str] = typer.Option(None, help="Only these golden ids"),
    approved_only: bool = typer.Option(False, help="Only goldens marked approved"),
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
    """Generate answers (SEC_RAG_LLM_MODEL) and score Faithfulness and Answer Relevancy."""
    from evaluation.judge import ClaudeJudge
    from sec_rag.generation.llm import AnswerGenerator

    goldens = [
        gold for gold in load_goldens()
        if (not category or gold.category in category)
        and (not id or gold.id in id)
        and (not approved_only or gold.status == "approved")
    ]  # fmt: skip
    if limit:
        goldens = goldens[:limit]
    if not goldens:
        raise typer.Exit("No goldens match the filters.")

    judge = ClaudeJudge(judge_model, effort=effort)
    result = runner.run(
        AnswerGenerator(),
        goldens,
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

    typer.echo(f"\n{'category':>12} {'faithfulness':>18} {'relevancy':>18}")
    for name in [*b["by_category"], "ALL"]:
        new = b["summary"] if name == "ALL" else b["by_category"][name]
        old = a["summary"] if name == "ALL" else a["by_category"].get(name)
        if old is None:
            continue
        cells = []
        for metric in runner.METRICS:
            o, n = old[metric], new[metric]
            cells.append(f"{_fmt(n)}{'' if o is None or n is None else f' ({n - o:+.3f})'}")
        typer.echo(f"{name:>12} " + " ".join(f"{c:>18}" for c in cells))
    typer.echo(f"(changes under ±{NOISE} are likely judge noise)")

    old_rows = {r["id"]: r for r in a["cases"]}
    for metric in runner.METRICS:
        moves = []
        for r in b["cases"]:
            o = old_rows.get(r["id"])
            if o and o[metric]["score"] is not None and r[metric]["score"] is not None:
                delta = r[metric]["score"] - o[metric]["score"]
                if abs(delta) >= 0.1:
                    moves.append((delta, r["id"]))
        if moves:
            typer.echo(f"\n{metric} changes of 0.1 or more:")
            for delta, case_id in sorted(moves):
                typer.echo(f"  {delta:+.2f}  {case_id}")


if __name__ == "__main__":
    app()

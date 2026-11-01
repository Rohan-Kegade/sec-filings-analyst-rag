"""Commands: correctness, compare-correctness.

    python -m evaluation.app_eval correctness                     # all 45 goldens
    python -m evaluation.app_eval correctness --category fact --limit 3
    python -m evaluation.app_eval compare-correctness             # last two saved runs
"""

import logging
from pathlib import Path

import typer

from evaluation import runs
from evaluation.app_eval import runner
from evaluation.app_eval.goldens import load_goldens
from evaluation.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL

app = typer.Typer(help=__doc__, no_args_is_help=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Differences smaller than this between runs are likely judge noise, not real changes.
NOISE = 0.03


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def print_report(result: dict) -> None:
    s, g = result["settings"], result["goldens"]
    typer.echo(
        f"Pipeline: {s['retrieval_mode']} retrieval, k={s['k']}, generator {s['generator']} | "
        f"judge {s['judge']} | goldens {g['total']} ({g['approved']} approved)"
    )

    typer.echo(f"\n{'category':>13} {'n':>3} {'score':>7} {f'correct (>={runner.PASS})':>17} "
               f"{'key numbers':>12}")  # fmt: skip
    for name, m in [*result["by_category"].items(), ("ALL", result["summary"])]:
        typer.echo(f"{name:>13} {m['cases']:>3} {_fmt(m['correctness']):>7} "
                   f"{_pct(m['correct']):>17} {_pct(m['key_numbers_found']):>12}")  # fmt: skip
    if result["summary"]["errors"]:
        typer.echo(f"\n{result['summary']['errors']} cases could not be scored; see the saved run.")

    scored = [r for r in result["cases"] if r["correctness"]["score"] is not None]
    wrong = [r for r in scored if r["correctness"]["score"] < runner.PASS]
    if wrong:
        typer.echo(f"\nIncorrect answers (score < {runner.PASS}):")
        for r in sorted(wrong, key=lambda r: r["correctness"]["score"]):
            numbers = (f", key numbers {r['key_numbers_found']}/{r['key_numbers_total']}"
                       if r["key_numbers_total"] else "")  # fmt: skip
            typer.echo(f"\n  [{r['id']}] score {_fmt(r['correctness']['score'], 2)}{numbers}")
            typer.echo(f"     judge: {(r['correctness']['reason'] or '')[:260]}")
            typer.echo(f"     answer: {(r['answer'] or '')[:200]!r}")


@app.command()
def correctness(
    k: int = typer.Option(8, help="Chunks retrieved per question"),
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
    concurrency: int = typer.Option(4, help="Answers generated / judged in parallel"),
    save: bool = typer.Option(True, help="Save the run for `compare-correctness`"),
):
    """Run the real pipeline and score answer correctness against the goldens."""
    from evaluation.app_eval.correctness import correctness_metric
    from evaluation.judge import ClaudeJudge
    from sec_rag.pipeline import RAGPipeline

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

    pipeline = RAGPipeline().warm_up()
    judge = ClaudeJudge(judge_model, effort=effort)
    result = runner.run(
        pipeline,
        goldens,
        k,
        metric_factory=lambda: {"correctness": correctness_metric(judge)},
        concurrency=concurrency,
        judge_name=judge_model,
    )
    print_report(result)
    typer.echo(f"\n{judge.cost_summary()} (generation by {result['settings']['generator']} "
               "is not included)")  # fmt: skip
    if save:
        typer.echo(f"Saved {runs.save(result, runner.RUN_KIND)}")


@app.command("compare-correctness")
def compare_correctness(before: Path = typer.Argument(None), after: Path = typer.Argument(None)):
    """Compare two saved correctness runs (default: the last two)."""
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

    typer.echo(f"\n{'category':>13} {'score':>17} {'correct':>9}")
    for name in [*b["by_category"], "ALL"]:
        new = b["summary"] if name == "ALL" else b["by_category"][name]
        old = a["summary"] if name == "ALL" else a["by_category"].get(name)
        if old is None:
            continue
        o, n = old["correctness"], new["correctness"]
        delta = "" if o is None or n is None else f" ({n - o:+.3f})"
        typer.echo(f"{name:>13} {_fmt(n) + delta:>17} {_pct(new['correct']):>9}")
    typer.echo(f"(changes under ±{NOISE} are likely judge noise)")

    old_rows = {r["id"]: r for r in a["cases"]}
    moves = []
    for r in b["cases"]:
        o = old_rows.get(r["id"])
        if o and o["correctness"]["score"] is not None and r["correctness"]["score"] is not None:
            delta = r["correctness"]["score"] - o["correctness"]["score"]
            if abs(delta) >= 0.1:
                moves.append((delta, r["id"]))
    if moves:
        typer.echo("\ncorrectness changes of 0.1 or more:")
        for delta, case_id in sorted(moves):
            typer.echo(f"  {delta:+.2f}  {case_id}")


if __name__ == "__main__":
    app()

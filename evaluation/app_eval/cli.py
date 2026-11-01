"""Commands: run, compare.

    python -m evaluation.app_eval run                                  # all metrics, 45 goldens
    python -m evaluation.app_eval run --metric completeness --category qualitative
    python -m evaluation.app_eval compare                              # last two saved runs
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
    s, g, names = result["settings"], result["goldens"], result["metrics"]
    typer.echo(
        f"Pipeline: {s['retrieval_mode']} retrieval, k={s['k']}, generator {s['generator']} | "
        f"judge {s['judge']} | goldens {g['total']} ({g['approved']} approved)"
    )

    header = " ".join(f"{name[:12]:>12} {'pass':>5}" for name in names)
    numbers = "correctness" in names
    typer.echo(f"\n{'category':>13} {'n':>3} {header}" + (f" {'key nums':>9}" if numbers else ""))
    for category, m in [*result["by_category"].items(), ("ALL", result["summary"])]:
        cells = " ".join(f"{_fmt(m[n]['score']):>12} {_pct(m[n]['passed']):>5}" for n in names)
        extra = f" {_pct(m['key_numbers_found']):>9}" if numbers else ""
        typer.echo(f"{category:>13} {m['cases']:>3} {cells}{extra}")
    typer.echo(f"(pass = score >= {runner.PASS}; '-' where a metric doesn't apply)")

    errors = sum(result["summary"][n]["errors"] for n in names)
    if errors:
        typer.echo(f"\n{errors} scores could not be computed; see the saved run.")

    for name in names:
        weak = [r for r in result["cases"]
                if r[name]["score"] is not None and r[name]["score"] < runner.PASS]  # fmt: skip
        if not weak:
            continue
        typer.echo(f"\n{name}: answers below {runner.PASS}")
        for r in sorted(weak, key=lambda r: r[name]["score"]):
            typer.echo(f"  [{r['id']}] {_fmt(r[name]['score'], 2)}: "
                       f"{(r[name]['reason'] or '')[:240]}")  # fmt: skip


@app.command()
def run(
    metric: list[str] = typer.Option(
        list(runner.METRICS), help=f"Metrics to score: {', '.join(runner.METRICS)}"
    ),
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
    save: bool = typer.Option(True, help="Save the run for `compare`"),
):
    """Run the real pipeline once and score the selected metrics on its answers."""
    from evaluation.judge import ClaudeJudge
    from sec_rag.pipeline import RAGPipeline

    unknown = [m for m in metric if m not in runner.METRICS]
    if unknown:
        raise typer.BadParameter(f"unknown metric(s) {unknown}; choose from {list(runner.METRICS)}")
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
        metrics={name: runner.METRICS[name] for name in metric},
        judge=judge,
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

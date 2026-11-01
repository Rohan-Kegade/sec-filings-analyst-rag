"""Commands: run, compare.

    python -m evaluation.scope_eval run                            # all 20 cases
    python -m evaluation.scope_eval run --category prompt_injection
    python -m evaluation.scope_eval compare                        # last two saved runs
"""

import logging
from pathlib import Path

import typer

from evaluation import runs
from evaluation.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL
from evaluation.scope_eval import runner
from evaluation.scope_eval.goldens import load_cases

app = typer.Typer(help=__doc__, no_args_is_help=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Differences smaller than this between runs are likely judge noise, not real changes.
NOISE = 0.03


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def print_report(result: dict) -> None:
    s = result["settings"]
    typer.echo(
        f"Pipeline: {s['retrieval_mode']} retrieval, k={s['k']}, generator {s['generator']} | "
        f"judge {s['judge']}"
    )
    typer.echo(f"\n{'category':>20} {'n':>3} {'score':>7} {'handled':>8} {'leaks':>6}")
    for name, m in [*result["by_category"].items(), ("ALL", result["summary"])]:
        typer.echo(f"{name:>20} {m['cases']:>3} {_fmt(m['score']):>7} {_pct(m['handled']):>8} "
                   f"{m['prompt_leaks']:>6}")  # fmt: skip
    typer.echo(f"(handled = score >= {runner.PASS}; leaks = answers repeating the system prompt)")
    if result["summary"]["errors"]:
        typer.echo(f"\n{result['summary']['errors']} cases could not be scored; see the saved run.")

    failed = [r for r in result["cases"] if r["prompt_leak"] or (
        r["scope"]["score"] is not None and r["scope"]["score"] < runner.PASS)]  # fmt: skip
    if failed:
        typer.echo("\nNot handled correctly:")
        for r in failed:
            leak = " [PROMPT LEAK]" if r["prompt_leak"] else ""
            typer.echo(f"\n  [{r['id']}] {r['category']}, score "
                       f"{_fmt(r['scope']['score'], 2)}{leak}")  # fmt: skip
            typer.echo(f"     judge: {(r['scope']['reason'] or '')[:240]}")
            typer.echo(f"     answer: {(r['answer'] or '')[:200]!r}")


@app.command()
def run(
    k: int = typer.Option(8, help="Chunks retrieved per question"),
    limit: int = typer.Option(0, help="Only the first N cases (0 = all)"),
    category: list[str] = typer.Option(None, help="Only these categories"),
    id: list[str] = typer.Option(None, help="Only these case ids"),
    judge_model: str = typer.Option(
        DEFAULT_JUDGE_MODEL, help="Claude model used as judge (default: SEC_RAG_JUDGE_MODEL)"
    ),
    effort: str = typer.Option(
        DEFAULT_JUDGE_EFFORT, help="Judge effort: low, medium, high (default: SEC_RAG_JUDGE_EFFORT)"
    ),
    concurrency: int = typer.Option(4, help="Answers generated / judged in parallel"),
    save: bool = typer.Option(True, help="Save the run for `compare`"),
):
    """Run the real pipeline on the scope cases and judge scope adherence."""
    from evaluation.judge import ClaudeJudge
    from evaluation.scope_eval.metric import scope_metric
    from sec_rag.pipeline import RAGPipeline

    cases = [
        c for c in load_cases()
        if (not category or c.category in category) and (not id or c.id in id)
    ]  # fmt: skip
    if limit:
        cases = cases[:limit]
    if not cases:
        raise typer.Exit("No cases match the filters.")

    pipeline = RAGPipeline().warm_up()
    judge = ClaudeJudge(judge_model, effort=effort)
    result = runner.run(
        pipeline,
        cases,
        k,
        metric_factory=lambda: {"scope": scope_metric(judge)},
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

    typer.echo(f"\n{'category':>20} {'score':>17} {'handled':>8}")
    for name in [*b["by_category"], "ALL"]:
        new = b["summary"] if name == "ALL" else b["by_category"][name]
        old = a["summary"] if name == "ALL" else a["by_category"].get(name)
        if old is None:
            continue
        o, n = old["score"], new["score"]
        delta = "" if o is None or n is None else f" ({n - o:+.3f})"
        typer.echo(f"{name:>20} {_fmt(n) + delta:>17} {_pct(new['handled']):>8}")
    typer.echo(f"(changes under ±{NOISE} are likely judge noise)")

    old_rows = {r["id"]: r for r in a["cases"]}
    for r in b["cases"]:
        o = old_rows.get(r["id"])
        if not o or o["scope"]["score"] is None or r["scope"]["score"] is None:
            continue
        was, now = o["scope"]["score"] >= runner.PASS, r["scope"]["score"] >= runner.PASS
        if was != now:
            typer.echo(f"  {'now handled' if now else 'REGRESSED'}: {r['id']}")


if __name__ == "__main__":
    app()

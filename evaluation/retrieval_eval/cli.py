"""Commands: run, compare.

    python -m evaluation.retrieval_eval run                 # hybrid, all 40 goldens
    python -m evaluation.retrieval_eval run --mode dense --mode hybrid --limit 5
    python -m evaluation.retrieval_eval compare             # last two saved runs
"""

import logging
from pathlib import Path

import typer

from evaluation.retrieval_eval import runner
from evaluation.retrieval_eval.goldens import load_goldens
from evaluation.retrieval_eval.judge import DEFAULT_JUDGE_MODEL

app = typer.Typer(help=__doc__, no_args_is_help=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Differences smaller than this between runs are likely judge noise, not real changes.
NOISE = 0.03


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _retrievers(modes: list[str]) -> dict:
    # Imported here so `compare` doesn't load the embedding model.
    from sec_rag.indexing.embedder import Embedder
    from sec_rag.indexing.keyword_index import KeywordIndex
    from sec_rag.indexing.vector_store import VectorStore
    from sec_rag.retrieval.retriever import Retriever

    shared = {"embedder": Embedder(), "store": VectorStore(), "keyword_index": KeywordIndex()}
    retrievers = {mode: Retriever(**shared, mode=mode) for mode in modes}
    next(iter(retrievers.values())).warm_up()  # load the model before latency is measured
    return retrievers


def print_report(result: dict) -> None:
    s, g = result["settings"], result["goldens"]
    typer.echo(
        f"Goldens: {g['total']} ({g['approved']} approved) | k={s['k']} | judge {s['judge']}"
    )
    typer.echo(f"\n{'mode':>8} {'recall':>8} {'precision':>10} {'errors':>7} {'ms':>6}")
    for mode, m in result["summary"].items():
        errors = m["recall_errors"] + m["precision_errors"]
        typer.echo(
            f"{mode:>8} {_fmt(m['recall']):>8} {_fmt(m['precision']):>10} "
            f"{errors:>7} {_fmt(m['latency_ms'], 0):>6}"
        )

    modes = list(result["by_category"])
    typer.echo("\nBy category (recall / precision):")
    typer.echo(f"{'category':>12} " + " ".join(f"{m:>15}" for m in modes))
    for category in next(iter(result["by_category"].values()), {}):
        cells = []
        for mode in modes:
            m = result["by_category"][mode][category]
            cells.append(f"{_fmt(m['recall'], 2)} / {_fmt(m['precision'], 2)}")
        typer.echo(f"{category:>12} " + " ".join(f"{c:>15}" for c in cells))

    for mode, rows in result["cases"].items():
        weak = [r for r in rows if (r["recall"]["score"] or 0) < 0.5 or r["recall"]["error"]]
        if weak:
            typer.echo(f"\n{mode}: cases with recall < 0.5")
            for r in weak:
                detail = r["recall"]["error"] or r["recall"]["reason"] or ""
                typer.echo(f"  {_fmt(r['recall']['score'], 2):>5}  {r['id']}: {detail[:160]}")


@app.command()
def run(
    mode: list[str] = typer.Option(["hybrid"], help="Retrieval modes: hybrid, dense, keyword"),
    k: int = typer.Option(8, help="Chunks retrieved per question"),
    limit: int = typer.Option(0, help="Only the first N goldens (0 = all)"),
    category: list[str] = typer.Option(None, help="Only these categories"),
    id: list[str] = typer.Option(None, help="Only these golden ids"),
    approved_only: bool = typer.Option(False, help="Only goldens marked approved"),
    judge_model: str = typer.Option(DEFAULT_JUDGE_MODEL, help="Claude model used as judge"),
    effort: str = typer.Option("medium", help="Judge effort: low, medium, high"),
    reasons: bool = typer.Option(True, help="Ask the judge for a reason per score (costs more)"),
    concurrency: int = typer.Option(4, help="Test cases judged in parallel"),
    save: bool = typer.Option(True, help="Save the run for `compare`"),
):
    """Score retrieval with DeepEval Contextual Recall and Contextual Precision."""
    from evaluation.retrieval_eval.judge import ClaudeJudge

    goldens = load_goldens()
    goldens = [
        gold for gold in goldens
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
        _retrievers(mode),
        goldens,
        k,
        metric_factory=lambda: runner.deepeval_metrics(judge, include_reason=reasons),
        concurrency=concurrency,
        judge_name=judge_model,
    )
    print_report(result)
    typer.echo(f"\n{judge.cost_summary()}")
    if save:
        typer.echo(f"Saved {runner.save(result)}")


@app.command()
def compare(before: Path = typer.Argument(None), after: Path = typer.Argument(None)):
    """Compare two saved runs (default: the last two)."""
    if before is None or after is None:
        runs = runner.list_runs()
        if len(runs) < 2:
            raise typer.Exit("Need at least two saved runs to compare.")
        before, after = runs[-2], runs[-1]
    a, b = runner.load_run(before), runner.load_run(after)
    typer.echo(f"before: {before.name}\nafter:  {after.name}")
    for key, value in b["settings"].items():
        if a["settings"].get(key) != value:
            typer.echo(f"  setting {key}: {a['settings'].get(key)} -> {value}")

    typer.echo(f"\n{'mode':>8} {'recall':>17} {'precision':>17}")
    for mode, m in b["summary"].items():
        if mode not in a["summary"]:
            continue
        cells = []
        for metric in runner.METRICS:
            old, new = a["summary"][mode][metric], m[metric]
            delta = "" if old is None or new is None else f" ({new - old:+.3f})"
            cells.append(f"{_fmt(new)}{delta}")
        typer.echo(f"{mode:>8} " + " ".join(f"{c:>17}" for c in cells))
    typer.echo(f"(changes under ±{NOISE} are likely judge noise)")

    for mode, rows in b["cases"].items():
        old_rows = {r["id"]: r for r in a["cases"].get(mode, [])}
        moves = []
        for r in rows:
            o = old_rows.get(r["id"])
            if o and o["recall"]["score"] is not None and r["recall"]["score"] is not None:
                delta = r["recall"]["score"] - o["recall"]["score"]
                if abs(delta) >= 0.1:
                    moves.append((delta, r["id"]))
        if moves:
            typer.echo(f"\n{mode}: recall changes of 0.1 or more")
            for delta, case_id in sorted(moves):
                typer.echo(f"  {delta:+.2f}  {case_id}")


if __name__ == "__main__":
    app()

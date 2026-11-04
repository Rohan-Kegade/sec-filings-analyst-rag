"""Retriever eval (component): DeepEval Contextual Recall and Contextual Precision.

Each golden's question goes through the retriever only (no answer generation); a Claude
judge then scores the retrieved chunks against the golden's reference answer.

    python -m evals.eval_retriever run                  # hybrid, all 40 goldens
    python -m evals.eval_retriever run --mode dense --mode hybrid --limit 5
    python -m evals.eval_retriever compare              # last two saved runs
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import typer
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from evals import GOLDENS_DIR, runs
from evals.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL, PRICES, score_case
from evals.runs import git_commit
from sec_rag.config import settings
from sec_rag.retrieval.retriever import Retriever

# --- goldens ---------------------------------------------------------------------------------

GOLDENS_FILE = GOLDENS_DIR / "retrieval_goldens.json"


@dataclass(frozen=True)
class Golden:
    id: str
    category: str
    input: str
    expected_output: str


def load_goldens(path: Path = GOLDENS_FILE) -> list[Golden]:
    goldens = [
        Golden(id=raw["id"], category=raw["category"], input=raw["input"],
               expected_output=raw["expected_output"])
        for raw in json.loads(path.read_text(encoding="utf-8"))
    ]  # fmt: skip
    ids = [g.id for g in goldens]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate golden ids in {path.name}: {duplicates}")
    return goldens


# --- runner ----------------------------------------------------------------------------------

RUN_KIND = "retriever"  # saved under evals/runs/retriever/
METRICS = ("recall", "precision")


def deepeval_metrics(judge: DeepEvalBaseLLM, include_reason: bool) -> dict:
    """Fresh metric objects for one test case (DeepEval metrics hold per-case state)."""
    common = {"model": judge, "include_reason": include_reason, "eval_mode": "llm"}
    return {
        "recall": ContextualRecallMetric(**common),
        "precision": ContextualPrecisionMetric(**common),
    }


async def evaluate_mode(
    retriever: Retriever,
    goldens: list[Golden],
    k: int,
    metric_factory: Callable[[], dict],
    concurrency: int,
    on_judged: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """on_judged(done, total) is called each time a case finishes judging."""
    # Retrieval is fast and local, so do it up front; only the judge calls run concurrently.
    # Hybrid also records where dense and keyword search ranked each chunk, and what fusion
    # dropped, so the report can show why a case failed.
    traced = getattr(retriever, "mode", None) == "hybrid"
    retrieved, dropped, latencies = [], [], []
    for golden in goldens:
        start = time.perf_counter()
        if traced:
            results, left_out = retriever.hybrid_trace(golden.input, k=k)
        else:
            results, left_out = retriever.retrieve(golden.input, k=k), []
        latencies.append((time.perf_counter() - start) * 1000)
        retrieved.append(results)
        dropped.append(left_out)

    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    if on_judged:
        on_judged(0, len(goldens))

    async def judge(golden: Golden, results: list) -> dict:
        nonlocal done
        scores = await score_case(
            LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                retrieval_context=[rc.chunk.text for rc in results],
            ),
            metric_factory(),
            semaphore,
        )
        done += 1
        if on_judged:
            on_judged(done, len(goldens))
        return scores

    scored = await asyncio.gather(
        *(judge(golden, results) for golden, results in zip(goldens, retrieved))
    )
    rows = []
    for golden, results, left_out, latency, scores in zip(
        goldens, retrieved, dropped, latencies, scored
    ):
        row = {
            "id": golden.id,
            "category": golden.category,
            "latency_ms": latency,
            "retrieved": [chunk_label(rc) for rc in results],
            **scores,
        }
        if traced:
            row["sources"] = [rc.ranks for rc in results]
            row["dropped"] = [{"chunk": chunk_label(rc), "ranks": rc.ranks} for rc in left_out]
        rows.append(row)
    return rows


def chunk_label(rc) -> str:
    c = rc.chunk
    return f"{c.ticker} FY{c.fiscal_year} Item {c.item} ({c.chunk_id})"


def source_mix(rows: list[dict], k: int) -> dict | None:
    """For hybrid rows: would each search alone (its own top k) have returned the final chunks?

    "neither" counts chunks that fusion promoted from deeper in both candidate lists, in
    place of a strong hit from one search.
    """
    traced = [r for r in rows if "sources" in r]
    if not traced:
        return None
    mix = {f"both top {k}s": 0, f"dense top {k} only": 0, f"keyword top {k} only": 0,
           f"neither top {k}": 0}  # fmt: skip
    for r in traced:
        for ranks in r["sources"]:
            dense = ranks.get("dense", k + 1) <= k
            keyword = ranks.get("keyword", k + 1) <= k
            if dense and keyword:
                mix[f"both top {k}s"] += 1
            elif dense:
                mix[f"dense top {k} only"] += 1
            elif keyword:
                mix[f"keyword top {k} only"] += 1
            else:
                mix[f"neither top {k}"] += 1
    return mix


def summarize(rows: list[dict]) -> dict:
    summary = {"cases": len(rows)}
    for metric in METRICS:
        values = [r[metric]["score"] for r in rows if r[metric]["score"] is not None]
        summary[metric] = mean(values) if values else None
        summary[f"{metric}_errors"] = sum(r[metric]["error"] is not None for r in rows)
    summary["latency_ms"] = mean(r["latency_ms"] for r in rows) if rows else None
    return summary


def run(
    retrievers: dict[str, Retriever],
    goldens: list[Golden],
    k: int,
    metric_factory: Callable[[], dict],
    concurrency: int,
    judge_name: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """progress(mode, done, total) is called as each mode's cases finish judging."""

    async def run_all() -> dict[str, list[dict]]:
        # One event loop for every mode: the async Claude client's connections belong to it.
        results = {}
        for mode, retriever in retrievers.items():
            on_judged = (lambda done, total, m=mode: progress(m, done, total)) if progress else None
            results[mode] = await evaluate_mode(
                retriever, goldens, k, metric_factory, concurrency, on_judged
            )
        return results

    results = asyncio.run(run_all())
    categories = sorted({g.category for g in goldens})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "k": k,
            "judge": judge_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "hybrid_candidates": settings.hybrid_candidates,
            "rrf_k": settings.rrf_k,
            "rerank": rerank_label(),
            "git_commit": git_commit(),
        },
        "goldens": {"total": len(goldens)},
        "summary": {mode: summarize(rows) for mode, rows in results.items()},
        "by_category": {
            mode: {c: summarize([r for r in rows if r["category"] == c]) for c in categories}
            for mode, rows in results.items()
        },
        "cases": results,
    }


# --- cli -------------------------------------------------------------------------------------

app = typer.Typer(help=__doc__, no_args_is_help=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Differences smaller than this between runs are likely judge noise, not real changes.
NOISE = 0.03


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def rerank_label() -> str:
    """What reranked the hybrid results, for the run's settings ("off" when nothing did)."""
    if not settings.rerank:
        return "off"
    return f"{settings.rerank_model} ({settings.rerank_effort}), top {settings.rerank_candidates}"


def rerank_cost(retrievers: dict) -> str | None:
    """Token use of the reranker during the run, priced like the judge."""
    reranker = next((r.reranker for r in retrievers.values() if r.rerank), None)
    if reranker is None:
        return None
    summary = f"reranker tokens: {reranker.input_tokens:,} in / {reranker.output_tokens:,} out"
    if reranker.model in PRICES:
        price_in, price_out = PRICES[reranker.model]
        dollars = reranker.input_tokens / 1e6 * price_in + reranker.output_tokens / 1e6 * price_out
        summary += f" (~${dollars:.2f})"
    return summary


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
        f"Goldens: {g['total']} | k={s['k']} | judge {s['judge']}"
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


def _md_cell(text) -> str:
    """Make text safe for one Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def markdown_report(result: dict, cost: str | None = None) -> str:
    """The run as a Markdown report: settings, summary, by category, weak cases, all cases."""
    s, modes = result["settings"], list(result["summary"])
    lines = [f"# Retriever eval: {result['timestamp']}", ""]
    lines += ["| setting | value |", "|---|---|"]
    lines += [f"| {key} | {_md_cell(value)} |" for key, value in s.items()]
    lines += [f"| goldens | {result['goldens']['total']} |"]
    if cost:
        lines += [f"| cost | {_md_cell(cost)} |"]

    lines += ["", "## Summary", "", "| mode | recall | precision | errors | latency ms |",
              "|---|---|---|---|---|"]  # fmt: skip
    for mode, m in result["summary"].items():
        errors = m["recall_errors"] + m["precision_errors"]
        lines.append(f"| {mode} | {_fmt(m['recall'])} | {_fmt(m['precision'])} | {errors} "
                     f"| {_fmt(m['latency_ms'], 0)} |")  # fmt: skip

    # Every mode runs the same goldens, so the case count is one column, not one per mode.
    lines += ["", "## By category (recall / precision)", "",
              "| category | cases | " + " | ".join(modes) + " |",
              "|---" * (len(modes) + 2) + "|"]  # fmt: skip
    for category, first in next(iter(result["by_category"].values()), {}).items():
        cells = []
        for mode in modes:
            m = result["by_category"][mode][category]
            cells.append(f"{_fmt(m['recall'], 2)} / {_fmt(m['precision'], 2)}")
        lines.append(f"| {category} | {first['cases']} | " + " | ".join(cells) + " |")

    for mode, rows in result["cases"].items():
        mix = source_mix(rows, s["k"])
        if mix:
            total = sum(mix.values()) or 1
            lines += ["", f"## {mode}: where the final chunks came from", "",
                      (f"Would each search alone (its own top {s['k']}) have returned the "
                       "chunk? \"neither\" = fusion promoted it from deeper in both lists."), "",
                      "| found by | chunks | share |", "|---|---|---|"]  # fmt: skip
            lines += [f"| {key} | {n} | {n / total:.0%} |" for key, n in mix.items()]

    # Summary only: retrieved chunks, search ranks and fusion drops stay in the run's JSON.
    for mode, rows in result["cases"].items():
        weak = [r for r in rows if (r["recall"]["score"] or 0) < 0.5 or r["recall"]["error"]]
        lines += ["", f"## {mode}: cases with recall < 0.5", ""]
        if not weak:
            lines.append("None.")
            continue
        lines += ["| id | category | recall | judge's reason |", "|---|---|---|---|"]
        for r in weak:
            recall = "error" if r["recall"]["error"] else _fmt(r["recall"]["score"], 2)
            detail = r["recall"]["error"] or r["recall"]["reason"] or ""
            lines.append(f"| {r['id']} | {r['category']} | {recall} | {_md_cell(detail)} |")

    for mode, rows in result["cases"].items():
        lines += ["", f"## {mode}: all cases", "", "| id | category | recall | precision |",
                  "|---|---|---|---|"]  # fmt: skip
        for r in rows:
            scores = ["error" if r[m]["error"] else _fmt(r[m]["score"], 2) for m in METRICS]
            lines.append(f"| {r['id']} | {r['category']} | " + " | ".join(scores) + " |")
    return "\n".join(lines) + "\n"


@app.command("run")
def run_command(
    mode: list[str] = typer.Option(["hybrid"], help="Retrieval modes: hybrid, dense, keyword"),
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
    reasons: bool = typer.Option(True, help="Ask the judge for a reason per score (costs more)"),
    concurrency: int = typer.Option(4, help="Test cases judged in parallel"),
    save: bool = typer.Option(True, help="Save the run for `compare`"),
):
    """Score retrieval with DeepEval Contextual Recall and Contextual Precision."""
    from evals.judge import ClaudeJudge

    goldens = load_goldens()
    goldens = [
        gold for gold in goldens
        if (not category or gold.category in category)
        and (not id or gold.id in id)
    ]  # fmt: skip
    if limit:
        goldens = goldens[:limit]
    if not goldens:
        raise typer.Exit("No goldens match the filters.")

    judge = ClaudeJudge(judge_model, effort=effort)
    typer.echo(f"{len(goldens)} questions x {len(mode)} mode(s): {', '.join(mode)} | "
               f"judge {judge_model} ({effort}) | k={k}", err=True)  # fmt: skip
    typer.echo("Loading retrievers...", err=True)
    retrievers = _retrievers(mode)
    started = time.perf_counter()

    def progress(mode_name: str, done: int, total: int) -> None:
        # One line per mode, rewritten in place; a newline once the mode is finished.
        elapsed = time.perf_counter() - started
        end = "\n" if done == total else ""
        typer.echo(f"\r  {mode_name:>8}: {done}/{total} judged  ({elapsed:.0f}s)",
                   nl=False, err=True)  # fmt: skip
        typer.echo(end, nl=False, err=True)

    result = run(
        retrievers,
        goldens,
        k,
        metric_factory=lambda: deepeval_metrics(judge, include_reason=reasons),
        concurrency=concurrency,
        judge_name=judge_model,
        progress=progress,
    )
    typer.echo("", err=True)
    print_report(result)
    cost = "; ".join(c for c in [judge.cost_summary(), rerank_cost(retrievers)] if c)
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

    typer.echo(f"\n{'mode':>8} {'recall':>17} {'precision':>17}")
    for mode, m in b["summary"].items():
        if mode not in a["summary"]:
            continue
        cells = []
        for metric in METRICS:
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

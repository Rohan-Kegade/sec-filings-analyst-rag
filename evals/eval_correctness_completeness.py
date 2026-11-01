"""Correctness and completeness evals (application quality), with DeepEval G-Eval.

Each golden's question goes through the real pipeline once, and every selected metric scores
the same answers, so results are comparable and generation is paid for once:
  correctness   key facts stated correctly, with no contradictions (plus a free number check)
  completeness  everything the question asks for is covered (coverage only)

    python -m evals.eval_correctness_completeness run   # both metrics, 40 goldens
    python -m evals.eval_correctness_completeness run --metric completeness --category qualitative
    python -m evals.eval_correctness_completeness compare
"""

import asyncio
import json
import logging
import re
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
from evals.eval_rag_triad import run_pipeline
from evals.eval_retriever import score_case
from evals.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL
from evals.runs import git_commit
from sec_rag.config import settings
from sec_rag.pipeline import RAGPipeline

# --- goldens ---------------------------------------------------------------------------------

RETRIEVAL_GOLDENS = GOLDENS_DIR / "retrieval_goldens.json"


@dataclass(frozen=True)
class AppGolden:
    id: str
    category: str
    status: str
    input: str
    expected_output: str
    key_facts: tuple[str, ...]  # what must be right (correctness)
    required_points: tuple[str, ...] = ()  # what must be covered (completeness)


def load_goldens(path: Path = RETRIEVAL_GOLDENS) -> list[AppGolden]:
    goldens = []
    for raw in json.loads(path.read_text(encoding="utf-8")):
        meta = raw["additional_metadata"]
        goldens.append(AppGolden(
            id=meta["id"], category=meta["category"], status=meta.get("status", "draft"),
            input=raw["input"], expected_output=raw["expected_output"],
            key_facts=tuple(meta["key_facts"]), required_points=tuple(meta["required_points"]),
        ))  # fmt: skip
    ids = [g.id for g in goldens]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate golden ids: {duplicates}")
    return goldens


# --- correctness -----------------------------------------------------------------------------

CORRECTNESS_STEPS = [
    "Read the question, the expected output (a reference answer based on the company's 10-K "
    "filing) and the key facts listed at its end.",
    "For each key fact, check whether the actual output states it correctly. Equivalent "
    "wording, rounding and units are fine (for example 'about 228,000' matches "
    "'approximately 228,000'). A key fact that is missing counts against correctness.",
    "Heavily penalize any statement that contradicts the expected output, or that gives a "
    "wrong number, date, fiscal year or company for a key fact.",
    "Do not penalize additional details that the expected output does not mention, unless "
    "they contradict it.",
]

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def expected_with_key_facts(golden: AppGolden) -> str:
    """The reference answer followed by its key facts, as the judge sees it."""
    facts = "\n".join(f"- {fact}" for fact in golden.key_facts)
    return f"{golden.expected_output}\n\nKey facts:\n{facts}"


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


def _normalize_number(token: str) -> str:
    token = token.replace(",", "")
    return token[:-2] if token.endswith(".0") else token  # "$95.0 billion" == "$95 billion"


def key_numbers(golden: AppGolden) -> set[str]:
    return {_normalize_number(n) for fact in golden.key_facts for n in _NUMBER.findall(fact)}


def number_check(golden: AppGolden, answer: str | None) -> tuple[int, int]:
    """(numbers from the key facts found in the answer, total). A free, exact pre-check."""
    expected = key_numbers(golden)
    found = {_normalize_number(n) for n in _NUMBER.findall(answer or "")}
    return len(expected & found), len(expected)


# --- completeness ----------------------------------------------------------------------------

COMPLETENESS_STEPS = [
    "Read the question and the required points listed in the expected output: everything a "
    "complete answer to this question must cover.",
    "For each required point, decide whether the actual output covers it fully, partly or "
    "not at all. Wording does not matter, only whether the point is addressed.",
    "The score reflects the share of required points covered; partly covered points count "
    "for less than fully covered ones.",
    "Do not judge whether figures or claims are accurate; correctness is evaluated "
    "separately. A point counts as covered if the answer addresses it.",
    "Do not penalize additional details beyond the required points.",
    "If the answer says the information for a point is not available instead of covering "
    "it, count that point as not covered.",
]


def expected_with_points(golden: AppGolden) -> str:
    """The reference answer followed by the required points, as the judge sees it."""
    points = "\n".join(f"- {point}" for point in golden.required_points)
    return f"{golden.expected_output}\n\nRequired points:\n{points}"


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

RUN_KIND = "app"  # saved under evals/runs/app/
PASS = 0.7  # a score at or above this counts as a pass


@dataclass(frozen=True)
class AppMetric:
    """One application metric: its judge, the expected output it sees, and where it applies."""

    build: Callable[[DeepEvalBaseLLM], object]
    expected: Callable[[AppGolden], str]
    applies: Callable[[AppGolden], bool]


METRICS = {
    "correctness": AppMetric(
        build=correctness_metric,
        expected=expected_with_key_facts,
        applies=lambda golden: True,
    ),
    "completeness": AppMetric(
        build=completeness_metric,
        expected=expected_with_points,
        applies=lambda golden: bool(golden.required_points),
    ),
}

NOT_APPLICABLE = {"score": None, "reason": None, "error": None, "applies": False}


async def score_all(goldens: list[AppGolden], outputs: list[dict], metrics: dict[str, AppMetric],
                    judge: DeepEvalBaseLLM, concurrency: int) -> list[dict]:  # fmt: skip
    semaphore = asyncio.Semaphore(concurrency)

    async def score(golden: AppGolden, output: dict, name: str) -> dict:
        metric = metrics[name]
        if not metric.applies(golden):
            return NOT_APPLICABLE
        if output["answer"] is None:
            return {"score": None, "reason": None, "error": output["error"] or "no answer",
                    "applies": True}  # fmt: skip
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=output["answer"],
            expected_output=metric.expected(golden),
        )
        scored = await score_case(test_case, {name: metric.build(judge)}, semaphore)
        return {**scored[name], "applies": True}

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
            "applies": sum(r[name]["applies"] for r in rows),
            "errors": sum(r[name]["error"] is not None for r in rows),
        }
    if "correctness" in metric_names:
        found = sum(r["key_numbers_found"] for r in rows)
        total = sum(r["key_numbers_total"] for r in rows)
        summary["key_numbers_found"] = found / total if total else None
    return summary


def run(pipeline: RAGPipeline, goldens: list[AppGolden], k: int, metrics: dict[str, AppMetric],
        judge: DeepEvalBaseLLM, concurrency: int, judge_name: str) -> dict:  # fmt: skip
    """`metrics` maps names to definitions, normally a selection from METRICS."""
    metric_names = list(metrics)
    outputs = run_pipeline(pipeline, goldens, k, concurrency)
    scores = asyncio.run(score_all(goldens, outputs, metrics, judge, concurrency))

    rows = []
    for golden, output, scored in zip(goldens, outputs, scores):
        found, total = number_check(golden, output["answer"])
        rows.append({
            "id": golden.id,
            "category": golden.category,
            "status": golden.status,
            "question": golden.input,
            "answer": output["answer"],
            "generation_error": output["error"],
            "sources": [f"{rc.chunk.ticker} FY{rc.chunk.fiscal_year} Item {rc.chunk.item}"
                        for rc in output["sources"]],
            "key_numbers_found": found,
            "key_numbers_total": total,
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
        "goldens": {
            "total": len(goldens),
            "approved": sum(g.status == "approved" for g in goldens),
        },
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
        f"judge {s['judge']} | goldens {g['total']} ({g['approved']} approved)"
    )

    header = " ".join(f"{name[:12]:>12} {'pass':>5}" for name in names)
    numbers = "correctness" in names
    typer.echo(f"\n{'category':>13} {'n':>3} {header}" + (f" {'key nums':>9}" if numbers else ""))
    for category, m in [*result["by_category"].items(), ("ALL", result["summary"])]:
        cells = " ".join(f"{_fmt(m[n]['score']):>12} {_pct(m[n]['passed']):>5}" for n in names)
        extra = f" {_pct(m['key_numbers_found']):>9}" if numbers else ""
        typer.echo(f"{category:>13} {m['cases']:>3} {cells}{extra}")
    typer.echo(f"(pass = score >= {PASS}; '-' where a metric doesn't apply)")

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
    from evals.judge import ClaudeJudge
    from sec_rag.pipeline import RAGPipeline

    unknown = [m for m in metric if m not in METRICS]
    if unknown:
        raise typer.BadParameter(f"unknown metric(s) {unknown}; choose from {list(METRICS)}")
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

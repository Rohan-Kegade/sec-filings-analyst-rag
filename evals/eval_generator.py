"""Generator eval (component): DeepEval Faithfulness and Answer Relevancy.

Each golden has a frozen set of excerpts. The generator answers from those excerpts only (no
retrieval), and a Claude judge scores the answer against them and against the question.

    python -m evals.eval_generator run                  # all 54 goldens
    python -m evals.eval_generator run --category partial --limit 3
    python -m evals.eval_generator compare              # last two saved runs

Each saved run (evals/runs/generator/) gets a Markdown report next to it, listing the
claims the judge found unsupported in each weak answer.
"""

import asyncio
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import typer
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from evals import GOLDENS_DIR, runs
from evals.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL, score_case, token_cost
from evals.runs import git_commit
from sec_rag.config import settings
from sec_rag.generation.llm import AnswerGenerator
from sec_rag.models import Chunk, RetrievedChunk

# --- goldens ---------------------------------------------------------------------------------

GOLDENS_FILE = GOLDENS_DIR / "generator_goldens.json"

# Every chunk starts with the header the chunker adds, e.g.
# "Microsoft Corporation (MSFT) 10-K FY2024 - Item 1: Business"
HEADER = re.compile(
    r"^(?P<company>.+?) \((?P<ticker>[A-Z.]+)\) 10-K FY(?P<year>\d{4}) - "
    r"Item (?P<item>[0-9A-Z]+): ?(?P<title>.*)$"
)


# Goldens in this category have no answer in their excerpts: the generator should say so.
NO_ANSWER = "no_answer"


@dataclass(frozen=True)
class GeneratorGolden:
    id: str
    category: str
    input: str
    context: tuple[str, ...]  # the frozen excerpts: evidence plus distractors

    @property
    def expected_behavior(self) -> str:
        return "abstain" if self.category == NO_ANSWER else "answer"


def load_goldens(path: Path = GOLDENS_FILE) -> list[GeneratorGolden]:
    goldens = [
        GeneratorGolden(id=raw["id"], category=raw["category"], input=raw["input"],
                        context=tuple(raw["context"]))
        for raw in json.loads(path.read_text(encoding="utf-8"))
    ]  # fmt: skip
    ids = [g.id for g in goldens]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate golden ids in {path.name}: {duplicates}")
    return goldens


def to_chunks(golden: GeneratorGolden) -> list[RetrievedChunk]:
    """Rebuild the chunks the generator's prompt expects from the frozen excerpt text.

    The prompt shows company, ticker, fiscal year and Item per excerpt, all taken from the
    header line every chunk starts with. Ids and positions in the index aren't used.
    """
    chunks = []
    for index, text in enumerate(golden.context):
        match = HEADER.match(text.split("\n", 1)[0])
        if not match:
            raise ValueError(f"{golden.id}: excerpt without a chunk header: {text[:80]!r}")
        chunk = Chunk(
            chunk_id=f"{golden.id}-{index}",
            text=text,
            ticker=match["ticker"],
            company_name=match["company"],
            fiscal_year=int(match["year"]),
            item=match["item"],
            item_title=match["title"],
            accession_number="",
            source_url="",
            chunk_index=index,
        )
        chunks.append(RetrievedChunk(chunk=chunk, score=1.0))
    return chunks


# --- runner ----------------------------------------------------------------------------------

RUN_KIND = "generator"  # saved under evals/runs/generator/
METRICS = ("faithfulness", "relevancy")

# Rough check that an answer says the excerpts don't contain the information. Used to report
# the no-answer cases, where Faithfulness alone can't tell a refusal from a vague answer.
ABSTAIN = re.compile(
    r"\b(do(es)? not|don't|doesn't|did not|no)\b.{0,40}\b(contain|include|disclose|provide|"
    r"state|mention|report|give|specify|information|data|details)",
    re.IGNORECASE,
)


def deepeval_metrics(judge: DeepEvalBaseLLM, include_reason: bool) -> dict:
    """Fresh metric objects for one test case (DeepEval metrics hold per-case state)."""
    common = {"model": judge, "include_reason": include_reason, "eval_mode": "llm"}
    return {
        "faithfulness": FaithfulnessMetric(**common),
        "relevancy": AnswerRelevancyMetric(**common),
    }


def _verdict(item) -> str:
    """A DeepEval verdict as plain text: "yes", "no" or "borderline"."""
    value = getattr(item, "verdict", "")
    return str(getattr(value, "value", value)).lower()


def metric_details(name: str, metric) -> dict | None:
    """What the judge did not accept, item by item, so a report can show *why* a score is low.

    Faithfulness: answer claims the excerpts don't support. Relevancy: answer statements that
    don't address the question. DeepEval keeps one verdict per claim/statement, in order.

    Only "no" verdicts lower the score; "borderline" (partly supported / partly relevant)
    counts as passing in both metrics, so those are kept apart as context.
    """
    items = getattr(metric, "claims" if name == "faithfulness" else "statements", None)
    verdicts = getattr(metric, "verdicts", None)
    if not items or not verdicts or len(items) != len(verdicts):
        return None

    def matching(verdict: str) -> list[dict]:
        return [
            {"text": text, "reason": getattr(v, "reason", None)}
            for text, v in zip(items, verdicts)
            if _verdict(v) == verdict
        ]

    key = "unsupported" if name == "faithfulness" else "irrelevant"
    return {"checked": len(items), key: matching("no"), "borderline": matching("borderline")}


def generate_answers(
    generator: AnswerGenerator,
    goldens: list[GeneratorGolden],
    concurrency: int,
    on_done: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Answer every golden from its own excerpts. Failures are recorded, not raised.

    on_done(done, total) is called as each answer finishes.
    """
    done, lock = 0, threading.Lock()
    if on_done:
        on_done(0, len(goldens))

    def answer(golden: GeneratorGolden) -> dict:
        nonlocal done
        start = time.perf_counter()
        try:
            text, error = generator.generate(golden.input, to_chunks(golden)), None
        except Exception as exc:  # API or model failure for this golden only
            text, error = None, str(exc)
        latency = (time.perf_counter() - start) * 1000
        with lock:
            done += 1
            if on_done:
                on_done(done, len(goldens))
        return {"answer": text, "error": error, "latency_ms": latency}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(answer, goldens))


async def score_all(
    goldens: list[GeneratorGolden], answers: list[dict], metric_factory: Callable[[], dict],
    concurrency: int, on_judged: Callable[[int, int], None] | None = None,
) -> list[dict | None]:  # fmt: skip
    """on_judged(done, total) is called as each answer finishes judging."""
    semaphore = asyncio.Semaphore(concurrency)
    to_judge = sum(a["answer"] is not None for a in answers)
    done = 0
    if on_judged:
        on_judged(0, to_judge)

    async def score(golden: GeneratorGolden, generated: dict) -> dict | None:
        nonlocal done
        if generated["answer"] is None:
            return None  # nothing to judge
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=generated["answer"],
            retrieval_context=list(golden.context),  # the same excerpts the generator saw
        )
        scores = await score_case(test_case, metric_factory(), semaphore, details=metric_details)
        done += 1
        if on_judged:
            on_judged(done, to_judge)
        return scores

    return await asyncio.gather(*(score(g, a) for g, a in zip(goldens, answers)))


def summarize(rows: list[dict]) -> dict:
    summary = {"cases": len(rows)}
    for metric in METRICS:
        values = [r[metric]["score"] for r in rows if r[metric]["score"] is not None]
        summary[metric] = mean(values) if values else None
        # Judge failures only: a failed generation is counted once, in generation_errors.
        summary[f"{metric}_errors"] = sum(
            r[metric]["error"] is not None and r["generation_error"] is None for r in rows
        )
    summary["generation_errors"] = sum(r["generation_error"] is not None for r in rows)
    abstain_rows = [r for r in rows if r["expected_behavior"] == "abstain"]
    summary["abstained"] = (
        sum(r["abstained"] for r in abstain_rows) / len(abstain_rows) if abstain_rows else None
    )
    summary["latency_ms"] = mean(r["latency_ms"] for r in rows) if rows else None
    return summary


def run(
    generator: AnswerGenerator,
    goldens: list[GeneratorGolden],
    metric_factory: Callable[[], dict],
    concurrency: int,
    judge_name: str,
    progress: Callable[[str, int, int], None] | None = None,
    judge_effort: str = DEFAULT_JUDGE_EFFORT,
) -> dict:
    """progress(stage, done, total) is called as answers are generated, then judged."""
    on_generated = (lambda done, total: progress("generated", done, total)) if progress else None
    on_judged = (lambda done, total: progress("judged", done, total)) if progress else None
    answers = generate_answers(generator, goldens, concurrency, on_generated)
    scores = asyncio.run(score_all(goldens, answers, metric_factory, concurrency, on_judged))

    not_run = {"score": None, "reason": None, "error": "no answer was generated"}
    rows = []
    for golden, generated, scored in zip(goldens, answers, scores):
        scored = scored or {metric: not_run for metric in METRICS}
        rows.append({
            "id": golden.id,
            "category": golden.category,
            "expected_behavior": golden.expected_behavior,
            "question": golden.input,
            "answer": generated["answer"],
            "generation_error": generated["error"],
            "abstained": bool(generated["answer"] and ABSTAIN.search(generated["answer"])),
            "latency_ms": generated["latency_ms"],
            **scored,
        })  # fmt: skip

    categories = sorted({g.category for g in goldens})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "generator": getattr(generator, "model", settings.llm_model),
            "generator_effort": settings.llm_effort,
            "generator_max_tokens": settings.llm_max_tokens,
            "judge": judge_name,
            "judge_effort": judge_effort,
            "git_commit": git_commit(),
        },
        "goldens": {"total": len(goldens)},
        "summary": summarize(rows),
        "by_category": {c: summarize([r for r in rows if r["category"] == c]) for c in categories},
        "cases": rows,
    }


# --- cli -------------------------------------------------------------------------------------

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
    typer.echo(f"Goldens: {g['total']} | generator {s['generator']} "
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
            for metric in METRICS:
                if r[metric]["reason"] and (r[metric]["score"] or 0) < 1:
                    typer.echo(f"     {metric}: {r[metric]['reason'][:220]}")

    abstain = [r for r in rows if r["expected_behavior"] == "abstain"]
    if abstain:
        typer.echo("\nNo-answer cases (should say the excerpts don't contain it):")
        for r in abstain:
            mark = "ok  " if r["abstained"] else "MISS"
            text = (r["answer"] or r["generation_error"] or "")[:160]
            typer.echo(f"  {mark} [{r['id']}] {text!r}")


def _md_cell(text) -> str:
    """Make text safe for one Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _weak(row: dict) -> bool:
    def below(metric: str, limit: float) -> bool:
        return row[metric]["score"] is not None and row[metric]["score"] < limit

    return below("faithfulness", WEAK_FAITHFULNESS) or below("relevancy", WEAK_RELEVANCY)


def _summary_row(name: str, m: dict) -> str:
    errors = m["faithfulness_errors"] + m["relevancy_errors"] + m["generation_errors"]
    return (f"| {name} | {m['cases']} | {_fmt(m['faithfulness'])} | {_fmt(m['relevancy'])} "
            f"| {_pct(m['abstained'])} | {errors} | {_fmt(m['latency_ms'], 0)} |")  # fmt: skip


def markdown_report(result: dict, cost: str | None = None) -> str:
    """The run as a Markdown report: settings, summary, by category, no-answer cases,
    weak answers with the claims the judge rejected, and every case."""
    s, rows = result["settings"], result["cases"]
    lines = [f"# Generator eval: {result['timestamp']}", ""]
    lines += ["| setting | value |", "|---|---|"]
    lines += [f"| {key} | {_md_cell(value)} |" for key, value in s.items()]
    lines += [f"| goldens | {result['goldens']['total']} |"]
    if cost:
        lines += [f"| cost | {_md_cell(cost)} |"]

    header = ["| category | cases | faithfulness | relevancy | abstained | errors | latency ms |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    lines += ["", "## Summary", "",
              ("- **faithfulness**: share of the answer's claims supported by the excerpts "
               "(1.0 = nothing made up)"),
              "- **relevancy**: share of the answer's statements that address the question",
              ("- **abstained**: share of no-answer questions where the answer said the "
               "excerpts don't contain it (should be 100%)"), "",
              *header, _summary_row("ALL", result["summary"])]  # fmt: skip
    lines += ["", "## By category", "", *header]
    lines += [_summary_row(name, m) for name, m in result["by_category"].items()]

    abstain = [r for r in rows if r["expected_behavior"] == "abstain"]
    if abstain:
        lines += ["", "## No-answer cases", "",
                  "These excerpts don't contain the answer, so the generator should say so.", "",
                  "| id | said so? | answer |", "|---|---|---|"]  # fmt: skip
        for r in abstain:
            text = r["answer"] or r["generation_error"] or ""
            lines.append(f"| {r['id']} | {'yes' if r['abstained'] else '**no**'} "
                         f"| {_md_cell(text[:300])} |")  # fmt: skip

    weak = [r for r in rows if _weak(r)]
    lines += ["", (f"## Weak answers (faithfulness < {WEAK_FAITHFULNESS} or "
                   f"relevancy < {WEAK_RELEVANCY})"), ""]  # fmt: skip
    if not weak:
        lines.append("None.")
    for r in weak:
        lines += [f"### {r['id']} ({r['category']})", "",
                  f"- **question:** {_md_cell(r['question'])}",
                  (f"- **faithfulness** {_fmt(r['faithfulness']['score'], 2)}, "
                   f"**relevancy** {_fmt(r['relevancy']['score'], 2)}")]  # fmt: skip
        for metric, key, label in [("faithfulness", "unsupported", "Claims not supported"),
                                   ("relevancy", "irrelevant", "Statements off the question")]:
            details = r[metric].get("details", {})
            for items, title in [(details.get(key, []), f"{label} (lower the score)"),
                                 (details.get("borderline", []),
                                  "Borderline: partly supported, not counted against it")]:
                if items:
                    noun = "claim" if metric == "faithfulness" else "statement"
                    lines += ["", f"{metric} - {title}:", "", f"| {noun} | judge's reason |",
                              "|---|---|"]  # fmt: skip
                    lines += [f"| {_md_cell(c['text'])} | {_md_cell(c['reason'] or '')} |"
                              for c in items]  # fmt: skip
            if not details and r[metric]["reason"] and (r[metric]["score"] or 0) < 1:
                lines += ["", f"{metric}: {_md_cell(r[metric]['reason'])}"]
        lines += ["", "<details><summary>Answer</summary>", "", r["answer"] or "", "",
                  "</details>", ""]  # fmt: skip

    lines += ["", "## All cases", "",
              "| id | category | faithfulness | relevancy | claims rejected | latency ms |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        scores = ["error" if r[m]["error"] else _fmt(r[m]["score"], 2) for m in METRICS]
        details = r["faithfulness"].get("details")
        rejected = f"{len(details['unsupported'])} / {details['checked']}" if details else "-"
        lines.append(f"| {r['id']} | {r['category']} | " + " | ".join(scores)
                     + f" | {rejected} | {_fmt(r['latency_ms'], 0)} |")  # fmt: skip
    return "\n".join(lines) + "\n"


@app.command("run")
def run_command(
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
    concurrency: int = typer.Option(4, help="Answers generated / judged in parallel"),
    save: bool = typer.Option(True, help="Save the run for `compare`"),
):
    """Generate answers (SEC_RAG_LLM_MODEL) and score Faithfulness and Answer Relevancy."""
    from evals.judge import ClaudeJudge
    from sec_rag.generation.llm import AnswerGenerator

    goldens = [
        gold for gold in load_goldens()
        if (not category or gold.category in category)
        and (not id or gold.id in id)
    ]  # fmt: skip
    if limit:
        goldens = goldens[:limit]
    if not goldens:
        raise typer.Exit("No goldens match the filters.")

    judge, generator = ClaudeJudge(judge_model, effort=effort), AnswerGenerator()
    typer.echo(f"{len(goldens)} questions | generator {generator.model} ({settings.llm_effort}) "
               f"| judge {judge_model} ({effort})", err=True)  # fmt: skip
    started = time.perf_counter()

    def progress(stage: str, done: int, total: int) -> None:
        # One line per stage, rewritten in place; a newline once the stage is finished.
        elapsed = time.perf_counter() - started
        typer.echo(f"\r  {done}/{total} {stage}  ({elapsed:.0f}s)", nl=False, err=True)
        if done == total:
            typer.echo("", err=True)

    result = run(
        generator,
        goldens,
        metric_factory=lambda: deepeval_metrics(judge, include_reason=reasons),
        concurrency=concurrency,
        judge_name=judge_model,
        progress=progress,
        judge_effort=effort,
    )
    typer.echo("", err=True)
    print_report(result)
    cost = "; ".join([
        token_cost("generator", generator.model, generator.input_tokens, generator.output_tokens),
        judge.cost_summary(),
    ])  # fmt: skip
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

    typer.echo(f"\n{'category':>12} {'faithfulness':>18} {'relevancy':>18}")
    for name in [*b["by_category"], "ALL"]:
        new = b["summary"] if name == "ALL" else b["by_category"][name]
        old = a["summary"] if name == "ALL" else a["by_category"].get(name)
        if old is None:
            continue
        cells = []
        for metric in METRICS:
            o, n = old[metric], new[metric]
            cells.append(f"{_fmt(n)}{'' if o is None or n is None else f' ({n - o:+.3f})'}")
        typer.echo(f"{name:>12} " + " ".join(f"{c:>18}" for c in cells))
    typer.echo(f"(changes under ±{NOISE} are likely judge noise)")

    old_rows = {r["id"]: r for r in a["cases"]}
    for metric in METRICS:
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

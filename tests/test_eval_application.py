"""Application evals (correctness, completeness): goldens and runner, with fakes (no API)."""

import json
import re
from types import SimpleNamespace

import pytest

from evals import GOLDENS_DIR, eval_application
from evals.eval_application import (
    COMPLETENESS_STEPS,
    CORRECTNESS_STEPS,
    AppGolden,
    load_goldens,
)
from tests.test_eval_rag_pipeline import FakeGenerator, FakeRetriever
from tests.test_eval_retriever import FakeMetric

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    """Numbers in a text, normalized: "228,000" == "228000" and "$95.0" == "$95"."""
    return {n.replace(",", "").removesuffix(".0") for n in NUMBER.findall(text)}


def full_references() -> dict[str, dict]:
    raw = json.loads((GOLDENS_DIR / "retrieval_goldens.json").read_text(encoding="utf-8"))
    return {r["id"]: r for r in raw}


def test_goldens_load():
    goldens = load_goldens()
    assert len(goldens) == 40  # out-of-scope questions live in the scope eval
    assert all(g.input and g.expected_output for g in goldens)


def test_goldens_match_the_retrieval_goldens():
    """Same 40 questions as the retrieval goldens, so the two files can't drift apart."""
    full = full_references()
    goldens = load_goldens()
    assert {g.id for g in goldens} == set(full)
    for g in goldens:
        assert g.input == full[g.id]["input"], g.id
        assert g.category == full[g.id]["category"], g.id


def test_references_add_no_numbers_beyond_the_full_reference():
    """A trimmed reference keeps only what the question asks for; it must not add figures."""
    full = full_references()
    for g in load_goldens():
        extra = numbers(g.expected_output) - numbers(full[g.id]["expected_output"])
        assert not extra, f"{g.id}: {sorted(extra)}"


def test_metrics_split_accuracy_and_coverage():
    """Correctness ignores what's missing; completeness ignores accuracy."""
    assert any("leaves out" in step and "completeness" in step for step in CORRECTNESS_STEPS)
    assert any("accurate" in step and "correctness" in step for step in COMPLETENESS_STEPS)


def golden(gid="g", question=None):
    return AppGolden(id=gid, category="fact", input=question or f"Question {gid}?",
                     expected_output="Microsoft employed about 228,000 people.")  # fmt: skip


def fake_metric(score):
    """A judge builder returning a FakeMetric with a fixed score; records what it was shown."""
    seen = []

    def build(judge):
        metric = FakeMetric(score=score)
        seen.append(metric)
        return metric

    return seen, build


def test_run_scores_every_metric_on_the_same_answers():
    pipeline = SimpleNamespace(retriever=FakeRetriever(), generator=FakeGenerator())
    seen_c, build_c = fake_metric(0.9)
    seen_p, build_p = fake_metric(0.5)

    result = eval_application.run(
        pipeline, [golden("ok"), golden("broken", "a broken one")], k=3,
        metrics={"correctness": build_c, "completeness": build_p}, judge=None,
        concurrency=2, judge_name="claude-sonnet-5-5",
    )  # fmt: skip

    # Both metrics judge the same answer against the same reference.
    reference = "Microsoft employed about 228,000 people."
    for seen in (seen_c, seen_p):
        assert [m.seen.expected_output for m in seen] == [reference]
        assert [m.seen.actual_output for m in seen] == ["An answer [1]."]

    rows = {r["id"]: r for r in result["cases"]}
    assert rows["ok"]["reference"] == "Microsoft employed about 228,000 people."
    assert rows["broken"]["correctness"]["error"] == "overloaded"  # generation failed: not judged
    summary = result["summary"]
    assert summary["correctness"]["score"] == pytest.approx(0.9)
    assert summary["completeness"]["score"] == pytest.approx(0.5)
    assert summary["completeness"]["passed"] == 0.0  # 0.5 < PASS
    assert summary["correctness"]["errors"] == 1
    assert result["metrics"] == ["correctness", "completeness"]
    assert result["goldens"] == {"total": 2}

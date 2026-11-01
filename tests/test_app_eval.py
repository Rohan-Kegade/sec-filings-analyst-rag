"""Correctness eval: goldens, key facts, number check and runner, with fakes (no API calls)."""

from types import SimpleNamespace

import pytest

from evaluation.app_eval import runner
from evaluation.app_eval.correctness import expected_with_key_facts, key_numbers, number_check
from evaluation.app_eval.goldens import CorrectnessGolden, load_goldens
from tests.test_rag_eval import FakeGenerator, FakeRetriever
from tests.test_retrieval_eval import FakeMetric


def test_goldens_load_with_key_facts():
    goldens = load_goldens()
    assert len(goldens) == 45
    assert all(g.key_facts and g.expected_output for g in goldens)
    assert sum(g.expected_behavior != "answer" for g in goldens) == 5


def test_key_fact_numbers_come_from_the_reference_answer():
    """Key facts must not introduce numbers the (filing-verified) reference doesn't contain."""
    for golden in load_goldens():
        reference = CorrectnessGolden(**{**golden.__dict__, "key_facts": (golden.expected_output,)})
        missing = key_numbers(golden) - key_numbers(reference)
        assert not missing, f"{golden.id}: {sorted(missing)}"


def golden(gid="g", facts=("approximately 228,000 employees",), behavior="answer"):
    return CorrectnessGolden(
        id=gid, category="fact", expected_behavior=behavior, status="draft",
        input=f"Question {gid}?", expected_output="Microsoft employed about 228,000 people.",
        key_facts=tuple(facts),
    )  # fmt: skip


def test_expected_output_carries_the_key_facts():
    text = expected_with_key_facts(golden(facts=("fact one", "fact two")))
    assert text.startswith("Microsoft employed about 228,000 people.")
    assert text.endswith("Key facts:\n- fact one\n- fact two")


@pytest.mark.parametrize("facts, answer, expected", [
    (("approximately 228,000 employees",), "About 228,000 people [1].", (1, 1)),
    (("repurchased $95.0 billion",), "Apple bought back $95 billion.", (1, 1)),
    (("12%, 11% and 11% of revenue",), "Customer A was 12% of revenue.", (1, 2)),
    (("Ernst & Young LLP",), "EY audits Apple.", (0, 0)),  # no numbers to check
    (("approximately 228,000",), None, (0, 1)),
])  # fmt: skip
def test_number_check(facts, answer, expected):
    assert number_check(golden(facts=facts), answer) == expected


def test_run_scores_correctness_against_reference_and_key_facts():
    goldens = [golden("ok"), golden("weak"), golden("broken")]
    goldens[2] = CorrectnessGolden(**{**goldens[2].__dict__, "input": "a broken one"})
    pipeline = SimpleNamespace(retriever=FakeRetriever(), generator=FakeGenerator())
    seen = []

    def factory():
        metric = FakeMetric(score=0.9 if not seen else 0.4)
        seen.append(metric)
        return {"correctness": metric}

    result = runner.run(pipeline, goldens, k=3, metric_factory=factory, concurrency=2,
                        judge_name="claude-sonnet-5-5")  # fmt: skip

    case = seen[0].seen
    assert case.input == "Question ok?" and case.actual_output == "An answer [1]."
    assert case.expected_output == expected_with_key_facts(goldens[0])
    assert len(seen) == 2  # the failed generation wasn't judged

    rows = {r["id"]: r for r in result["cases"]}
    assert rows["broken"]["correctness"]["score"] is None
    assert rows["broken"]["correctness"]["error"] == "overloaded"
    summary = result["summary"]
    assert summary["correctness"] == pytest.approx(0.65)
    assert summary["correct"] == pytest.approx(0.5)  # 0.9 passes, 0.4 doesn't
    assert summary["errors"] == 1

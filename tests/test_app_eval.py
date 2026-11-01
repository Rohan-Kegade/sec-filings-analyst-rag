"""Application evals (correctness, completeness): goldens and runner, with fakes (no API)."""

from types import SimpleNamespace

import pytest

from evaluation.app_eval import runner
from evaluation.app_eval.completeness import expected_with_points
from evaluation.app_eval.correctness import expected_with_key_facts, key_numbers, number_check
from evaluation.app_eval.goldens import AppGolden, load_goldens
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
        reference = AppGolden(**{**golden.__dict__, "key_facts": (golden.expected_output,)})
        missing = key_numbers(golden) - key_numbers(reference)
        assert not missing, f"{golden.id}: {sorted(missing)}"


def golden(gid="g", facts=("approximately 228,000 employees",), behavior="answer"):
    return AppGolden(
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


def test_required_points_exist_and_add_no_new_numbers():
    for g in load_goldens():
        if g.expected_behavior != "answer":
            assert g.required_points == ()  # completeness doesn't apply to out-of-scope cases
            continue
        assert g.required_points, g.id
        as_facts = AppGolden(**{**g.__dict__, "key_facts": g.required_points})
        reference = AppGolden(**{**g.__dict__, "key_facts": (g.expected_output,)})
        missing = key_numbers(as_facts) - key_numbers(reference)
        assert not missing, f"{g.id}: {sorted(missing)}"


def test_expected_output_carries_the_required_points():
    g = AppGolden(**{**golden().__dict__, "required_points": ("part one", "part two")})
    assert expected_with_points(g).endswith("Required points:\n- part one\n- part two")


def fake_metric(score):
    """An AppMetric whose judge is a FakeMetric with a fixed score; records what it was shown."""
    seen = []

    def build(judge):
        metric = FakeMetric(score=score)
        seen.append(metric)
        return metric

    return seen, build


def test_run_scores_every_metric_on_the_same_answers():
    ok = AppGolden(**{**golden("ok").__dict__, "required_points": ("the total",)})
    broken = AppGolden(**{**golden("broken").__dict__, "input": "a broken one",
                          "required_points": ("the total",)})  # fmt: skip
    oos = golden("oos", behavior="abstain")  # no required points
    pipeline = SimpleNamespace(retriever=FakeRetriever(), generator=FakeGenerator())
    seen_c, build_c = fake_metric(0.9)
    seen_p, build_p = fake_metric(0.5)
    metrics = {
        "correctness": runner.AppMetric(build_c, expected_with_key_facts, lambda g: True),
        "completeness": runner.AppMetric(build_p, expected_with_points,
                                         lambda g: bool(g.required_points)),
    }  # fmt: skip

    result = runner.run(pipeline, [ok, broken, oos], k=3, metrics=metrics, judge=None,
                        concurrency=2, judge_name="claude-sonnet-5-5")  # fmt: skip

    # Each metric sees its own expected output for the same generated answer.
    assert {m.seen.expected_output for m in seen_c} == {
        expected_with_key_facts(ok), expected_with_key_facts(oos)}  # fmt: skip
    assert [m.seen.expected_output for m in seen_p] == [expected_with_points(ok)]
    assert {m.seen.actual_output for m in seen_c + seen_p} == {"An answer [1]."}  # one answer

    rows = {r["id"]: r for r in result["cases"]}
    assert rows["oos"]["completeness"]["applies"] is False  # not judged for out-of-scope
    assert rows["broken"]["correctness"]["error"] == "overloaded"
    summary = result["summary"]
    assert summary["correctness"]["score"] == pytest.approx(0.9)
    assert summary["completeness"]["score"] == pytest.approx(0.5)
    assert summary["completeness"]["passed"] == 0.0  # 0.5 < PASS
    assert summary["completeness"]["applies"] == 2  # ok and broken, not oos
    assert summary["correctness"]["errors"] == 1
    assert result["metrics"] == ["correctness", "completeness"]

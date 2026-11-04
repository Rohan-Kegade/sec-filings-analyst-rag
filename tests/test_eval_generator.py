"""DeepEval generator eval: goldens, excerpt parsing and runner, with fakes (no API calls)."""

import asyncio
from types import SimpleNamespace

import pytest

from evals import eval_generator
from evals.eval_generator import GeneratorGolden, load_goldens, to_chunks
from tests.test_eval_retriever import FakeMetric


def test_goldens_file_loads_and_every_excerpt_parses():
    goldens = load_goldens()
    assert len(goldens) == 54
    assert sum(g.expected_behavior == "abstain" for g in goldens) == 4
    for golden in goldens:
        chunks = to_chunks(golden)  # raises if any excerpt lacks its chunk header
        assert len(chunks) == len(golden.context) >= 4


def golden(gid="g1", category="fact", context=None):
    context = context or (
        "Microsoft Corporation (MSFT) 10-K FY2024 - Item 1: Business\n\nWe employed 228,000.",
    )
    return GeneratorGolden(id=gid, category=category, input=f"Question {gid}?",
                           context=tuple(context))  # fmt: skip


def test_no_answer_category_means_abstain():
    assert golden(category="no_answer").expected_behavior == "abstain"
    assert golden(category="partial").expected_behavior == "answer"


def test_to_chunks_restores_metadata_from_the_header():
    chunk = to_chunks(golden())[0].chunk
    assert (chunk.company_name, chunk.ticker, chunk.fiscal_year) == (
        "Microsoft Corporation", "MSFT", 2024
    )  # fmt: skip
    assert (chunk.item, chunk.item_title) == ("1", "Business")
    assert chunk.chunk_id == "g1-0" and chunk.chunk_index == 0


def test_to_chunks_rejects_excerpts_without_header():
    with pytest.raises(ValueError, match="header"):
        to_chunks(golden(context=("no header here",)))


@pytest.mark.parametrize("answer, expected", [
    ("The excerpts do not disclose Vision Pro revenue.", True),
    ("The provided excerpts don't include a breakdown by function.", True),
    ("There is no information about attrition in the excerpts.", True),
    ("Microsoft employed approximately 228,000 people [1].", False),
])  # fmt: skip
def test_abstain_heuristic(answer, expected):
    assert bool(eval_generator.ABSTAIN.search(answer)) is expected


class FakeGenerator:
    model = "claude-haiku-5-5"

    def __init__(self):
        self.calls = []

    def generate(self, question, chunks, history=None):
        self.calls.append((question, chunks))
        if question == "Question broken?":
            raise RuntimeError("overloaded")
        if question == "Question empty?":
            return "The excerpts do not contain this information."
        return "Microsoft employed approximately 228,000 people [1]."


def test_run_generates_from_frozen_context_and_scores():
    goldens = [golden("ok"), golden("empty", category="no_answer"), golden("broken")]
    generator, seen = FakeGenerator(), []

    def factory():
        metrics = {"faithfulness": FakeMetric(score=1.0), "relevancy": FakeMetric(score=0.5)}
        seen.append(metrics)
        return metrics

    result = eval_generator.run(generator, goldens, metric_factory=factory, concurrency=2,
                        judge_name="claude-sonnet-5-5")  # fmt: skip

    # The generator got the golden's own excerpts, and the judge got those same excerpts.
    question, chunks = generator.calls[0]
    assert [rc.chunk.text for rc in chunks] == list(goldens[0].context)
    case = seen[0]["faithfulness"].seen
    assert case.retrieval_context == list(goldens[0].context)
    assert case.actual_output.startswith("Microsoft employed")

    rows = {r["id"]: r for r in result["cases"]}
    assert rows["broken"]["generation_error"] == "overloaded"
    assert rows["broken"]["faithfulness"]["score"] is None  # nothing to judge
    assert rows["empty"]["abstained"] is True and rows["ok"]["abstained"] is False

    summary = result["summary"]
    assert summary["faithfulness"] == 1.0 and summary["relevancy"] == 0.5
    assert summary["generation_errors"] == 1
    # The failed generation is not counted again as two judge errors.
    assert summary["faithfulness_errors"] == summary["relevancy_errors"] == 0
    assert summary["abstained"] == 1.0  # the one no-answer case abstained
    assert len(seen) == 2  # the failed generation was not sent to the judge
    assert result["settings"]["generator"] == "claude-haiku-5-5"


class FakeFaithfulness(FakeMetric):
    """Records claims and verdicts the way DeepEval's FaithfulnessMetric does."""

    async def a_measure(self, test_case, _show_indicator=True):
        await super().a_measure(test_case, _show_indicator)
        self.claims = ["Microsoft employed 228,000 people.", "Most work in Redmond.",
                       "Headcount grew."]  # fmt: skip
        self.verdicts = [
            SimpleNamespace(verdict="yes", reason=None),
            SimpleNamespace(verdict=SimpleNamespace(value="no"), reason="Not in the excerpts."),
            SimpleNamespace(verdict="borderline", reason="Implied only."),
        ]


def faithfulness_factory():
    return {"faithfulness": FakeFaithfulness(score=0.5), "relevancy": FakeMetric(score=1.0)}


def test_metric_details_lists_rejected_claims_only():
    metric = FakeFaithfulness(score=0.5)
    asyncio.run(metric.a_measure(None))
    details = eval_generator.metric_details("faithfulness", metric)
    # Only "no" lowers the score; "borderline" passes in DeepEval, so it is kept apart.
    assert details == {
        "checked": 3,
        "unsupported": [{"text": "Most work in Redmond.", "reason": "Not in the excerpts."}],
        "borderline": [{"text": "Headcount grew.", "reason": "Implied only."}],
        "matched": True,
    }


def test_metric_details_without_matching_claims_still_lists_rejections():
    # The judge returned 3 verdicts for 2 claims: the score uses the verdicts, so keep them.
    metric = FakeFaithfulness(score=0.5)
    asyncio.run(metric.a_measure(None))
    metric.claims = metric.claims[:2]
    details = eval_generator.metric_details("faithfulness", metric)
    assert details["checked"] == 3 and details["matched"] is False
    assert details["unsupported"] == [{"text": None, "reason": "Not in the excerpts."}]
    # Metrics without per-item verdicts (or a mismatch) give no details.
    assert eval_generator.metric_details("relevancy", FakeMetric(score=1.0)) is None


def test_run_records_details_and_reports_progress():
    events = []
    result = eval_generator.run(
        FakeGenerator(), [golden("a"), golden("b"), golden("broken")],
        metric_factory=faithfulness_factory, concurrency=2, judge_name="claude-haiku-5-5",
        progress=lambda stage, done, total: events.append((stage, done, total)),
        judge_effort="high",
    )  # fmt: skip
    rows = {r["id"]: r for r in result["cases"]}
    assert rows["a"]["faithfulness"]["details"]["unsupported"][0]["text"] == "Most work in Redmond."
    assert "details" not in rows["a"]["relevancy"]
    assert result["settings"]["judge_effort"] == "high"
    # Generation counts every golden; judging skips the one whose generation failed.
    assert events[0] == ("generated", 0, 3) and ("generated", 3, 3) in events
    assert ("judged", 0, 2) in events and events[-1] == ("judged", 2, 2)


def test_markdown_report_shows_rejected_claims_and_no_answer_cases(tmp_path):
    goldens = [golden("weak"), golden("empty", category="no_answer"), golden("broken")]
    result = eval_generator.run(FakeGenerator(), goldens, metric_factory=faithfulness_factory,
                                concurrency=1, judge_name="claude-haiku-5-5")  # fmt: skip
    report = eval_generator.markdown_report(result, cost="judge tokens: 1 in / 1 out (~$0.00)")

    assert report.startswith("# Generator eval: ")
    assert "| cost | judge tokens: 1 in / 1 out (~$0.00) |" in report
    assert "## No-answer cases" in report and "| empty | yes |" in report
    assert "### weak (fact)" in report
    assert "faithfulness - Claims not supported (lower the score):" in report
    assert "| Most work in Redmond. | Not in the excerpts. |" in report
    assert "| Headcount grew. | Implied only. |" in report
    assert "| weak | fact | 0.50 | 1.00 | 1 / 3 |" in report
    assert "| broken | fact | error | error | - |" in report

    from evals import runs
    path = runs.save_report(report, tmp_path / "run.json")
    assert path.suffix == ".md" and path.read_text(encoding="utf-8") == report

"""DeepEval generator eval: goldens, excerpt parsing and runner, with fakes (no API calls)."""

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


def golden(gid="g1", behavior="answer", context=None, chunk_ids=None):
    context = context or (
        "Microsoft Corporation (MSFT) 10-K FY2024 - Item 1: Business\n\nWe employed 228,000.",
    )
    return GeneratorGolden(
        id=gid, category="fact", stress=None, expected_behavior=behavior,
        input=f"Question {gid}?", context=tuple(context),
        context_chunks=tuple(chunk_ids or ["0000950170-24-087843-1-20"]), expected_output="",
    )  # fmt: skip


def test_to_chunks_restores_metadata_from_the_header():
    chunk = to_chunks(golden())[0].chunk
    assert (chunk.company_name, chunk.ticker, chunk.fiscal_year) == (
        "Microsoft Corporation", "MSFT", 2024
    )  # fmt: skip
    assert (chunk.item, chunk.item_title) == ("1", "Business")
    assert chunk.accession_number == "0000950170-24-087843" and chunk.chunk_index == 20


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
    goldens = [golden("ok"), golden("empty", behavior="abstain"), golden("broken")]
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
    assert summary["abstained"] == 1.0  # the one no-answer case abstained
    assert len(seen) == 2  # the failed generation was not sent to the judge
    assert result["settings"]["generator"] == "claude-haiku-5-5"

"""RAG triad eval: questions, pipeline run, diagnosis and scoring, with fakes (no API calls)."""

from types import SimpleNamespace

import pytest

from evaluation.rag_eval import runner
from evaluation.rag_eval.questions import Question, load_questions
from sec_rag.models import RetrievedChunk
from sec_rag.pipeline import NO_RESULTS
from tests.test_keyword_index import make_chunk
from tests.test_retrieval_eval import FakeMetric


def test_questions_are_the_40_goldens_plus_out_of_scope():
    questions = load_questions()
    assert len(questions) == 45
    oos = [q for q in questions if q.category == "out_of_scope"]
    assert {q.expected_behavior for q in oos} == {"abstain", "decline_advice"}
    assert all(q.expected_behavior == "answer" for q in questions if q.category != "out_of_scope")


class FakeRetriever:
    mode = "hybrid"

    def retrieve(self, question, k):
        if "2019" in question:
            return []  # year not downloaded: nothing retrieved
        return [RetrievedChunk(chunk=make_chunk(f"c{i}", f"excerpt {i}"), score=1.0)
                for i in range(k)]  # fmt: skip


class FakeGenerator:
    model = "claude-haiku-5-5"

    def generate(self, question, chunks, history=None):
        if "broken" in question:
            raise RuntimeError("overloaded")
        if "GOOGL" in question:
            return "The excerpts do not contain information about GOOGL."
        if "buy" in question:
            return "I can't give investment advice, but the filing describes these risks [1]."
        return "An answer [1]."


QUESTIONS = [
    Question("good", "fact", "answer", "MSFT employees?"),
    Question("weak-retrieval", "fact", "answer", "AAPL margins?"),
    Question("googl", "out_of_scope", "abstain", "GOOGL revenue?"),
    Question("y2019", "out_of_scope", "abstain", "AAPL risks in 2019?"),
    Question("advice", "out_of_scope", "decline_advice", "Should I buy NVDA?"),
    Question("broken", "fact", "answer", "a broken one"),
]


def test_run_scores_the_triad_and_diagnoses_failures():
    pipeline = SimpleNamespace(retriever=FakeRetriever(), generator=FakeGenerator())
    seen = []

    def factory():
        n = len(seen)
        context = 0.1 if n == 1 else 0.6  # the 2nd scored question has poor retrieval
        metrics = {"context_relevance": FakeMetric(score=context),
                   "groundedness": FakeMetric(score=1.0),
                   "answer_relevance": FakeMetric(score=0.9)}  # fmt: skip
        seen.append(metrics)
        return metrics

    result = runner.run(pipeline, QUESTIONS, k=3, metric_factory=factory, concurrency=2,
                        judge_name="claude-sonnet-5-5")  # fmt: skip
    rows = {r["id"]: r for r in result["cases"]}

    # The judge sees exactly what the pipeline retrieved and generated.
    case = seen[0]["groundedness"].seen
    assert case.retrieval_context == ["excerpt 0", "excerpt 1", "excerpt 2"]
    assert case.actual_output == "An answer [1]."

    assert rows["good"]["diagnosis"] == "ok"
    assert rows["weak-retrieval"]["diagnosis"] == "retrieval"
    assert rows["y2019"]["answer"] == NO_RESULTS and rows["y2019"]["diagnosis"] == "no_context"
    assert rows["broken"]["diagnosis"] == "generation_error"
    assert len(seen) == 4  # no judge calls for the empty-context or failed cases

    assert rows["googl"]["handled"] and rows["y2019"]["handled"] and rows["advice"]["handled"]
    assert rows["good"]["handled"] is None
    assert result["summary"]["handled"] == 1.0
    assert result["summary"]["groundedness"] == pytest.approx(1.0)
    assert result["settings"]["retrieval_mode"] == "hybrid"
    assert result["settings"]["generator"] == "claude-haiku-5-5"


@pytest.mark.parametrize("answer, expected", [
    ("I can't tell you whether to buy NVDA. That depends on your goals.", True),
    ("I can't give investment advice, but the filing lists these risks [1].", True),
    ("NVDA is a strong buy given its Data Center growth [2].", False),
])  # fmt: skip
def test_advice_check(answer, expected):
    assert bool(runner.ADVICE.search(answer)) is expected


@pytest.mark.parametrize("scores, expected", [
    ((0.6, 1.0, 0.9), "ok"),
    ((0.2, 1.0, 0.9), "retrieval"),
    ((0.6, 0.5, 0.9), "not_grounded"),
    ((0.6, 1.0, 0.4), "off_target"),
])  # fmt: skip
def test_diagnose_checks_links_in_pipeline_order(scores, expected):
    row = {"generation_error": None, "sources": ["s"]}
    for metric, score in zip(runner.METRICS, scores):
        row[metric] = {"score": score}
    assert runner.diagnose(row) == expected

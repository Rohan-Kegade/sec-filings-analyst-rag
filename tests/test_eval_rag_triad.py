"""RAG triad eval: questions, pipeline run, diagnosis and scoring, with fakes (no API calls)."""

from types import SimpleNamespace

import pytest

from evals import eval_rag_triad
from evals.eval_rag_triad import Question, load_questions
from sec_rag.models import RetrievedChunk
from sec_rag.pipeline import NO_RESULTS
from tests.test_eval_retriever import FakeMetric
from tests.test_keyword_index import make_chunk


def test_questions_are_the_40_goldens():
    questions = load_questions()
    assert len(questions) == 40  # out-of-scope questions live in the scope eval
    assert "out_of_scope" not in {q.category for q in questions}


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
    Question("good", "fact", "MSFT employees?"),
    Question("weak-retrieval", "fact", "AAPL margins?"),
    Question("y2019", "fact", "AAPL risks in 2019?"),  # retrieval finds nothing
    Question("broken", "fact", "a broken one"),
]


def test_run_scores_the_triad_and_diagnoses_failures():
    pipeline = SimpleNamespace(retriever=FakeRetriever(), generator=FakeGenerator())
    seen = []

    def factory():
        n = len(seen)
        context = 0.1 if n == 1 else 0.6  # the 2nd scored question has poor retrieval
        metrics = {"context_relevance": FakeMetric(score=context),
                   "faithfulness": FakeMetric(score=1.0),
                   "answer_relevance": FakeMetric(score=0.9)}  # fmt: skip
        seen.append(metrics)
        return metrics

    result = eval_rag_triad.run(pipeline, QUESTIONS, k=3, metric_factory=factory, concurrency=2,
                        judge_name="claude-sonnet-5-5")  # fmt: skip
    rows = {r["id"]: r for r in result["cases"]}

    # The judge sees exactly what the pipeline retrieved and generated.
    case = seen[0]["faithfulness"].seen
    assert case.retrieval_context == ["excerpt 0", "excerpt 1", "excerpt 2"]
    assert case.actual_output == "An answer [1]."

    assert rows["good"]["diagnosis"] == "ok"
    assert rows["weak-retrieval"]["diagnosis"] == "retrieval"
    assert rows["y2019"]["answer"] == NO_RESULTS and rows["y2019"]["diagnosis"] == "no_context"
    assert rows["broken"]["diagnosis"] == "generation_error"
    assert len(seen) == 2  # no judge calls for the empty-context or failed cases

    assert result["summary"]["faithfulness"] == pytest.approx(1.0)
    assert result["settings"]["retrieval_mode"] == "hybrid"
    assert result["settings"]["generator"] == "claude-haiku-5-5"


@pytest.mark.parametrize("scores, expected", [
    ((0.6, 1.0, 0.9), "ok"),
    ((0.2, 1.0, 0.9), "retrieval"),
    ((0.6, 0.5, 0.9), "not_faithful"),
    ((0.6, 1.0, 0.4), "off_target"),
])  # fmt: skip
def test_diagnose_checks_links_in_pipeline_order(scores, expected):
    row = {"generation_error": None, "sources": ["s"]}
    for metric, score in zip(eval_rag_triad.METRICS, scores):
        row[metric] = {"score": score}
    assert eval_rag_triad.diagnose(row) == expected

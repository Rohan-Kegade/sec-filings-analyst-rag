"""RAG pipeline eval (RAG triad): questions, pipeline run, diagnosis and scoring, with fakes."""

from types import SimpleNamespace

import pytest

from evals import eval_rag_pipeline
from evals.eval_rag_pipeline import Question, load_questions
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

    result = eval_rag_pipeline.run(pipeline, QUESTIONS, k=3, metric_factory=factory, concurrency=2,
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


def triad_factory(context=0.6):
    from tests.test_eval_generator import FakeFaithfulness

    return lambda: {"context_relevance": FakeMetric(score=context),
                    "faithfulness": FakeFaithfulness(score=0.5),
                    "answer_relevance": FakeMetric(score=0.9)}  # fmt: skip


def test_run_records_settings_progress_and_rejected_claims():
    pipeline = SimpleNamespace(retriever=FakeRetriever(), generator=FakeGenerator())
    events = []
    result = eval_rag_pipeline.run(
        pipeline, QUESTIONS, k=3, metric_factory=triad_factory(), concurrency=2,
        judge_name="claude-haiku-5-5", judge_effort="high",
        progress=lambda stage, done, total: events.append((stage, done, total)),
    )  # fmt: skip
    s = result["settings"]
    # Everything from .env that shapes retrieval or judging is recorded with the run.
    for key in ("hybrid_candidates", "rrf_k", "rerank", "min_chunks_per_entity",
                "max_comparison_chunks", "generator_max_tokens"):  # fmt: skip
        assert key in s
    assert s["rerank"] == "off"  # the fake retriever doesn't rerank
    assert s["judge_effort"] == "high" and result["questions"]["total"] == 4

    # Each stage reports progress up to its total; judging skips no-context and failed cases.
    assert ("retrieved", 4, 4) in events and ("generated", 4, 4) in events
    assert events[-1] == ("judged", 2, 2)

    rows = {r["id"]: r for r in result["cases"]}
    unsupported = rows["good"]["faithfulness"]["details"]["unsupported"]
    assert unsupported == [{"text": "Most work in Redmond.", "reason": "Not in the excerpts."}]
    assert rows["good"]["generation_ms"] is not None
    assert result["summary"]["retrieval_ms"] is not None and result["summary"]["judge_errors"] == 0


def test_markdown_report_lists_failing_cases_with_rejected_claims():
    pipeline = SimpleNamespace(retriever=FakeRetriever(), generator=FakeGenerator())
    result = eval_rag_pipeline.run(pipeline, QUESTIONS, k=3, metric_factory=triad_factory(),
                                   concurrency=1, judge_name="claude-haiku-5-5")  # fmt: skip
    report = eval_rag_pipeline.markdown_report(result, cost="judge tokens: 1 in / 1 out (~$0.00)")

    assert report.startswith("# RAG pipeline eval: ")
    assert "| rrf_k |" in report and "| rerank | off |" in report
    assert "| cost | judge tokens: 1 in / 1 out (~$0.00) |" in report
    assert "## Diagnosis" in report and "| not_faithful | 2 |" in report
    assert "### good (fact): not_faithful" in report
    assert "| Most work in Redmond. | Not in the excerpts. |" in report
    assert "### y2019 (fact): no_context" in report and "**retrieved:** nothing" in report
    assert "### broken (fact): generation_error" in report
    assert "| good | fact | 0.60 | 0.50 | 0.90 | not_faithful | 1 / 3 |" in report


@pytest.mark.parametrize("scores, expected", [
    ((0.6, 1.0, 0.9), "ok"),
    ((0.2, 1.0, 0.9), "retrieval"),
    ((0.6, 0.5, 0.9), "not_faithful"),
    ((0.6, 1.0, 0.4), "off_target"),
])  # fmt: skip
def test_diagnose_checks_links_in_pipeline_order(scores, expected):
    row = {"generation_error": None, "sources": ["s"]}
    for metric, score in zip(eval_rag_pipeline.METRICS, scores):
        row[metric] = {"score": score}
    assert eval_rag_pipeline.diagnose(row) == expected

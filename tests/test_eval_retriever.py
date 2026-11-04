"""DeepEval retrieval eval: goldens, Claude judge and runner, with fakes (no API calls)."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from evals import eval_retriever
from evals.eval_retriever import Golden, load_goldens
from evals.judge import ClaudeJudge
from sec_rag.config import Settings, settings
from sec_rag.generation.llm import FALLBACK_BETA
from sec_rag.models import RetrievedChunk
from tests.test_keyword_index import make_chunk


def test_goldens_file_loads():
    goldens = load_goldens()
    assert len(goldens) == 40
    assert all(g.input and g.expected_output for g in goldens)
    assert {g.category for g in goldens} == {
        "fact", "qualitative", "exact_term", "comparison", "follow_up"
    }  # fmt: skip


# --- judge -------------------------------------------------------------------------------


class Verdict(BaseModel):
    verdict: str


def fake_message(parsed=None, text="", stop_reason="end_turn"):
    return SimpleNamespace(
        parsed_output=parsed,
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=10),
    )


class FakeMessages:
    def __init__(self, message):
        self.message, self.calls = message, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.message

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.message


class FakeAsyncMessages(FakeMessages):
    async def parse(self, **kwargs):
        return FakeMessages.parse(self, **kwargs)

    async def create(self, **kwargs):
        return FakeMessages.create(self, **kwargs)


def make_judge(message):
    sync, aio = FakeMessages(message), FakeAsyncMessages(message)
    judge = ClaudeJudge(
        client=SimpleNamespace(beta=SimpleNamespace(messages=sync)),
        async_client=SimpleNamespace(beta=SimpleNamespace(messages=aio)),
    )
    return judge, sync, aio


def test_judge_returns_schema_instance_and_sends_no_temperature():
    judge, sync, _ = make_judge(fake_message(parsed=Verdict(verdict="yes")))
    assert judge.generate("prompt", schema=Verdict) == Verdict(verdict="yes")
    request = sync.calls[0]
    assert request["model"] == settings.judge_model
    assert request["output_format"] is Verdict
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": settings.judge_effort}
    assert request["betas"] == [FALLBACK_BETA] and request["fallbacks"] == "default"
    assert "temperature" not in request


def test_judge_model_and_effort_come_from_env(monkeypatch):
    monkeypatch.setenv("SEC_RAG_JUDGE_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("SEC_RAG_JUDGE_EFFORT", "high")
    configured = Settings(_env_file=None)
    assert (configured.judge_model, configured.judge_effort) == ("claude-opus-5-5", "high")


def test_judge_caps_concurrent_requests():
    """Metrics may fire many judge calls at once; at most max_concurrent_requests run."""
    in_flight = peak = 0

    class SlowMessages:
        async def create(self, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return fake_message(text="ok")

    judge = ClaudeJudge(
        client=SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(None))),
        async_client=SimpleNamespace(beta=SimpleNamespace(messages=SlowMessages())),
        max_concurrent_requests=3,
    )

    async def burst():
        return await asyncio.gather(*(judge.a_generate("p") for _ in range(12)))

    assert asyncio.run(burst()) == ["ok"] * 12
    assert peak == 3


def test_judge_async_and_text_paths_count_tokens():
    judge, _, aio = make_judge(fake_message(text="plain answer"))
    assert asyncio.run(judge.a_generate("prompt")) == "plain answer"
    assert "output_format" not in aio.calls[0]
    assert judge.input_tokens == 100 and judge.output_tokens == 10
    assert "~$" in judge.cost_summary()


def test_judge_raises_on_refusal():
    judge, _, _ = make_judge(fake_message(stop_reason="refusal"))
    with pytest.raises(RuntimeError, match="refusal"):
        judge.generate("prompt", schema=Verdict)


# --- runner ------------------------------------------------------------------------------


class FakeRetriever:
    def retrieve(self, question, k):
        return [RetrievedChunk(chunk=make_chunk(f"c{i}", f"text {i}"), score=1.0) for i in range(k)]


class FakeMetric:
    def __init__(self, score=None, error=None):
        self.score_value, self.error = score, error
        self.score = self.reason = None
        self.seen = None

    async def a_measure(self, test_case, _show_indicator=True):
        self.seen = test_case
        if self.error:
            raise RuntimeError(self.error)
        self.score, self.reason = self.score_value, "because"


GOLDENS = [
    Golden(id="a", category="fact", input="Q a", expected_output="A a"),
    Golden(id="b", category="fact", input="Q b", expected_output="A b"),
    Golden(id="c", category="comparison", input="Q c", expected_output="A c"),
]


def test_run_scores_each_golden_and_records_errors():
    created = []

    def factory():
        n = len(created)
        metrics = {
            "recall": FakeMetric(score=[1.0, 0.5, 0.0][n]),
            "precision": FakeMetric(error="judge failed") if n == 2 else FakeMetric(score=0.8),
        }
        created.append(metrics)
        return metrics

    result = eval_retriever.run({"hybrid": FakeRetriever()}, GOLDENS, k=3, metric_factory=factory,
                        concurrency=2, judge_name="claude-sonnet-5-5")  # fmt: skip

    # The test case holds the question, the golden's reference and only retrieved text.
    case = created[0]["recall"].seen
    assert case.input == "Q a" and case.expected_output == "A a"
    assert case.retrieval_context == ["text 0", "text 1", "text 2"]
    assert case.actual_output is None  # retriever tested in isolation

    summary = result["summary"]["hybrid"]
    assert summary["recall"] == pytest.approx(0.5)
    assert summary["precision"] == pytest.approx(0.8)  # the failed case is left out
    assert summary["precision_errors"] == 1
    assert result["by_category"]["hybrid"]["fact"]["recall"] == pytest.approx(0.75)
    assert result["goldens"] == {"total": 3}
    rows = {r["id"]: r for r in result["cases"]["hybrid"]}
    assert rows["c"]["precision"]["error"] == "judge failed"
    assert len(rows["a"]["retrieved"]) == 3


def test_run_reports_progress_per_mode():
    calls = []

    def factory():
        return {"recall": FakeMetric(score=1.0), "precision": FakeMetric(score=1.0)}

    eval_retriever.run({"dense": FakeRetriever(), "hybrid": FakeRetriever()}, GOLDENS, k=2,
                       metric_factory=factory, concurrency=2, judge_name="j",
                       progress=lambda mode, done, total: calls.append((mode, done, total)))  # fmt: skip
    assert calls == [("dense", n, 3) for n in range(4)] + [("hybrid", n, 3) for n in range(4)]


def test_markdown_report_and_save(tmp_path):
    def factory():
        return {"recall": FakeMetric(score=0.0), "precision": FakeMetric(error="judge | failed")}

    result = eval_retriever.run({"hybrid": FakeRetriever()}, GOLDENS, k=2, metric_factory=factory,
                                concurrency=2, judge_name="claude-haiku-5-5")  # fmt: skip
    report = eval_retriever.markdown_report(result, cost="judge tokens: 1 in / 1 out (~$0.00)")

    assert report.startswith("# Retriever eval: ")
    assert "| judge | claude-haiku-5-5 |" in report
    assert "| cost | judge tokens: 1 in / 1 out (~$0.00) |" in report
    assert "| hybrid | 0.000 | - | 3 |" in report  # recall 0, precision all errors
    assert "## hybrid: cases with recall < 0.5" in report
    assert "| a | fact | 0.00 | because |" in report
    assert "| c | comparison | 0.00 | error |" in report

    assert "where the final chunks came from" not in report  # no hybrid trace here

    path = eval_retriever.runs.save_report(report, tmp_path / "2026-01-01T00-00-00.json")
    assert path.name == "2026-01-01T00-00-00.md"
    assert path.read_text(encoding="utf-8") == report


class TracedRetriever(FakeRetriever):
    mode = "hybrid"

    def hybrid_trace(self, question, k):
        results = self.retrieve(question, k)
        results[0].ranks, results[1].ranks = {"dense": 1, "keyword": 3}, {"keyword": 1}
        dropped = RetrievedChunk(chunk=make_chunk("x9", "dropped"), score=0.0, ranks={"dense": 2})
        return results, [dropped]


def test_hybrid_rows_and_report_show_sources_and_drops():
    def factory():
        return {"recall": FakeMetric(score=0.0), "precision": FakeMetric(score=0.5)}

    result = eval_retriever.run({"hybrid": TracedRetriever()}, GOLDENS[:1], k=2,
                                metric_factory=factory, concurrency=1, judge_name="j")  # fmt: skip
    row = result["cases"]["hybrid"][0]
    assert row["sources"] == [{"dense": 1, "keyword": 3}, {"keyword": 1}]
    assert row["dropped"][0]["ranks"] == {"dense": 2}

    report = eval_retriever.markdown_report(result)
    # k=2: c0 is dense #1 but keyword #3 (outside keyword's top 2); c1 is keyword #1.
    assert "| dense top 2 only | 1 | 50% |" in report
    assert "| keyword top 2 only | 1 | 50% |" in report
    assert "| both top 2s | 0 | 0% |" in report and "| neither top 2 | 0 | 0% |" in report
    assert "(c0)" not in report and "x9" not in report  # chunks stay in the JSON only


def test_source_mix_counts_against_each_search_top_k():
    rows = [{"sources": [{"dense": 1, "keyword": 2}, {"dense": 3}, {"keyword": 9},
                         {"dense": 14, "keyword": 20}]}]  # fmt: skip
    assert eval_retriever.source_mix(rows, k=8) == {
        "both top 8s": 1, "dense top 8 only": 1, "keyword top 8 only": 0, "neither top 8": 2,
    }  # fmt: skip
    assert eval_retriever.source_mix([{"id": "no trace"}], k=8) is None



def test_token_cost_prices_known_models_only():
    from evals.judge import token_cost

    assert token_cost("judge", "claude-haiku-5-5", 1_000_000, 1_000_000) == (
        "judge tokens: 1,000,000 in / 1,000,000 out (~$0.60)"
    )
    assert token_cost("generator", "some-other-model", 10, 5) == (
        "generator tokens: 10 in / 5 out"
    )

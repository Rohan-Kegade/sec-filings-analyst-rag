"""DeepEval retrieval eval: goldens, Claude judge and runner, with fakes (no API calls)."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from evaluation.retrieval_eval import runner
from evaluation.retrieval_eval.goldens import Golden, load_goldens
from evaluation.retrieval_eval.judge import ClaudeJudge
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
    assert request["model"] == "claude-sonnet-5-5"
    assert request["output_format"] is Verdict
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "medium"}
    assert request["betas"] == [FALLBACK_BETA] and request["fallbacks"] == "default"
    assert "temperature" not in request


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
    Golden(id="a", category="fact", status="approved", input="Q a", expected_output="A a"),
    Golden(id="b", category="fact", status="draft", input="Q b", expected_output="A b"),
    Golden(id="c", category="comparison", status="draft", input="Q c", expected_output="A c"),
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

    result = runner.run({"hybrid": FakeRetriever()}, GOLDENS, k=3, metric_factory=factory,
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
    assert result["goldens"] == {"total": 3, "approved": 1}
    rows = {r["id"]: r for r in result["cases"]["hybrid"]}
    assert rows["c"]["precision"]["error"] == "judge failed"
    assert len(rows["a"]["retrieved"]) == 3

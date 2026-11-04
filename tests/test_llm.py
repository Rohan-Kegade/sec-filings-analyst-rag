"""AnswerGenerator against a fake Anthropic client (no API calls)."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from sec_rag.generation.llm import FALLBACK_BETA, REFUSAL_MESSAGE, AnswerGenerator, GenerationError


class FakeStream:
    def __init__(self, texts, stop_reason="end_turn", error=None):
        self.texts, self.stop_reason, self.error = texts, stop_reason, error

    def __enter__(self):
        if self.error:
            raise self.error
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        yield from self.texts

    def get_final_message(self):
        content = [SimpleNamespace(type="thinking", thinking="")]
        content += [SimpleNamespace(type="text", text=t) for t in self.texts]
        return SimpleNamespace(stop_reason=self.stop_reason, content=content)


class FakeClient:
    def __init__(self, stream):
        self.stream_obj, self.kwargs = stream, None
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.kwargs = kwargs
        return self.stream_obj


def test_request_uses_adaptive_thinking_effort_and_fallback():
    client = FakeClient(FakeStream(["Hi"]))
    AnswerGenerator(client=client, model="claude-opus-5-5").complete("system", "prompt")
    assert client.kwargs["model"] == "claude-opus-5-5"
    assert client.kwargs["thinking"] == {"type": "adaptive"}
    assert client.kwargs["output_config"]["effort"] == "medium"
    assert client.kwargs["betas"] == [FALLBACK_BETA]
    assert client.kwargs["fallbacks"] == "default"


def test_complete_joins_text_blocks_only():
    generator = AnswerGenerator(client=FakeClient(FakeStream(["Apple ", "relies on [1]."])))
    assert generator.complete("system", "prompt") == "Apple relies on [1]."


def test_stream_yields_text():
    generator = AnswerGenerator(client=FakeClient(FakeStream(["a", "b"])))
    assert list(generator.complete_stream("system", "prompt")) == ["a", "b"]


def test_refusal_without_text_returns_message():
    client = FakeClient(FakeStream([], stop_reason="refusal"))
    generator = AnswerGenerator(client=client)
    assert generator.complete("system", "prompt") == REFUSAL_MESSAGE
    assert list(generator.complete_stream("system", "prompt")) == [REFUSAL_MESSAGE]


def test_api_errors_become_generation_errors():
    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    generator = AnswerGenerator(client=FakeClient(FakeStream([], error=error)))
    with pytest.raises(GenerationError):
        generator.complete("system", "prompt")
    with pytest.raises(GenerationError):
        list(generator.complete_stream("system", "prompt"))


class UsageStream(FakeStream):
    def get_final_message(self):
        message = super().get_final_message()
        message.usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        return message


def test_token_use_is_counted_across_calls():
    generator = AnswerGenerator(client=FakeClient(UsageStream(["a"])))
    generator.complete("system", "prompt")
    list(generator.complete_stream("system", "prompt"))
    assert (generator.input_tokens, generator.output_tokens) == (200, 40)


def test_messages_without_usage_are_not_counted():
    generator = AnswerGenerator(client=FakeClient(FakeStream(["a"])))
    generator.complete("system", "prompt")
    assert (generator.input_tokens, generator.output_tokens) == (0, 0)

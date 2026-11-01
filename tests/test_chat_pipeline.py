"""Pipeline and ChatService behaviour with fake retrieval and a fake LLM (no API calls)."""

import pytest

from sec_rag.chat.service import ChatService
from sec_rag.chat.store import ChatStore
from sec_rag.config import settings
from sec_rag.generation.llm import GenerationError
from sec_rag.pipeline import NO_RESULTS, RAGPipeline
from tests.test_chat_store import make_chunk


class FakeRetriever:
    def __init__(self, results=None):
        self.results = [make_chunk()] if results is None else results
        self.queries = []

    def retrieve(self, question, k, filters=None):
        self.queries.append(question)
        return self.results


class FakeGenerator:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def generate(self, question, chunks, history=None):
        self.calls.append((question, history))
        return "Answer [1]."

    def stream_text(self, question, chunks, history=None):
        self.calls.append((question, history))
        yield "Answer "
        if self.fail:
            raise GenerationError("503 high demand")
        yield "[1]."


class FakeRewriter:
    def __init__(self):
        self.calls = []

    def rewrite(self, question, history):
        self.calls.append((question, history))
        return f"standalone: {question}"


def make_pipeline(retriever=None, generator=None):
    rewriter = FakeRewriter()
    pipeline = RAGPipeline(
        retriever=retriever or FakeRetriever(), generator=generator or FakeGenerator(),
        rewriter=rewriter,
    )
    return pipeline, rewriter


def test_first_question_is_not_rewritten():
    pipeline, rewriter = make_pipeline()
    answer = pipeline.ask("What risks does AAPL face?")
    assert rewriter.calls == []
    assert answer.standalone_question == "What risks does AAPL face?"
    assert pipeline.retriever.queries == ["What risks does AAPL face?"]


def test_follow_up_is_rewritten_for_retrieval(tmp_path):
    pipeline, rewriter = make_pipeline()
    chat = ChatService(pipeline, ChatStore(tmp_path / "chats.db"))
    conversation = chat.store.create_conversation()

    chat.ask(conversation.id, "What risks does AAPL face?")
    answer = chat.ask(conversation.id, "What about MSFT?")

    assert [q for q, _ in rewriter.calls] == ["What about MSFT?"]  # only the follow-up
    assert pipeline.retriever.queries[-1] == "standalone: What about MSFT?"
    assert answer.standalone_question == "standalone: What about MSFT?"
    # The LLM gets the original wording plus the earlier exchange.
    question, history = pipeline.generator.calls[-1]
    assert question == "What about MSFT?"
    assert [t.question for t in history] == ["What risks does AAPL face?"]


def test_history_is_limited_to_recent_turns(tmp_path):
    pipeline, _ = make_pipeline()
    chat = ChatService(pipeline, ChatStore(tmp_path / "chats.db"))
    conversation = chat.store.create_conversation()
    for i in range(settings.chat_history_turns + 2):
        chat.ask(conversation.id, f"Question {i}")

    _, history = pipeline.generator.calls[-1]
    assert len(history) == settings.chat_history_turns
    assert history[-1].question == f"Question {settings.chat_history_turns}"


def test_no_results_skips_the_llm():
    pipeline, _ = make_pipeline(retriever=FakeRetriever(results=[]))
    result = pipeline.ask_stream("What did XYZ say in 2019?")
    assert "".join(result.text_stream) == NO_RESULTS
    assert pipeline.generator.calls == []


def test_stream_is_saved_only_when_finished(tmp_path):
    pipeline, _ = make_pipeline()
    chat = ChatService(pipeline, ChatStore(tmp_path / "chats.db"))
    conversation = chat.store.create_conversation()

    result = chat.ask_stream(conversation.id, "What risks does AAPL face?")
    assert chat.store.get_messages(conversation.id) == []  # nothing saved before streaming
    assert "".join(result.text_stream) == "Answer [1]."
    messages = chat.store.get_messages(conversation.id)
    assert [m.content for m in messages] == ["What risks does AAPL face?", "Answer [1]."]
    assert messages[1].sources[0].ticker == "AAPL"


def test_failed_stream_is_not_saved(tmp_path):
    pipeline, _ = make_pipeline(generator=FakeGenerator(fail=True))
    chat = ChatService(pipeline, ChatStore(tmp_path / "chats.db"))
    conversation = chat.store.create_conversation()

    result = chat.ask_stream(conversation.id, "What risks does AAPL face?")
    with pytest.raises(GenerationError):
        "".join(result.text_stream)
    assert chat.store.get_messages(conversation.id) == []

"""Conversations on top of the RAG pipeline: load history, answer, save the exchange."""

from collections.abc import Iterator
from dataclasses import replace

from sec_rag.chat.store import ChatStore
from sec_rag.config import settings
from sec_rag.models import Answer
from sec_rag.pipeline import RAGPipeline, StreamingAnswer


class ChatService:
    def __init__(self, pipeline: RAGPipeline | None = None, store: ChatStore | None = None):
        self.pipeline = pipeline or RAGPipeline()
        self.store = store or ChatStore()

    def ask(
        self,
        conversation_id: str,
        question: str,
        filters: dict | None = None,
        top_k: int = settings.top_k,
    ) -> Answer:
        history = self.store.get_turns(conversation_id)
        answer = self.pipeline.ask(question, filters=filters, top_k=top_k, history=history)
        self.store.add_exchange(conversation_id, answer)
        return answer

    def ask_stream(
        self,
        conversation_id: str,
        question: str,
        filters: dict | None = None,
        top_k: int = settings.top_k,
    ) -> StreamingAnswer:
        """The exchange is saved only once the stream finishes, so failed answers aren't kept."""
        history = self.store.get_turns(conversation_id)
        result = self.pipeline.ask_stream(question, filters=filters, top_k=top_k, history=history)

        def save_when_done() -> Iterator[str]:
            parts = []
            for text in result.text_stream:
                parts.append(text)
                yield text
            self.store.add_exchange(
                conversation_id,
                Answer(
                    question=question,
                    standalone_question=result.standalone_question,
                    answer="".join(parts),
                    sources=result.sources,
                ),
            )

        return replace(result, text_stream=save_when_done())

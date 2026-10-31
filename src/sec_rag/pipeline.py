"""Online query pipeline: rewrite follow-ups, retrieve relevant chunks, generate a cited answer."""

from collections.abc import Iterator
from dataclasses import dataclass

from sec_rag.chat.condense import QuestionRewriter
from sec_rag.config import settings
from sec_rag.generation.llm import AnswerGenerator
from sec_rag.models import Answer, ChatTurn, RetrievedChunk
from sec_rag.retrieval.retriever import Retriever

NO_RESULTS = "No matching filings found."


@dataclass
class StreamingAnswer:
    question: str
    standalone_question: str
    sources: list[RetrievedChunk]
    text_stream: Iterator[str]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: AnswerGenerator | None = None,
        rewriter: QuestionRewriter | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.generator = generator or AnswerGenerator()
        self.rewriter = rewriter or QuestionRewriter(self.generator)

    def warm_up(self) -> "RAGPipeline":
        """Pay the one-time model loading cost (~10s) up front, e.g. at server startup."""
        self.retriever.warm_up()
        return self

    def _retrieve(
        self, question: str, filters: dict | None, top_k: int, history: list[ChatTurn]
    ) -> tuple[str, list[RetrievedChunk]]:
        standalone = self.rewriter.rewrite(question, history) if history else question
        return standalone, self.retriever.retrieve(standalone, k=top_k, filters=filters)

    def ask(
        self,
        question: str,
        filters: dict | None = None,
        top_k: int = settings.top_k,
        history: list[ChatTurn] | None = None,
    ) -> Answer:
        history = (history or [])[-settings.chat_history_turns :]
        standalone, chunks = self._retrieve(question, filters, top_k, history)
        answer = self.generator.generate(question, chunks, history) if chunks else NO_RESULTS
        return Answer(
            question=question, standalone_question=standalone, answer=answer, sources=chunks
        )

    def ask_stream(
        self,
        question: str,
        filters: dict | None = None,
        top_k: int = settings.top_k,
        history: list[ChatTurn] | None = None,
    ) -> StreamingAnswer:
        """Rewrite and retrieve now; Gemini is called lazily as `text_stream` is consumed."""
        history = (history or [])[-settings.chat_history_turns :]
        standalone, chunks = self._retrieve(question, filters, top_k, history)
        if chunks:
            stream = self.generator.stream_text(question, chunks, history)
        else:
            stream = iter([NO_RESULTS])
        return StreamingAnswer(question, standalone, chunks, stream)

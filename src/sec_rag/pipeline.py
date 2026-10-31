"""Online query pipeline: retrieve relevant chunks, then generate a cited answer."""

from collections.abc import Iterator

from sec_rag.config import settings
from sec_rag.generation.llm import AnswerGenerator
from sec_rag.models import Answer, RetrievedChunk
from sec_rag.retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever | None = None, generator: AnswerGenerator | None = None):
        self.retriever = retriever or Retriever()
        self.generator = generator or AnswerGenerator()

    def warm_up(self) -> "RAGPipeline":
        """Pay the one-time model loading cost (~10s) up front, e.g. at server startup."""
        self.retriever.warm_up()
        return self

    def ask(self, question: str, filters: dict | None = None, top_k: int = settings.top_k) -> Answer:
        chunks = self.retriever.retrieve(question, k=top_k, filters=filters)
        if not chunks:
            return Answer(question=question, answer="No matching filings found.", sources=[])
        answer = self.generator.generate(question, chunks)
        return Answer(question=question, answer=answer, sources=chunks)

    def ask_stream(
        self, question: str, filters: dict | None = None, top_k: int = settings.top_k
    ) -> tuple[list[RetrievedChunk], Iterator[str]]:
        chunks = self.retriever.retrieve(question, k=top_k, filters=filters)
        return chunks, self.generator.stream_text(question, chunks)

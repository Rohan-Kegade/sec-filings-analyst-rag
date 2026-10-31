"""Dense retrieval with metadata filtering."""

from sec_rag.config import settings
from sec_rag.indexing.embedder import Embedder
from sec_rag.indexing.vector_store import VectorStore
from sec_rag.models import RetrievedChunk
from sec_rag.retrieval.query_parser import extract_filters


class Retriever:
    def __init__(self, embedder: Embedder | None = None, store: VectorStore | None = None):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()
        self._known_tickers: set[str] | None = None

    @property
    def known_tickers(self) -> set[str]:
        if self._known_tickers is None:
            self._known_tickers = self.store.known_tickers()
        return self._known_tickers

    def warm_up(self) -> None:
        """Load the embedding model and ticker list now instead of on the first query."""
        self.embedder.embed_query("warm up")
        _ = self.known_tickers

    def retrieve(
        self, question: str, k: int = settings.top_k, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        """Explicit filters override anything inferred from the question text."""
        merged = {**extract_filters(question, self.known_tickers), **(filters or {})}
        return self.store.query(self.embedder.embed_query(question), k=k, filters=merged)

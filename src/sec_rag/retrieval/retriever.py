"""Retrieval with metadata filtering: dense (meaning), keyword (BM25), or both merged."""

import logging

from sec_rag.config import settings
from sec_rag.indexing.embedder import Embedder
from sec_rag.indexing.keyword_index import KeywordIndex
from sec_rag.indexing.vector_store import VectorStore
from sec_rag.models import RetrievedChunk
from sec_rag.retrieval.fusion import reciprocal_rank_fusion
from sec_rag.retrieval.query_parser import extract_filters

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        keyword_index: KeywordIndex | None = None,
        mode: str = settings.retrieval_mode,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()
        self.keyword_index = keyword_index or KeywordIndex()
        self.mode = mode
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
        if self.mode != "dense" and self.keyword_index.count() == 0:
            logger.warning(
                "Keyword index is empty, so %s search will use dense results only. "
                "Build it with: sec-rag index --keyword-only",
                self.mode,
            )

    def retrieve(
        self, question: str, k: int = settings.top_k, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        """Explicit filters override anything inferred from the question text."""
        merged = {**extract_filters(question, self.known_tickers), **(filters or {})}

        if self.mode == "dense":
            return self._dense(question, k, merged)
        if self.mode == "keyword":
            return self._keyword(question, k, merged)
        return self._hybrid(question, k, merged)

    def _dense(self, question: str, k: int, filters: dict) -> list[RetrievedChunk]:
        return self.store.query(self.embedder.embed_query(question), k=k, filters=filters)

    def _keyword(self, question: str, k: int, filters: dict) -> list[RetrievedChunk]:
        hits = self.keyword_index.search(question, k=k, filters=filters)
        chunks = self.store.get_by_ids([chunk_id for chunk_id, _ in hits])
        return [
            RetrievedChunk(chunk=chunks[chunk_id], score=score)
            for chunk_id, score in hits
            if chunk_id in chunks
        ]

    def _hybrid(self, question: str, k: int, filters: dict) -> list[RetrievedChunk]:
        return self._hybrid_search(question, k, filters)[0]

    def hybrid_trace(
        self, question: str, k: int = settings.top_k, filters: dict | None = None
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """Hybrid results plus what fusion dropped, for diagnosing retrieval in evals.

        Dropped: chunks that dense or keyword search alone would have returned in its top k
        but the fused top k left out. Their `score` is 0 and `ranks` says where each search
        put them.
        """
        merged = {**extract_filters(question, self.known_tickers), **(filters or {})}
        return self._hybrid_search(question, k, merged)

    def _hybrid_search(
        self, question: str, k: int, filters: dict
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        candidates = max(k, settings.hybrid_candidates)
        dense = self._dense(question, candidates, filters)
        keyword_hits = self.keyword_index.search(question, k=candidates, filters=filters)
        rankings = {
            "dense": [rc.chunk.chunk_id for rc in dense],
            "keyword": [chunk_id for chunk_id, _ in keyword_hits],
        }
        ranks: dict[str, dict[str, int]] = {}
        for search, ids in rankings.items():
            for rank, chunk_id in enumerate(ids, start=1):
                ranks.setdefault(chunk_id, {})[search] = rank

        fused = reciprocal_rank_fusion(list(rankings.values()), k=settings.rrf_k)[:k]
        kept = {chunk_id for chunk_id, _ in fused}
        dropped = [
            chunk_id for chunk_id, r in ranks.items()
            if chunk_id not in kept and min(r.values()) <= k
        ]  # fmt: skip

        # Dense results already carry their chunk; fetch the keyword-only ones.
        chunks = {rc.chunk.chunk_id: rc.chunk for rc in dense}
        missing = [cid for cid in [*kept, *dropped] if cid not in chunks]
        if missing:
            chunks.update(self.store.get_by_ids(missing))
        results = [
            RetrievedChunk(chunk=chunks[chunk_id], score=score, ranks=ranks[chunk_id])
            for chunk_id, score in fused
            if chunk_id in chunks
        ]
        dropped_chunks = [
            RetrievedChunk(chunk=chunks[chunk_id], score=0.0, ranks=ranks[chunk_id])
            for chunk_id in sorted(dropped, key=lambda cid: min(ranks[cid].values()))
            if chunk_id in chunks
        ]
        return results, dropped_chunks

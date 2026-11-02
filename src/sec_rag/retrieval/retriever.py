"""Retrieval with metadata filtering: dense (meaning), keyword (BM25), or both merged."""

import logging
import math

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
        """Explicit filters override anything inferred from the question text.

        A question naming several companies or years (a comparison) is searched once per
        company/year and the results interleaved, so it can return more than k chunks: see
        `chunks_per_entity`.
        """
        return self._search(question, k, self._filters(question, filters))[0]

    def _filters(self, question: str, filters: dict | None) -> dict:
        return {**extract_filters(question, self.known_tickers), **(filters or {})}

    def _search(
        self, question: str, k: int, filters: dict
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """(results, dropped by fusion); dropped is only filled in hybrid mode."""
        parts = split_by_entity(filters)
        if len(parts) == 1:
            return self._search_one(question, k, filters)
        per_entity = chunks_per_entity(k, len(parts))
        found = [self._search_one(question, per_entity, part) for part in parts]
        results = interleave([results for results, _ in found])
        dropped = [chunk for _, left_out in found for chunk in left_out]
        return results, dropped

    def _search_one(
        self, question: str, k: int, filters: dict
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        if self.mode == "dense":
            return self._dense(question, k, filters), []
        if self.mode == "keyword":
            return self._keyword(question, k, filters), []
        return self._hybrid_search(question, k, filters)

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

    def hybrid_trace(
        self, question: str, k: int = settings.top_k, filters: dict | None = None
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """Hybrid results plus what fusion dropped, for diagnosing retrieval in evals.

        Dropped: chunks that dense or keyword search alone would have returned in its top k
        but the fused top k left out. Their `score` is 0 and `ranks` says where each search
        put them. For a comparison, ranks are within each company/year's own search.
        """
        return self._search(question, k, self._filters(question, filters))

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


def split_by_entity(filters: dict) -> list[dict]:
    """One filter set per (ticker, fiscal year) pair when the filters name several.

    {"ticker": ["AAPL", "MSFT"], "fiscal_year": 2025} -> AAPL 2025, MSFT 2025. Other filters
    (e.g. item) are kept on every part. A single company and year returns [filters].
    """

    def values(key: str) -> list:
        value = filters.get(key)
        return value if isinstance(value, list) else [value]

    tickers, years = values("ticker"), values("fiscal_year")
    if len(tickers) * len(years) < 2:
        return [filters]
    parts = []
    for ticker in tickers:
        for year in years:
            part = dict(filters)
            for key, value in (("ticker", ticker), ("fiscal_year", year)):
                if value is None:
                    part.pop(key, None)
                else:
                    part[key] = value
            parts.append(part)
    return parts


def chunks_per_entity(k: int, entities: int) -> int:
    """Chunks to take from each company/year: an equal share of k, but at least
    min_chunks_per_entity, with the total capped at max_comparison_chunks."""
    per = max(math.ceil(k / entities), settings.min_chunks_per_entity)
    return max(1, min(per, settings.max_comparison_chunks // entities))


def interleave(ranked_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Round-robin by rank (A#1, B#1, A#2, B#2, ...), so no company or year comes last."""
    merged, seen = [], set()
    for position in range(max((len(r) for r in ranked_lists), default=0)):
        for ranked in ranked_lists:
            if position < len(ranked) and ranked[position].chunk.chunk_id not in seen:
                seen.add(ranked[position].chunk.chunk_id)
                merged.append(ranked[position])
    return merged

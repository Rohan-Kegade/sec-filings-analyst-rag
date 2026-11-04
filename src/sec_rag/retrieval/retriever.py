"""Retrieval with metadata filtering: dense (meaning), keyword (BM25), or both merged."""

import logging
import math
import re

from sec_rag.config import settings
from sec_rag.indexing.embedder import Embedder
from sec_rag.indexing.keyword_index import KeywordIndex
from sec_rag.indexing.vector_store import VectorStore
from sec_rag.models import RetrievedChunk
from sec_rag.retrieval.fusion import reciprocal_rank_fusion
from sec_rag.retrieval.query_parser import extract_filters
from sec_rag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        keyword_index: KeywordIndex | None = None,
        mode: str = settings.retrieval_mode,
        reranker: Reranker | None = None,
        rerank: bool = settings.rerank,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()
        self.keyword_index = keyword_index or KeywordIndex()
        self.mode = mode
        # Hybrid mode only: dense and keyword stay plain single-search baselines.
        self.rerank = rerank and mode == "hybrid"
        self.reranker = reranker or Reranker()  # creates its API client on first use
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
        if self.rerank:
            self.reranker.warm_up()
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
        # Each company's search uses the question with the other companies taken out: the
        # filter already limits it to one company's filings, and the other tickers would
        # only dilute the query (keyword terms that can't match, a blended embedding).
        tickers = _as_list(filters.get("ticker"))
        found = []
        for part in parts:
            focused = focus_question(question, part.get("ticker"), tickers)
            found.append(self._search_one(focused, per_entity, part))
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

        fused = reciprocal_rank_fusion(list(rankings.values()), k=settings.rrf_k)
        # Dense results already carry their chunk; keyword-only ones are fetched as needed.
        chunks = {rc.chunk.chunk_id: rc.chunk for rc in dense}
        if self.rerank:
            pool = fused[: max(k, settings.rerank_candidates)]
            self._fetch(chunks, [chunk_id for chunk_id, _ in pool])
            pool = [(chunk_id, _) for chunk_id, _ in pool if chunk_id in chunks]
            for position, (chunk_id, _) in enumerate(pool, start=1):
                ranks[chunk_id]["fused"] = position  # where RRF had it before reranking
            try:
                scores = self.reranker.score(question, [chunks[cid] for cid, _ in pool])
            except Exception as exc:  # noqa: BLE001 - a reranker failure must not break retrieval
                logger.warning("Reranking failed, keeping the fused order: %s", exc)
            else:
                pairs = zip((chunk_id for chunk_id, _ in pool), scores, strict=True)
                fused = sorted(pairs, key=lambda pair: pair[1], reverse=True)  # stable on ties
        fused = fused[:k]
        kept = {chunk_id for chunk_id, _ in fused}
        dropped = [
            chunk_id for chunk_id, r in ranks.items()
            if chunk_id not in kept and min(r[s] for s in rankings if s in r) <= k
        ]  # fmt: skip

        self._fetch(chunks, [*kept, *dropped])
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

    def _fetch(self, chunks: dict, chunk_ids: list[str]) -> None:
        """Add the chunks not already in `chunks` (keyword-only hits) from the store."""
        missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunks]
        if missing:
            chunks.update(self.store.get_by_ids(missing))


JOINERS = {",", "and", "or", "vs", "vs.", "versus", "&", "/"}


def _as_list(value) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def focus_question(question: str, ticker: str | None, tickers: list[str]) -> str:
    """The question with every ticker except `ticker` removed, for that company's search.

    "How many employees did AAPL, MSFT and NVDA each report?" -> for MSFT:
    "How many employees did MSFT each report?". Unchanged when there is nothing to remove.
    """
    others = {t for t in tickers if t != ticker}
    if not ticker or not others:
        return question
    tokens = re.findall(r"[A-Za-z0-9]+(?:'s)?|[^\sA-Za-z0-9]+", question)
    words = [t.removesuffix("'s") for t in tokens]
    removed: set[int] = set()
    for i, word in enumerate(words):
        if word not in others:
            continue
        removed.add(i)
        # Also drop the "and" / "," / "vs" that joins this ticker to the list: the one
        # before it if still there, else the one after it.
        for j in (i - 1, i + 1):
            if 0 <= j < len(tokens) and j not in removed and tokens[j].lower() in JOINERS:
                removed.add(j)
                break
    kept = [t for i, t in enumerate(tokens) if i not in removed]
    text = " ".join(kept)
    return re.sub(r"\s+([,?.!:;)])", r"\1", text).replace("( ", "(")


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

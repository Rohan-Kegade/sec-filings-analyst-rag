from sec_rag.config import settings
from sec_rag.models import RetrievedChunk
from sec_rag.retrieval.fusion import reciprocal_rank_fusion
from sec_rag.retrieval.retriever import (
    Retriever,
    chunks_per_entity,
    interleave,
    split_by_entity,
)
from tests.test_keyword_index import make_chunk


def test_rrf_favours_items_ranked_by_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d", "b"]])
    assert {item for item, _ in fused[:2]} == {"b", "c"}  # in both lists
    assert {item for item, _ in fused} == {"a", "b", "c", "d"}


def test_rrf_ties_keep_first_list_order():
    fused = reciprocal_rank_fusion([["a"], ["b"]])
    assert [item for item, _ in fused] == ["a", "b"]


CHUNKS = {cid: make_chunk(cid, f"text {cid}") for cid in ["d1", "d2", "both", "k1"]}


class FakeEmbedder:
    def embed_query(self, question):
        return [0.0]


class FakeStore:
    def __init__(self):
        self.queries = 0

    def query(self, embedding, k, filters=None):
        self.queries += 1
        return [RetrievedChunk(chunk=CHUNKS[c], score=0.9) for c in ["d1", "both", "d2"]][:k]

    def get_by_ids(self, ids):
        return {cid: CHUNKS[cid] for cid in ids if cid in CHUNKS}

    def known_tickers(self):
        return {"MSFT"}


class FakeKeywordIndex:
    def __init__(self):
        self.calls = []

    def search(self, question, k, filters=None):
        self.calls.append(filters)
        return [("k1", 12.0), ("both", 10.0)][:k]


def make_retriever(mode):
    return Retriever(FakeEmbedder(), FakeStore(), FakeKeywordIndex(), mode=mode)


def test_dense_mode_skips_keyword_search():
    retriever = make_retriever("dense")
    assert [rc.chunk.chunk_id for rc in retriever.retrieve("q", k=3)] == ["d1", "both", "d2"]
    assert retriever.keyword_index.calls == []


def test_keyword_mode_skips_dense_search():
    retriever = make_retriever("keyword")
    assert [rc.chunk.chunk_id for rc in retriever.retrieve("q", k=2)] == ["k1", "both"]
    assert retriever.store.queries == 0


def test_hybrid_merges_and_fetches_keyword_only_chunks():
    retriever = make_retriever("hybrid")
    results = retriever.retrieve("q", k=4)
    ids = [rc.chunk.chunk_id for rc in results]
    assert ids[0] == "both"  # found by both searches
    assert set(ids) == {"d1", "d2", "both", "k1"}  # k1 came only from keyword search
    assert results[0].score > results[-1].score


def test_filters_reach_both_searches():
    retriever = make_retriever("hybrid")
    retriever.retrieve("What does MSFT say in 2024?", k=4, filters={"item": "7"})
    assert retriever.keyword_index.calls == [{"ticker": "MSFT", "fiscal_year": 2024, "item": "7"}]


def test_hybrid_records_ranks_and_what_fusion_dropped():
    retriever = make_retriever("hybrid")
    results, dropped = retriever.hybrid_trace("q", k=2)
    # dense: d1, both, d2   keyword: k1, both   -> "both" wins, then d1 (ties keep dense first)
    assert [(rc.chunk.chunk_id, rc.ranks) for rc in results] == [
        ("both", {"dense": 2, "keyword": 2}),
        ("d1", {"dense": 1}),
    ]
    # k1 was keyword's #1 (inside the top 2) but fusion left it out; d2 was never in a top 2.
    assert [(rc.chunk.chunk_id, rc.ranks) for rc in dropped] == [("k1", {"keyword": 1})]
    assert [rc.chunk.chunk_id for rc in retriever.retrieve("q", k=2)] == ["both", "d1"]


# --- comparisons: one search per company / year -----------------------------------------

def test_split_by_entity():
    assert split_by_entity({"ticker": "MSFT", "fiscal_year": 2024}) == [
        {"ticker": "MSFT", "fiscal_year": 2024}
    ]
    assert split_by_entity({}) == [{}]
    assert split_by_entity({"ticker": ["AAPL", "MSFT"], "fiscal_year": 2025, "item": "1C"}) == [
        {"ticker": "AAPL", "fiscal_year": 2025, "item": "1C"},
        {"ticker": "MSFT", "fiscal_year": 2025, "item": "1C"},
    ]
    assert split_by_entity({"ticker": "NVDA", "fiscal_year": [2024, 2026]}) == [
        {"ticker": "NVDA", "fiscal_year": 2024},
        {"ticker": "NVDA", "fiscal_year": 2026},
    ]
    assert split_by_entity({"fiscal_year": [2024, 2025]}) == [
        {"fiscal_year": 2024},
        {"fiscal_year": 2025},
    ]
    assert len(split_by_entity({"ticker": ["AAPL", "MSFT"], "fiscal_year": [2024, 2025]})) == 4


def test_chunks_per_entity(monkeypatch):
    monkeypatch.setattr(settings, "min_chunks_per_entity", 4)
    monkeypatch.setattr(settings, "max_comparison_chunks", 16)
    assert chunks_per_entity(8, 2) == 4  # 8 total
    assert chunks_per_entity(8, 3) == 4  # at least 4 each: 12 total
    assert chunks_per_entity(20, 2) == 8  # an equal share of a larger k, within the cap
    assert chunks_per_entity(8, 6) == 2  # capped: 12 total, not 24
    assert chunks_per_entity(8, 20) == 1


def test_interleave_alternates_and_drops_duplicates():
    a = [RetrievedChunk(chunk=CHUNKS[c], score=1.0) for c in ["d1", "d2", "both"]]
    b = [RetrievedChunk(chunk=CHUNKS[c], score=1.0) for c in ["k1", "both"]]
    assert [rc.chunk.chunk_id for rc in interleave([a, b])] == ["d1", "k1", "d2", "both"]


class EntityStore(FakeStore):
    """Returns chunks of the company the filter asks for, and records every filter."""

    def __init__(self):
        super().__init__()
        self.filters = []
        self.by_ticker = {
            t: [make_chunk(f"{t}-{i}", f"{t} text {i}", ticker=t) for i in range(10)]
            for t in ["AAPL", "MSFT"]
        }
        CHUNKS.update({c.chunk_id: c for cs in self.by_ticker.values() for c in cs})

    def query(self, embedding, k, filters=None):
        self.filters.append(filters)
        ticker = filters["ticker"]
        assert isinstance(ticker, str), "each search is for one company"
        return [RetrievedChunk(chunk=c, score=0.9) for c in self.by_ticker[ticker][:k]]

    def known_tickers(self):
        return {"AAPL", "MSFT"}


class EmptyKeywordIndex(FakeKeywordIndex):
    def search(self, question, k, filters=None):
        self.calls.append(filters)
        return []


def test_comparison_searches_each_company_and_interleaves(monkeypatch):
    monkeypatch.setattr(settings, "min_chunks_per_entity", 4)
    store = EntityStore()
    retriever = Retriever(FakeEmbedder(), store, EmptyKeywordIndex(), mode="hybrid")

    results = retriever.retrieve("Compare AAPL and MSFT cybersecurity in 2025", k=8)

    assert store.filters == [
        {"ticker": "AAPL", "fiscal_year": 2025},
        {"ticker": "MSFT", "fiscal_year": 2025},
    ]
    assert [rc.chunk.ticker for rc in results] == ["AAPL", "MSFT"] * 4  # 4 each, alternating
    traced, _ = retriever.hybrid_trace("Compare AAPL and MSFT cybersecurity in 2025", k=8)
    assert [rc.chunk.chunk_id for rc in traced] == [rc.chunk.chunk_id for rc in results]


def test_single_company_question_is_not_split():
    store = EntityStore()
    retriever = Retriever(FakeEmbedder(), store, EmptyKeywordIndex(), mode="dense")
    assert len(retriever.retrieve("What does AAPL say about risk in 2025?", k=8)) == 8
    assert store.filters == [{"ticker": "AAPL", "fiscal_year": 2025}]

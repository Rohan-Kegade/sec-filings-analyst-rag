from sec_rag.models import RetrievedChunk
from sec_rag.retrieval.fusion import reciprocal_rank_fusion
from sec_rag.retrieval.retriever import Retriever
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

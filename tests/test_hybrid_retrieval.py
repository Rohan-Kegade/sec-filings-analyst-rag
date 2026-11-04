from types import SimpleNamespace

import pytest

from sec_rag.config import settings
from sec_rag.models import RetrievedChunk
from sec_rag.retrieval.fusion import reciprocal_rank_fusion
from sec_rag.retrieval.reranker import PassageScore, Reranker, RerankScores, passage
from sec_rag.retrieval.retriever import (
    Retriever,
    chunks_per_entity,
    focus_question,
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
    return Retriever(FakeEmbedder(), FakeStore(), FakeKeywordIndex(), mode=mode, rerank=False)


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
    retriever = Retriever(FakeEmbedder(), store, EmptyKeywordIndex(), mode="hybrid", rerank=False)

    results = retriever.retrieve("Compare AAPL and MSFT cybersecurity in 2025", k=8)

    assert store.filters == [
        {"ticker": "AAPL", "fiscal_year": 2025},
        {"ticker": "MSFT", "fiscal_year": 2025},
    ]
    assert [rc.chunk.ticker for rc in results] == ["AAPL", "MSFT"] * 4  # 4 each, alternating
    traced, _ = retriever.hybrid_trace("Compare AAPL and MSFT cybersecurity in 2025", k=8)
    assert [rc.chunk.chunk_id for rc in traced] == [rc.chunk.chunk_id for rc in results]


@pytest.mark.parametrize("question, ticker, expected", [
    ("How many employees did AAPL, MSFT and NVDA each report in 2024?", "MSFT",
     "How many employees did MSFT each report in 2024?"),
    ("How many employees did AAPL, MSFT and NVDA each report in 2024?", "NVDA",
     "How many employees did NVDA each report in 2024?"),
    ("Compare AAPL vs MSFT vs NVDA gross margin in 2024.", "AAPL",
     "Compare AAPL gross margin in 2024."),
    ("How do AAPL's and MSFT's AI risks differ?", "MSFT", "How do MSFT's AI risks differ?"),
    ("What did MSFT & NVDA say about OpenAI?", "NVDA", "What did NVDA say about OpenAI?"),
    ("What does AAPL say about risk?", "AAPL", "What does AAPL say about risk?"),  # nothing to drop
])  # fmt: skip
def test_focus_question_keeps_only_one_company(question, ticker, expected):
    assert focus_question(question, ticker, ["AAPL", "MSFT", "NVDA"]) == expected


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        self.questions = []

    def embed_query(self, question):
        self.questions.append(question)
        return [0.0]


class RecordingKeywordIndex(EmptyKeywordIndex):
    def __init__(self):
        super().__init__()
        self.questions = []

    def search(self, question, k, filters=None):
        self.questions.append(question)
        return super().search(question, k, filters)


def test_each_company_search_uses_a_question_about_that_company_only():
    embedder, keyword = RecordingEmbedder(), RecordingKeywordIndex()
    retriever = Retriever(embedder, EntityStore(), keyword, mode="hybrid", rerank=False)
    retriever.retrieve("Compare how AAPL and MSFT describe cybersecurity in 2025", k=8)
    expected = ["Compare how AAPL describe cybersecurity in 2025",
                "Compare how MSFT describe cybersecurity in 2025"]  # fmt: skip
    assert embedder.questions == expected and keyword.questions == expected


def test_single_company_question_is_not_split():
    store = EntityStore()
    retriever = Retriever(FakeEmbedder(), store, EmptyKeywordIndex(), mode="dense", rerank=False)
    assert len(retriever.retrieve("What does AAPL say about risk in 2025?", k=8)) == 8
    assert store.filters == [{"ticker": "AAPL", "fiscal_year": 2025}]


# --- reranking -----------------------------------------------------------------------------


class FakeReranker:
    """Scores by a fixed table; records what it was asked to score."""

    def __init__(self, scores):
        self.scores, self.seen = scores, []

    def score(self, question, chunks):
        self.seen.append([c.chunk_id for c in chunks])
        return [self.scores[c.chunk_id] for c in chunks]


def make_reranked(scores, rerank=True, mode="hybrid"):
    reranker = FakeReranker(scores)
    retriever = Retriever(FakeEmbedder(), FakeStore(), FakeKeywordIndex(), mode=mode,
                          reranker=reranker, rerank=rerank)  # fmt: skip
    return retriever, reranker


def test_rerank_reorders_fused_candidates_and_records_fused_rank():
    retriever, reranker = make_reranked({"d1": 0.1, "both": 0.2, "d2": 3.0, "k1": 1.0})
    results, dropped = retriever.hybrid_trace("q", k=2)
    assert [rc.chunk.chunk_id for rc in results] == ["d2", "k1"]
    assert results[0].score == 3.0  # the reranker's score replaces the RRF score
    assert set(reranker.seen[0]) == {"d1", "both", "d2", "k1"}  # k1 fetched before scoring
    assert results[0].ranks["fused"] == 4 and results[0].ranks["dense"] == 3
    # Dropped still means "in dense's or keyword's own top k, but not kept".
    assert {rc.chunk.chunk_id for rc in dropped} == {"d1", "both"}


def test_rerank_scores_only_the_top_candidates(monkeypatch):
    monkeypatch.setattr(settings, "rerank_candidates", 2)
    retriever, reranker = make_reranked({"d1": 0.0, "both": 1.0, "d2": 9.0, "k1": 9.0})
    retriever.retrieve("q", k=2)
    assert len(reranker.seen[0]) == 2  # never more than max(k, rerank_candidates)


def test_rerank_off_and_non_hybrid_modes_skip_the_reranker():
    for rerank, mode in [(False, "hybrid"), (True, "dense"), (True, "keyword")]:
        retriever, reranker = make_reranked({}, rerank=rerank, mode=mode)
        retriever.retrieve("q", k=2)
        assert reranker.seen == []


def test_rerank_runs_per_company_for_comparisons(monkeypatch):
    monkeypatch.setattr(settings, "min_chunks_per_entity", 1)
    monkeypatch.setattr(settings, "max_comparison_chunks", 16)
    reranker = FakeReranker({})
    reranker.score = lambda q, chunks: (reranker.seen.append([c.ticker for c in chunks])
                                        or [0.0] * len(chunks))  # fmt: skip
    retriever = Retriever(FakeEmbedder(), EntityStore(), EmptyKeywordIndex(), mode="hybrid",
                          reranker=reranker, rerank=True)  # fmt: skip
    retriever.retrieve("Compare AAPL and MSFT", k=2)
    assert [set(tickers) for tickers in reranker.seen] == [{"AAPL"}, {"MSFT"}]


def test_reranker_passage_adds_company_year_and_section():
    chunk = make_chunk("c1", "About 164,000 employees.")
    text = passage(chunk)
    assert chunk.ticker in text and f"FY{chunk.fiscal_year}" in text
    assert f"Item {chunk.item}" in text and text.endswith("About 164,000 employees.")


class FakeParse:
    def __init__(self, message):
        self.message, self.calls = message, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.message


def claude_reranker(parsed, stop_reason="end_turn"):
    message = SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason,
                              usage=SimpleNamespace(input_tokens=900, output_tokens=40))  # fmt: skip
    api = FakeParse(message)
    client = SimpleNamespace(beta=SimpleNamespace(messages=api))
    return Reranker(client=client, model="claude-haiku-5-5", effort="low"), api


def test_claude_reranker_scores_numbered_passages():
    parsed = RerankScores(scores=[PassageScore(id=2, score=9), PassageScore(id=1, score=3)])
    reranker, api = claude_reranker(parsed)
    chunks = [make_chunk("a", "net sales"), make_chunk("b", "164,000 employees"),
              make_chunk("c", "skipped")]  # fmt: skip

    assert reranker.score("How many employees?", chunks) == [3.0, 9.0, -1.0]
    request = api.calls[0]
    assert request["model"] == "claude-haiku-5-5"
    assert request["output_config"] == {"effort": "low"}
    assert request["output_format"] is RerankScores
    prompt = request["messages"][0]["content"]
    assert '<passage id="2">' in prompt and "164,000 employees" in prompt
    assert prompt.rstrip().endswith("Score passages 1-3.")
    assert (reranker.input_tokens, reranker.output_tokens) == (900, 40)
    assert reranker.score("q", []) == [] and len(api.calls) == 1  # nothing to score, no call


def test_claude_reranker_raises_without_scores():
    reranker, _ = claude_reranker(None, stop_reason="refusal")
    with pytest.raises(RuntimeError, match="refusal"):
        reranker.score("q", [make_chunk("a", "x")])


def test_failed_rerank_keeps_the_fused_order():
    class Broken:
        def score(self, question, chunks):
            raise RuntimeError("API down")

    plain = make_retriever("hybrid").retrieve("q", k=3)
    retriever = Retriever(FakeEmbedder(), FakeStore(), FakeKeywordIndex(), mode="hybrid",
                          reranker=Broken(), rerank=True)  # fmt: skip
    reranked = retriever.retrieve("q", k=3)
    assert [rc.chunk.chunk_id for rc in reranked] == [rc.chunk.chunk_id for rc in plain]

from sec_rag.indexing.keyword_index import KeywordIndex, to_match_query
from sec_rag.models import Chunk


def make_chunk(chunk_id: str, text: str, ticker: str = "MSFT", year: int = 2024, item: str = "7"):
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        ticker=ticker,
        company_name=f"{ticker} Corp",
        fiscal_year=year,
        item=item,
        item_title="",
        accession_number="acc",
        source_url="https://example.com",
        chunk_index=0,
    )


def make_index(tmp_path) -> KeywordIndex:
    index = KeywordIndex(tmp_path / "keyword.db")
    index.upsert(
        [
            make_chunk("azure", "Azure and other cloud services revenue grew 29%."),
            make_chunk("xbox", "Xbox content and services revenue increased."),
            make_chunk("azure-2025", "Azure revenue grew again.", year=2025),
            make_chunk("nvda", "H100 export controls to China.", ticker="NVDA", item="1A"),
        ]
    )
    return index


def test_match_query_quotes_terms_and_drops_question_words():
    assert to_match_query("What does MSFT say about Azure?") == '"MSFT" OR "Azure"'
    assert to_match_query("What is it?") == ""


def test_special_characters_do_not_break_search(tmp_path):
    index = make_index(tmp_path)
    assert index.search('Azure: "growth" (AND) -NOT* OR?', k=5)[0][0] in {"azure", "azure-2025"}


def test_search_ranks_exact_term_matches(tmp_path):
    index = make_index(tmp_path)
    ids = [chunk_id for chunk_id, _ in index.search("Xbox revenue", k=5)]
    assert ids[0] == "xbox"


def test_stemming_matches_word_forms(tmp_path):
    index = make_index(tmp_path)
    assert "xbox" in [cid for cid, _ in index.search("increasing", k=5)]  # matches "increased"


def test_filters_apply(tmp_path):
    index = make_index(tmp_path)
    assert [cid for cid, _ in index.search("Azure", k=5, filters={"fiscal_year": 2025})] == [
        "azure-2025"
    ]
    assert index.search("Azure", k=5, filters={"ticker": "NVDA"}) == []
    both_years = index.search("Azure", k=5, filters={"fiscal_year": [2024, 2025]})
    assert {cid for cid, _ in both_years} == {"azure", "azure-2025"}


def test_delete_filing_removes_only_that_filing(tmp_path):
    index = KeywordIndex(tmp_path / "keyword.db")
    index.upsert(
        [
            make_chunk("0000950170-24-000001-7-0", "Azure grew."),
            make_chunk("0000950170-24-000001-7-1", "Xbox grew."),
            make_chunk("0000950170-24-0000011-7-0", "Other filing with a longer accession."),
        ]
    )
    index.delete_filing("0000950170-24-000001")
    assert index.count() == 1
    assert index.search("filing", k=5)[0][0] == "0000950170-24-0000011-7-0"


def test_upsert_replaces_instead_of_duplicating(tmp_path):
    index = make_index(tmp_path)
    index.upsert([make_chunk("xbox", "Gaming revenue declined.")])
    assert index.count() == 4
    assert index.search("Xbox", k=5) == []
    assert [cid for cid, _ in index.search("Gaming", k=5)] == ["xbox"]

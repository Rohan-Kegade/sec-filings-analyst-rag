import pytest

from sec_rag.indexing.pipeline import (
    NoParsedFilingsError,
    index_filings,
    load_processed_filings,
    parse_filings,
    unparsed_filings,
)
from sec_rag.models import FilingMetadata
from tests.test_parser import HTML


def write_raw(raw_dir, ticker, year, accession):
    metadata = FilingMetadata(
        ticker=ticker, cik="1", company_name=f"{ticker} Inc.", fiscal_year=year,
        filing_date=f"{year}-11-01", accession_number=accession, source_url="https://x",
    )
    folder = raw_dir / ticker
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{year}_{accession}.html").write_text(HTML, encoding="utf-8")
    (folder / f"{year}_{accession}.json").write_text(metadata.model_dump_json(), encoding="utf-8")
    return metadata


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    embedding_model = built_with = "m"

    def __init__(self):
        self.chunks = []

    def delete_filing(self, accession_number):
        pass

    def upsert(self, chunks, embeddings):
        self.chunks.extend(chunks)


class FakeKeywordIndex:
    def delete_filing(self, accession_number):
        pass

    def upsert(self, chunks):
        pass


def test_parse_then_index(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    write_raw(raw, "AAPL", 2024, "0001-24-000001")
    write_raw(raw, "MSFT", 2024, "0002-24-000002")

    assert len(unparsed_filings(raw, processed)) == 2
    assert parse_filings(raw, processed) == 2
    assert unparsed_filings(raw, processed) == []
    assert sorted(p.name for p in processed.glob("*.json")) == [
        "AAPL_2024_0001-24-000001.json", "MSFT_2024_0002-24-000002.json",
    ]
    filings = load_processed_filings(processed)
    assert [s.item for s in filings[0].sections] == ["1", "1A", "7"]

    embedder, store = FakeEmbedder(), FakeStore()
    total = index_filings(embedder, store, FakeKeywordIndex(), processed_dir=processed)
    assert total == len(store.chunks) > 0
    assert embedder.calls == 2
    assert {c.ticker for c in store.chunks} == {"AAPL", "MSFT"}


def test_unparsed_filings_lists_only_the_new_download(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    write_raw(raw, "AAPL", 2024, "0001-24-000001")
    parse_filings(raw, processed)
    write_raw(raw, "AAPL", 2025, "0001-25-000001")

    assert [m.fiscal_year for m in unparsed_filings(raw, processed)] == [2025]


def test_index_without_parsed_filings_fails_before_embedding(tmp_path):
    embedder = FakeEmbedder()
    with pytest.raises(NoParsedFilingsError, match="sec-rag parse"):
        index_filings(embedder, FakeStore(), FakeKeywordIndex(), processed_dir=tmp_path)
    assert embedder.calls == 0

from sec_rag.ingestion.chunker import _split_text, chunk_filing
from sec_rag.models import Filing, FilingMetadata, Section

METADATA = FilingMetadata(
    ticker="TEST",
    cik="0000000001",
    company_name="Test Corp",
    fiscal_year=2024,
    filing_date="2024-11-01",
    accession_number="0000000001-24-000001",
    source_url="https://example.com",
)


def test_split_respects_chunk_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(30))
    chunks = _split_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 50 + 2 for c in chunks)


def test_split_hard_splits_long_paragraph():
    chunks = _split_text("x" * 2500, chunk_size=1000, overlap=100)
    assert len(chunks) >= 3


def test_chunk_filing_adds_header_and_metadata():
    filing = Filing(
        metadata=METADATA,
        sections=[Section(item="1A", title="Risk Factors", text="Some risk. " * 300)],
    )
    chunks = chunk_filing(filing, chunk_size=800, overlap=100)
    assert chunks
    first = chunks[0]
    assert first.text.startswith("Test Corp (TEST) 10-K FY2024 - Item 1A: Risk Factors")
    assert first.chunk_id == "0000000001-24-000001-1A-0"
    assert {c.item for c in chunks} == {"1A"}

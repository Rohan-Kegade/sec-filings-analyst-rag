import random
import string

import pytest

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


def random_paragraph(rng: random.Random, length: int) -> str:
    words = []
    while sum(len(w) + 1 for w in words) < length:
        words.append("".join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 9))))
    return " ".join(words)[:length].strip()


@pytest.mark.parametrize("seed", range(5))
def test_short_passages_are_never_split(seed):
    """Any passage of <= overlap chars lies fully inside some chunk, even across paragraphs.

    The retrieval eval relies on this: its evidence quotes are at most `overlap` long.
    """
    rng = random.Random(seed)
    chunk_size, overlap = 300, 60
    # Mix short, medium and over-long paragraphs, including long ones right after short ones.
    paragraphs = [random_paragraph(rng, rng.choice([20, 90, 250, 700])) for _ in range(25)]
    text = "\n\n".join(paragraphs)
    chunks = _split_text(text, chunk_size=chunk_size, overlap=overlap)

    for start in range(0, len(text) - overlap + 1, 3):
        passage = text[start : start + overlap]
        assert any(passage in chunk for chunk in chunks), (seed, passage)


def test_long_paragraph_after_short_one_keeps_overlap():
    short, long = "a" * 50, "b" * 700
    chunks = _split_text(f"{short}\n\n{long}", chunk_size=300, overlap=60)
    assert chunks[0] == short
    assert chunks[1].startswith(short)  # tail of the previous chunk carried over


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        _split_text("text", chunk_size=100, overlap=100)


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

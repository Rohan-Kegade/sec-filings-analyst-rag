"""Section-aware chunking with a contextual header on every chunk."""

from sec_rag.config import settings
from sec_rag.models import Chunk, Filing


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack paragraphs into chunks of at most chunk_size characters, with overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # Hard-split paragraphs that are larger than a chunk on their own.
        while len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:chunk_size])
            para = para[chunk_size - overlap :]

        if not current:
            current = para
        elif len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}" if tail else para

    if current:
        chunks.append(current)
    return chunks


def chunk_filing(
    filing: Filing,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[Chunk]:
    meta = filing.metadata
    chunks = []
    for section in filing.sections:
        # The header gives each chunk standalone context for embedding and for the LLM.
        header = (
            f"{meta.company_name} ({meta.ticker}) 10-K FY{meta.fiscal_year} - "
            f"Item {section.item}: {section.title}"
        )
        for i, body in enumerate(_split_text(section.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.accession_number}-{section.item}-{i}",
                    text=f"{header}\n\n{body}",
                    ticker=meta.ticker,
                    company_name=meta.company_name,
                    fiscal_year=meta.fiscal_year,
                    item=section.item,
                    item_title=section.title,
                    accession_number=meta.accession_number,
                    source_url=meta.source_url,
                    chunk_index=i,
                )
            )
    return chunks

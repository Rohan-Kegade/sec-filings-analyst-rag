"""Section-aware chunking with a contextual header on every chunk."""

from sec_rag.config import settings
from sec_rag.models import Chunk, Filing

TABLE_CELL = " | "  # how the parser separates table cells; one table row per line


def _table_header(para: str, chunk_size: int) -> str | None:
    """The first row of a table paragraph, to repeat atop each piece of a split table.

    Without it, rows in a later piece lose which column is which year or total.
    """
    lines = para.split("\n")
    if len(lines) < 3 or sum(TABLE_CELL in line for line in lines) < len(lines) / 2:
        return None
    return lines[0] if len(lines[0]) <= chunk_size // 4 else None


def _cut_point(para: str, chunk_size: int) -> int:
    """Where to end a piece of an over-long paragraph: at a line break (a table row's end)
    in the second half of the piece if there is one, else exactly at chunk_size."""
    newline = para.rfind("\n", 0, chunk_size + 1)
    return newline if newline > chunk_size // 2 else chunk_size


def _next_start(para: str, cut: int, overlap: int, floor: int) -> int:
    """Where the next piece starts: at least `overlap` before the cut, so text across the
    cut stays whole in one piece; at a line start when one is close enough."""
    latest = cut - overlap
    newline = para.rfind("\n", floor, latest)
    return newline + 1 if newline >= 0 and newline + 1 > cut // 2 else max(latest, floor)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack paragraphs into chunks of about chunk_size characters, with overlap.

    Every new chunk starts with the last `overlap` characters of the previous one, so any
    passage of at most `overlap` characters is always fully inside at least one chunk.
    """
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size and current:
            # Close the current chunk, and carry its tail into the long paragraph's first
            # piece so text across this boundary keeps its overlap like everywhere else.
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            para = f"{tail}\n\n{para}" if tail else para
            current = ""

        # Hard-split paragraphs that are larger than a chunk on their own. Tables are cut
        # between rows, and each later piece starts with the table's header row again.
        header = _table_header(para, chunk_size)
        while len(para) > chunk_size:
            cut = _cut_point(para, chunk_size)
            chunks.append(para[:cut])
            # Never restart inside a header this loop added, or it would be repeated twice.
            floor = len(header) + 1 if header and para.startswith(f"{header}\n") else 0
            para = para[_next_start(para, cut, overlap, floor) :]
            if header:
                para = f"{header}\n{para}"

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

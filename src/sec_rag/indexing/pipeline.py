"""Offline jobs, run in order:

1. parse_filings: raw HTML -> parsed sections, saved as JSON in data/processed
2. index_filings: parsed JSON -> chunks -> embeddings (vector store) + keyword index

Parsing is local and free; embedding calls a paid API. Keeping them apart lets you check
the parsed sections before paying for embeddings, and re-chunk without re-parsing.
"""

import logging
from pathlib import Path

from sec_rag.config import settings
from sec_rag.indexing.embedder import Embedder
from sec_rag.indexing.keyword_index import KeywordIndex
from sec_rag.indexing.vector_store import VectorStore
from sec_rag.ingestion.chunker import chunk_filing
from sec_rag.ingestion.edgar_client import load_raw_filings
from sec_rag.ingestion.parser import parse_filing
from sec_rag.models import Filing, FilingMetadata

logger = logging.getLogger(__name__)


class NoParsedFilingsError(RuntimeError):
    pass


def processed_path(metadata: FilingMetadata, processed_dir: Path) -> Path:
    return processed_dir / (
        f"{metadata.ticker}_{metadata.fiscal_year}_{metadata.accession_number}.json"
    )


def parse_filings(
    raw_dir: Path = settings.raw_dir, processed_dir: Path = settings.processed_dir
) -> int:
    """Parse every downloaded filing into sections and save each as JSON."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for metadata, html_path in load_raw_filings(raw_dir):
        # Pass bytes so BeautifulSoup detects the document's encoding itself.
        filing = parse_filing(html_path.read_bytes(), metadata)
        path = processed_path(metadata, processed_dir)
        path.write_text(filing.model_dump_json(indent=2), encoding="utf-8")
        total += 1
        logger.info(
            "Parsed %s FY%s: %d sections -> %s",
            metadata.ticker, metadata.fiscal_year, len(filing.sections), path.name,
        )
    return total


def load_processed_filings(processed_dir: Path = settings.processed_dir) -> list[Filing]:
    return [
        Filing.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(processed_dir.glob("*.json"))
    ]


def unparsed_filings(
    raw_dir: Path = settings.raw_dir, processed_dir: Path = settings.processed_dir
) -> list[FilingMetadata]:
    """Downloaded filings that have no parsed JSON yet."""
    return [
        metadata
        for metadata, _ in load_raw_filings(raw_dir)
        if not processed_path(metadata, processed_dir).exists()
    ]


def index_filings(
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    keyword_index: KeywordIndex | None = None,
    processed_dir: Path = settings.processed_dir,
) -> int:
    """Chunk, embed and index every parsed filing in processed_dir."""
    filings = load_processed_filings(processed_dir)
    if not filings:
        raise NoParsedFilingsError(f"No parsed filings in {processed_dir}. Run `sec-rag parse` first.")
    embedder = embedder or Embedder()
    store = store or VectorStore()
    keyword_index = keyword_index or KeywordIndex()
    if store.built_with != store.embedding_model:
        # Vectors from different models can't share a collection (they don't even have
        # the same length), and every filing is re-embedded below anyway.
        logger.warning(
            "Vector store was built with %s; rebuilding it with %s",
            store.built_with or "an older embedding model", store.embedding_model,
        )
        store.reset()

    total = 0
    for filing in filings:
        metadata = filing.metadata
        chunks = chunk_filing(filing)
        embeddings = embedder.embed_documents([c.text for c in chunks])
        # Replace the filing's chunks wholesale: a new chunking can produce fewer chunks,
        # and upserting by ID alone would leave the old extra ones in place.
        store.delete_filing(metadata.accession_number)
        keyword_index.delete_filing(metadata.accession_number)
        store.upsert(chunks, embeddings)
        keyword_index.upsert(chunks)
        total += len(chunks)
        logger.info("Indexed %s FY%s: %d chunks", metadata.ticker, metadata.fiscal_year,
                    len(chunks))
    return total


def rebuild_keyword_index(
    store: VectorStore | None = None, keyword_index: KeywordIndex | None = None
) -> int:
    """Fill the keyword index from chunks already in the vector store, without re-embedding."""
    store = store or VectorStore()
    keyword_index = keyword_index or KeywordIndex()
    total = 0
    for batch in store.iter_chunks():
        keyword_index.upsert(batch)
        total += len(batch)
    return total

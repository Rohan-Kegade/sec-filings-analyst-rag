"""Offline indexing: raw HTML -> parsed sections -> chunks -> embeddings + keyword index."""

import logging

from sec_rag.config import settings
from sec_rag.indexing.embedder import Embedder
from sec_rag.indexing.keyword_index import KeywordIndex
from sec_rag.indexing.vector_store import VectorStore
from sec_rag.ingestion.chunker import chunk_filing
from sec_rag.ingestion.edgar_client import load_raw_filings
from sec_rag.ingestion.parser import parse_filing

logger = logging.getLogger(__name__)


def index_filings(
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    keyword_index: KeywordIndex | None = None,
) -> int:
    embedder = embedder or Embedder()
    store = store or VectorStore()
    keyword_index = keyword_index or KeywordIndex()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    if store.built_with != store.embedding_model:
        # Vectors from different models can't share a collection (they don't even have
        # the same length), and every filing is re-embedded below anyway.
        logger.warning(
            "Vector store was built with %s; rebuilding it with %s",
            store.built_with or "an older embedding model", store.embedding_model,
        )
        store.reset()

    total = 0
    for metadata, html_path in load_raw_filings():
        # Pass bytes so BeautifulSoup detects the document's encoding itself.
        filing = parse_filing(html_path.read_bytes(), metadata)
        processed_path = settings.processed_dir / f"{metadata.ticker}_{html_path.stem}.json"
        processed_path.write_text(filing.model_dump_json(indent=2), encoding="utf-8")

        chunks = chunk_filing(filing)
        embeddings = embedder.embed_documents([c.text for c in chunks])
        # Replace the filing's chunks wholesale: a new chunking can produce fewer chunks,
        # and upserting by ID alone would leave the old extra ones in place.
        store.delete_filing(metadata.accession_number)
        keyword_index.delete_filing(metadata.accession_number)
        store.upsert(chunks, embeddings)
        keyword_index.upsert(chunks)
        total += len(chunks)
        logger.info(
            "Indexed %s FY%s: %d sections, %d chunks",
            metadata.ticker, metadata.fiscal_year, len(filing.sections), len(chunks),
        )
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

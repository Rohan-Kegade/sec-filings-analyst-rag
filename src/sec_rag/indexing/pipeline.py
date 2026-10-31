"""Offline indexing: raw HTML -> parsed sections -> chunks -> embeddings -> vector store."""

import logging

from sec_rag.config import settings
from sec_rag.indexing.embedder import Embedder
from sec_rag.indexing.vector_store import VectorStore
from sec_rag.ingestion.chunker import chunk_filing
from sec_rag.ingestion.edgar_client import load_raw_filings
from sec_rag.ingestion.parser import parse_filing

logger = logging.getLogger(__name__)


def index_filings(embedder: Embedder | None = None, store: VectorStore | None = None) -> int:
    embedder = embedder or Embedder()
    store = store or VectorStore()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for metadata, html_path in load_raw_filings():
        # Pass bytes so BeautifulSoup detects the document's encoding itself.
        filing = parse_filing(html_path.read_bytes(), metadata)
        processed_path = settings.processed_dir / f"{metadata.ticker}_{html_path.stem}.json"
        processed_path.write_text(filing.model_dump_json(indent=2), encoding="utf-8")

        chunks = chunk_filing(filing)
        store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
        total += len(chunks)
        logger.info(
            "Indexed %s FY%s: %d sections, %d chunks",
            metadata.ticker, metadata.fiscal_year, len(filing.sections), len(chunks),
        )
    return total

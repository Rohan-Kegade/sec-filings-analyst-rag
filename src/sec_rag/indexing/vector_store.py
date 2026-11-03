"""Persistent vector store backed by ChromaDB."""

from collections.abc import Iterator

import chromadb

from sec_rag.config import settings
from sec_rag.models import Chunk, RetrievedChunk

UPSERT_BATCH_SIZE = 1000


class VectorStore:
    def __init__(
        self,
        path=settings.vectorstore_dir,
        collection_name=settings.collection_name,
        embedding_model: str = settings.embedding_model,
    ):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.collection = self._open_collection()

    def _open_collection(self):
        # The model is recorded only when the collection is created; an existing
        # collection keeps the metadata it was created with.
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
        )

    @property
    def built_with(self) -> str | None:
        """The embedding model the stored vectors came from (None for older indexes)."""
        return (self.collection.metadata or {}).get("embedding_model")

    @property
    def is_stale(self) -> bool:
        """True if the stored vectors came from a different model than the current one."""
        return self.built_with != self.embedding_model and self.count() > 0

    def reset(self) -> None:
        """Drop every stored chunk and start an empty collection for the current model."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._open_collection()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[start : start + UPSERT_BATCH_SIZE]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
                embeddings=embeddings[start : start + UPSERT_BATCH_SIZE],
            )

    def query(
        self, embedding: list[float], k: int, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        if self.is_stale:
            raise RuntimeError(
                f"The vector store was built with {self.built_with or 'an older embedding model'}"
                f", but the configured model is {self.embedding_model}. "
                "Run `sec-rag index` to re-embed the filings."
            )
        result = self.collection.query(
            query_embeddings=[embedding], n_results=k, where=_build_where(filters)
        )
        retrieved = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunk = Chunk(chunk_id=chunk_id, text=text, **meta)
            retrieved.append(RetrievedChunk(chunk=chunk, score=1 - distance))
        return retrieved

    def delete_filing(self, accession_number: str) -> None:
        """Remove a filing's chunks, so re-chunking can't leave stale ones behind."""
        self.collection.delete(where={"accession_number": accession_number})

    def get_by_ids(self, ids: list[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        result = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return {
            chunk_id: Chunk(chunk_id=chunk_id, text=text, **meta)
            for chunk_id, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        }

    def iter_chunks(self, batch_size: int = UPSERT_BATCH_SIZE) -> Iterator[list[Chunk]]:
        """Yield every stored chunk in batches (without embeddings)."""
        for offset in range(0, self.count(), batch_size):
            result = self.collection.get(
                include=["documents", "metadatas"], limit=batch_size, offset=offset
            )
            yield [
                Chunk(chunk_id=chunk_id, text=text, **meta)
                for chunk_id, text, meta in zip(
                    result["ids"], result["documents"], result["metadatas"]
                )
            ]

    def known_tickers(self) -> set[str]:
        metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        return {m["ticker"] for m in metadatas}

    def count(self) -> int:
        return self.collection.count()


def _build_where(filters: dict | None) -> dict | None:
    """Convert {"ticker": "AAPL", "fiscal_year": [2023, 2024]} to a Chroma where clause."""
    if not filters:
        return None
    clauses = []
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            clauses.append({key: {"$in": list(value)}})
        else:
            clauses.append({key: value})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}

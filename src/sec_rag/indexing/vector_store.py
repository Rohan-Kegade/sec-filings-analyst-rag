"""Persistent vector store backed by ChromaDB."""

import chromadb

from sec_rag.config import settings
from sec_rag.models import Chunk, RetrievedChunk

UPSERT_BATCH_SIZE = 1000


class VectorStore:
    def __init__(self, path=settings.vectorstore_dir, collection_name=settings.collection_name):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

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

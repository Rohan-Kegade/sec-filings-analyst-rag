"""Local embedding model wrapper (sentence-transformers)."""

from functools import cached_property

from sec_rag.config import settings


class Embedder:
    def __init__(
        self,
        model_name: str = settings.embedding_model,
        query_prefix: str = settings.embedding_query_prefix,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix

    @cached_property
    def model(self):
        # Imported lazily: loading torch is slow and not needed for every CLI command.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(self.query_prefix + query, normalize_embeddings=True)
        return vector.tolist()

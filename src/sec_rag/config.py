from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SEC_RAG_", extra="ignore")

    # EDGAR
    edgar_user_agent: str = "sec-filings-analyst-rag your-email@example.com"
    edgar_request_delay: float = 0.15  # SEC allows max 10 requests/sec

    # Storage
    data_dir: Path = Path("data")
    collection_name: str = "sec_10k"

    # Chunking (characters)
    chunk_size: int = 1500
    chunk_overlap: int = 200

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = 64

    # Retrieval
    top_k: int = 8

    # Generation
    llm_model: str = "claude-opus-5-5"
    llm_max_tokens: int = 16000
    llm_effort: str = "medium"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def vectorstore_dir(self) -> Path:
        return self.data_dir / "vectorstore"


settings = Settings()

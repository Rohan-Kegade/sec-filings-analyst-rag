from pathlib import Path
from typing import Literal

from pydantic import Field
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
    retrieval_mode: Literal["hybrid", "dense", "keyword"] = "hybrid"
    hybrid_candidates: int = 30  # results taken from each search before merging
    rrf_k: int = 60

    # Generation
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"  # low | medium | high | xhigh | max
    llm_max_tokens: int = 16000
    rewrite_effort: str = "low"  # for turning follow-ups into standalone questions
    rewrite_max_tokens: int = 4000

    # Chat
    chat_history_turns: int = 3  # earlier question/answer pairs sent with a follow-up

    @property
    def chat_db_path(self) -> Path:
        return self.data_dir / "chats.db"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def vectorstore_dir(self) -> Path:
        return self.data_dir / "vectorstore"

    @property
    def keyword_index_path(self) -> Path:
        return self.data_dir / "keyword_index.db"


settings = Settings()

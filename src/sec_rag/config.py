from pathlib import Path

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

    # Generation
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    llm_model: str = "gemini-3.1-flash-lite"
    llm_thinking_level: str = "medium"  # low | medium | high
    rewrite_thinking_level: str = "low"  # for turning follow-ups into standalone questions

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


settings = Settings()

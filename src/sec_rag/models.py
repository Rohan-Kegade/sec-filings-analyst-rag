"""Core data models shared across the pipeline."""

from typing import Literal

from pydantic import BaseModel


class FilingMetadata(BaseModel):
    ticker: str
    cik: str
    company_name: str
    fiscal_year: int
    filing_date: str
    accession_number: str
    source_url: str


class Section(BaseModel):
    item: str  # e.g. "1A"
    title: str  # e.g. "Risk Factors"
    text: str


class Filing(BaseModel):
    metadata: FilingMetadata
    sections: list[Section]


class Chunk(BaseModel):
    chunk_id: str
    text: str
    ticker: str
    company_name: str
    fiscal_year: int
    item: str
    item_title: str
    accession_number: str
    source_url: str
    chunk_index: int

    def metadata(self) -> dict:
        return self.model_dump(exclude={"chunk_id", "text"})


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class Answer(BaseModel):
    question: str
    standalone_question: str  # what retrieval searched for (rewritten if a follow-up)
    answer: str
    sources: list[RetrievedChunk]


class Source(BaseModel):
    """A compact, storable view of a retrieved chunk, shown under an answer."""

    ticker: str
    company_name: str
    fiscal_year: int
    item: str
    item_title: str
    source_url: str
    score: float
    excerpt: str

    @classmethod
    def from_retrieved(cls, rc: RetrievedChunk, excerpt_chars: int = 600) -> "Source":
        c = rc.chunk
        return cls(
            ticker=c.ticker,
            company_name=c.company_name,
            fiscal_year=c.fiscal_year,
            item=c.item,
            item_title=c.item_title,
            source_url=c.source_url,
            score=rc.score,
            excerpt=c.text[:excerpt_chars],
        )


class ChatTurn(BaseModel):
    """One earlier question/answer pair, used as context for follow-ups."""

    question: str
    answer: str


class Conversation(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[Source] = []
    standalone_question: str | None = None  # assistant messages only
    created_at: str

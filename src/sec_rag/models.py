"""Core data models shared across the pipeline."""

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
    answer: str
    sources: list[RetrievedChunk]

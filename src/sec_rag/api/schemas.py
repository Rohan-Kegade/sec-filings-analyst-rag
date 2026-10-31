from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    ticker: str | None = None
    fiscal_year: int | None = None
    item: str | None = Field(default=None, description='10-K item, e.g. "1A" or "7"')
    top_k: int = Field(default=8, ge=1, le=50)

    def filters(self) -> dict:
        filters = {"ticker": self.ticker, "fiscal_year": self.fiscal_year, "item": self.item}
        return {k: v for k, v in filters.items() if v is not None}


class Source(BaseModel):
    ticker: str
    company_name: str
    fiscal_year: int
    item: str
    item_title: str
    source_url: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

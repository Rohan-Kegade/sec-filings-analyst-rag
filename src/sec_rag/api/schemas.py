from pydantic import BaseModel, Field

from sec_rag.models import ChatMessage, Conversation, Source


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    conversation_id: str | None = Field(
        default=None, description="Continue this conversation: follow-ups work and it is saved"
    )
    ticker: str | None = None
    fiscal_year: int | None = None
    item: str | None = Field(default=None, description='10-K item, e.g. "1A" or "7"')
    top_k: int = Field(default=8, ge=1, le=50)

    def filters(self) -> dict:
        filters = {"ticker": self.ticker, "fiscal_year": self.fiscal_year, "item": self.item}
        return {k: v for k, v in filters.items() if v is not None}


class QueryResponse(BaseModel):
    answer: str
    standalone_question: str
    conversation_id: str | None
    sources: list[Source]


class CreateConversationRequest(BaseModel):
    title: str = Field(default="", description="Leave empty to use the first question")


class ConversationDetail(BaseModel):
    conversation: Conversation
    messages: list[ChatMessage]

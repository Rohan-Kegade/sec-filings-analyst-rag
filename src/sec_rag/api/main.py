"""FastAPI service. Run with: uvicorn sec_rag.api.main:app --reload"""

import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

from sec_rag.api.schemas import (
    ConversationDetail,
    CreateConversationRequest,
    QueryRequest,
    QueryResponse,
)
from sec_rag.chat.service import ChatService
from sec_rag.generation.llm import GenerationError
from sec_rag.models import Conversation, Source

logger = logging.getLogger(__name__)

GEMINI_UNAVAILABLE = "Couldn't get an answer from Gemini. Please try again in a moment."


@lru_cache
def get_chat() -> ChatService:
    return ChatService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models before accepting requests so the first query isn't slow.
    get_chat().pipeline.warm_up()
    yield


app = FastAPI(title="SEC 10-K Analyst RAG", lifespan=lifespan)


def _require_conversation(conversation_id: str) -> Conversation:
    conversation = get_chat().store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": get_chat().pipeline.retriever.store.count()}


@app.post("/conversations", response_model=Conversation, status_code=201)
def create_conversation(request: CreateConversationRequest) -> Conversation:
    return get_chat().store.create_conversation(request.title)


@app.get("/conversations", response_model=list[Conversation])
def list_conversations() -> list[Conversation]:
    return get_chat().store.list_conversations()


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str) -> ConversationDetail:
    conversation = _require_conversation(conversation_id)
    return ConversationDetail(
        conversation=conversation, messages=get_chat().store.get_messages(conversation_id)
    )


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> Response:
    if not get_chat().store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=204)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    chat = get_chat()
    try:
        if request.conversation_id:
            _require_conversation(request.conversation_id)
            result = chat.ask(
                request.conversation_id, request.question, request.filters(), request.top_k
            )
        else:
            result = chat.pipeline.ask(request.question, request.filters(), request.top_k)
    except GenerationError as exc:
        raise HTTPException(status_code=503, detail=f"{GEMINI_UNAVAILABLE} ({exc})") from exc
    return QueryResponse(
        answer=result.answer,
        standalone_question=result.standalone_question,
        conversation_id=request.conversation_id,
        sources=[Source.from_retrieved(rc, excerpt_chars=500) for rc in result.sources],
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    chat = get_chat()
    if request.conversation_id:
        _require_conversation(request.conversation_id)
        result = chat.ask_stream(
            request.conversation_id, request.question, request.filters(), request.top_k
        )
    else:
        result = chat.pipeline.ask_stream(request.question, request.filters(), request.top_k)

    def stream_with_errors() -> Iterator[str]:
        # Headers are already sent once streaming starts, so a failure is reported in the body.
        try:
            yield from result.text_stream
        except GenerationError:
            logger.warning("Gemini failed mid-stream", exc_info=True)
            yield f"\n\n[error] {GEMINI_UNAVAILABLE}"

    return StreamingResponse(stream_with_errors(), media_type="text/plain")

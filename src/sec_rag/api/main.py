"""FastAPI service. Run with: uvicorn sec_rag.api.main:app --reload"""

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from sec_rag.api.schemas import QueryRequest, QueryResponse, Source
from sec_rag.pipeline import RAGPipeline


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models before accepting requests so the first query isn't slow.
    get_pipeline().warm_up()
    yield


app = FastAPI(title="SEC 10-K Analyst RAG", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": get_pipeline().retriever.store.count()}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = get_pipeline().ask(request.question, filters=request.filters(), top_k=request.top_k)
    sources = [
        Source(
            ticker=rc.chunk.ticker,
            company_name=rc.chunk.company_name,
            fiscal_year=rc.chunk.fiscal_year,
            item=rc.chunk.item,
            item_title=rc.chunk.item_title,
            source_url=rc.chunk.source_url,
            score=rc.score,
            excerpt=rc.chunk.text[:500],
        )
        for rc in result.sources
    ]
    return QueryResponse(answer=result.answer, sources=sources)


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    _, text_stream = get_pipeline().ask_stream(
        request.question, filters=request.filters(), top_k=request.top_k
    )
    return StreamingResponse(text_stream, media_type="text/plain")

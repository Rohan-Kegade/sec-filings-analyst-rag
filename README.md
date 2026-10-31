# SEC Filings Analyst RAG

Ask natural-language questions about companies' SEC 10-K filings and get answers with citations.

```
EDGAR ──► download ──► parse into Items ──► chunk ──► embed ──► ChromaDB
                                                                   │
question ──► rewrite follow-up ──► extract filters ──► vector search ◄──┘
               (uses chat history)   (ticker/year)          │
                                              top-k chunks ─┴─► Gemini ──► cited answer
                                       + last 3 exchanges ──┘                │
                                                                 saved to data/chats.db
```

## Project structure

```
sec-filings-analyst-rag/
├── src/sec_rag/
│   ├── config.py              # All settings (env vars prefixed SEC_RAG_)
│   ├── models.py              # Pydantic models: filings, chunks, answers, chat messages
│   ├── ingestion/
│   │   ├── edgar_client.py    # Ticker -> CIK lookup, list & download 10-Ks from EDGAR
│   │   ├── parser.py          # HTML -> clean text -> Item sections (1, 1A, 7, 7A, 8, ...)
│   │   └── chunker.py         # Section-aware chunking with a contextual header per chunk
│   ├── indexing/
│   │   ├── embedder.py        # sentence-transformers embeddings (local, free)
│   │   ├── vector_store.py    # ChromaDB persistent store with metadata filters
│   │   └── pipeline.py        # Offline job: raw -> processed -> chunks -> vectors
│   ├── retrieval/
│   │   ├── query_parser.py    # Pull tickers / fiscal years out of the question
│   │   └── retriever.py       # Dense search + metadata filtering
│   ├── generation/
│   │   ├── prompts.py         # System, rewrite and context/history prompts
│   │   └── llm.py             # Gemini calls (streaming); failures raise GenerationError
│   ├── chat/
│   │   ├── store.py           # SQLite chat history: conversations + messages with sources
│   │   ├── condense.py        # Rewrite follow-ups into standalone questions
│   │   └── service.py         # ChatService: load history, answer, save the exchange
│   ├── pipeline.py            # RAGPipeline: rewrite + retrieve + generate
│   ├── api/                   # FastAPI service (queries, conversations, health)
│   └── cli.py                 # `sec-rag download | index | ask`
├── app/streamlit_app.py       # Chat UI with saved conversations
├── evaluation/                # Retrieval eval set + Hit@k / MRR script
├── tests/                     # Unit tests (parser, chunker, prompts, chat)
├── notebooks/                 # Exploration
└── data/
    ├── raw/                   # Downloaded 10-K HTML + metadata JSON (per ticker)
    ├── processed/             # Parsed filings as JSON
    ├── vectorstore/           # ChromaDB files
    └── chats.db               # Saved conversations (SQLite)
```

## Setup

Use Python 3.11–3.13 (PyTorch and ChromaDB wheels may lag behind the newest Python release).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -e ".[ui,dev]"
copy .env.example .env          # then fill in GEMINI_API_KEY and your EDGAR user agent
```

## Usage

```bash
sec-rag download AAPL MSFT NVDA --years 3
sec-rag index
sec-rag ask "What supply chain risks does Apple describe in FY2024?"

uvicorn sec_rag.api.main:app --reload   # API at http://localhost:8000/docs
streamlit run app/streamlit_app.py      # Chat UI
python evaluation/run_eval.py           # Retrieval quality
pytest
```

## Chats

Conversations are saved in `data/chats.db` and survive restarts. Follow-up questions
("What about MSFT?") are rewritten into standalone questions before retrieval, and the last
3 exchanges are sent to Gemini as context (`SEC_RAG_CHAT_HISTORY_TURNS`). An exchange is
saved only once its answer finishes, so failed answers aren't kept.

| Endpoint | Purpose |
|---|---|
| `POST /conversations` | Start a conversation (title defaults to the first question) |
| `GET /conversations` | List conversations, newest activity first |
| `GET /conversations/{id}` | A conversation with its messages and sources |
| `DELETE /conversations/{id}` | Delete a conversation |
| `POST /query`, `POST /query/stream` | Ask; pass `conversation_id` for follow-ups and saving |

Without `conversation_id`, queries are one-off and nothing is saved.

## Next steps

- **Hybrid search**: add BM25 next to dense vectors; exact terms like "EBITDA" or segment names matter in filings.
- **Reranking**: a cross-encoder (e.g. `BAAI/bge-reranker-base`) over the top 30 → top 8.
- **Tables**: Item 8 financial statements lose structure as plain text; use XBRL company facts (`data.sec.gov/api/xbrl/companyfacts`) for numeric questions.
- **Multi-company and multi-year questions**: break the question into one retrieval per company/year, then synthesize.

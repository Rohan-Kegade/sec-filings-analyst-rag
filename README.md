# SEC Filings Analyst RAG

Ask natural-language questions about companies' SEC 10-K filings and get answers with citations.

```
EDGAR ──► download ──► parse into Items ──► chunk ──► embed ──► ChromaDB
                                                                   │
                                                         + keyword index (BM25)
                                                                   │
question ──► rewrite follow-up ──► extract filters ──► vector + keyword search
               (uses chat history)   (ticker/year)          │ merged by rank (RRF)
                                              top-k chunks ─┴─► Claude ──► cited answer
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
│   │   ├── keyword_index.py   # SQLite FTS5 keyword index (BM25) with the same filters
│   │   └── pipeline.py        # Offline job: raw -> processed -> chunks -> vectors + keywords
│   ├── retrieval/
│   │   ├── query_parser.py    # Pull tickers / fiscal years out of the question
│   │   ├── fusion.py          # Reciprocal Rank Fusion of ranked result lists
│   │   └── retriever.py       # Hybrid (default), dense or keyword search + filters
│   ├── generation/
│   │   ├── prompts.py         # System, rewrite and context/history prompts
│   │   └── llm.py             # Claude calls (streaming); failures raise GenerationError
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
    ├── keyword_index.db       # Keyword search index (SQLite FTS5)
    └── chats.db               # Saved conversations (SQLite)
```

## Setup

Use Python 3.11–3.13 (PyTorch and ChromaDB wheels may lag behind the newest Python release).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -e ".[ui,dev]"
copy .env.example .env          # then fill in ANTHROPIC_API_KEY and your EDGAR user agent
```

## Usage

```bash
sec-rag download AAPL MSFT NVDA --years 3
sec-rag index
sec-rag ask "What supply chain risks does Apple describe in FY2024?"

uvicorn sec_rag.api.main:app --reload   # API at http://localhost:8000/docs
streamlit run app/streamlit_app.py      # Chat UI
python evaluation/run_eval.py           # Compare dense / keyword / hybrid retrieval
pytest
```

## Hybrid search

Retrieval runs a meaning-based vector search and a keyword (BM25) search, takes the top 30
from each, and merges them with Reciprocal Rank Fusion: chunks ranked well by both rise to
the top. Keyword search catches exact terms that meaning-based search can miss, such as
product names and codes ("CUDA", "H100", "Vision Pro").

- `SEC_RAG_RETRIEVAL_MODE`: `hybrid` (default), `dense` or `keyword`
- `sec-rag index` fills both indexes. For chunks indexed before hybrid search existed, run
  `sec-rag index --keyword-only` to build the keyword index without re-embedding.
- Source scores are merged rank scores (small numbers like 0.03), not percentages.

## Chats

Conversations are saved in `data/chats.db` and survive restarts. Follow-up questions
("What about MSFT?") are rewritten into standalone questions before retrieval, and the last
3 exchanges are sent to Claude as context (`SEC_RAG_CHAT_HISTORY_TURNS`). An exchange is
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

- **Reranking**: a cross-encoder (e.g. `BAAI/bge-reranker-base`) over the top 30 → top 8.
- **Tables**: Item 8 financial statements lose structure as plain text; use XBRL company facts (`data.sec.gov/api/xbrl/companyfacts`) for numeric questions.
- **Multi-company and multi-year questions**: break the question into one retrieval per company/year, then synthesize.

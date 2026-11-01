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
├── evaluation/
│   ├── goldens/               # 40 retrieval goldens (DeepEval format, with source quotes)
│   ├── retrieval_eval/        # DeepEval Contextual Recall / Precision with a Claude judge
│   ├── generator_eval/        # DeepEval Faithfulness / Answer Relevancy on frozen contexts
│   ├── rag_eval/              # RAG triad on the full pipeline (context/answer relevance, groundedness)
│   ├── app_eval/              # Application evals on the full pipeline (correctness so far)
│   ├── judge.py               # ClaudeJudge: Claude as a DeepEval judge model
│   └── run_eval.py            # Quick free check: section + exact-term hits per mode
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

## Retrieval evaluation (DeepEval)

`pip install -e ".[eval]"` adds DeepEval. Each golden's question goes through the
retriever only, then a Claude judge (`claude-sonnet-5-5` by default) scores the top-k
chunks against the golden's reference answer:

- **Contextual Recall**: share of the reference answer's statements supported by the
  retrieved chunks.
- **Contextual Precision**: whether relevant chunks are ranked above irrelevant ones.

```bash
python -m evaluation.retrieval_eval run                          # hybrid, all goldens
python -m evaluation.retrieval_eval run --mode dense --mode hybrid --category fact
python -m evaluation.retrieval_eval compare                      # last two runs
```

Every run calls the judge, so it costs money (roughly $0.05 per golden per mode with
Sonnet 5.5) and scores vary slightly between runs; differences under ~0.03 are likely
noise. Runs are saved to `evaluation/runs/retrieval/` with their settings.

## Generator evaluation (DeepEval)

Tests the answer generator alone: each of the 54 goldens in
`evaluation/goldens/generator_goldens.json` has a frozen set of excerpts (evidence plus
same-year distractors), so retrieval can't affect the scores. The generator
(`SEC_RAG_LLM_MODEL`) answers from those excerpts, then the judge scores:

- **Faithfulness**: share of the answer's claims supported by the excerpts it was given.
- **Answer Relevancy**: share of the answer's statements that address the question.

Besides the 40 regular questions, the goldens include partial-context, wrong-year,
model-already-knows and no-answer cases. For no-answer cases the report also shows whether
the answer said the excerpts don't contain the information (a simple text check).

```bash
python -m evaluation.generator_eval run                     # all 54 goldens
python -m evaluation.generator_eval run --category partial  # one category
python -m evaluation.generator_eval compare                 # last two runs
```

Roughly $0.04 per golden for the Sonnet judge, plus generation. Runs are saved to
`evaluation/runs/generator/`.

## End-to-end evaluation: the RAG triad (DeepEval)

Runs each question through the real pipeline (filters, hybrid retrieval, generation) and
scores the three links of the triad with the judge:

| Triad metric | DeepEval metric | Low score points to |
|---|---|---|
| Context relevance (question → excerpts) | `ContextualRelevancyMetric` | retrieval |
| Groundedness (excerpts → answer) | `FaithfulnessMetric` | the generator making things up |
| Answer relevance (question → answer) | `AnswerRelevancyMetric` | an off-target answer |

All three are reference-free, so only questions are needed: the 40 golden questions plus
5 out-of-scope ones in `evaluation/goldens/out_of_scope_goldens.json` (an unindexed company, a year
not downloaded, a long-range forecast, investment advice). Each case gets a diagnosis of
the first failing link, and out-of-scope questions are checked for not answering.

```bash
python -m evaluation.rag_eval run                         # all 45 questions
python -m evaluation.rag_eval run --category out_of_scope
python -m evaluation.rag_eval compare
```

Context relevance counts every statement in the excerpts, and 10-K chunks are long and only
partly on-topic, so its scores run low; compare it between runs rather than reading it as an
absolute grade. About $0.11 per question with the Sonnet judge (context relevance reads every
excerpt). The judge caps itself at 4 concurrent requests and retries rate limits.

## Application evaluation: correctness and completeness (DeepEval)

Runs each golden's question through the full pipeline **once**, then scores the answers with
each selected metric (DeepEval G-Eval with fixed steps):

| Metric | Golden field | Asks |
|---|---|---|
| **Correctness** | `key_facts`: what must be right | Are the key facts stated correctly, with no wrong numbers, years, companies or contradictions? Out-of-scope questions must decline instead of inventing an answer. |
| **Completeness** | `required_points`: what must be covered | Does the answer cover everything the question asks for? Coverage only; accuracy is correctness's job. Not applied to out-of-scope questions. |

Neither penalizes extra details beyond the reference. A free code check also reports whether
the key facts' numbers appear in the answer. The 40 goldens store `key_facts` and
`required_points` next to their reference answers in `retrieval_goldens.json`; the 5
out-of-scope questions are in `out_of_scope_goldens.json`.

```bash
python -m evaluation.app_eval run                             # all metrics, 45 goldens
python -m evaluation.app_eval run --metric completeness --category qualitative
python -m evaluation.app_eval compare                         # last two runs
```

Correctness is defined by the filings: the reference is what the company's 10-K for that
fiscal year states. About $0.01 per golden for both metrics with the Sonnet judge, plus
generation. Runs are saved to `evaluation/runs/app/`.

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

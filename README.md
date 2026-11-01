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
├── goldens/                   # Eval goldens: retrieval, generator, scope (hand-checked)
├── evals/                     # One file per eval (levels: see Evaluation below)
│   ├── eval_retriever.py      # 1. Contextual Recall / Precision
│   ├── eval_generator.py      # 1. Faithfulness / Answer Relevancy on frozen contexts
│   ├── eval_rag_triad.py      # 2. Context relevance, faithfulness, answer relevance
│   ├── eval_correctness_completeness.py  # 3. Quality: correctness, completeness
│   ├── eval_scope_adherence.py           # 3. Safety: scope adherence
│   ├── judge.py               # ClaudeJudge: Claude as a DeepEval judge model
│   └── runs.py                # Saving and loading eval runs (evals/runs/, not in git)
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
pytest
```

## Evaluation (DeepEval)

`pip install -e ".[eval]"` adds DeepEval. The evals are grouped in four levels, from single
components up to the whole suite:

| Level | Eval | File |
|---|---|---|
| 1. Component | retriever | `evals/eval_retriever.py` |
| | generator | `evals/eval_generator.py` |
| 2. Pipeline | RAG triad: context relevance, faithfulness, answer relevance | `evals/eval_rag_triad.py` |
| 3. Application | quality: correctness, completeness (style skipped) | `evals/eval_correctness_completeness.py` |
| | safety: scope adherence (toxicity, leakage TODO) | `evals/eval_scope_adherence.py` |
| | operations: latency, cost, tokens | TODO |
| 4. Regression | the whole suite against a baseline | TODO |

Each eval runs on its own: `python -m evals.<file> run`, then `compare` for its last two
runs. Every command has `--help`.

### 1. Component evals

#### Retriever

Each golden's question goes through the retriever only, then a Claude judge
(`claude-sonnet-5-5` by default) scores the top-k chunks against the golden's reference
answer:

- **Contextual Recall**: share of the reference answer's statements supported by the
  retrieved chunks.
- **Contextual Precision**: whether relevant chunks are ranked above irrelevant ones.

```bash
python -m evals.eval_retriever run                          # hybrid, all goldens
python -m evals.eval_retriever run --mode dense --mode hybrid --category fact
python -m evals.eval_retriever compare                      # last two runs
```

Every run calls the judge, so it costs money (roughly $0.05 per golden per mode with
Sonnet 5.5) and scores vary slightly between runs; differences under ~0.03 are likely
noise. Runs are saved to `evals/runs/retrieval/` with their settings.

#### Generator

Tests the answer generator alone: each of the 54 goldens in
`goldens/generator_goldens.json` has a frozen set of excerpts (evidence plus
same-year distractors), so retrieval can't affect the scores. The generator
(`SEC_RAG_LLM_MODEL`) answers from those excerpts, then the judge scores:

- **Faithfulness**: share of the answer's claims supported by the excerpts it was given.
- **Answer Relevancy**: share of the answer's statements that address the question.

Besides the 40 regular questions, the goldens include partial-context, wrong-year,
model-already-knows and no-answer cases. For no-answer cases the report also shows whether
the answer said the excerpts don't contain the information (a simple text check).

```bash
python -m evals.eval_generator run                     # all 54 goldens
python -m evals.eval_generator run --category partial  # one category
python -m evals.eval_generator compare                 # last two runs
```

Roughly $0.04 per golden for the Sonnet judge, plus generation. Runs are saved to
`evals/runs/generator/`.

### 2. Pipeline evals: the RAG triad

Runs each question through the real pipeline (filters, hybrid retrieval, generation) and
scores the three links of the triad with the judge:

| Triad metric | DeepEval metric | Low score points to |
|---|---|---|
| Context relevance (question → excerpts) | `ContextualRelevancyMetric` | retrieval |
| Faithfulness (excerpts → answer) | `FaithfulnessMetric` | the generator making things up |
| Answer relevance (question → answer) | `AnswerRelevancyMetric` | an off-target answer |

All three are reference-free, so only questions are needed: the 40 golden questions. Each
case gets a diagnosis of the first failing link. (Out-of-scope questions have their own
eval, below.)

```bash
python -m evals.eval_rag_triad run                         # all 40 questions
python -m evals.eval_rag_triad run --category comparison
python -m evals.eval_rag_triad compare
```

Context relevance counts every statement in the excerpts, and 10-K chunks are long and only
partly on-topic, so its scores run low; compare it between runs rather than reading it as an
absolute grade. About $0.11 per question with the Sonnet judge (context relevance reads every
excerpt). The judge caps itself at 4 concurrent requests and retries rate limits.

### 3. Application evals

#### Quality: correctness and completeness

Runs each golden's question through the full pipeline **once**, then scores the answers with
each selected metric (DeepEval G-Eval with fixed steps):

| Metric | Golden field | Asks |
|---|---|---|
| **Correctness** | `key_facts`: what must be right | Are the key facts stated correctly, with no wrong numbers, years, companies or contradictions? |
| **Completeness** | `required_points`: what must be covered | Does the answer cover everything the question asks for? Coverage only; accuracy is correctness's job. |

Neither penalizes extra details beyond the reference. A free code check also reports whether
the key facts' numbers appear in the answer. The 40 goldens store `key_facts` and
`required_points` next to their reference answers in `retrieval_goldens.json`.

```bash
python -m evals.eval_correctness_completeness run      # both metrics, 40 goldens
python -m evals.eval_correctness_completeness run --metric completeness --category qualitative
python -m evals.eval_correctness_completeness compare   # last two runs
```

Correctness is defined by the filings: the reference is what the company's 10-K for that
fiscal year states. About $0.01 per golden for both metrics with the Sonnet judge, plus
generation. Runs are saved to `evals/runs/app/`.

Style (tone, format, citations) is skipped for now.

#### Safety: scope adherence

Does the app stay within its job, answering questions about the indexed 10-Ks? Each case is
a request plus the behavior a safe answer shows; the real pipeline answers it and a G-Eval
judge checks the answer against that behavior. The 20 cases in `goldens/scope_goldens.json`
cover companies and years not indexed, predictions, investment advice, facts the model may
"know" but the filings don't contain, off-topic requests, prompt injection, provocation
(toxicity), and system-prompt and data leakage probes. A free code check also flags answers
that reproduce sentences of the app's own instructions.

```bash
python -m evals.eval_scope_adherence run                            # all 20 cases
python -m evals.eval_scope_adherence run --category prompt_injection
python -m evals.eval_scope_adherence compare                        # lists regressions
```

About $0.005 per case with the Sonnet judge, plus generation. Runs are saved to
`evals/runs/safety/`. Toxicity and leakage as their own metrics are TODO.

#### Operations (TODO)

Latency, cost and tokens per answer.

### 4. Regression (TODO)

Run the whole suite and compare each level against a saved baseline.

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

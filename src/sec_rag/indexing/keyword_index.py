"""Keyword search (BM25) over chunks, using SQLite's built-in FTS5 full-text index."""

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sec_rag.config import settings
from sec_rag.models import Chunk

# Porter stemming lets "risks" match "risk"; unicode61 handles accents and punctuation.
SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    chunk_id UNINDEXED,
    text,
    ticker UNINDEXED,
    fiscal_year UNINDEXED,
    item UNINDEXED,
    tokenize = 'porter unicode61'
);
"""

FILTER_COLUMNS = ("ticker", "fiscal_year", "item")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
# Question words that carry no topic. BM25 already down-weights common words, but dropping
# these keeps chunks that merely repeat "what does ... describe" from matching.
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can", "describe", "described",
    "did", "do", "does", "for", "from", "has", "have", "how", "in", "is", "it", "its",
    "mention", "mentioned", "of", "on", "or", "say", "said", "tell", "that", "the", "their",
    "these", "this", "those", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "with",
}  # fmt: skip
WRITE_BATCH_SIZE = 500


def to_match_query(question: str) -> str:
    """Turn free text into a safe FTS5 query: quoted terms joined by OR.

    Quoting every term means characters like ? : " or - in the question can never be read
    as FTS5 query syntax.
    """
    terms = dict.fromkeys(
        t for t in TOKEN_PATTERN.findall(question) if t.lower() not in STOPWORDS
    )
    return " OR ".join(f'"{t}"' for t in terms)


class KeywordIndex:
    def __init__(self, path: Path = settings.keyword_index_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            with conn:  # commit on success, roll back on error
                yield conn
        finally:
            conn.close()

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert chunks, replacing any already indexed under the same chunk_id."""
        with self._connect() as conn:
            for start in range(0, len(chunks), WRITE_BATCH_SIZE):
                batch = chunks[start : start + WRITE_BATCH_SIZE]
                placeholders = ",".join("?" * len(batch))
                conn.execute(
                    f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})",
                    [c.chunk_id for c in batch],
                )
                conn.executemany(
                    "INSERT INTO chunks (chunk_id, text, ticker, fiscal_year, item) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(c.chunk_id, c.text, c.ticker, str(c.fiscal_year), c.item) for c in batch],
                )

    def search(
        self, question: str, k: int, filters: dict | None = None
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, score) for the best keyword matches, highest score first."""
        match = to_match_query(question)
        if not match:
            return []

        where, params = ["chunks MATCH ?"], [match]
        for key, value in (filters or {}).items():
            if key not in FILTER_COLUMNS:
                raise ValueError(f"Unsupported keyword filter: {key}")
            values = value if isinstance(value, (list, tuple, set)) else [value]
            # Values are stored as text so integer years compare reliably.
            where.append(f"{key} IN ({','.join('?' * len(values))})")
            params.extend(str(v) for v in values)

        with self._connect() as conn:
            rows = conn.execute(
                # bm25() is lower-is-better, so negate it for a higher-is-better score.
                f"SELECT chunk_id, -bm25(chunks) AS score FROM chunks "
                f"WHERE {' AND '.join(where)} ORDER BY score DESC LIMIT ?",
                [*params, k],
            ).fetchall()
        return [(chunk_id, score) for chunk_id, score in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

"""Persistent chat history in a local SQLite file."""

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sec_rag.config import settings
from sec_rag.models import Answer, ChatMessage, ChatTurn, Conversation, Source

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources TEXT,
    standalone_question TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, id);
"""

TITLE_MAX_CHARS = 60


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _title_from(question: str) -> str:
    question = " ".join(question.split())
    return question if len(question) <= TITLE_MAX_CHARS else question[: TITLE_MAX_CHARS - 1] + "…"


class ChatStore:
    def __init__(self, path: Path = settings.chat_db_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A short-lived connection per call keeps this safe to use from the
        # worker threads that Streamlit and FastAPI run requests on.
        conn = sqlite3.connect(self.path)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:  # commit on success, roll back on error
                yield conn
        finally:
            conn.close()

    def create_conversation(self, title: str = "") -> Conversation:
        """Create a conversation. An empty title is filled from the first question."""
        now = _now()
        conversation = Conversation(
            id=uuid.uuid4().hex, title=title, created_at=now, updated_at=now
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation.id, conversation.title, now, now),
            )
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return Conversation(**row) if row else None

    def list_conversations(self) -> list[Conversation]:
        """Newest activity first."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()
        return [Conversation(**row) for row in rows]

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
            )
        return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cursor.rowcount > 0

    def add_exchange(self, conversation_id: str, answer: Answer) -> None:
        """Save a question and its answer together, so a chat never has half an exchange."""
        now = _now()
        sources = json.dumps([Source.from_retrieved(rc).model_dump() for rc in answer.sources])
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (?, 'user', ?, ?)",
                (conversation_id, answer.question, now),
            )
            conn.execute(
                "INSERT INTO messages "
                "(conversation_id, role, content, sources, standalone_question, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?, ?)",
                (conversation_id, answer.answer, sources, answer.standalone_question, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ?, "
                "title = CASE WHEN title = '' THEN ? ELSE title END WHERE id = ?",
                (now, _title_from(answer.question), conversation_id),
            )

    def get_messages(self, conversation_id: str) -> list[ChatMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
        return [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                sources=[Source(**s) for s in json.loads(row["sources"] or "[]")],
                standalone_question=row["standalone_question"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_turns(self, conversation_id: str) -> list[ChatTurn]:
        """All question/answer pairs in order, for follow-up context."""
        messages = self.get_messages(conversation_id)
        return [
            ChatTurn(question=user.content, answer=assistant.content)
            for user, assistant in zip(messages[::2], messages[1::2])
            if user.role == "user" and assistant.role == "assistant"
        ]

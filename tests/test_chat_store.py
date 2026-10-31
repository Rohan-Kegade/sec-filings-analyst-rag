from sec_rag.chat.store import TITLE_MAX_CHARS, ChatStore
from sec_rag.models import Answer, Chunk, RetrievedChunk


def make_chunk(ticker: str = "AAPL", item: str = "1A") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"acc-{item}-0",
        text=f"{ticker} 10-K FY2024 - Item {item}: Risk Factors\n\nSome text.",
        ticker=ticker,
        company_name="Apple Inc.",
        fiscal_year=2024,
        item=item,
        item_title="Risk Factors",
        accession_number="acc",
        source_url="https://example.com/filing",
        chunk_index=0,
    )
    return RetrievedChunk(chunk=chunk, score=0.8)


def make_answer(question: str, answer: str = "An answer [1].") -> Answer:
    return Answer(
        question=question, standalone_question=question, answer=answer, sources=[make_chunk()]
    )


def test_exchange_round_trip(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    conversation = store.create_conversation()
    store.add_exchange(conversation.id, make_answer("What risks does AAPL face?"))

    messages = store.get_messages(conversation.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "What risks does AAPL face?"
    assert messages[1].content == "An answer [1]."
    assert messages[1].sources[0].ticker == "AAPL"
    assert messages[1].sources[0].item == "1A"
    assert messages[1].standalone_question == "What risks does AAPL face?"


def test_title_comes_from_first_question(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    conversation = store.create_conversation()
    store.add_exchange(conversation.id, make_answer("First question about AAPL"))
    store.add_exchange(conversation.id, make_answer("Second question"))
    assert store.get_conversation(conversation.id).title == "First question about AAPL"

    long = store.create_conversation()
    store.add_exchange(long.id, make_answer("x" * 200))
    assert len(store.get_conversation(long.id).title) == TITLE_MAX_CHARS


def test_explicit_title_is_kept(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    conversation = store.create_conversation("Apple research")
    store.add_exchange(conversation.id, make_answer("What risks does AAPL face?"))
    assert store.get_conversation(conversation.id).title == "Apple research"


def test_turns_pair_questions_with_answers(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    conversation = store.create_conversation()
    store.add_exchange(conversation.id, make_answer("Q1", "A1"))
    store.add_exchange(conversation.id, make_answer("Q2", "A2"))
    turns = store.get_turns(conversation.id)
    assert [(t.question, t.answer) for t in turns] == [("Q1", "A1"), ("Q2", "A2")]


def test_list_orders_by_latest_activity(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    older = store.create_conversation()
    newer = store.create_conversation()
    store.add_exchange(older.id, make_answer("Q"))  # activity moves it to the top
    assert [c.id for c in store.list_conversations()] == [older.id, newer.id]


def test_delete_removes_conversation_and_messages(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    conversation = store.create_conversation()
    store.add_exchange(conversation.id, make_answer("Q"))

    assert store.delete_conversation(conversation.id)
    assert store.get_conversation(conversation.id) is None
    assert store.get_messages(conversation.id) == []
    assert not store.delete_conversation(conversation.id)


def test_rename(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    conversation = store.create_conversation()
    assert store.rename_conversation(conversation.id, "Renamed")
    assert store.get_conversation(conversation.id).title == "Renamed"

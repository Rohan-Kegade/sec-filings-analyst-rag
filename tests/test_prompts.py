from sec_rag.generation.prompts import build_rewrite_message, build_user_message, format_history
from sec_rag.models import ChatTurn
from tests.test_chat_store import make_chunk


def test_no_conversation_block_without_history():
    message = build_user_message("What risks?", [make_chunk()])
    assert "<conversation>" not in message
    assert message.endswith("Question: What risks?")


def test_history_is_included_before_excerpts():
    history = [ChatTurn(question="What risks does AAPL face?", answer="Supplier risk [2].")]
    message = build_user_message("Which is most serious?", [make_chunk()], history)
    assert message.index("<conversation>") < message.index("<excerpts>")
    assert "User: What risks does AAPL face?" in message
    assert message.endswith("Question: Which is most serious?")


def test_old_citations_are_stripped():
    history = [ChatTurn(question="Q", answer="Suppliers [2]. Tariffs [1, 3]. China [4-6].")]
    assert format_history(history) == "User: Q\nAssistant: Suppliers. Tariffs. China."


def test_rewrite_message_trims_long_answers():
    history = [ChatTurn(question="Q", answer="a" * 5000)]
    message = build_rewrite_message("What about MSFT?", history)
    assert len(message) < 2000
    assert message.endswith("Latest question: What about MSFT?")

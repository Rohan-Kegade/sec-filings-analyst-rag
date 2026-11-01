"""Chat UI with saved conversations. Run with: streamlit run app/streamlit_app.py"""

import streamlit as st

from sec_rag.chat.service import ChatService
from sec_rag.generation.llm import GenerationError
from sec_rag.models import Source


@st.cache_resource(show_spinner="Loading models (first start only)...")
def get_chat() -> ChatService:
    chat = ChatService()
    chat.pipeline.warm_up()
    return chat


def select_conversation(conversation_id: str | None) -> None:
    st.session_state.conversation_id = conversation_id


def delete_conversation(conversation_id: str) -> None:
    get_chat().store.delete_conversation(conversation_id)
    st.session_state.conversation_id = None


def render_sources(sources: list[Source]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {s.ticker} FY{s.fiscal_year} - Item {s.item}: {s.item_title}** "
                f"· relevance {s.score:.3f} · [filing]({s.source_url})"
            )
            st.caption(s.excerpt)


def render_searched_for(question: str, standalone_question: str | None) -> None:
    if standalone_question and standalone_question != question:
        st.caption(f"Searched for: {standalone_question}")


st.set_page_config(page_title="10-K Analyst", layout="wide")
st.title("SEC 10-K Analyst")

# Load models when the page opens, not when the first question is asked.
chat = get_chat()
store = chat.store
st.session_state.setdefault("conversation_id", None)
current_id = st.session_state.conversation_id

with st.sidebar:
    st.button("New chat", on_click=select_conversation, args=(None,), width="stretch")

    st.subheader("Chats")
    conversations = store.list_conversations()
    if not conversations:
        st.caption("No saved chats yet.")
    for conversation in conversations:
        st.button(
            conversation.title or "Untitled chat",
            key=f"conversation-{conversation.id}",
            on_click=select_conversation,
            args=(conversation.id,),
            type="primary" if conversation.id == current_id else "secondary",
            width="stretch",
        )
    if current_id:
        st.button(
            "Delete this chat", on_click=delete_conversation, args=(current_id,), width="stretch"
        )

    st.divider()
    ticker = st.text_input("Ticker filter (optional)").strip().upper() or None
    year = st.number_input("Fiscal year (0 = any)", min_value=0, max_value=2100, value=0)
    top_k = st.slider("Chunks to retrieve", 3, 20, 8)

# Earlier messages of the selected chat.
messages = store.get_messages(current_id) if current_id else []
for user_msg, assistant_msg in zip(messages[::2], messages[1::2]):
    st.chat_message("user").markdown(user_msg.content)
    with st.chat_message("assistant"):
        render_searched_for(user_msg.content, assistant_msg.standalone_question)
        st.markdown(assistant_msg.content)
        render_sources(assistant_msg.sources)

question = st.chat_input("Ask about a company's 10-K...")
if question:
    started_new_chat = current_id is None
    if started_new_chat:
        current_id = st.session_state.conversation_id = store.create_conversation().id

    st.chat_message("user").markdown(question)
    filters = {k: v for k, v in {"ticker": ticker, "fiscal_year": year or None}.items() if v}
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the filings..."):
                result = chat.ask_stream(current_id, question, filters=filters, top_k=top_k)
            render_searched_for(question, result.standalone_question)
            st.write_stream(result.text_stream)  # the exchange is saved when this finishes
        except GenerationError as exc:  # e.g. Gemini 503 "high demand" or a network error
            st.error(f"Couldn't get an answer from Gemini. Please try again in a moment.\n\n{exc}")
            if started_new_chat:  # don't leave an empty chat in the sidebar
                delete_conversation(current_id)
        else:
            st.rerun()  # redraw from the saved chat, with sources and the updated sidebar

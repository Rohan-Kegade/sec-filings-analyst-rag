"""Minimal chat UI. Run with: streamlit run app/streamlit_app.py"""

import streamlit as st

from sec_rag.pipeline import RAGPipeline


@st.cache_resource(show_spinner="Loading models (first start only)...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline().warm_up()


st.set_page_config(page_title="10-K Analyst", layout="wide")
st.title("SEC 10-K Analyst")

# Load models when the page opens, not when the first question is asked.
pipeline = get_pipeline()

with st.sidebar:
    ticker = st.text_input("Ticker filter (optional)").strip().upper() or None
    year = st.number_input("Fiscal year (0 = any)", min_value=0, max_value=2100, value=0)
    top_k = st.slider("Chunks to retrieve", 3, 20, 8)

question = st.chat_input("Ask about a company's 10-K...")
if question:
    st.chat_message("user").write(question)
    filters = {k: v for k, v in {"ticker": ticker, "fiscal_year": year or None}.items() if v}
    chunks, text_stream = pipeline.ask_stream(question, filters=filters, top_k=top_k)
    with st.chat_message("assistant"):
        try:
            st.write_stream(text_stream)
        except Exception as exc:  # e.g. Gemini 503 "high demand" or a network error
            st.error(f"Couldn't get an answer from Gemini. Please try again in a moment.\n\n{exc}")

    with st.expander("Sources"):
        for i, rc in enumerate(chunks, start=1):
            c = rc.chunk
            st.markdown(f"**[{i}] {c.ticker} FY{c.fiscal_year} - Item {c.item}: {c.item_title}**")
            st.caption(c.text[:600])

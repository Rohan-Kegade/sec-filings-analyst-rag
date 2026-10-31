from sec_rag.models import RetrievedChunk

SYSTEM_PROMPT = """\
You are a financial analyst answering questions about companies using excerpts from \
their SEC 10-K filings.

Answer using only the numbered excerpts provided. Cite every claim with the excerpt \
number in square brackets, e.g. [2]. Quote exact figures as they appear in the filing, \
including units and fiscal periods. If the excerpts do not contain enough information \
to answer, say what is missing rather than guessing. When comparing companies or years, \
keep each company's and year's figures clearly attributed."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        blocks.append(
            f'<excerpt index="{i}" company="{c.company_name}" ticker="{c.ticker}" '
            f'fiscal_year="{c.fiscal_year}" item="{c.item}">\n{c.text}\n</excerpt>'
        )
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"<excerpts>\n{format_context(chunks)}\n</excerpts>\n\nQuestion: {question}"

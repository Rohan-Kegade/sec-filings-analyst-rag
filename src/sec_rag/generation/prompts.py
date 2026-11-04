import re

from sec_rag.models import ChatTurn, RetrievedChunk

SYSTEM_PROMPT = """\
You are a financial analyst answering questions about companies using excerpts from \
their SEC 10-K filings.

Answer using only the numbered excerpts provided. Cite every claim with the excerpt \
number in square brackets, e.g. [2]. Quote exact figures as they appear in the filing, \
including units and fiscal periods. When comparing companies or years, keep each \
company's and year's figures clearly attributed.

Answer the question that was asked, directly and first. Add other facts only when they \
are needed to understand the answer; do not add figures for other years, other \
companies or related topics that the question did not ask about.

If the excerpts do not contain enough information to answer, say so in one sentence, \
naming what is missing, rather than guessing. Do not comment on the excerpts themselves \
(how they are numbered, cut off or labeled).

Tables appear one row per line with cells separated by " | "; the first row of a table \
usually holds the column headers. Read every figure against its column header and row \
label before using it.

A <conversation> block, if present, shows earlier questions and answers. Use it only to \
understand what the latest question refers to. Take facts and citation numbers only from \
the excerpts in this message: earlier answers were based on different excerpts."""

REWRITE_INSTRUCTIONS = """\
Rewrite the user's latest question so it can be understood without the conversation. \
Resolve references such as "they", "these", "that year" or "what about MSFT?" using the \
conversation. Refer to companies by their ticker symbol in capitals (e.g. AAPL) and keep \
any fiscal years and 10-K terms. If the question already stands alone, return it \
unchanged. Reply with only the rewritten question."""

# Matches citations like [2], [1, 3] or [4-6] so earlier answers' numbers are not reused.
CITATION_PATTERN = re.compile(r"\s*\[\d+(?:\s*[,\-–]\s*\d+)*\]")


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        blocks.append(
            f'<excerpt index="{i}" company="{c.company_name}" ticker="{c.ticker}" '
            f'fiscal_year="{c.fiscal_year}" item="{c.item}">\n{c.text}\n</excerpt>'
        )
    return "\n\n".join(blocks)


def format_history(turns: list[ChatTurn], max_answer_chars: int | None = None) -> str:
    lines = []
    for turn in turns:
        answer = CITATION_PATTERN.sub("", turn.answer).strip()
        if max_answer_chars and len(answer) > max_answer_chars:
            answer = answer[:max_answer_chars] + "…"
        lines.append(f"User: {turn.question}\nAssistant: {answer}")
    return "\n\n".join(lines)


def build_user_message(
    question: str, chunks: list[RetrievedChunk], history: list[ChatTurn] | None = None
) -> str:
    parts = []
    if history:
        parts.append(f"<conversation>\n{format_history(history)}\n</conversation>")
    parts.append(f"<excerpts>\n{format_context(chunks)}\n</excerpts>")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def build_rewrite_message(question: str, history: list[ChatTurn]) -> str:
    # The topic is all the rewrite needs, so long earlier answers are trimmed.
    return (
        f"<conversation>\n{format_history(history, max_answer_chars=1500)}\n</conversation>\n\n"
        f"Latest question: {question}"
    )

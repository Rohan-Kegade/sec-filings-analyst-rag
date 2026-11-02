"""Claude reranker: reads the question and each candidate chunk together and scores the match.

Dense and keyword search score the question and each chunk separately (a vector each, or
word counts), so a chunk that answers the question can still rank low. One Claude call
(Haiku by default) scores all the top fused candidates at once; the best k are kept.
"""

from functools import cached_property

import anthropic
from pydantic import BaseModel, Field

from sec_rag.config import settings
from sec_rag.generation.llm import FALLBACK_BETA
from sec_rag.models import Chunk

RERANK_SYSTEM = """\
You rank passages from SEC 10-K filings for a retrieval system.

For each passage, score how useful it is for answering the question:
10 = directly contains the facts the answer needs
7  = clearly relevant, contains part of the answer
4  = same topic, but not the information asked for
0  = unrelated, or about a different company or year than the question asks for

Judge each passage on its own content. Give every passage a score."""


class PassageScore(BaseModel):
    id: int = Field(description="The passage number")
    score: int = Field(description="0-10")


class RerankScores(BaseModel):
    scores: list[PassageScore]


def passage(chunk: Chunk) -> str:
    """Chunk text with its company, year and section, which the text alone often lacks."""
    return (
        f"{chunk.company_name} ({chunk.ticker}) FY{chunk.fiscal_year} 10-K, "
        f"Item {chunk.item} {chunk.item_title}\n{chunk.text}"
    )


class Reranker:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = settings.rerank_model,
        effort: str = settings.rerank_effort,
    ):
        self._client = client
        self.model = model
        self.effort = effort
        self.input_tokens = 0
        self.output_tokens = 0

    @cached_property
    def client(self) -> anthropic.Anthropic:
        return self._client or anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def warm_up(self) -> None:
        _ = self.client

    def score(self, question: str, chunks: list[Chunk]) -> list[float]:
        """One relevance score per chunk (higher = better match), in the given order."""
        if not chunks:
            return []
        passages = "\n\n".join(
            f'<passage id="{i}">\n{passage(chunk)}\n</passage>'
            for i, chunk in enumerate(chunks, start=1)
        )
        prompt = f"{passages}\n\nQuestion: {question}\n\nScore passages 1-{len(chunks)}."
        message = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=RERANK_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            output_format=RerankScores,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        self.input_tokens += message.usage.input_tokens
        self.output_tokens += message.usage.output_tokens
        if message.stop_reason in ("refusal", "max_tokens") or message.parsed_output is None:
            raise RuntimeError(f"Reranker returned no scores (stop_reason={message.stop_reason})")
        by_id = {s.id: float(s.score) for s in message.parsed_output.scores}
        # A passage the model skipped gets -1, below every scored one.
        return [by_id.get(i, -1.0) for i in range(1, len(chunks) + 1)]

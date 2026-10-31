"""Answer generation with Claude."""

from collections.abc import Iterator

import anthropic

from sec_rag.config import settings
from sec_rag.generation.prompts import SYSTEM_PROMPT, build_user_message
from sec_rag.models import RetrievedChunk

# Server-side fallback: if a request is declined by a safety classifier, the API
# re-runs it on a fallback model chosen for the refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnswerGenerator:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = settings.llm_model):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def _stream(self, question: str, chunks: list[RetrievedChunk]):
        return self.client.beta.messages.stream(
            model=self.model,
            max_tokens=settings.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, chunks)}],
            thinking={"type": "adaptive"},
            output_config={"effort": settings.llm_effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        with self._stream(question, chunks) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            return "The model declined to answer this question."
        return "".join(block.text for block in message.content if block.type == "text")

    def stream_text(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        with self._stream(question, chunks) as stream:
            yield from stream.text_stream

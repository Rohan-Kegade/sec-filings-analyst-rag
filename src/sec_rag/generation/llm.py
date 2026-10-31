"""Answer generation with Google Gemini (Interactions API)."""

from collections.abc import Iterator

from google import genai

from sec_rag.config import settings
from sec_rag.generation.prompts import SYSTEM_PROMPT, build_user_message
from sec_rag.models import RetrievedChunk


class AnswerGenerator:
    def __init__(self, client: genai.Client | None = None, model: str = settings.llm_model):
        # GEMINI_API_KEY comes from the environment or .env via settings.
        self.client = client or genai.Client(api_key=settings.gemini_api_key)
        self.model = model

    def _create(self, question: str, chunks: list[RetrievedChunk], stream: bool):
        return self.client.interactions.create(
            model=self.model,
            system_instruction=SYSTEM_PROMPT,
            input=build_user_message(question, chunks),
            generation_config={"thinking_level": settings.llm_thinking_level},
            stream=stream,
        )

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        interaction = self._create(question, chunks, stream=False)
        return interaction.output_text or "The model returned no answer for this question."

    def stream_text(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        for event in self._create(question, chunks, stream=True):
            if event.event_type == "step.delta" and event.delta.type == "text":
                yield event.delta.text

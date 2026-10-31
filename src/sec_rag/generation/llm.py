"""Answer generation with Google Gemini (Interactions API)."""

from collections.abc import Iterator

from google import genai

from sec_rag.config import settings
from sec_rag.generation.prompts import SYSTEM_PROMPT, build_user_message
from sec_rag.models import ChatTurn, RetrievedChunk


class GenerationError(RuntimeError):
    """A Gemini call failed: model overloaded (503), rate limit, network, bad API key, ..."""


class AnswerGenerator:
    def __init__(self, client: genai.Client | None = None, model: str = settings.llm_model):
        # GEMINI_API_KEY comes from the environment or .env via settings.
        self.client = client or genai.Client(api_key=settings.gemini_api_key)
        self.model = model

    # The SDK's Interactions API error classes are only exported from a private module,
    # so any failure inside a Gemini call is re-raised as GenerationError.

    def complete(
        self,
        system_instruction: str,
        prompt: str,
        thinking_level: str = settings.llm_thinking_level,
    ) -> str:
        try:
            interaction = self.client.interactions.create(
                model=self.model,
                system_instruction=system_instruction,
                input=prompt,
                generation_config={"thinking_level": thinking_level},
            )
        except Exception as exc:
            raise GenerationError(str(exc)) from exc
        return interaction.output_text or ""

    def complete_stream(
        self,
        system_instruction: str,
        prompt: str,
        thinking_level: str = settings.llm_thinking_level,
    ) -> Iterator[str]:
        try:
            for event in self.client.interactions.create(
                model=self.model,
                system_instruction=system_instruction,
                input=prompt,
                generation_config={"thinking_level": thinking_level},
                stream=True,
            ):
                if event.event_type == "step.delta" and event.delta.type == "text":
                    yield event.delta.text
        except Exception as exc:
            raise GenerationError(str(exc)) from exc

    def generate(
        self, question: str, chunks: list[RetrievedChunk], history: list[ChatTurn] | None = None
    ) -> str:
        answer = self.complete(SYSTEM_PROMPT, build_user_message(question, chunks, history))
        return answer or "The model returned no answer for this question."

    def stream_text(
        self, question: str, chunks: list[RetrievedChunk], history: list[ChatTurn] | None = None
    ) -> Iterator[str]:
        return self.complete_stream(SYSTEM_PROMPT, build_user_message(question, chunks, history))

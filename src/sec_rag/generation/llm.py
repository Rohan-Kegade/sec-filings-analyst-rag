"""Answer generation with Claude (Anthropic Messages API, streaming)."""

import threading
from collections.abc import Iterator

import anthropic

from sec_rag.config import settings
from sec_rag.generation.prompts import SYSTEM_PROMPT, build_user_message
from sec_rag.models import ChatTurn, RetrievedChunk

# Server-side fallback: if a safety classifier declines a request, the API re-runs it on a
# fallback model chosen for the refusal category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
REFUSAL_MESSAGE = "The model declined to answer this question."


class GenerationError(RuntimeError):
    """A Claude call failed: overloaded, rate limit, network, bad API key, ..."""


class AnswerGenerator:
    def __init__(
        self, client: anthropic.Anthropic | None = None, model: str = settings.llm_model
    ):
        # ANTHROPIC_API_KEY comes from the environment or .env via settings. When unset, the
        # SDK falls back to its own credential lookup (e.g. an `ant auth login` profile).
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = model
        # Token use across all calls, e.g. for an eval's cost line. Calls may run in threads.
        self.input_tokens = 0
        self.output_tokens = 0
        self._usage_lock = threading.Lock()

    def _count_usage(self, message) -> None:
        usage = getattr(message, "usage", None)
        if usage is None:
            return
        with self._usage_lock:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens

    def _stream(self, system: str, prompt: str, effort: str, max_tokens: int):
        return self.client.beta.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )

    def complete(
        self,
        system: str,
        prompt: str,
        effort: str = settings.llm_effort,
        max_tokens: int = settings.llm_max_tokens,
    ) -> str:
        try:
            with self._stream(system, prompt, effort, max_tokens) as stream:
                message = stream.get_final_message()
        except anthropic.APIError as exc:
            raise GenerationError(str(exc)) from exc
        self._count_usage(message)
        if message.stop_reason == "refusal":
            return REFUSAL_MESSAGE
        return "".join(block.text for block in message.content if block.type == "text")

    def complete_stream(
        self,
        system: str,
        prompt: str,
        effort: str = settings.llm_effort,
        max_tokens: int = settings.llm_max_tokens,
    ) -> Iterator[str]:
        try:
            with self._stream(system, prompt, effort, max_tokens) as stream:
                produced_text = False
                for text in stream.text_stream:
                    produced_text = True
                    yield text
                message = stream.get_final_message()
                self._count_usage(message)
                # A refusal that even the fallback model declines ends with no text.
                if message.stop_reason == "refusal" and not produced_text:
                    yield REFUSAL_MESSAGE
        except anthropic.APIError as exc:
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

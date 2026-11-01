"""Claude as a DeepEval judge model."""

import asyncio

import anthropic
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from sec_rag.config import settings
from sec_rag.generation.llm import FALLBACK_BETA

# Override with SEC_RAG_JUDGE_MODEL / SEC_RAG_JUDGE_EFFORT in .env or the environment.
DEFAULT_JUDGE_MODEL = settings.judge_model
DEFAULT_JUDGE_EFFORT = settings.judge_effort
# $ per million input/output tokens, for the cost estimate printed after a run.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0),
          "claude-haiku-5-5": (0.10, 0.50)}  # fmt: skip
# Some metrics (e.g. Contextual Relevancy) make one judge call per excerpt, all at once, which
# can exceed the account's concurrent-request limit (HTTP 429). Cap in-flight calls and let
# the SDK retry rate-limited requests longer than its default 2 times.
MAX_CONCURRENT_REQUESTS = 4
MAX_RETRIES = 6


class ClaudeJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by Claude.

    DeepEval passes a Pydantic `schema` when it needs structured output; Claude's structured
    outputs (`messages.parse`) return an instance of it directly. Current Claude models reject
    `temperature`, so none is sent; depth is set with `effort` instead.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        effort: str = DEFAULT_JUDGE_EFFORT,
        max_tokens: int = 8000,
        client: anthropic.Anthropic | None = None,
        async_client: anthropic.AsyncAnthropic | None = None,
        max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
    ):
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client
        self.async_client = async_client or anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key, max_retries=MAX_RETRIES
        )
        self.max_concurrent_requests = max_concurrent_requests
        self._limit: asyncio.Semaphore | None = None
        self._limit_loop: asyncio.AbstractEventLoop | None = None
        self.input_tokens = 0
        self.output_tokens = 0
        super().__init__(model)  # sets self.name and calls load_model()

    def load_model(self) -> anthropic.Anthropic:
        return self._client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key, max_retries=MAX_RETRIES
        )

    def _concurrency_limit(self) -> asyncio.Semaphore:
        # A semaphore belongs to one event loop; make a new one if the loop changed.
        loop = asyncio.get_running_loop()
        if self._limit is None or self._limit_loop is not loop:
            self._limit = asyncio.Semaphore(self.max_concurrent_requests)
            self._limit_loop = loop
        return self._limit

    def get_model_name(self) -> str:
        return self.name

    def _request(self, prompt: str) -> dict:
        return {
            "model": self.name,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def _result(self, message, schema: type[BaseModel] | None):
        self.input_tokens += message.usage.input_tokens
        self.output_tokens += message.usage.output_tokens
        if message.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"Judge stopped early (stop_reason={message.stop_reason})")
        if schema is not None:
            if message.parsed_output is None:
                raise RuntimeError("Judge returned no parsable output")
            return message.parsed_output
        return "".join(block.text for block in message.content if block.type == "text")

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        messages = self.model.beta.messages
        if schema is not None:
            message = messages.parse(**self._request(prompt), output_format=schema)
            return self._result(message, schema)
        return self._result(messages.create(**self._request(prompt)), None)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        messages = self.async_client.beta.messages
        async with self._concurrency_limit():
            if schema is not None:
                message = await messages.parse(**self._request(prompt), output_format=schema)
                return self._result(message, schema)
            return self._result(await messages.create(**self._request(prompt)), None)

    def cost_summary(self) -> str:
        summary = f"judge tokens: {self.input_tokens:,} in / {self.output_tokens:,} out"
        if self.name in PRICES:
            price_in, price_out = PRICES[self.name]
            cost = self.input_tokens / 1e6 * price_in + self.output_tokens / 1e6 * price_out
            summary += f" (~${cost:.2f})"
        return summary

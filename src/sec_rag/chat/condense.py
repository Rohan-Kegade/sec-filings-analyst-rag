"""Turn a follow-up question into a standalone question for retrieval."""

import logging

from sec_rag.config import settings
from sec_rag.generation.llm import AnswerGenerator, GenerationError
from sec_rag.generation.prompts import REWRITE_INSTRUCTIONS, build_rewrite_message
from sec_rag.models import ChatTurn

logger = logging.getLogger(__name__)


class QuestionRewriter:
    def __init__(self, generator: AnswerGenerator):
        self.generator = generator

    def rewrite(self, question: str, history: list[ChatTurn]) -> str:
        """Return a standalone version of `question`, or `question` itself on any failure."""
        if not history:
            return question
        try:
            rewritten = self.generator.complete(
                REWRITE_INSTRUCTIONS,
                build_rewrite_message(question, history),
                thinking_level=settings.rewrite_thinking_level,
            )
        except GenerationError:
            # A failed rewrite shouldn't block the answer; search with the original text.
            logger.warning("Question rewrite failed; using the original question", exc_info=True)
            return question
        return rewritten.strip().strip('"').strip() or question

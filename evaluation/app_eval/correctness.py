"""Correctness: does the answer get the golden's key facts right, without contradictions?

Scored by DeepEval's G-Eval with fixed evaluation steps, plus a free code check that the
numbers in the key facts appear in the answer.
"""

import re

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import SingleTurnParams

from evaluation.app_eval.goldens import AppGolden

EVALUATION_STEPS = [
    "Read the question, the expected output (a reference answer based on the company's 10-K "
    "filing) and the key facts listed at its end.",
    "For each key fact, check whether the actual output states it correctly. Equivalent "
    "wording, rounding and units are fine (for example 'about 228,000' matches "
    "'approximately 228,000'). A key fact that is missing counts against correctness.",
    "Heavily penalize any statement that contradicts the expected output, or that gives a "
    "wrong number, date, fiscal year or company for a key fact.",
    "Do not penalize additional details that the expected output does not mention, unless "
    "they contradict it.",
    "If the expected output says the information is not available, or describes how to "
    "respond instead of giving facts (for example, not giving investment advice), the actual "
    "output is correct only if it responds that way and does not invent an answer.",
]

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def expected_with_key_facts(golden: AppGolden) -> str:
    """The reference answer followed by its key facts, as the judge sees it."""
    facts = "\n".join(f"- {fact}" for fact in golden.key_facts)
    return f"{golden.expected_output}\n\nKey facts:\n{facts}"


def correctness_metric(judge: DeepEvalBaseLLM) -> GEval:
    return GEval(
        name="Correctness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=EVALUATION_STEPS,
        model=judge,
    )


def _normalize_number(token: str) -> str:
    token = token.replace(",", "")
    return token[:-2] if token.endswith(".0") else token  # "$95.0 billion" == "$95 billion"


def key_numbers(golden: AppGolden) -> set[str]:
    return {_normalize_number(n) for fact in golden.key_facts for n in _NUMBER.findall(fact)}


def number_check(golden: AppGolden, answer: str | None) -> tuple[int, int]:
    """(numbers from the key facts found in the answer, total). A free, exact pre-check."""
    expected = key_numbers(golden)
    found = {_normalize_number(n) for n in _NUMBER.findall(answer or "")}
    return len(expected & found), len(expected)

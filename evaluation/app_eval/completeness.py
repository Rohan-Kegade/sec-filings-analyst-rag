"""Completeness: does the answer cover everything the question asks for?

Scored by DeepEval's G-Eval against the golden's required points. Whether stated figures are
right is correctness's job, so completeness only checks coverage.
"""

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import SingleTurnParams

from evaluation.app_eval.goldens import AppGolden

EVALUATION_STEPS = [
    "Read the question and the required points listed in the expected output: everything a "
    "complete answer to this question must cover.",
    "For each required point, decide whether the actual output covers it fully, partly or "
    "not at all. Wording does not matter, only whether the point is addressed.",
    "The score reflects the share of required points covered; partly covered points count "
    "for less than fully covered ones.",
    "Do not judge whether figures or claims are accurate; correctness is evaluated "
    "separately. A point counts as covered if the answer addresses it.",
    "Do not penalize additional details beyond the required points.",
    "If the answer says the information for a point is not available instead of covering "
    "it, count that point as not covered.",
]


def expected_with_points(golden: AppGolden) -> str:
    """The reference answer followed by the required points, as the judge sees it."""
    points = "\n".join(f"- {point}" for point in golden.required_points)
    return f"{golden.expected_output}\n\nRequired points:\n{points}"


def completeness_metric(judge: DeepEvalBaseLLM) -> GEval:
    return GEval(
        name="Completeness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=EVALUATION_STEPS,
        model=judge,
    )

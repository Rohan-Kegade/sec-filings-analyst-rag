"""Questions for the RAG triad: the retrieval goldens' questions.

Out-of-scope questions live in the separate scope adherence eval (evaluation/scope_eval).
"""

from dataclasses import dataclass

from evaluation.retrieval_eval.goldens import load_goldens


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    input: str


def load_questions() -> list[Question]:
    return [Question(id=g.id, category=g.category, input=g.input) for g in load_goldens()]

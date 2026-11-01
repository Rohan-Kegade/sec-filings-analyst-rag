"""Questions for the RAG triad: the retrieval goldens' questions plus out-of-scope ones."""

import json
from dataclasses import dataclass
from pathlib import Path

from evaluation.retrieval_eval.goldens import load_goldens

OUT_OF_SCOPE_FILE = Path(__file__).parents[1] / "goldens" / "out_of_scope_goldens.json"


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    expected_behavior: str  # "answer", "abstain" or "decline_advice"
    input: str


def load_questions(out_of_scope_file: Path = OUT_OF_SCOPE_FILE) -> list[Question]:
    questions = [
        Question(id=g.id, category=g.category, expected_behavior="answer", input=g.input)
        for g in load_goldens()
    ]
    for raw in json.loads(out_of_scope_file.read_text(encoding="utf-8")):
        questions.append(
            Question(
                id=raw["id"],
                category=raw["category"],
                expected_behavior=raw["expected_behavior"],
                input=raw["input"],
            )
        )
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate question ids: {duplicates}")
    return questions

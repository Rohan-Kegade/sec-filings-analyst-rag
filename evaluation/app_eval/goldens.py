"""Correctness goldens: question, reference answer and key facts.

Read from the existing golden files so each reference answer lives in one place:
retrieval_goldens.json (40 questions) and out_of_scope_goldens.json (5 out-of-scope questions).
"""

import json
from dataclasses import dataclass
from pathlib import Path

GOLDENS_DIR = Path(__file__).parents[1] / "goldens"
RETRIEVAL_GOLDENS = GOLDENS_DIR / "retrieval_goldens.json"
OUT_OF_SCOPE = GOLDENS_DIR / "out_of_scope_goldens.json"


@dataclass(frozen=True)
class CorrectnessGolden:
    id: str
    category: str
    expected_behavior: str  # "answer", "abstain" or "decline_advice"
    status: str
    input: str
    expected_output: str
    key_facts: tuple[str, ...]


def load_goldens(
    retrieval_file: Path = RETRIEVAL_GOLDENS, out_of_scope_file: Path = OUT_OF_SCOPE
) -> list[CorrectnessGolden]:
    goldens = []
    for raw in json.loads(retrieval_file.read_text(encoding="utf-8")):
        meta = raw["additional_metadata"]
        goldens.append(CorrectnessGolden(
            id=meta["id"], category=meta["category"], expected_behavior="answer",
            status=meta.get("status", "draft"), input=raw["input"],
            expected_output=raw["expected_output"], key_facts=tuple(meta["key_facts"]),
        ))  # fmt: skip
    for raw in json.loads(out_of_scope_file.read_text(encoding="utf-8")):
        goldens.append(CorrectnessGolden(
            id=raw["id"], category=raw["category"], expected_behavior=raw["expected_behavior"],
            status=raw.get("status", "draft"), input=raw["input"],
            expected_output=raw["expected_output"], key_facts=tuple(raw["key_facts"]),
        ))  # fmt: skip
    ids = [g.id for g in goldens]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate golden ids: {duplicates}")
    return goldens

"""Application goldens: question, reference answer, key facts and required points.

Read from retrieval_goldens.json, so each reference answer lives in one place. Out-of-scope
questions live in the separate scope adherence eval (evaluation/scope_eval).
"""

import json
from dataclasses import dataclass
from pathlib import Path

RETRIEVAL_GOLDENS = Path(__file__).parents[1] / "goldens" / "retrieval_goldens.json"


@dataclass(frozen=True)
class AppGolden:
    id: str
    category: str
    status: str
    input: str
    expected_output: str
    key_facts: tuple[str, ...]  # what must be right (correctness)
    required_points: tuple[str, ...] = ()  # what must be covered (completeness)


def load_goldens(path: Path = RETRIEVAL_GOLDENS) -> list[AppGolden]:
    goldens = []
    for raw in json.loads(path.read_text(encoding="utf-8")):
        meta = raw["additional_metadata"]
        goldens.append(AppGolden(
            id=meta["id"], category=meta["category"], status=meta.get("status", "draft"),
            input=raw["input"], expected_output=raw["expected_output"],
            key_facts=tuple(meta["key_facts"]), required_points=tuple(meta["required_points"]),
        ))  # fmt: skip
    ids = [g.id for g in goldens]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate golden ids: {duplicates}")
    return goldens

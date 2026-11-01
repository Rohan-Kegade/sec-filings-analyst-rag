"""Load the retrieval goldens (DeepEval golden format) from evaluation/goldens/."""

import json
from dataclasses import dataclass
from pathlib import Path

GOLDENS_FILE = Path(__file__).parents[1] / "goldens" / "retrieval_goldens.json"


@dataclass(frozen=True)
class Golden:
    id: str
    category: str
    status: str
    input: str
    expected_output: str


def load_goldens(path: Path = GOLDENS_FILE) -> list[Golden]:
    goldens = []
    for raw in json.loads(path.read_text(encoding="utf-8")):
        meta = raw["additional_metadata"]
        goldens.append(
            Golden(
                id=meta["id"],
                category=meta["category"],
                status=meta.get("status", "draft"),
                input=raw["input"],
                expected_output=raw["expected_output"],
            )
        )
    ids = [g.id for g in goldens]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate golden ids in {path.name}: {duplicates}")
    return goldens

"""Load the scope adherence cases."""

import json
from dataclasses import dataclass
from pathlib import Path

SCOPE_GOLDENS = Path(__file__).parents[1] / "goldens" / "scope_goldens.json"


@dataclass(frozen=True)
class ScopeCase:
    id: str
    category: str  # the kind of out-of-scope request, e.g. "investment_advice"
    input: str
    expected_output: str  # what correct behavior looks like for this request


def load_cases(path: Path = SCOPE_GOLDENS) -> list[ScopeCase]:
    cases = [
        ScopeCase(id=raw["id"], category=raw["category"], input=raw["input"],
                  expected_output=raw["expected_output"])
        for raw in json.loads(path.read_text(encoding="utf-8"))
    ]  # fmt: skip
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate scope case ids: {duplicates}")
    return cases

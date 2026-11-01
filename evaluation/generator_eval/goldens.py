"""Load the generator goldens and turn their frozen excerpts back into chunks."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from sec_rag.models import Chunk, RetrievedChunk

GOLDENS_FILE = Path(__file__).parents[1] / "goldens" / "generator_goldens.json"

# Every chunk starts with the header the chunker adds, e.g.
# "Microsoft Corporation (MSFT) 10-K FY2024 - Item 1: Business"
HEADER = re.compile(
    r"^(?P<company>.+?) \((?P<ticker>[A-Z.]+)\) 10-K FY(?P<year>\d{4}) - "
    r"Item (?P<item>[0-9A-Z]+): ?(?P<title>.*)$"
)


@dataclass(frozen=True)
class GeneratorGolden:
    id: str
    category: str
    stress: str | None
    expected_behavior: str  # "answer" or "abstain"
    status: str
    input: str
    context: tuple[str, ...]
    context_chunks: tuple[str, ...]
    expected_output: str


def load_goldens(path: Path = GOLDENS_FILE) -> list[GeneratorGolden]:
    goldens = []
    for raw in json.loads(path.read_text(encoding="utf-8")):
        meta = raw["additional_metadata"]
        goldens.append(
            GeneratorGolden(
                id=meta["id"],
                category=meta["category"],
                stress=meta.get("stress"),
                expected_behavior=meta.get("expected_behavior", "answer"),
                status=meta.get("status", "draft"),
                input=raw["input"],
                context=tuple(raw["context"]),
                context_chunks=tuple(meta["context_chunks"]),
                expected_output=raw.get("expected_output", ""),
            )
        )
    ids = [g.id for g in goldens]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate golden ids in {path.name}: {duplicates}")
    return goldens


def to_chunks(golden: GeneratorGolden) -> list[RetrievedChunk]:
    """Rebuild the chunks the generator's prompt expects from the frozen excerpt text."""
    chunks = []
    for chunk_id, text in zip(golden.context_chunks, golden.context):
        match = HEADER.match(text.split("\n", 1)[0])
        if not match:
            raise ValueError(f"{golden.id}: excerpt without a chunk header: {text[:80]!r}")
        accession, _, index = chunk_id.rsplit("-", 2)
        chunk = Chunk(
            chunk_id=chunk_id,
            text=text,
            ticker=match["ticker"],
            company_name=match["company"],
            fiscal_year=int(match["year"]),
            item=match["item"],
            item_title=match["title"],
            accession_number=accession,
            source_url="",
            chunk_index=int(index),
        )
        chunks.append(RetrievedChunk(chunk=chunk, score=1.0))
    return chunks

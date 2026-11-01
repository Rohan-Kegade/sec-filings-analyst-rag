"""Retrieval evaluation comparing dense, keyword and hybrid retrieval. No LLM calls.

Two checks, each applied to the cases that define its field:
  - section ("ticker" + "item"): rank of the first chunk from the expected company and Item.
    "item" may list several acceptable Items. Reported as Hit@k and MRR.
  - term ("term"): how many of the top-k chunks contain an exact term, e.g. "H100". This is
    where keyword search helps, since meaning-based search can miss product names and codes.

Run with: python evaluation/run_eval.py
"""

import json
from pathlib import Path

from sec_rag.indexing.embedder import Embedder
from sec_rag.indexing.keyword_index import KeywordIndex
from sec_rag.indexing.vector_store import VectorStore
from sec_rag.models import RetrievedChunk
from sec_rag.retrieval.retriever import Retriever

EVAL_FILE = Path(__file__).parent / "eval_set.jsonl"
MODES = ("dense", "keyword", "hybrid")


def load_cases() -> list[dict]:
    lines = EVAL_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def section_rank(results: list[RetrievedChunk], case: dict) -> int | None:
    items = case["item"] if isinstance(case["item"], list) else [case["item"]]
    return next(
        (
            rank
            for rank, rc in enumerate(results, start=1)
            if rc.chunk.ticker == case["ticker"] and rc.chunk.item in items
        ),
        None,
    )


def term_count(results: list[RetrievedChunk], case: dict) -> int:
    term = case["term"].lower()
    return sum(term in rc.chunk.text.lower() for rc in results)


def main(k: int = 8) -> None:
    cases = load_cases()
    # Share one embedder, store and index so the model loads only once.
    shared = {"embedder": Embedder(), "store": VectorStore(), "keyword_index": KeywordIndex()}
    retrievers = {mode: Retriever(**shared, mode=mode) for mode in MODES}
    results = {
        mode: [r.retrieve(case["question"], k=k) for case in cases]
        for mode, r in retrievers.items()
    }

    section_cases = [i for i, case in enumerate(cases) if "item" in case]
    term_cases = [i for i, case in enumerate(cases) if "term" in case]
    header = "  ".join(f"{mode:>7}" for mode in MODES)

    print("Section check: rank of first chunk from the expected company + Item (- = miss)")
    print(f"{header}  question")
    ranks = {mode: {i: section_rank(results[mode][i], cases[i]) for i in section_cases}
             for mode in MODES}  # fmt: skip
    for i in section_cases:
        row = "  ".join(f"{ranks[mode][i] or '-':>7}" for mode in MODES)
        print(f"{row}  {cases[i]['question']}")

    print(f"\nTerm check: chunks in the top {k} containing the term")
    print(f"{header}  term")
    counts = {mode: {i: term_count(results[mode][i], cases[i]) for i in term_cases}
              for mode in MODES}  # fmt: skip
    for i in term_cases:
        row = "  ".join(f"{counts[mode][i]:>7}" for mode in MODES)
        print(f"{row}  {cases[i]['term']}")

    print(f"\n{'':>7}  {'Hit@' + str(k):>7}  {'MRR':>6}  {'terms':>9}")
    for mode in MODES:
        hits = sum(rank is not None for rank in ranks[mode].values())
        mrr = sum(1 / rank for rank in ranks[mode].values() if rank) / len(section_cases)
        terms = sum(counts[mode].values())
        print(
            f"{mode:>7}  {hits / len(section_cases):>7.1%}  {mrr:>6.3f}  "
            f"{terms:>3}/{k * len(term_cases):<5}"
        )


if __name__ == "__main__":
    main()

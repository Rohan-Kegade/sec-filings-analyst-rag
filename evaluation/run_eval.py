"""Retrieval evaluation: does top-k contain a chunk from the expected company + Item?

Runs locally with no LLM calls. Run with: python evaluation/run_eval.py
"""

import json
from pathlib import Path

from sec_rag.retrieval.retriever import Retriever

EVAL_FILE = Path(__file__).parent / "eval_set.jsonl"


def main(k: int = 8) -> None:
    retriever = Retriever()
    cases = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]

    hits, reciprocal_ranks = 0, []
    for case in cases:
        results = retriever.retrieve(case["question"], k=k)
        rank = next(
            (
                i
                for i, rc in enumerate(results, start=1)
                if rc.chunk.ticker == case["ticker"] and rc.chunk.item == case["item"]
            ),
            None,
        )
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        print(f"{'HIT ' if rank else 'MISS'} rank={rank} | {case['question']}")

    print(f"\nHit@{k}: {hits / len(cases):.2%}   MRR: {sum(reciprocal_ranks) / len(cases):.3f}")


if __name__ == "__main__":
    main()

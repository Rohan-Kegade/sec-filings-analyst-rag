"""Evals (DeepEval), one file per eval, in four levels:

    1. component    eval_retriever, eval_generator
    2. pipeline     eval_rag_triad
    3. application  quality:    eval_correctness_completeness (style: skipped)
                    safety:     eval_scope_adherence (toxicity, leakage: TODO)
                    operations: latency, cost, tokens (TODO)
    4. regression   the whole suite against a previous run (TODO)

Run one with `python -m evals.<file> run`, e.g. `python -m evals.eval_rag_triad run`.
"""

import os
from pathlib import Path

GOLDENS_DIR = Path(__file__).parents[1] / "goldens"  # project root: shared test data

# Keep DeepEval local: no usage telemetry, no update nags. Set before deepeval is imported.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "0")

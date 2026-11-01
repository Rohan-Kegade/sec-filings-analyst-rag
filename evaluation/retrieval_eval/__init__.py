"""Retriever evaluation with DeepEval's Contextual Recall and Contextual Precision.

Each golden's question goes through the retriever only (no answer generation); a Claude
judge then scores the retrieved chunks against the golden's reference answer.
"""

import os

# Keep DeepEval local: no usage telemetry, no update nags. Set before deepeval is imported.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "0")

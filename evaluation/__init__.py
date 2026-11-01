"""Evaluation suites (DeepEval) for the retriever and the answer generator."""

import os

# Keep DeepEval local: no usage telemetry, no update nags. Set before deepeval is imported.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "0")

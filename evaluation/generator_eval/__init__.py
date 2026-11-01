"""Answer generator evaluation with DeepEval's Faithfulness and Answer Relevancy.

Each golden has a frozen set of excerpts. The generator answers from those excerpts only (no
retrieval), and a Claude judge scores the answer against them and against the question.
"""

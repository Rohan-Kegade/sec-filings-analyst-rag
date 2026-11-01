"""Retriever evaluation with DeepEval's Contextual Recall and Contextual Precision.

Each golden's question goes through the retriever only (no answer generation); a Claude
judge then scores the retrieved chunks against the golden's reference answer.
"""

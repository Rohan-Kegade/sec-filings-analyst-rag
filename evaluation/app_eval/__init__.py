"""Application-level evaluation of the full pipeline's answers (correctness first).

Each question goes through the real pipeline; a Claude judge compares the answer with the
golden's reference answer and key facts, taken from the company's 10-K.
"""

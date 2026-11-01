"""End-to-end RAG evaluation with the RAG triad, using DeepEval metrics.

Each question goes through the real pipeline (filters, retrieval, generation). A Claude judge
then scores the three links of the triad:
  context relevance  (question -> retrieved excerpts)   DeepEval ContextualRelevancyMetric
  groundedness       (excerpts -> answer)               DeepEval FaithfulnessMetric
  answer relevance   (question -> answer)               DeepEval AnswerRelevancyMetric
All three are reference-free, so only questions are needed.
"""

"""Scope adherence eval: does the app stay within answering questions about the indexed 10-Ks?

Each case (evaluation/goldens/scope_goldens.json) goes through the real pipeline, and a
Claude judge checks the answer against the case's expected behavior: declining off-topic
requests, investment advice and predictions, not answering from memory, resisting prompt
injection and provocation, and not leaking its prompt or data.
"""

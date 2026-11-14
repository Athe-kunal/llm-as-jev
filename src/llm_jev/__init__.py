"""llm-jev: decision models that pick one option of a question per pass.

Two families share one interface (`models.base.DecisionModel`): LLM-as-Jev
decoders scored through a candidate trie, and masked-LM encoders with one
mask per option, plus an evidence head on the encoder.
"""

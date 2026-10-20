"""Span generation: labelling evidence sentences for Reason+Long-12k rows (SelfCite best-of-N).

The data pipeline, one file per step:
  serve.sh      deploy the labelling LLM (vLLM, OpenAI-compatible, local only)
  cite.j2       the prompt: numbered sentences, question, gold answer -> {"sentences": [...]}
  best_of_n.py  propose N citation sets per row with the LLM, score each with
                the SelfCite reward (necessity + sufficiency of the gold
                answer under the decision model's trie), keep the best

Sentences are context_cite.split_sources (nltk, content merged); the chosen
set is written in the row's `spans` format. Chuang et al., "SelfCite"
(arXiv 2502.09604). Work in progress: see the TODOs in best_of_n.py.
"""

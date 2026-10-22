"""Span generation: label evidence sentences for Reason+Long-12k rows (SelfCite best-of-N).

Pipeline: serve.sh (labelling LLM), cite.j2 (prompt), best_of_n.py (sampling and
scoring). Chuang et al., SelfCite, arXiv 2502.09604. Work in progress (see TODOs
in best_of_n.py).
"""

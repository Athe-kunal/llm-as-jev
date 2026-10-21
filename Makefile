PY := .venv/bin/python
DATA := data/splits/reason_long_12k

.PHONY: help download data

help:
	@echo "make download    fetch the Reason+Long-12k sources (and the benchmarks used to decontaminate them) into data/external"
	@echo "make data        build $(DATA)/{train,eval}.jsonl and manifest.json (CPU, ~10 min)"
	@echo "training and evaluation: scripts/train.sh, scripts/evaluate.sh"

download:
	$(PY) -m llm_jev.dataloader.build_data --download

data: download
	$(PY) -m llm_jev.dataloader.build_data

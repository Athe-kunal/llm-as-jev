PY := PYTHONPATH=src .venv/bin/python
DATA := data/splits/reason_long_12k

.PHONY: help download data data-view

help:
	@echo "make download    fetch the Reason+Long-12k sources and benchmarks into data/external"
	@echo "make data        build $(DATA)/{train,eval}.jsonl and manifest.json (CPU, ~10 min; docs/data.md)"
	@echo "make data-view   browse 5 examples per dataset at http://127.0.0.1:8790"
	@echo "training, evaluation and viewers: scripts/train.sh, scripts/evaluate.sh, scripts/view.sh"

download:
	$(PY) -m dataloader.build_data --download

data: download
	$(PY) -m dataloader.build_data

data-view:
	$(PY) -m viewers.data --port 8790


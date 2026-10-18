PY := PYTHONPATH=src .venv/bin/python
DATA := data/splits/reason_long_12k
BLOGS_PORT := 8765

.PHONY: help download data blogs

help:
	@echo "make download    fetch the Reason+Long-12k sources and benchmarks into data/external"
	@echo "make data        build $(DATA)/{train,eval}.jsonl and manifest.json (CPU, ~10 min; docs/data.md)"
	@echo "make blogs       serve blogs/ at http://127.0.0.1:$(BLOGS_PORT)/ (Ctrl-C to stop)"
	@echo "training, evaluation, model viewer: scripts/train.sh, scripts/evaluate.sh, scripts/view.sh"
	@echo "dataset examples: blogs/01_prompt_and_data/index.html (rebuild with its build.py)"

download:
	$(PY) -m dataloader.build_data --download

data: download
	$(PY) -m dataloader.build_data

blogs:
	@echo "Serving blogs/ at http://127.0.0.1:$(BLOGS_PORT)/  (Ctrl-C to stop)"
	@echo "  open http://127.0.0.1:$(BLOGS_PORT)/01_prompt_and_data/"
	cd blogs && python3 -m http.server $(BLOGS_PORT) --bind 127.0.0.1


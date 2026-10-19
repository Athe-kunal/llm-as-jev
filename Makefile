PY := PYTHONPATH=src .venv/bin/python
DATA := data/splits/reason_long_12k
BLOGS_PORT := 8765

.PHONY: help download data blogs

help:
	@echo "make download    fetch the Reason+Long-12k sources and benchmarks into data/external"
	@echo "make data        build $(DATA)/{train,eval}.jsonl and manifest.json (CPU, ~10 min; docs/data.md)"
	@echo "make blogs       serve blogs/ at http://127.0.0.1:$(BLOGS_PORT)/ (Ctrl-C to stop)"
	@echo "training, evaluation, model viewer: scripts/train.sh, scripts/evaluate.sh, scripts/view.sh"

download:
	$(PY) -m dataloader.build_data --download

data: download
	$(PY) -m dataloader.build_data

blogs:
	cd blogs && ( [ -d node_modules ] || npm install --no-fund --no-audit ) \
		&& ( [ -f data/prompt_and_data.json ] || npm run data ) \
		&& npx vite --port $(BLOGS_PORT) --strictPort


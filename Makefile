PY := .venv/bin/python
ACCELERATE := .venv/bin/accelerate
GPUS ?= 0
comma := ,
NPROC = $(words $(subst $(comma), ,$(GPUS)))
# Training targets: CONFIG picks the run, ARGS adds dotlist overrides,
# e.g. make train-encoder CONFIG=configs/train/ultrabert_380m_sigmoid_norm.yaml ARGS="optim.lr=8e-6"
CONFIG ?=
ARGS ?=

.PHONY: help install data download train-decoder train-encoder evaluate benchmarks \
	evidence test lint format paper paper-figures arxiv

help:
	@echo "make install         uv sync (CUDA 12.8 torch) with the dev extra"
	@echo "make data            download the pinned sources, build data/splits/reason_long_nimble"
	@echo "make train-decoder   CONFIG=configs/train/qwen3_0.6b_lora.yaml GPUS=0,1,2,3 (FSDP2)"
	@echo "make train-encoder   CONFIG=configs/train/ultrabert_380m_sigmoid_norm.yaml GPUS=0 (DDP)"
	@echo "make evaluate        CONFIG=configs/evaluate/encoder.yaml ARGS='model.path=runs/...'"
	@echo "make benchmarks      CONFIG=configs/evaluate/benchmarks.yaml ARGS='model.kind=... model.path=...'"
	@echo "make evidence        CONFIG=configs/evidence/ultrabert_380m.yaml"
	@echo "make test / lint / format"
	@echo "make paper           paper/generated/ (from runs/), then paper/main.pdf"

install:
	uv sync --extra dev

download:
	$(PY) -m llm_jev.data.build --config configs/data/reason_long_nimble.yaml download_only=true

data: download
	$(PY) -m llm_jev.data.build --config configs/data/reason_long_nimble.yaml $(ARGS)

train-decoder:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(ACCELERATE) launch --config_file configs/accelerate/fsdp2.yaml \
		--num_processes $(NPROC) -m llm_jev.training.decoder --config $(CONFIG) $(ARGS)

train-encoder:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(ACCELERATE) launch --config_file configs/accelerate/ddp.yaml \
		--num_processes $(NPROC) -m llm_jev.training.encoder --config $(CONFIG) $(ARGS)

evaluate:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(PY) -m llm_jev.evaluation.evaluate --config $(CONFIG) $(ARGS)

benchmarks:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(PY) -m llm_jev.evaluation.benchmarks --config $(CONFIG) $(ARGS)

evidence:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(PY) -m llm_jev.evidence.train --config $(CONFIG) $(ARGS)

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

format:
	.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests

paper-figures:
	$(PY) paper/figures/make_figures.py

paper: paper-figures
	cd paper && if command -v latexmk >/dev/null; then latexmk -pdf -interaction=nonstopmode main.tex; \
		elif command -v tectonic >/dev/null; then tectonic main.tex; \
		else echo "no LaTeX: install TeX Live (latexmk) or tectonic"; exit 1; fi

arxiv: paper-figures
	cd paper && tectonic --keep-intermediates --outdir . main.tex
	rm -rf paper/arxiv paper/arxiv.tar.gz && mkdir -p paper/arxiv/sections paper/arxiv/generated
	grep -v '% TODO' paper/main.tex > paper/arxiv/main.tex
	cp paper/main.bbl paper/arxiv/
	cp paper/sections/*.tex paper/arxiv/sections/
	cp paper/generated/*.tex paper/arxiv/generated/
	cd paper/arxiv && tar czf ../arxiv.tar.gz *
	@echo "wrote paper/arxiv.tar.gz"

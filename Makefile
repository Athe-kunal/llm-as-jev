PY := .venv/bin/python
DATA := data/splits/reason_long_nimble
# make sglang-merge / sglang-serve: the trained adapter, its base model, the merged checkpoint
BASE ?= Qwen/Qwen3-4B
ADAPTER ?= runs/qwen3_4b/adapter
MERGED ?= $(dir $(ADAPTER))merged
GPU ?= 0
PORT ?= 30000

.PHONY: help download data sglang-install sglang-merge sglang-serve sglang-stop

help:
	@echo "make download        fetch the pinned training sources (Reason+Long-12k + Nimble) (and the benchmarks used to decontaminate them) from the internet into the Hugging Face cache"
	@echo "make data            build $(DATA)/{train,eval}.parquet and manifest.json (CPU, ~6 min with 8 workers)"
	@echo "make sglang-install  SGLang as a uv tool (its own environment)"
	@echo "make sglang-merge    merge ADAPTER into BASE -> MERGED (CPU)"
	@echo "make sglang-serve    serve MERGED with SGLang on GPU (http://127.0.0.1:PORT), merging first if needed"
	@echo "make sglang-stop     stop the server on PORT"
	@echo "training and evaluation: scripts/train.sh, scripts/evaluate.sh"

download:
	$(PY) -m llm_jev.dataloader.build --download

data: download
	$(PY) -m llm_jev.dataloader.build --out $(DATA)

sglang-install:
	uv tool install --prerelease=allow --python 3.11 sglang==0.5.22

$(MERGED)/config.json:
	$(PY) -m llm_jev.serving.merge --model $(BASE) --adapter $(ADAPTER) --out $(MERGED)

sglang-merge: $(MERGED)/config.json

sglang-serve: $(MERGED)/config.json
	scripts/sglang_serve.sh $(MERGED) $(GPU) $(PORT)
	@echo "waiting for http://127.0.0.1:$(PORT)/health ..."
	@for i in $$(seq 600); do curl -sf http://127.0.0.1:$(PORT)/health >/dev/null && echo ready && exit 0; \
		sleep 2; done; echo "not ready after 20 min: see logs/sglang_$(PORT).log"; exit 1

sglang-stop:
	@pidfile=logs/sglang_$(PORT).pid; [ -f $$pidfile ] || { echo "no $$pidfile"; exit 0; }; \
	pid=$$(cat $$pidfile); kill -- -$$pid 2>/dev/null && echo "stopped server group $$pid"; \
	for i in $$(seq 30); do ps -p $$pid >/dev/null || { rm -f $$pidfile; echo "gone"; exit 0; }; sleep 1; done; \
	echo "pid $$pid still alive after 30 s"

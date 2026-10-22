"""Merges a LoRA adapter into its base model and saves it for SGLang.

Usage (or: make sglang-merge):
  .venv/bin/python -m llm_jev.serving.merge --model Qwen/Qwen3-4B \
      --adapter runs/qwen3_4b/adapter --out runs/qwen3_4b/merged

SGLang's score API cannot select a LoRA adapter, so it serves the merged weights.
"""

import argparse
import pathlib

from loguru import logger

from llm_jev.modeling import modeling


def main() -> None:
    """Writes the merged model and tokenizer to --out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="the adapter's base model (HF id)")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    out = pathlib.Path(args.out)
    model = modeling.load(args.model, args.adapter, device="cpu")
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(out)
    logger.info("merged {} + {} -> {}", args.model, args.adapter, out)


if __name__ == "__main__":
    main()

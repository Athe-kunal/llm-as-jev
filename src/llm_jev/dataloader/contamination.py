"""Dropping training examples that overlap the evaluation benchmarks.

An example sharing a 13-word n-gram, or an identical sentence of six or more
words, with any evaluation benchmark is dropped (paper section 3.5, Appendix D).
Benchmarks are the pinned downloads of `sources.download.BENCH_SOURCES`.
"""

import collections
import csv
import json
import pathlib
import re
from collections.abc import Iterable, Sequence
from typing import Any

from llm_jev.dataloader import normalize
from llm_jev.dataloader import schema
from llm_jev.dataloader.sources import download


def _parquet(path: pathlib.Path) -> list[dict[str, Any]]:
    import pandas as pd  # pylint: disable=import-outside-toplevel

    data = pd.read_parquet(path)
    return data.drop(
        columns=[c for c in ("image",) if c in data]).to_dict("records")


def _jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _strings(value: Any) -> Iterable[str]:
    """Every string inside a (nested) value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        for v in (value.tolist() if hasattr(value, "tolist") else value):
            yield from _strings(v)


def benchmark_items() -> dict[str, list[list[str]]]:
    """The text fields of every item of each benchmark's evaluation splits.

    Whole splits are used, since the paper's samples are not released.
    """
    out: dict[str, list[list[str]]] = {}

    def fields(name: str, records: Iterable[dict], keys: tuple[str, ...]):
        out.setdefault(name, []).extend(
            [s for k in keys if k in r for s in _strings(r[k])]
            for r in records)

    fields("boolq", _parquet(download.benchmark_path(
        "boolq", "data/validation-00000-of-00001.parquet")),
           ("question", "passage"))
    for split in ("test", "validation", "dev"):
        fields("mmlu", _parquet(download.benchmark_path(
            "mmlu", f"all/{split}-00000-of-00001.parquet")),
               ("question", "choices"))
    for split in ("test", "validation"):
        fields("mmlu_pro", _parquet(download.benchmark_path(
            "mmlu_pro", f"data/{split}-00000-of-00001.parquet")),
               ("question", "options", "cot_content"))
    fields("arc_challenge", _parquet(download.benchmark_path(
        "arc_challenge", "ARC-Challenge/test-00000-of-00001.parquet")),
           ("question", "choices"))
    fields("winogrande", _parquet(download.benchmark_path(
        "winogrande", "winogrande_xl/validation-00000-of-00001.parquet")),
           ("sentence", "option1", "option2"))
    fields("sciq", _parquet(download.benchmark_path(
        "sciq", "data/test-00000-of-00001.parquet")),
           ("question", "support", "correct_answer", "distractor1",
            "distractor2", "distractor3"))
    with open(download.benchmark_path("banking77", "test.csv"), newline="") as f:
        fields("banking77", csv.DictReader(f), ("text",))
    for split in ("easy", "original", "hard"):
        fields("jevbench", _jsonl(download.benchmark_path("jevbench", f"{split}.jsonl")),
               ("state", "question", "labels"))
    fields("gsm8k", _parquet(download.benchmark_path(
        "gsm8k", "main/test-00000-of-00001.parquet")),
           ("question", "answer"))
    fields("ifeval", _jsonl(download.benchmark_path("ifeval", "ifeval_input_data.jsonl")),
           ("prompt",))
    trivia = _parquet(download.benchmark_path(
        "triviaqa", "rc.nocontext/validation-00000-of-00001.parquet"))
    fields("triviaqa", ({"q": r["question"], "a": r["answer"]["value"]}
                        for r in trivia), ("q", "a"))
    fields("lambada", _parquet(download.benchmark_path("lambada", "default/test/default.parquet")),
           ("text",))
    # WikiText-2 test: one item per article (a " = Title = " line opens one)
    articles, current = [], []
    for r in _parquet(download.benchmark_path(
            "wikitext2", "wikitext-2-raw-v1/test-00000-of-00001.parquet")):
        text = r["text"]
        if re.fullmatch(r" = [^=].* = \n", text) and current:
            articles.append({"text": current})
            current = []
        if text.strip():
            current.append(text)
    articles.append({"text": current})
    fields("wikitext2", articles, ("text",))
    fields("mmbench", _parquet(download.benchmark_path(
        "mmbench", "en/dev-00000-of-00001.parquet")),
           ("question", "hint", "A", "B", "C", "D"))
    fields("mmstar", _parquet(download.benchmark_path("mmstar", "mmstar.parquet")),
           ("question",))
    # Behavior suite: items are prompts only; reference responses are not checked (App. B).
    fields("dolly", _jsonl(download.benchmark_path("dolly", "databricks-dolly-15k.jsonl")),
           ("instruction", "context"))
    fields("mt_bench", _jsonl(download.benchmark_path("mt_bench", "raw/question.jsonl")),
           ("prompt",))
    return out


class OverlapIndex:
    """13-word n-grams and >= 6-word sentences of the evaluation benchmarks.

    Each key maps to a bit mask over `names`: bit b is names[b].
    """

    def __init__(self, items: dict[str, list[list[str]]]):
        self.names = sorted(items)
        self.ngrams: dict[int, int] = {}
        self.sentences: dict[int, int] = {}
        self.sizes: dict[str, int] = {}
        for b, name in enumerate(self.names):
            bit = 1 << b
            self.sizes[name] = len(items[name])
            for texts in items[name]:
                for text in texts:
                    for gram in normalize.ngrams(text):
                        h = hash(gram)
                        self.ngrams[h] = self.ngrams.get(h, 0) | bit
                    for sentence in normalize.sentences(text):
                        h = hash(sentence)
                        self.sentences[h] = self.sentences.get(h, 0) | bit
        self._cache: dict[str, tuple[int, int]] = {}

    def text_hits(self, text: str) -> tuple[int, int]:
        """(n-gram mask, sentence mask) of one text (cached)."""
        hit = self._cache.get(text)
        if hit is None:
            ngram_mask = sentence_mask = 0
            for gram in normalize.ngrams(text):
                ngram_mask |= self.ngrams.get(hash(gram), 0)
            for sentence in normalize.sentences(text):
                sentence_mask |= self.sentences.get(hash(sentence), 0)
            hit = self._cache[text] = (ngram_mask, sentence_mask)
        return hit

    def example_hits(self, example: schema.Example) -> tuple[int, int]:
        """Masks for an example's overlap text (or parts + question + options)."""
        texts = example.overlap_text or [
            *(p.text for p in example.parts), example.question,
            *example.options]
        ngram_mask = sentence_mask = 0
        for text in texts:
            n, s = self.text_hits(text)
            ngram_mask, sentence_mask = ngram_mask | n, sentence_mask | s
        return ngram_mask, sentence_mask

    def decode(self, mask: int) -> list[str]:
        """The benchmark names in a mask."""
        return [n for b, n in enumerate(self.names) if mask >> b & 1]


def contamination_stats(
    ids: Sequence[str], ngram_masks: Sequence[int],
    sentence_masks: Sequence[int], index: OverlapIndex,
) -> dict[str, Any]:
    """Drop counts for examples that overlap a benchmark.

    Returns:
      Counts by kind (an example with both counts as "ngram") and by benchmark
      (an example can count for several), with the first three example ids per
      benchmark.
    """
    by_bench: collections.Counter[str] = collections.Counter()
    by_kind: collections.Counter[str] = collections.Counter()
    examples_by_bench: dict[str, list[str]] = collections.defaultdict(list)
    for example_id, ngram_mask, sentence_mask in zip(
            ids, ngram_masks, sentence_masks, strict=True):
        if not (ngram_mask or sentence_mask):
            continue
        by_kind["ngram" if ngram_mask else "sentence"] += 1
        for name in index.decode(ngram_mask | sentence_mask):
            by_bench[name] += 1
            if len(examples_by_bench[name]) < 3:
                examples_by_bench[name].append(example_id)
    return {
        "contaminated": sum(by_kind.values()),
        "contaminated_by_kind": dict(by_kind),
        "contaminated_by_benchmark": dict(sorted(by_bench.items())),
        "contaminated_examples": dict(examples_by_bench),
    }

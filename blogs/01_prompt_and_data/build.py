"""Builds blogs/01_prompt_and_data/index.html: prompt format, data notes, dataset browser.

Usage (from the repo root):
  PYTHONPATH=src .venv/bin/python blogs/01_prompt_and_data/build.py
"""

import collections
import html
import pathlib
import random
from typing import Any

import transformers
from loguru import logger

from dataloader import answers, data

_OUT = pathlib.Path(__file__).with_name("index.html")
_MODEL = "Qwen/Qwen3.5-4B"
_PER, _SEED = 2, 0
_COLLAPSE_AT = 20  # option lists longer than this fold (gold stays visible)
# Span-carrying sources: this many of the --per rows come from the span pool.
_SPAN_SOURCES = {"proofwriter", "contractnli", "musique"}
_SPAN_ROWS = 2

# source -> (title, description); descriptions follow docs/data.md.
DATASETS: dict[str, tuple[str, str]] = {
    "clinc150": (
        "CLINC150",
        "Short user messages, each labelled with one of 151 intents across 10 "
        "domains, including out-of-scope (`oos`). The option ids are the intent "
        "names.",
    ),
    "massive": (
        "MASSIVE (en-US)",
        "Voice-assistant utterances labelled with one of 60 intents. The option "
        "ids are the intent names.",
    ),
    "bitext": (
        "Bitext",
        "E-commerce customer-support messages labelled with one of 27 intents. "
        "Messages keep the release's `{{...}}` placeholders.",
    ),
    "reclor": (
        "ReClor",
        "Logical-reasoning argument passages with a 4-option question about the "
        "argument. Research-only license.",
    ),
    "logiqa2": (
        "LogiQA 2.0",
        "Short logical-reasoning passages (civil-service style) with a 4-option "
        "question.",
    ),
    "cosmosqa": (
        "CosmosQA",
        "Commonsense reading comprehension: a short everyday context with a "
        "question and 4 options.",
    ),
    "piqa": (
        "PIQA",
        "Physical commonsense: a goal and two candidate solutions; pick the one "
        "that better achieves the goal.",
    ),
    "anli": (
        "αNLI (ART)",
        "Abductive reasoning: two observations and two hypotheses; pick the one "
        "that best explains what happened between them.",
    ),
    "proofwriter": (
        "ProofWriter (OWA, depth 5)",
        "A theory of facts and rules, and a statement; is it true, false, or "
        "unknown given the theory? True and false rows carry proof-step sentences "
        "as spans.",
    ),
    "contractnli": (
        "ContractNLI",
        "A long contract and a hypothesis statement: entailed, contradicted, or "
        "not mentioned. Entailed and contradicted rows carry the human evidence "
        "spans.",
    ),
    "musique": (
        "MuSiQue (full v1.0)",
        "Twenty Wikipedia paragraphs and a multi-hop question; yes or no, are the "
        "paragraphs enough to answer it? Answerable rows carry the supporting "
        "paragraphs as spans.",
    ),
    "quality": (
        "QuALITY (v1.0.1, HTML-stripped)",
        "Long story articles with a 4-option reading question; the gold answer is "
        "the annotators' majority.",
    ),
}


_NOTES = [
    "12 datasets, 1,000 training rows each: 3 intent, 6 short reasoning, "
    "3 long-input (Reason+Long-12k).",
    "Options are shuffled per example at training time, so gold lands on any "
    "<code>[k]</code>.",
    "Rows store <code>parts</code>, <code>question</code> and options "
    "separately; the prompt is rendered from them.",
    "State parts carry no names, are separated by a blank line, and the "
    "question always sits just before <code>Options:</code>.",
    "Inputs are never truncated; rows are capped at 10,240 tokens.",
    "ProofWriter, ContractNLI and MuSiQue rows keep native evidence spans; "
    "all others have none.",
]

_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1f23;--muted:#6b6f76;--card:#fff;--line:#e2e0da;
--code:#f4f3ef;--gold:#d8f5e1;--goldline:#1f7a3d;--c-sys:#8a8f98;--c-state:#5b4bb7;
--c-q:#b3541e;--c-opt:#1f6f8b;--c-ins:#8a5a00;--c-pre:#1f7a3d}
@media (prefers-color-scheme:dark){:root{--bg:#15171b;--fg:#e6e6e6;--muted:#9aa1ac;
--card:#1e2127;--line:#343842;--code:#22252c;--gold:#173a25;--goldline:#7fdca0;
--c-sys:#7b8190;--c-state:#a99bff;--c-q:#ffa06b;--c-opt:#6fcbe8;--c-ins:#f0c674;
--c-pre:#7fdca0}}
*{box-sizing:border-box}
body{margin:0 auto;max-width:960px;padding:24px 16px 64px;background:var(--bg);
color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
h1{font-size:24px;margin:0 0 2px}h2{font-size:18px;margin:32px 0 8px}
.sub,.muted{color:var(--muted)}
code,pre,.prompt{font-family:ui-monospace,"SF Mono",Menlo,monospace}
code{background:var(--code);padding:0 4px;border-radius:3px;font-size:13px}
ul{padding-left:20px;margin:0}li{margin:3px 0}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px;margin:8px 0}
.legend i{font-style:normal;border-left:4px solid;padding-left:6px}
.prompt{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 14px;font-size:12.5px;line-height:1.5;white-space:pre-wrap;
word-break:break-word;margin:0}
.prompt span{border-left:3px solid;padding-left:5px;margin-left:-8px}
.s-sys{color:var(--c-sys);border-color:var(--c-sys)}
.s-state{color:var(--c-state);border-color:var(--c-state);font-weight:600}
.s-q{color:var(--c-q);border-color:var(--c-q)}
.s-opt{color:var(--c-opt);border-color:var(--c-opt)}
.s-ins{color:var(--c-ins);border-color:var(--c-ins)}
.s-pre{color:var(--c-pre);border-color:var(--c-pre);font-weight:600}
.s-opt.gold{background:var(--gold);border-color:var(--goldline)}
.doc{display:block;max-height:220px;overflow:auto;background:var(--code);
border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin:4px 0;
color:var(--fg)}
.ans{margin-top:8px;font-size:13px}
.ans b{font-family:ui-monospace,monospace;background:var(--gold);
border-left:4px solid var(--goldline);padding:1px 6px}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 12px}
.tabs button{font:inherit;font-size:13px;color:var(--fg);background:var(--card);
border:1px solid var(--line);border-radius:6px;padding:3px 10px;cursor:pointer}
.tabs button[aria-selected=true]{background:var(--fg);color:var(--bg)}
.tabs small{opacity:.65;margin-left:4px}
.ds{display:none}.ds.on{display:block}
.ex{margin:14px 0}.ex h4{margin:0 0 4px;font-size:12px;color:var(--muted);
font-weight:normal}
details summary{cursor:pointer;color:var(--muted)}
"""

_JS = """
const tabs=[...document.querySelectorAll('.tabs button')];
function show(id){tabs.forEach(b=>b.setAttribute('aria-selected',b.dataset.id===id));
document.querySelectorAll('.ds').forEach(d=>d.classList.toggle('on',d.id===id));
history.replaceState(null,'','#'+id)}
tabs.forEach(b=>b.onclick=()=>show(b.dataset.id));
show(tabs.some(b=>b.dataset.id===location.hash.slice(1))?location.hash.slice(1):tabs[0].dataset.id);
"""

_esc = html.escape


def _span(cls: str, text: str) -> str:
    return f'<span class="s-{cls}">{_esc(text)}</span>'


def _options(lines: list[str], gold: int) -> str:
    """Option lines; a long list folds into a <details> with gold shown outside."""
    def one(i: int) -> str:
        c = "opt gold" if i == gold else "opt"
        return f'<span class="s-{c}">{_esc(lines[i])}</span>'

    if len(lines) <= _COLLAPSE_AT:
        return "".join(one(i) for i in range(len(lines)))
    inner = "".join(one(i) for i in range(len(lines)))
    return (f"<details><summary>{len(lines)} options (click to expand)</summary>"
            f"{inner}</details>{one(gold)}")


def pick_rows(
    rows: list[dict[str, Any]], per: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Seeded sample of `per` rows for each dataset, in DATASETS order."""
    rng = random.Random(seed)
    by_src: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    picked: dict[str, list[dict[str, Any]]] = {}
    for src in DATASETS:
        pool = by_src[src]
        chosen: list[dict[str, Any]] = []
        if src in _SPAN_SOURCES:
            with_spans = [r for r in pool if r["spans"]]
            chosen = rng.sample(with_spans, min(_SPAN_ROWS, per, len(with_spans)))
        taken = {r["id"] for r in chosen}
        rest = [r for r in pool if r["id"] not in taken]
        chosen += rng.sample(rest, min(per - len(chosen), len(rest)))
        picked[src] = chosen
    return picked


def render_prompt(row: dict[str, Any], order: list[str], tokenizer: Any) -> str:
    """The exact build_prompt text as HTML, each piece wrapped in a labeled span."""
    text, where = answers.build_prompt(row, order, tokenizer)
    gold = order.index(row["label"])
    parts = sorted(where.values())
    out = [_span("sys", text[: text.index("State:")])]
    out.append(_span("state", "State:") + _esc(text[text.index("State:") + 6 : parts[0][0]]))
    for i, (s, e) in enumerate(parts):
        out.append(f'<span class="doc">{_esc(text[s:e])}</span>')
        nxt = parts[i + 1][0] if i + 1 < len(parts) else None
        if nxt is not None:
            out.append(_esc(text[e:nxt]))
    q = text.index(row["question"], parts[-1][1])
    out.append(_esc(text[parts[-1][1] : q]))
    qe = q + len(row["question"])
    out.append(_span("q", text[q:qe]))
    opt = text.index("Options:", qe)
    out.append(_esc(text[qe:opt]) + _span("state", "Options:") + "\n")
    cur = opt + len("Options:\n")
    lines = [f"[{i + 1}] {row['labels'][k]}\n" for i, k in enumerate(order)]
    if not text.startswith("".join(lines), cur):
        raise ValueError(f"option lines do not match the prompt of {row['id']}")
    out.append(_options(lines, gold))
    cur += sum(map(len, lines))
    tail = text.index("<|im_end|>", cur)
    out.append(_span("ins", text[cur:tail].rstrip("\n")) + "\n")
    pre = text.index(answers.ANSWER_PREFIX, tail)
    out.append(_span("sys", text[tail:pre]) + _span("pre", text[pre:]))
    return (f'<pre class="prompt">{"".join(out)}</pre>'
            f'<div class="ans">model answers <b>{gold + 1}]</b> '
            f'<span class="muted">(gold: {_esc(row["label"])})</span></div>')


def _dataset(src: str, rows: list[dict], count: int, tok: Any, rng: random.Random) -> str:
    title, desc = DATASETS[src]
    ex = ""
    for n, row in enumerate(rows, 1):
        order = list(row["labels"])
        rng.shuffle(order)  # as at training time
        ex += (f'<div class="ex"><h4>#{n} · <code>{_esc(row["id"])}</code> · '
               f'{len(order)} options</h4>{render_prompt(row, order, tok)}</div>')
    return (f'<section class="ds" id="{src}"><p><b>{_esc(title)}</b> · '
            f'<span class="muted">{count:,} train rows. {_esc(desc)}</span></p>{ex}</section>')


def main() -> None:
    """Writes index.html."""
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL)
    rows = data.load_rows()
    counts = collections.Counter(r["source"] for r in rows)
    picked = pick_rows(rows, _PER, _SEED)
    rng = random.Random(_SEED)

    demo = picked["piqa"][0]
    demo_order = list(demo["labels"])
    rng.shuffle(demo_order)
    legend = "".join(f'<i class="s-{c}">{t}</i>' for c, t in [
        ("sys", "chat template"), ("state", "State: / Options:"),
        ("q", "question"), ("opt", "[k] options"), ("ins", "fixed instruction"),
        ("pre", "assistant prefill")])
    tabs = "".join(
        f'<button data-id="{s}">{_esc(DATASETS[s][0])}<small>{counts[s]:,}</small></button>'
        for s in DATASETS)
    sections = "".join(_dataset(s, picked[s], counts[s], tok, rng) for s in DATASETS)
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Prompt and data</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>1. Prompt and data</h1>"
        '<div class="sub">LLM-as-Jev reproduction</div>'
        "<h2>Prompt</h2>"
        f'<div class="legend">{legend}</div>{render_prompt(demo, demo_order, tok)}'
        '<p class="muted">The answer is the suffix <code>k]</code> after the prefill. '
        "Yes/no tasks use <code>[1]</code>/<code>[2]</code> too. "
        f"Rendered with <code>{_MODEL}</code>'s chat template.</p>"
        f'<h2>Data</h2><ul>{"".join(f"<li>{n}</li>" for n in _NOTES)}</ul>'
        "<h2>Browser</h2>"
        f'<div class="tabs" role="tablist">{tabs}</div>{sections}'
        f"<script>{_JS}</script></body></html>"
    )
    _OUT.write_text(page)
    logger.info("wrote {} ({:.0f} KB), {} datasets", _OUT, len(page) / 1024, len(counts))


if __name__ == "__main__":
    main()

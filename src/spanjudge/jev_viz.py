"""Viewer: a trained LLM-as-Jev adapter against its base model on dev rows.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python -m spanjudge.jev_viz \
      --run runs/jev_qwen3_0.6b

Each request picks an eval row (optionally from one dataset, or by index),
renders the paper prompt (options in release order), and shows for the
trained model and the base model (the same model with the adapter disabled):
each option's probability from Eq. 1, the argmax, and the answer decoded
inside the trie after "Best answer: [" (greedy over the legal tokens, ending
at the closing "]"). Local only (real data).
"""

import argparse
import html
import json
import pathlib
import random
import re
import threading
import time
from typing import Any

import fastapi
from fastapi import responses
import torch
import uvicorn

from spanjudge import jev
from spanjudge import train_joint

# Options shown per model: the top ones by probability plus the gold one.
_SHOW = 12

_CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--mute:#78716c;--card:#fff;--line:#e7e5e4;
--gold:#15803d;--bad:#b91c1c;--bar:#a8a29e;--barg:#22c55e}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#e7e5e4;
--mute:#a8a29e;--card:#292524;--line:#44403c;--bar:#78716c}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;
margin:0;padding:16px;max-width:1200px;margin:auto}
form{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
select,input,button{font:inherit;padding:4px 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px;margin-bottom:12px}
pre{white-space:pre-wrap;word-break:break-word;max-height:360px;overflow:auto;
margin:0;font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:800px){.grid{grid-template-columns:1fr}}
.opt{display:grid;grid-template-columns:3.5em 1fr 4.5em;gap:6px;
align-items:center;margin:2px 0}
.bar{height:6px;background:var(--bar);border-radius:3px}
.isgold .bar{background:var(--barg)}.isgold{color:var(--gold);font-weight:600}
.ok{color:var(--gold);font-weight:600}.bad{color:var(--bad);font-weight:600}
.mute{color:var(--mute)}
"""


def _esc(x: Any) -> str:
    return html.escape(str(x))


class _Model:
    """Base model + the run's LoRA adapter on one GPU."""

    def __init__(self, run: str):
        from peft import PeftModel
        from transformers import AutoTokenizer

        args = json.loads((pathlib.Path(run) / "args.json").read_text())
        self.name = args["model"]
        self.tok = AutoTokenizer.from_pretrained(args["model"],
                                                 revision=args["revision"])
        base = train_joint.load_base(args["model"], args["revision"]).to("cuda")
        self.model = PeftModel.from_pretrained(
            base, str(pathlib.Path(run) / "adapter")).eval()
        self.lock = threading.Lock()

    def _one(self, prompt: list[int], k: int) -> dict[str, Any]:
        s, mass, decoded = jev.score(self.model, prompt, self.tok, k, "cuda")
        return {"p": torch.softmax(s, 0).tolist(), "mass": mass,
                "gen": f"{decoded + 1}]"}

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Trained and base predictions for one row (release option order)."""
        ex = jev.encode(row, self.tok, None)
        prompt = ex["input_ids"][: ex["path"][0][0] + 1]
        with self.lock, torch.no_grad():
            trained = self._one(prompt, ex["n_options"])
            with self.model.disable_adapter():
                base = self._one(prompt, ex["n_options"])
        return {"gold": ex["gold"], "trained": trained, "base": base,
                "prompt": self.tok.decode(prompt), "tokens": len(prompt)}


def _side(title: str, r: dict, gold: int, options: list[str]) -> str:
    p = r["p"]
    pred = max(range(len(p)), key=p.__getitem__)
    m = re.match(r"(\d+)\]", r["gen"])
    gen_ok = bool(m) and int(m.group(1)) == gold + 1
    show = sorted(range(len(p)), key=lambda i: -p[i])[:_SHOW]
    if gold not in show:
        show.append(gold)
    rows = "".join(
        f'<div class="opt{" isgold" if i == gold else ""}">'
        f"<span>[{i + 1}]</span><span>{_esc(options[i])}"
        f'<div class="bar" style="width:{max(p[i] * 100, 0.5):.1f}%"></div>'
        f"</span><span>{p[i]:.3f}</span></div>" for i in show)
    more = (f'<div class="mute">{len(p) - len(show)} more options not shown'
            "</div>" if len(p) > len(show) else "")
    verdict = '<span class="ok">correct</span>' if pred == gold else \
        '<span class="bad">wrong</span>'
    return (
        f'<div class="card"><b>{title}</b> — scored answer [{pred + 1}] '
        f"{verdict} <span class='mute'>(root legal mass {r['mass']:.4f})"
        f"</span>{rows}{more}<p><b>Decoded answer</b> "
        f"{'<span class=ok>correct</span>' if gen_ok else '<span class=bad>wrong / invalid</span>'}"
        f"<pre>Best answer: [{_esc(r['gen'])}</pre></p></div>")


def make_app(run: str, data: str) -> fastapi.FastAPI:
    with open(data) as f:
        rows = [json.loads(line) for line in f]
    sources = sorted({r["source"] for r in rows})
    model = _Model(run)
    app = fastapi.FastAPI()

    @app.get("/", response_class=responses.HTMLResponse)
    def index(source: str = "", idx: str = "") -> str:
        pool = [i for i, r in enumerate(rows)
                if not source or r["source"] == source]
        i = int(idx) if idx.strip().isdigit() and int(idx) < len(rows) \
            else random.choice(pool)
        row = rows[i]
        start = time.time()
        res = model.predict(row)
        options = [row["labels"][label] for label in row["labels"]]
        opts = "".join(
            f'<option value="{s}"{" selected" if s == source else ""}>{s}'
            "</option>" for s in sources)
        return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev Dev Viewer</title><style>{_CSS}</style></head><body>
<h2>{_esc(model.name)} — trained vs base <span class="mute">({_esc(run)})</span></h2>
<form><select name="source"><option value="">all datasets</option>{opts}</select>
<button>random row</button><span class="mute">or row index</span>
<input name="idx" size="6" placeholder="{i}"><button>go</button></form>
<div class="card"><b>{_esc(row['source'])}</b> row {i} · id {_esc(row['id'])}
· {len(options)} options · {res['tokens']} prompt tokens ·
<span class="mute">{(time.time() - start):.1f}s</span><br>
gold: <span class="ok">[{res['gold'] + 1}] {_esc(options[res['gold']])}</span>
<details><summary>prompt</summary><pre>{_esc(res['prompt'])}</pre></details></div>
<div class="grid">{_side("Trained", res["trained"], res["gold"], options)}
{_side("Base (adapter off)", res["base"], res["gold"], options)}</div>
</body></html>"""

    return app


def main() -> None:
    """Serves the viewer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--data",
                        default="data/splits/reason_long_12k/eval.jsonl")
    parser.add_argument("--port", type=int, default=8781)
    args = parser.parse_args()
    app = make_app(args.run, args.data)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

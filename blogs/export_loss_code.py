"""Exports a line-by-line trace of train/losses.compute for the blog.

Runs the real loss code on one training example, with a trained adapter as
p_theta and the same model with the adapter disabled as p0. A line tracer
records every local variable after each line, so the values the blog shows are
the ones the code produced. The steps below say which lines to show.

Usage (one GPU):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python blogs/export_loss_code.py
"""

import argparse
import ast
import inspect
import json
import math
import os
import pathlib
import random
import sys

os.environ["TORCHDYNAMO_DISABLE"] = "1"  # trace the plain Python of loss_terms

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from llm_jev.dataloader import answers  # noqa: E402
from llm_jev.dataloader import storage  # noqa: E402
from llm_jev.modeling import modeling  # noqa: E402
from llm_jev.train import losses  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
ADAPTER = "runs/qwen3_0.6b/adapter"
EXAMPLE = 4161  # a short four-option ReClor row in the training split
SEED = f"42:{EXAMPLE}"
SKIP = {"model", "device", "base", "head", "pad", "top", "self"}

FUNCTIONS = {
    "compute": losses.compute,
    "_logits": losses._logits,
    "batched_hidden": modeling.batched_hidden,
    "legal_mask": losses.legal_mask,
    "_loss_terms": losses._loss_terms,
    "node_stats": losses.node_stats,
    "_split": losses._split,
    "node_anchor": losses.node_anchor,
    "_kl_top": losses._kl_top,
    "tree_loss": losses.tree_loss,
    "pos_stats": losses.pos_stats,
    "pos_anchor": losses.pos_anchor,
}

# (function, anchor text of the line, options, title, body). Options: call
# (which call of the function, default the first), occ="last" (last pass of a
# loop), to (a loop: the anchor is its header, `to` its last line, and the step
# shows the state after the loop), ret (show the returned value), entry (show
# the arguments, before any line runs). Steps are ordered by when their line
# finishes.
STEPS = [
    ("compute", "", {"entry": True}, "The inputs",
     "`ex` is the dict from `answers.encode`: token ids (prompt + gold suffix), the gold path (position, legal tokens, gold token at each step), the reply position and the sampled off-path nodes. Token ids are decoded here for reading; the code works on ids."),
    ("compute", 'prompt = ex["input_ids"]', {}, "Cut the suffix off",
     "`path[0][0]` is the index of the last prompt token, the `[`. Keeping `input_ids[:that + 1]` drops the gold suffix and leaves the bare prompt that every off-path sequence will start with."),
    ("compute", 'seqs = [ex["input_ids"]]', {}, "One sequence per branch",
     "Row 0 is the teacher-forced gold sequence. Each other row is the prompt plus one off-path prefix, so the logits at its last position are the model's distribution at that trie node. S = 1 + the number of off-path nodes."),

    ("batched_hidden", "n = max(", {}, "Pad length",
     "The S sequences are padded to the length of the longest."),
    ("batched_hidden", "ids = torch.full(", {}, "A pad-filled token matrix",
     "`ids` is an `[S, n]` matrix of the pad token. The real tokens are written over it next."),
    ("batched_hidden", "mask = torch.zeros(", {}, "An empty attention mask",
     "`mask` is 1 where a real token will sit and 0 on padding."),
    ("batched_hidden", "for i, x in enumerate(seqs):", {"to": "mask[i, n - len(x):] = 1"}, "Left-pad each row",
     "The loop writes each sequence into the right end of its row of `ids` and sets `mask` to 1 over the real tokens. The padding sits on the left, so every row's last real token is in column `n - 1`; that alignment lets `last[:, -1]` read the final position of every row. The panel shows the state after the loop."),
    ("batched_hidden", "ids, mask = ids.to(device)", {}, "Move to the GPU", "Both matrices go to the device."),
    ("batched_hidden", "pos = (mask.cumsum", {}, "Position ids that skip padding",
     "`cumsum - 1` numbers the real tokens 0, 1, 2, … from the first real token, and `clamp(min=0)` puts padded slots at 0. Without it, a left-padded row would get shifted rotary positions and different logits than the same sequence run alone."),
    ("batched_hidden", "base = model.get_base_model()", {}, "Skip the wrappers",
     "`get_base_model()` unwraps the PEFT wrapper. The LoRA layers inside stay active, or are switched off by `disable_adapter()` in the reference pass."),
    ("batched_hidden", "out = base.model(", {}, "The transformer forward",
     "One pass over all S rows through the transformer stack only (`base.model`, not the LM head), so the vocabulary-sized projection is not applied at every position. This is where nearly all the time goes."),
    ("batched_hidden", "last = out.last_hidden_state", {}, "Last layer's states",
     "`[S, n, H]`: one hidden vector per token."),
    ("batched_hidden", "logits = base.lm_head(last[:, -1])", {}, "LM head on the last column only",
     "Column `n - 1` is the position after each row's final real token. For an off-path row that is the trie node we want; for the gold row it is the position after the closing `]`. Only these S vectors go through the vocabulary-wide head."),
    ("batched_hidden", "last = last.float()", {}, "States to fp32",
     "The states are cast so the gold-path head projection below is computed in fp32."),
    ("batched_hidden", "return [last[i", {"ret": True}, "Remove the padding again",
     "Returns each row's hidden states without its left padding (`[L_i, H]`) and the `[S, V]` last-position logits."),

    ("_logits", "hs, last = modeling.batched_hidden", {}, "Hidden states and last logits",
     "`hs[i]` are row i's hidden states and `last[i]` is its next-token distribution at the end. Row 0 is the gold sequence, so `last[0]` is the distribution after the final `]`."),
    ("_logits", "head = model.get_base_model().lm_head", {}, "The LM head",
     "The same output projection, applied below to chosen gold-path positions."),
    ("_logits", "pos = [p for p, _, _ in", {}, "Which positions we need",
     "The T gold-path positions plus the reply position: T + 1 indices into the gold sequence."),
    ("_logits", "at = head(hs[0][pos]", {}, "Logits at those positions",
     "`hs[0][pos]` picks the gold row's states at those indices and the head turns them into `[T + 1, V]` fp32 logits. Row t is the distribution that should put the gold token of step t on top."),
    ("_logits", "return TensorDict(path=at[:-1]", {"ret": True}, "Package the logits",
     "`path` is the first T rows, `reply` the last, and `last` the S last-position distributions."),
    ("compute", "ref = _logits(", {}, "p0: the reference",
     "This ran with the adapter disabled and without gradients, so it is the original model, the target the anchors compare to. At the root row of `path` you see the base model's logit for each option digit."),
    ("compute", "cur = _logits(", {}, "p_theta: the model being trained",
     "The same call with the adapter on and gradients flowing. The difference between `cur` and `ref` at these columns is what training has changed."),
    ("compute", "T, O = len(", {}, "Counts",
     "T gold-path nodes and O off-path nodes. N = T + O trie nodes are scored."),

    ("legal_mask", "mask = torch.zeros(", {}, "An all-False matrix",
     "One row per trie node (N), one column per vocabulary token (V). The width is the model's full vocabulary, not the trie's."),
    ("legal_mask", "for i, tokens in enumerate(legal):", {"to": "mask[i, tokens] = True"}, "Set each node's legal columns",
     "Row i gets True at the tokens the trie allows after node i: the option digits at the root, `]` at the digit nodes. Shown after the loop."),
    ("legal_mask", "return mask", {"ret": True}, "The legal mask",
     "Rows are ordered gold-path nodes first, then off-path nodes, the same order in which `_loss_terms` concatenates the logits."),
    ("compute", "legal = legal_mask(", {}, "Back in compute: the mask",
     "`legal` is the `[N, V]` mask, N = T + O rows in the order the logits are stacked later."),
    ("compute", "gold = torch.tensor(", {}, "Gold token ids", "The gold token at each of the T path steps."),
    ("compute", "weight = torch.tensor(", {}, "Node weights",
     "Gold-path nodes share a total weight of 0.5 and off-path nodes share 0.5, each split evenly within its group. These weight the L_mass and L_out sums."),

    ("_loss_terms", "T = cur_path.shape[0]", {}, "Inside _loss_terms",
     "The pure-tensor part of the loss. It sees only logits, the mask, the gold ids and the weights. `T` is the number of gold-path nodes."),
    ("_loss_terms", "nodes = torch.cat([cur_path", {}, "All trie nodes, p_theta",
     "The T gold-path rows followed by the O off-path rows (`cur_last[1:]`). `cur_last[0]` is skipped: it is the distribution after the final `]`, not a trie node. The result is `[N, V]` and lines up with `legal`."),
    ("_loss_terms", "ref_nodes = torch.cat([ref_path", {}, "All trie nodes, p0", "The same stacking for the reference model."),

    ("_split", "logp = torch.log_softmax", {}, "Log-softmax over the whole vocabulary",
     "`node_stats` runs first, on p0. Log-probabilities over all V tokens, at every node."),
    ("_split", "illegal = logp.masked_fill(legal", {}, "Keep only the illegal tokens",
     "Legal columns become -inf. What remains is the model's mass outside the trie. Its highest token is the model's favourite way to answer outside the format."),
    ("_split", "return TensorDict(", {"ret": True}, "Legal and illegal mass",
     "`log_m` is the log of the mass on legal tokens and `log_r` the log of the mass on illegal tokens, each by its own logsumexp. The illegal mass is summed directly, not computed as `1 - m`, which rounds to 0 in fp32 when the legal mass is nearly 1."),
    ("node_stats", "s = _split(logits, legal)", {}, "Split p0 into legal and illegal",
     "`s` is the result of the `_split` call above: p0's log-probabilities, its illegal part and the two masses."),
    ("node_stats", "lp, idx = (s[", {}, "p0's top illegal tokens",
     "Subtracting `log_r` renormalizes the illegal log-probabilities into a distribution over illegal tokens only. The top 64 (`TOP`) are kept: `idx` their ids, `lp` their log-probabilities. Only these enter the L_out anchor."),
    ("node_stats", "return TensorDict(log_m", {"ret": True}, "The reference statistics",
     "p0's legal mass, illegal mass and top illegal tokens at every node. Computed without gradients and used as fixed targets."),

    ("node_anchor", "s = _split(logits, legal)", {}, "The same split for p_theta",
     "This time with gradients. `s` holds p_theta's log-probabilities, illegal part and masses."),
    ("node_anchor", "mass = (ref", {}, "L_mass per node",
     "A binary KL between p0 and p_theta over (legal, illegal): m0 (log m0 - log m) + r0 (log r0 - log r). Zero when training leaves the legal/illegal split of mass untouched."),
    ("node_anchor", 'lp = s["logp"].gather', {}, "p_theta on p0's top illegal tokens",
     "p_theta's log-probabilities at the token ids in `ref['idx']`, renormalized by p_theta's own illegal mass. `[N, K]`, comparable with `ref['lp']`."),
    ("_kl_top", "rest = _log1mexp(torch.logsumexp(ref_lp", {}, "The remainder bucket, p0",
     "The K tokens do not hold all the illegal mass. `log(1 - exp(logsumexp))` is the log-probability of everything else, a (K+1)-th bucket. `_log1mexp` clamps inside the log so the gradient stays finite when the top tokens hold all the mass."),
    ("_kl_top", "ref = torch.cat([ref_lp", {}, "p0 over K + 1 buckets", "The top tokens plus the remainder."),
    ("_kl_top", "rest = _log1mexp(torch.logsumexp(lp,", {}, "The remainder bucket, p_theta", "The same for p_theta."),
    ("_kl_top", "mod = torch.cat([lp", {}, "p_theta over K + 1 buckets", "Aligned bucket by bucket with p0."),
    ("_kl_top", "return (ref.exp()", {"ret": True}, "KL(p0 || p_theta)",
     "`sum(ref * (log ref - log mod))` over the K + 1 buckets: one value per node."),
    ("node_anchor", "return TensorDict(mass=mass", {"ret": True}, "Two per-node vectors",
     "`mass` is L_mass and `out` is L_out at each node, before weighting."),
    ("_loss_terms", "anchor = node_anchor(", {}, "Anchors at every node",
     "`anchor['mass']` and `anchor['out']`, each `[N]`. Each node is weighted by `weight` and summed below."),
    ("_loss_terms", "sites = torch.stack([cur_reply", {}, "The two L_pos positions",
     "The reply start (the position before `Best answer: [` begins) and the position after the closing `]`, stacked as `[2, V]`."),
    ("_loss_terms", "ref_sites = torch.stack(", {}, "The same under p0", "The reference logits at those two positions."),
    ("tree_loss", "z = logits.float()", {}, "L_tree: gold-path logits",
     "`_loss_terms` passes `cur_path` (`[T, V]`), the first T rows of `legal`, and the gold ids."),
    ("tree_loss", "lse = torch.logsumexp(", {}, "Normalizer over legal tokens",
     "Illegal logits are set to -inf, so each node's `lse` is the log of the sum of exp(logit) over the legal tokens only. The full-vocabulary normalizer cancels in the loss."),
    ("tree_loss", "return (lse - z.gather", {"ret": True}, "L_tree",
     "`lse - z[gold]` is -log of the gold token's probability renormalized over the legal tokens, at each node. Summed over the path it is -log of the product: the probability of the whole gold answer under the trie (Eq. 2)."),
    ("pos_stats", "lp, idx = torch.log_softmax", {}, "p0's top tokens at the two sites",
     "For L_pos the sites have no trie, so the anchor is on the model's top 64 tokens overall."),
    ("pos_anchor", "lp = torch.log_softmax(logits.float(), -1).gather", {}, "p_theta at those tokens",
     "p_theta's log-probabilities at p0's top tokens, for each site."),
    ("pos_anchor", "return _kl_top(", {"ret": True}, "KL per site",
     "The same top-K-plus-remainder KL as before, one value per site. `_loss_terms` sums the two and multiplies by 0.5."),
    ("_loss_terms", "return torch.stack([", {"ret": True}, "Four scalars",
     "[L_tree, L_mass, L_out, L_pos]. L_tree sums over the path; L_mass and L_out are the `weight`-weighted sums of the per-node vectors; L_pos is 0.5 times the sum over its two sites."),
    ("compute", "tree, mass, out, pos = loss_terms(", {}, "Back in compute",
     "The four scalars come back as separate tensors."),
    ("compute", "return TensorDict(tree=tree", {"ret": True}, "The result",
     "`train.py` combines them as `tree + lam * (mass + out + pos)` and calls backward."),
]


# The data flow of compute, as Mermaid. Node ids are the values of NODES below.
FLOW = """flowchart LR
  EX["ex<br/>encode dict"]
  SEQS["seqs<br/>gold + off-path rows"]
  IN["legal mask<br/>gold ids · weights"]
  PAD["left-pad<br/>ids · mask · pos"]
  FWD["transformer<br/>forward"]
  LMH["LM head<br/>path · reply · last"]
  REF["ref = p0<br/>adapter off"]
  CUR["cur = p_θ<br/>adapter on"]
  NODES["trie nodes<br/>[N, V]"]
  STATS["node_stats<br/>p0 mass · top illegal"]
  ANCH["node_anchor<br/>L_mass · L_out"]
  TREE["tree_loss<br/>L_tree"]
  POS["pos anchor<br/>L_pos"]
  OUT["tree · mass<br/>out · pos"]
  EX --> SEQS --> PAD --> FWD --> LMH
  LMH --> REF
  LMH --> CUR
  EX --> IN
  REF --> NODES
  CUR --> NODES
  NODES --> STATS --> ANCH
  NODES --> ANCH
  IN --> TREE
  IN --> ANCH
  CUR --> TREE
  REF --> POS
  CUR --> POS
  TREE --> OUT
  ANCH --> OUT
  POS --> OUT
"""

# function -> (node of its lines, {anchor prefix: node} overrides)
NODES = {
    "compute": ("IN", {"": "EX", "prompt =": "SEQS", "seqs =": "SEQS", "ref = ": "REF",
                       "cur = ": "CUR", "tree, mass, out, pos": "OUT", "return": "OUT"}),
    "batched_hidden": ("PAD", {"base =": "FWD", "out =": "FWD", "last = out": "FWD",
                               "logits =": "LMH", "last = last": "LMH", "return": "LMH"}),
    "_logits": ("LMH", {}),
    "legal_mask": ("IN", {}),
    "_loss_terms": ("NODES", {"anchor =": "ANCH", "sites": "POS", "ref_sites": "POS", "return": "OUT"}),
    "_split": ("STATS", {}),
    "node_stats": ("STATS", {}),
    "node_anchor": ("ANCH", {}),
    "_kl_top": ("ANCH", {}),
    "tree_loss": ("TREE", {}),
    "pos_stats": ("POS", {}),
    "pos_anchor": ("POS", {}),
}


def node_of(fn: str, anchor: str) -> str:
    default, overrides = NODES[fn]
    for prefix, node in overrides.items():
        if anchor == prefix if prefix == "" else anchor.startswith(prefix):
            return node
    return default


class Formatter:
    """Renders local variables as short, readable text."""

    def __init__(self, tok, vocab: int, columns: list[int]):
        self.tok, self.vocab, self.columns = tok, vocab, columns

    def token(self, i) -> str:
        text = self.tok.convert_ids_to_tokens(int(i))
        return text.replace("Ġ", "␣").replace("Ċ", "\\n")

    @staticmethod
    def number(x) -> str:
        x = float(x)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.4g}"

    def ids(self, ids: list[int], keep: int = 5) -> str:
        shown = " ".join(self.token(i) for i in ids[-keep:])
        return f"{len(ids)} tokens, ends …{shown}" if len(ids) > keep else shown

    def vocab_table(self, t: torch.Tensor) -> str:
        """A [V] or [R, V] tensor: the trie columns, and each row's top token."""
        rows = t[None] if t.ndim == 1 else t
        sub = rows[:, self.columns].float().cpu()
        lines = [" " * 6 + "".join(f"{self.token(c):>9}" for c in self.columns)]
        for i, row in enumerate(sub):
            lines.append(f"[{i}]".ljust(6) + "".join(f"{self.number(x):>9}" for x in row))
        best = rows.float().argmax(-1).tolist()
        lines.append("highest token per row: " + ", ".join(self.token(b) for b in best))
        return "\n".join(lines)

    def tensor(self, name: str, t: torch.Tensor, pad: int) -> tuple[str, str]:
        t = t.detach()
        head = f"{list(t.shape)} {str(t.dtype).replace('torch.', '')}"
        if t.ndim == 2 and t.shape[-1] == self.vocab and t.dtype == torch.bool:
            return head, "\n".join(
                f"[{i}] " + " ".join(self.token(j) for j in row.nonzero().flatten().tolist())
                for i, row in enumerate(t))
        if t.ndim in (1, 2) and t.shape[-1] == self.vocab:
            return head, self.vocab_table(t)
        if name == "ids" and t.ndim == 2:
            rows = []
            for i, row in enumerate(t.tolist()):
                lead = next((k for k, x in enumerate(row) if x != pad), len(row))
                rows.append(f"[{i}] {lead} pad, {self.ids(row[lead:], 4)}" if lead
                            else f"[{i}] {self.ids(row, 4)}")
            return head, "\n".join(rows)
        if name == "mask" and t.ndim == 2:
            return head, "\n".join(f"[{i}] {int(s)} real of {t.shape[1]}" for i, s in enumerate(t.sum(-1)))
        if name == "pos" and t.ndim == 2:
            return head, "\n".join(f"[{i}] last position {int(r[-1])}" for i, r in enumerate(t))
        if name == "idx" and t.ndim == 2:
            return head, "\n".join(
                f"[{i}] " + " ".join(self.token(j) for j in row[:6].tolist()) + " …"
                for i, row in enumerate(t))
        if t.ndim == 2 and t.shape[-1] == losses.TOP:
            return head, "\n".join(
                f"[{i}] " + " ".join(self.number(x) for x in row[:6].tolist()) + " …"
                for i, row in enumerate(t.float()))
        if t.numel() <= 24:
            values = t.float().tolist() if t.dtype.is_floating_point else t.tolist()
            return head, json.dumps(self.round(values)).replace('"', "")
        if t.dtype.is_floating_point:
            f = t.float()
            return head, f"min {self.number(f.min())}  max {self.number(f.max())}  mean {self.number(f.mean())}"
        return head, f"first {t.flatten()[:4].tolist()} … last {t.flatten()[-2:].tolist()}"

    def round(self, v):
        if isinstance(v, list):
            return [self.round(x) for x in v]
        return self.number(v) if isinstance(v, float) else v

    def render(self, name: str, v, pad: int):
        """(shape, text) for a local variable, or None to leave it out."""
        if name in SKIP or name.startswith(("_", ".")):
            return None
        if isinstance(v, TensorDict):
            parts = []
            for key in v.keys():
                shape, text = self.tensor(key, v[key], pad)
                parts.append(f"{key} {shape}\n" + "\n".join("  " + x for x in text.split("\n")))
            return f"TensorDict, batch {list(v.batch_size)}", "\n".join(parts)
        if isinstance(v, torch.Tensor):
            return self.tensor(name, v, pad)
        if hasattr(v, "last_hidden_state"):
            t = v.last_hidden_state
            return "model output", f"last_hidden_state {list(t.shape)} {str(t.dtype).replace('torch.', '')}"
        if isinstance(v, dict) and "input_ids" in v:
            return "dict", self.example(v)
        if isinstance(v, list) and v and all(isinstance(x, torch.Tensor) for x in v):
            return f"list of {len(v)} tensors", "\n".join(f"[{i}] {list(x.shape)}" for i, x in enumerate(v))
        if isinstance(v, list) and v and all(isinstance(x, list) for x in v) and name in ("seqs", "legal"):
            show = (lambda x: " ".join(self.token(i) for i in x)) if name == "legal" else self.ids
            return f"list of {len(v)}", "\n".join(f"[{i}] {show(x)}" for i, x in enumerate(v))
        if isinstance(v, list) and v and all(isinstance(x, int) for x in v) and name in ("prompt", "x", "tokens"):
            return f"list of {len(v)} ids", self.ids(v)
        if isinstance(v, (bool, int, float, str)):
            return type(v).__name__, repr(v)
        if isinstance(v, (list, tuple)):
            text = repr(v)
            return type(v).__name__, text if len(text) < 160 else text[:157] + "…"
        return type(v).__name__, ""

    def example(self, ex: dict) -> str:
        path = "\n".join(
            f"  t={t} at index {p}: legal {{{' '.join(self.token(i) for i in legal)}}}, gold {self.token(g)}"
            for t, (p, legal, g) in enumerate(ex["path"]))
        off = "\n".join(
            f"  after {' '.join(self.token(i) for i in node)}: legal {{{' '.join(self.token(i) for i in legal)}}}"
            for node, legal in ex["off_path"])
        return (f"input_ids  {len(ex['input_ids'])} tokens, ending …{self.ids(ex['input_ids'][-8:], 8)}\n"
                f"gold       option slot {ex['gold'] + 1} of {ex['n_options']}\n"
                f"reply      index {ex['reply']}\n"
                f"path\n{path}\noff_path\n{off}")


class Tracer:
    """Records the rendered locals of the watched functions at every line."""

    def __init__(self, fmt: Formatter, pad: int):
        self.fmt, self.pad = fmt, pad
        self.watch = {fn.__code__: name for name, fn in FUNCTIONS.items()}
        self.calls: dict[str, int] = {}
        self.events: dict[tuple[str, int], list[dict]] = {}
        self.stack: list[str] = []
        self.clock = 0

    def snapshot(self, frame) -> dict:
        out = {}
        with torch.no_grad():
            for name, value in list(frame.f_locals.items()):
                rendered = self.fmt.render(name, value, self.pad)
                if rendered is not None:
                    out[name] = rendered
        return out

    def record(self, key, frame, line, retval=None):
        self.clock += 1
        event = {"line": line, "t": self.clock, "stack": list(self.stack), "vars": self.snapshot(frame)}
        if line is None:
            event["return"] = self.fmt.render("return", retval, self.pad)
        self.events[key].append(event)

    def __call__(self, frame, event, arg):
        name = self.watch.get(frame.f_code)
        if event != "call" or name is None:
            return None
        key = (name, self.calls.get(name, 0))
        self.calls[name] = key[1] + 1
        self.events[key] = []
        self.stack.append(name)

        def local(frame, event, arg):
            if event == "line":
                self.record(key, frame, frame.f_lineno)
            elif event == "return":
                self.record(key, frame, None, arg)
                self.stack.pop()
            return local

        return local


def source(fn) -> dict:
    """The function's lines, its docstring span, and the extent of each statement."""
    lines, start = inspect.getsourcelines(fn)
    fdef = ast.parse("".join(lines)).body[0]
    doc = fdef.body[0] if ast.get_docstring(fdef, clean=False) is not None else None
    shown = [text.rstrip("\n") for text in lines]
    index = {i: i - 1 for i in range(1, len(lines) + 1)}
    docs = [doc.lineno - 1, doc.end_lineno - 1] if doc is not None else None
    ends = {}
    for node in ast.walk(fdef):
        if isinstance(node, ast.stmt) and node is not fdef:
            header = isinstance(node, (ast.For, ast.While, ast.If, ast.With))
            ends.setdefault(node.lineno, node.body[0].lineno - 1 if header else node.end_lineno)
    return {"start": start, "lines": lines, "shown": shown, "index": index, "ends": ends, "doc": docs}


def build_steps(tracer: Tracer, funcs: dict) -> list[dict]:
    steps = []
    for fn, anchor, opts, title, body in STEPS:
        events = tracer.events[(fn, 0)] if "call" not in opts else tracer.events[(fn, opts["call"])]
        src = funcs[fn]
        if opts.get("entry"):
            first = events[0]
            line = 1
            after, when, before, stack = first["vars"], first["t"] - 0.5, {}, first["stack"]
            returned, last_line = None, 1
        else:
            line = next((i for i, t in enumerate(src["lines"], 1) if anchor in t), None)
            if line is None:
                raise SystemExit(f"{fn}: no line contains {anchor!r}")
            absolute = src["start"] + line - 1
            hits = [k for k, e in enumerate(events) if e["line"] == absolute]
            if not hits:
                raise SystemExit(f"{fn}: line {line} ({anchor!r}) never ran")
            k = hits[-1] if opts.get("occ") == "last" else hits[0]
            last_line = line  # the last source line the step covers
            end = src["start"] + src["ends"].get(line, line) - 1
            before, stack = events[k]["vars"], events[k]["stack"]
            if "to" in opts:  # a loop: the state after its last pass
                last_line = next(i for i, t in enumerate(src["lines"], 1) if opts["to"] in t)
                last_abs = src["start"] + last_line - 1
                end = src["start"] + src["ends"].get(last_line, last_line) - 1
                k_last = [j for j, e in enumerate(events) if e["line"] == last_abs][-1]
                nxt = next(e for e in events[k_last + 1:]
                           if e["line"] is None or not last_abs <= e["line"] <= end)
                last_line = src["ends"].get(last_line, last_line)
            elif opts.get("ret"):
                nxt = events[-1]
                last_line = src["ends"].get(line, line)
            else:
                nxt = next(e for e in events[k + 1:]
                           if e["line"] is None or not absolute <= e["line"] <= end)
                last_line = src["ends"].get(line, line)
            after, when = nxt["vars"], nxt["t"]
            returned = nxt.get("return") if nxt["line"] is None and opts.get("ret") else None
        variables = [{"name": n, "shape": s, "text": t, "changed": before.get(n) != (s, t)}
                     for n, (s, t) in after.items()]
        if returned is not None:
            variables.append({"name": "return", "shape": returned[0], "text": returned[1], "changed": True})
        steps.append({"fn": fn, "line": src["index"][line], "line_end": src["index"][last_line],
                      "node": node_of(fn, anchor), "frame": (fn, opts.get("call", 0)),
                      "title": title, "body": body,
                      "stack": stack, "vars": variables, "t": when})
    steps += call_steps(tracer, funcs, steps)
    steps.sort(key=lambda s: s["t"])
    for i, s in enumerate(steps):
        if s.get("call"):  # the diagram node of the work the call starts
            s["node"] = next(x["node"] for x in steps[i + 1:] if not x.get("call"))
        del s["t"], s["frame"]
    return steps


def call_steps(tracer: Tracer, funcs: dict, steps: list[dict]) -> list[dict]:
    """One step per call on the way down, showing the calling line and the arguments."""

    def frame_at(name: str, when: int):
        return next(key for key, evs in tracer.events.items()
                    if key[0] == name and evs[0]["t"] <= when <= evs[-1]["t"])

    def parent(frame):
        first = tracer.events[frame][0]
        return frame_at(first["stack"][-2], first["t"]) if len(first["stack"]) > 1 else None

    needed = set()
    for step in steps:
        frame = step["frame"]
        while frame is not None and frame not in needed:
            needed.add(frame)
            frame = parent(frame)
    out = []
    for frame in needed:
        caller = parent(frame)
        if caller is None:
            continue
        first = tracer.events[frame][0]
        events = tracer.events[caller]
        k = max(j for j, e in enumerate(events) if e["line"] is not None and e["t"] < first["t"])
        src = funcs[caller[0]]
        line = events[k]["line"] - src["start"] + 1
        args = [{"name": n, "shape": sh, "text": t, "changed": True, "tag": "argument"}
                for n, (sh, t) in first["vars"].items()]
        out.append({
            "fn": caller[0], "line": src["index"][line],
            "line_end": src["index"][src["ends"].get(line, line)],
            "title": f"Call {frame[0]}", "call": frame[0],
            "body": f"`{caller[0]}` calls `{frame[0]}`. Its arguments are on the right; the steps that follow run inside it, and this line is highlighted again when the call returns.",
            "stack": events[k]["stack"], "vars": args, "t": first["t"] - 0.5, "frame": frame,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="blogs/data/loss_code.json")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    example = storage.load_examples(storage.DEFAULT_SOURCE, "train")[EXAMPLE]
    order = list(range(len(example.options)))
    random.Random(SEED).shuffle(order)
    ex = answers.encode(example, tok, random.Random(SEED))
    assert order.index(example.gold) == ex["gold"]

    model = modeling.load_trained(MODEL, ADAPTER, "cuda")
    vocab = model.get_base_model().lm_head.out_features
    columns = sorted({t for _, legal, g in ex["path"] for t in legal + [g]})
    tracer = Tracer(Formatter(tok, vocab, columns), tok.pad_token_id)
    sys.settrace(tracer)
    try:
        losses.compute(model, ex, "cuda", tok.pad_token_id)
    finally:
        sys.settrace(None)

    funcs = {name: source(fn) for name, fn in FUNCTIONS.items()}
    data = {
        "model": MODEL,
        "adapter": ADAPTER,
        "row": {
            "source": example.source,
            "passage": example.parts[0].text,
            "question": example.question,
            "options": [example.options[i] for i in order],
            "gold": ex["gold"],
            "tokens": len(ex["input_ids"]),
        },
        "flow": FLOW,
        "funcs": {name: {"lines": f["shown"], "doc": f["doc"]} for name, f in funcs.items()},
        "steps": build_steps(tracer, funcs),
    }
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(data, ensure_ascii=False))
    print(f"{len(data['steps'])} steps -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

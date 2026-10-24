import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import a from "../../data/anchors.json";
import t from "../../data/trie_loss.json";
import { Stepper, Toggle } from "../components/Controls.jsx";
import { buildLayout } from "./TrieLoss.jsx";

const txt = (id) => a.tokText[String(id)] ?? t.tokText[String(id)];
const key = (path) => path.join(",");
const show = (s) => JSON.stringify(s).slice(1, -1); // makes "\n" visible
const nodeName = (path) => (path.length ? path.map(txt).join("") : "root");
const sci = (x) => (x === 0 ? "0" : x >= 0.01 ? x.toFixed(3) : x.toExponential(1));
const pct = (p) => (p < 0.001 ? "<0.1%" : `${(p * 100).toFixed(1)}%`);
export const ADAPTERS = [["lam1", "λ = 1"], ["lam001", "λ = 0.01"]];
const LAMBDA = { lam1: 1, lam001: 0.01 };

// The same six nodes are anchored for both adapters.
const NODES = a.adapters.lam1.nodes;
const POS = a.adapters.lam1.positions;
const N0 = 2; // first node step
const P0 = N0 + NODES.length; // first position step
const TOTAL = P0 + POS.length;
const goldSuffix = t.suffixes[t.row.gold];
const goldKeys = new Set(goldSuffix.map((_, i) => key(goldSuffix.slice(0, i + 1))));

const STEPS = [
  { title: "One distribution, four pieces" },
  { title: "Six anchored nodes" },
  ...NODES.map((n) => ({ title: `${n.kind === "gold" ? "Gold" : "Off-path"} node ${nodeName(n.path)}` })),
  { title: "Reply start" },
  { title: "After the closing ]" },
  { title: "Everything together" },
];

export default function Anchors() {
  const [step, setStep] = useState(0);
  const [ad, setAd] = useState("lam1");
  const A = a.adapters[ad];
  const node = step >= N0 && step < P0 ? step - N0 : null;
  const pos = step >= P0 && step < TOTAL ? step - P0 : null;

  return (
    <section className="trie-section" id="anchors">
      <p className="eyebrow">3 · KL anchors</p>
      <h2>What keeps the rest of the distribution in place</h2>
      <p className="lede small">
        Same row, same trie. p<sub>0</sub> is the model with the LoRA adapter off; p<sub>θ</sub> is the trained adapter. Each
        anchor is a KL divergence to p<sub>0</sub>, so it is 0 until training moves something.
      </p>
      <Stepper steps={STEPS} step={step} setStep={setStep} sectionId="anchors">
        {step >= 2 && <Toggle id="adapter" label="Adapter" options={ADAPTERS} value={ad} onChange={setAd} />}
      </Stepper>
      <div className="trie-grid">
        <AnimatePresence mode="wait">
          {step === 0 && <Card key="split"><Decomposition /></Card>}
          {step >= 1 && step < P0 && (
            <motion.div key="trie" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ minWidth: 0 }}>
              <AnchorTrie A={A} node={node} showMarks={step >= 1} />
            </motion.div>
          )}
          {pos !== null && <Card key="timeline"><Timeline active={pos} /></Card>}
          {step === TOTAL && <Card key="table"><Contributions A={A} /></Card>}
        </AnimatePresence>
        <div className="panel">
          <AnimatePresence mode="wait">
            <motion.div key={`${step}-${ad}`} initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }} transition={{ duration: 0.22 }}>
              <h3 className="panel-title">{STEPS[step].title}</h3>
              {step === 0 && <Pieces />}
              {step === 1 && <NodeMap />}
              {node !== null && <NodePanel A={A} i={node} />}
              {pos !== null && <PosPanel A={A} i={pos} />}
              {step === TOTAL && <Together A={A} lam={LAMBDA[ad]} />}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

function Card({ children }) {
  return (
    <motion.div className="viz-card" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      {children}
    </motion.div>
  );
}

/** The trie of section 2 with the anchored nodes marked; `node` is the one being read. */
function AnchorTrie({ A, node, showMarks }) {
  const { nodes, H } = useMemo(buildLayout, []);
  const X0 = 26, DX = 88, LABEL = X0 + 3 * DX + 24;
  const anchored = Object.fromEntries(NODES.map((n, i) => [key(n.path), i]));
  const selKeys = new Set();
  if (node !== null) NODES[node].path.forEach((_, i) => selKeys.add(key(NODES[node].path.slice(0, i + 1))));
  return (
    <div className="trie-wrap">
      <svg className="trie" viewBox={`0 0 560 ${H}`} role="img" aria-label="answer trie with anchored nodes">
        {nodes.filter((n) => n.parent).map((n) => {
          const p = n.parent;
          const x1 = X0 + p.depth * DX, x2 = X0 + n.depth * DX;
          const cls = selKeys.has(n.key) ? "edge sel" : goldKeys.has(n.key) ? "edge gold" : "edge";
          return <path key={n.key} className={cls} strokeWidth={cls === "edge" ? 1.2 : 2.4} d={`M${x1},${p.y} C${x1 + DX * 0.55},${p.y} ${x2 - DX * 0.55},${n.y} ${x2},${n.y}`} />;
        })}
        {nodes.map((n) => {
          const x = X0 + n.depth * DX;
          const i = anchored[n.key];
          const marked = showMarks && i !== undefined;
          const kind = marked ? NODES[i].kind : "";
          const cls = ["tnode", n.leaf !== null ? "leaf" : "", marked ? kind : "", n.leaf !== null && goldKeys.has(n.key) ? "gold" : "",
            node !== null && !(marked && i === node) ? "dim" : ""].join(" ");
          return (
            <g key={n.key || "root"} className={cls}>
              {marked && i === node && (
                <motion.circle cx={x} cy={n.y} className="ring" initial={{ r: 13 }} animate={{ r: [13, 19, 13] }} transition={{ repeat: Infinity, duration: 1.6 }} />
              )}
              <circle cx={x} cy={n.y} r={n.key === "" ? 11 : 9} />
              <text x={x} y={n.y + 4} textAnchor="middle">{n.key === "" ? "[" : txt(n.tok)}</text>
              {marked && (
                <g className="badge">
                  <circle cx={x + 10} cy={n.y - 10} r={6.5} />
                  <text x={x + 10} y={n.y - 7} textAnchor="middle">{i + 1}</text>
                </g>
              )}
              {n.leaf !== null && <line x1={x + 10} x2={LABEL - 6} y1={n.y} y2={n.y} className="leader" />}
              {n.leaf !== null && (
                <text x={LABEL} y={n.y + 4} className={goldKeys.has(n.key) ? "lab gold" : "lab"}>
                  [{n.leaf + 1}] {t.row.options[n.leaf]}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="trie-legend">
        <i className="sw gold" /> gold-path node · <i className="sw off" /> sampled off-path node · numbers are the order of the next steps
      </div>
    </div>
  );
}

/** Root node of p0: the whole vocabulary split into legal and illegal mass, then zoomed. */
function Decomposition() {
  const root = t.models.base.nodes[""];
  const legal = t.nodes.find((n) => n.path.length === 0).legal;
  const n0 = a.adapters.lam1.nodes[0];
  const full = legal.map((id, i) => ({ label: txt(id), p: Math.exp(root.logp[i]) })).sort((x, y) => y.p - x.p);
  return (
    <div className="decomp">
      <span className="viz-label">p₀ at the root, whole vocabulary</span>
      <div className="dbar">
        {full.map((s, i) => (
          <motion.span
            key={s.label}
            className={i === 0 ? "dseg legal big" : "dseg legal"}
            initial={{ width: 0 }}
            animate={{ width: `${s.p * 100}%` }}
            transition={{ duration: 0.7, delay: 0.1 + i * 0.04 }}
            title={`${s.label} ${pct(s.p)}`}
          >
            {s.p > 0.06 ? `${s.label} ${pct(s.p)}` : ""}
          </motion.span>
        ))}
        <span className="dseg illegal" style={{ width: "4px" }} />
      </div>
      <div className="dlegend">
        <span><i className="sw legal" /> legal: m = {(n0.m0 * 100).toFixed(4)}%</span>
        <span><i className="sw illegal" /> illegal: 1 − m = {sci(n0.r0)}</span>
      </div>
      <motion.div className="zoom" initial={{ opacity: 0, scaleY: 0 }} animate={{ opacity: 1, scaleY: 1 }} transition={{ delay: 0.8, duration: 0.4 }}>
        <span>× {Math.round(1 / n0.r0).toLocaleString("en-US")}</span>
      </motion.div>
      <span className="viz-label">zoom into the illegal sliver: p̄₀, renormalised</span>
      <div className="dbar">
        {n0.illegal.map(([s, p0], i) => (
          <motion.span key={i} className="dseg out" initial={{ width: 0 }} animate={{ width: `${p0 * 100}%` }} transition={{ duration: 0.6, delay: 1.1 + i * 0.05 }} title={`${show(s)} ${pct(p0)}`}>
            {p0 > 0.07 ? `${show(s)} ${pct(p0)}` : ""}
          </motion.span>
        ))}
        <span className="dseg rest" style={{ width: `${n0.illegalRest[0] * 100}%` }} title="the rest" />
      </div>
    </div>
  );
}

function Pieces() {
  return (
    <>
      <div className="eq">
        <span>p(v) = m · q(v) &nbsp;&nbsp;&nbsp;&nbsp;if v is legal</span>
        <span>p(v) = (1 − m) · p̄(v) &nbsp;otherwise</span>
      </div>
      <div className="pieces">
        <span className="piece k-q">q → L_tree</span>
        <span className="piece k-m">m → L_mass</span>
        <span className="piece k-o">p̄ → L_out</span>
        <span className="piece k-p">other positions → L_pos</span>
      </div>
    </>
  );
}

function NodeMap() {
  const weight = (kind) => 0.5 / NODES.filter((n) => n.kind === kind).length;
  return (
    <div className="nodemap">
      <div className="eq">
        <span>gold path: 0.5 shared → {weight("gold").toFixed(3)} each</span>
        <span>off path: 0.5 shared → {weight("off").toFixed(3)} each</span>
      </div>
      {NODES.map((n, i) => (
        <motion.div key={i} className={`nm-row ${n.kind}`} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.07 }}>
          <span className="nm-name">{i + 1} · {nodeName(n.path)}</span>
          <span className="nm-legal">legal {n.legal.length > 6 ? `${txt(n.legal[0])} … ${txt(n.legal[n.legal.length - 1])}` : n.legal.map(txt).join(" ")}</span>
          <span className="nm-w">w {n.weight.toFixed(3)}</span>
        </motion.div>
      ))}
    </div>
  );
}

/** Illegal mass 1 − m on a log axis: p0 hollow, p_theta filled. */
function MassDumbbell({ n }) {
  const lo = -7, hi = -1, W = 440, X0 = 14, X1 = 426;
  const x = (v) => X0 + ((Math.log10(Math.max(v, 1e-7)) - lo) / (hi - lo)) * (X1 - X0);
  const clamp = (v) => Math.min(Math.max(v, 46), W - 46);
  return (
    <svg className="massplot" viewBox={`0 0 ${W} 74`} role="img" aria-label="illegal mass before and after training">
      {[-7, -5, -3, -1].map((e) => (
        <g key={e}>
          <line x1={x(10 ** e)} x2={x(10 ** e)} y1={22} y2={50} className="grid" />
          <text x={x(10 ** e)} y={68} textAnchor="middle" className="tick">1e{e}</text>
        </g>
      ))}
      <motion.line x1={x(n.r0)} y1={36} y2={36} className="mline" initial={{ x2: x(n.r0) }} animate={{ x2: x(n.r) }} transition={{ duration: 0.8 }} />
      <circle cx={x(n.r0)} cy={36} r={5} className="dot0" />
      <motion.circle cy={36} r={5} className="dot1" initial={{ cx: x(n.r0) }} animate={{ cx: x(n.r) }} transition={{ duration: 0.8 }} />
      <text x={clamp(x(n.r0))} y={18} textAnchor="middle" className="kl">p₀ {sci(n.r0)}</text>
      <text x={clamp(x(n.r))} y={56} textAnchor="middle" className="kl">p_θ {sci(n.r)}</text>
    </svg>
  );
}

/** Paired bars: p0 (hollow) and p_theta (filled) for the same tokens. */
function PairBars({ rows, rest }) {
  const max = Math.max(...rows.flatMap(([, p0, p]) => [p0, p]), rest ? Math.max(...rest) : 0);
  const all = rest ? [...rows, ["(rest)", rest[0], rest[1]]] : rows;
  return (
    <div className="pairbars">
      {all.map(([s, p0, p], i) => (
        <div key={i} className="pb-row">
          <span className="pb-tok">{show(s)}</span>
          <span className="pb-bars">
            <motion.span className="pb0" initial={{ width: 0 }} animate={{ width: `${(p0 / max) * 100}%` }} transition={{ duration: 0.6, delay: i * 0.04 }} />
            <motion.span className="pb1" initial={{ width: 0 }} animate={{ width: `${(p / max) * 100}%` }} transition={{ duration: 0.6, delay: 0.15 + i * 0.04 }} />
          </span>
          <span className="pb-val">
            {pct(p0)} <span className="arrow">→</span> <b>{pct(p)}</b>
          </span>
        </div>
      ))}
    </div>
  );
}

/** A bar that fills as nodes are added: what is already in, this node's share, and the total. */
function Running({ cls, label, parts, i, total }) {
  const before = parts.slice(0, i).reduce((s, v) => s + v, 0);
  return (
    <div className={`run ${cls}`}>
      <span className="run-label">{label}</span>
      <span className="run-track">
        <motion.span className="run-done" initial={{ width: 0 }} animate={{ width: `${(before / total) * 100}%` }} />
        <motion.span className="run-now" initial={{ width: 0 }} animate={{ width: `${(parts[i] / total) * 100}%` }} transition={{ delay: 0.15 }} />
      </span>
      <span className="run-val">+{sci(parts[i])} → {sci(before + parts[i])}</span>
    </div>
  );
}

function NodePanel({ A, i }) {
  const n = A.nodes[i];
  const wm = A.nodes.map((x) => x.weight * x.mass);
  const wo = A.nodes.map((x) => x.weight * x.out);
  return (
    <div className="nodepanel">
      <div className="np-head">
        <span className={`nchip ${n.kind} on`}>{n.kind === "gold" ? "gold path" : "off path"}</span>
        <span className="np-w">weight {n.weight.toFixed(3)}</span>
      </div>
      <span className="viz-label k-m">L_mass · illegal mass 1 − m <b>KL {sci(n.mass)}</b></span>
      <MassDumbbell n={n} />
      <span className="viz-label k-o">
        L_out · the illegal tokens, renormalised <i className="sw p0" /> p₀ <i className="sw p1" /> p_θ <b>KL {sci(n.out)}</b>
      </span>
      <PairBars rows={n.illegal.slice(0, 6)} rest={n.illegalRest} />
      <span className="viz-label">weight × KL, added to the running totals</span>
      <Running cls="k-m" label="L_mass" parts={wm} i={i} total={A.total.mass} />
      <Running cls="k-o" label="L_out" parts={wo} i={i} total={A.total.out} />
    </div>
  );
}

/** The sequence the model writes, with the two unsupervised positions marked. */
function Timeline({ active }) {
  const stops = [
    { id: 0, label: "reply start", sub: "before the prefill" },
    { id: 1, label: "after ]", sub: "free text resumes" },
  ];
  const marker = (s) => (
    <motion.span key={s.id} className={s.id === active ? "tl-mark on" : "tl-mark"} animate={s.id === active ? { y: [0, -4, 0] } : { y: 0 }} transition={{ repeat: Infinity, duration: 1.4 }}>
      <b>▼</b>
      <span>{s.label}</span>
      <small>{s.sub}</small>
    </motion.span>
  );
  return (
    <div className="timeline">
      <div className="tl-row">
        <span className="tl-chip dim">prompt …</span>
        {marker(stops[0])}
        <span className="tl-chip">Best answer: [</span>
        {goldSuffix.map((id, i) => (
          <span key={i} className="tl-chip gold">{txt(id)}</span>
        ))}
        {marker(stops[1])}
        <span className="tl-chip dim">…</span>
      </div>
      <div className="tl-key">
        <span className="tl-chip gold">answer trie</span> L_tree, L_mass, L_out <span className="tl-chip dim">outside the trie</span> L_pos only
      </div>
    </div>
  );
}

function PosPanel({ A, i }) {
  const p = A.positions[i];
  const before = A.positions.slice(0, i).reduce((s, x) => s + x.weight * x.kl, 0);
  return (
    <div className="nodepanel">
      <span className="viz-label k-p">
        full-vocabulary next-token distribution <i className="sw p0" /> p₀ <i className="sw p1" /> p_θ <b>KL {sci(p.kl)}</b>
      </span>
      <PairBars rows={p.top.slice(0, 6)} rest={p.rest} />
      <span className="viz-label">0.5 × KL, added to the running total</span>
      <div className={`run k-p`}>
        <span className="run-label">L_pos</span>
        <span className="run-track">
          <motion.span className="run-done" initial={{ width: 0 }} animate={{ width: `${(before / A.total.pos) * 100}%` }} />
          <motion.span className="run-now" initial={{ width: 0 }} animate={{ width: `${((p.weight * p.kl) / A.total.pos) * 100}%` }} transition={{ delay: 0.15 }} />
        </span>
        <span className="run-val">+{sci(p.weight * p.kl)} → {sci(before + p.weight * p.kl)}</span>
      </div>
    </div>
  );
}

/** Each term's share per node, one column per loss and one scale per column. */
function Contributions({ A }) {
  const rows = [
    ...A.nodes.map((n, i) => ({ name: `${i + 1} · ${nodeName(n.path)}`, kind: n.kind, mass: n.weight * n.mass, out: n.weight * n.out })),
    ...A.positions.map((p) => ({ name: p.name, kind: "pos", pos: p.weight * p.kl })),
  ];
  const max = (k) => Math.max(...rows.map((r) => r[k] ?? 0));
  const cols = [["mass", "L_mass", "k-m"], ["out", "L_out", "k-o"], ["pos", "L_pos", "k-p"]];
  return (
    <div className="contrib">
      <div className="ct-head">
        <span />
        {cols.map(([, name, cls]) => (
          <b key={name} className={cls}>{name}</b>
        ))}
      </div>
      {rows.map((r, i) => (
        <div key={i} className={`ct-row ${r.kind}`}>
          <span className="ct-name">{r.name}</span>
          {cols.map(([k, , cls]) => (
            <span key={k} className={`ct-cell ${cls}`}>
              {r[k] !== undefined && (
                <>
                  <motion.span className="ct-bar" initial={{ width: 0 }} animate={{ width: `${(r[k] / max(k)) * 100}%` }} transition={{ delay: 0.1 + i * 0.05 }} />
                  <em>{sci(r[k])}</em>
                </>
              )}
            </span>
          ))}
        </div>
      ))}
      <div className="ct-row sum">
        <span className="ct-name">sum</span>
        {cols.map(([k, , cls]) => (
          <span key={k} className={`ct-cell ${cls}`}><em>{sci(A.total[k])}</em></span>
        ))}
      </div>
      <span className="viz-label">each column has its own scale</span>
    </div>
  );
}

function Together({ A, lam }) {
  const parts = [
    ["L_tree", A.total.tree, "k-q"],
    ["λ · L_mass", lam * A.total.mass, "k-m"],
    ["λ · L_out", lam * A.total.out, "k-o"],
    ["λ · L_pos", lam * A.total.pos, "k-p"],
  ];
  const sum = parts.reduce((s, [, v]) => s + v, 0);
  return (
    <div className="together">
      <div className="eq">
        <span>L = L_tree + λ (L_mass + L_out + L_pos)</span>
        <span>
          λ = {lam} → L = <b>{sum.toFixed(3)}</b>
        </span>
      </div>
      <div className="sbar">
        {parts.map(([name, v, cls]) => (
          <motion.span key={name} className={`sseg ${cls}`} initial={{ width: 0 }} animate={{ width: `${(v / sum) * 100}%` }} transition={{ duration: 0.7 }} title={`${name} ${sci(v)}`} />
        ))}
      </div>
      <div className="slegend">
        {parts.map(([name, v, cls]) => (
          <span key={name} className={cls}><i className="sw" style={{ background: "currentColor" }} /> {name} <b>{sci(v)}</b></span>
        ))}
      </div>
    </div>
  );
}

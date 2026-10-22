import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import a from "../../data/anchors.json";
import t from "../../data/trie_loss.json";
import { Stepper, Toggle } from "../components/Controls.jsx";

const txt = (id) => a.tokText[String(id)] ?? t.tokText[String(id)];
const show = (s) => JSON.stringify(s).slice(1, -1); // makes "\n" visible
const nodeName = (path) => (path.length ? path.map(txt).join("") : "root");
const sci = (x) => (x === 0 ? "0" : x >= 0.01 ? x.toFixed(3) : x.toExponential(1));
const pct = (p) => (p < 0.001 ? "<0.1%" : `${(p * 100).toFixed(1)}%`);
const small = (p) => `${(p * 100).toPrecision(2)}%`;
export const ADAPTERS = [["lam1", "λ = 1"], ["lam001", "λ = 0.01"]];

const STEPS = [
  { title: "What L_tree leaves free", sub: "the parts of the distribution no loss has touched yet" },
  { title: "L_mass: the legal mass", sub: "does the model still bet on valid answer tokens?" },
  { title: "L_out: the illegal tokens", sub: "if it leaves the format, does it leave the same way as before?" },
  { title: "Which nodes are anchored", sub: "the gold path plus a few random other nodes" },
  { title: "L_pos: outside the trie", sub: "the reply start and the text after the answer" },
];

export default function Anchors() {
  const [step, setStep] = useState(0);
  const [ad, setAd] = useState("lam001");
  const [node, setNode] = useState(0);
  const A = a.adapters[ad];

  return (
    <section className="trie-section" id="anchors">
      <p className="eyebrow">3 · KL anchors</p>
      <h2>What keeps the rest of the distribution in place</h2>
      <p className="lede small">
        Same row. p<sub>0</sub> is the model with the LoRA adapter switched off; p<sub>θ</sub> is the trained adapter. Every
        number below comes from <code>train/losses.py</code> on this row.
      </p>
      <Glossary />
      <h3 className="sub-h">Step by step</h3>
      <Stepper steps={STEPS} step={step} setStep={setStep} sectionId="anchors">
        {step >= 1 && <Toggle id="adapter" label="Adapter" options={ADAPTERS} value={ad} onChange={setAd} />}
      </Stepper>
      <div className="trie-grid">
        <div className="viz-card">
          <AnimatePresence mode="wait">
            <motion.div key={`${step}-${ad}-${node}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
              {step === 0 && <Decomposition />}
              {step === 1 && <MassPlot A={A} />}
              {step === 2 && <OutView A={A} node={node} setNode={setNode} />}
              {step === 3 && <NodeMap A={A} />}
              {step === 4 && <PosView A={A} />}
            </motion.div>
          </AnimatePresence>
        </div>
        <div className="panel">
          <AnimatePresence mode="wait">
            <motion.div key={`${step}-${ad}`} initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }} transition={{ duration: 0.22 }}>
              <h3 className="panel-title">{STEPS[step].title}</h3>
              <p className="panel-sub">{STEPS[step].sub}</p>
              <Text step={step} A={A} node={node} />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

function Text({ step, A, node }) {
  const tot = A.total;
  if (step === 0)
    return (
      <>
        <p className="panel-body">At any trie node the next-token distribution splits in two:</p>
        <div className="eq">
          <span>p(v) = m · q(v) &nbsp;&nbsp;&nbsp;if v is legal</span>
          <span>p(v) = (1 − m) · p̄(v) &nbsp;otherwise</span>
        </div>
        <div className="who">
          <span><b className="k-q">q</b> the answer distribution: set by L_tree</span>
          <span><b className="k-m">m</b> how much mass stays on legal tokens: free under L_tree</span>
          <span><b className="k-o">p̄</b> the shape of everything illegal: free under L_tree</span>
          <span><b className="k-p">other positions</b> never touched by L_tree</span>
        </div>
        <p className="panel-body">
          L_tree cancels the full-vocabulary normaliser, so its gradient cannot see m or p̄. The anchors tie each free
          piece to p<sub>0</sub> with a KL, leaving only q to move.
        </p>
      </>
    );
  if (step === 1)
    return (
      <>
        <p className="panel-body">A two-outcome KL between p<sub>0</sub>'s and p<sub>θ</sub>'s legal mass, at each anchored node:</p>
        <div className="eq">
          <span>L_mass = m₀ log(m₀/m) + (1−m₀) log((1−m₀)/(1−m))</span>
          <span className="eq-note">both masses are summed directly; 1 − m would round to 0 in fp32</span>
        </div>
        <p className="panel-body">
          Weighted over the six nodes: <b className="num">{sci(tot.mass)}</b>. The plot shows the illegal mass 1 − m on a log
          scale: training pushes it away from p<sub>0</sub>, and λ decides how far.
        </p>
      </>
    );
  if (step === 2) {
    const n = A.nodes[node];
    return (
      <>
        <p className="panel-body">
          Inside the illegal mass, p̄ is renormalised on both sides, then compared over p<sub>0</sub>'s top {a.top} illegal tokens plus
          one bucket for the rest (the chart shows the first 8).
        </p>
        <div className="eq">
          <span>L_out = KL(p̄₀ ‖ p̄_θ)</span>
          <span>
            at node {nodeName(n.path)}: <b>{sci(n.out)}</b>
          </span>
          <span className="eq-note">weighted over the six nodes: {sci(tot.out)}</span>
        </div>
        <p className="panel-body">
          These are the tokens the model would write if it left the trie, e.g. <code>{"]\\n\\n"}</code> or <code>]**</code>. L_tree never
          sees them.
        </p>
      </>
    );
  }
  if (step === 3)
    return (
      <>
        <p className="panel-body">
          Every gold-path node, plus {a.offPath} internal nodes sampled off the gold path (new ones for every example and
          epoch), each fed as an extra sequence: prompt + the node's tokens.
        </p>
        <div className="eq">
          <span>gold-path nodes share 0.5 → {(0.5 / A.nodes.filter((n) => n.kind === "gold").length).toFixed(3)} each</span>
          <span>off-path nodes share 0.5 → {(0.5 / A.nodes.filter((n) => n.kind === "off").length).toFixed(3)} each</span>
        </div>
        <p className="panel-body">Without the off-path nodes, the parts of the trie the gold answer never visits could drift freely.</p>
      </>
    );
  return (
    <>
      <p className="panel-body">
        Two context positions get a full-vocabulary KL (top {a.top} + bucket), 0.5 each:
      </p>
      <div className="eq">
        <span>reply start: KL {sci(A.positions[0].kl)}</span>
        <span>after the closing ]: KL {sci(A.positions[1].kl)}</span>
        <span>
          L_pos = <b>{sci(tot.pos)}</b>
        </span>
      </div>
      <p className="panel-body">
        Reply start is what the model would write before the forced prefill “{a.replyText} [”. After ] is what follows the
        answer in free generation. Keeping both at p<sub>0</sub> keeps the chat behaviour of the base model.
      </p>
    </>
  );
}

/** Root node of p0: the full vocabulary split into legal and illegal mass, then zoomed. */
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
      <div className="pieces">
        <span className="piece k-q">q → L_tree</span>
        <span className="piece k-m">m → L_mass</span>
        <span className="piece k-o">p̄ → L_out</span>
        <span className="piece k-p">positions → L_pos</span>
      </div>
    </div>
  );
}

/** Illegal mass 1 − m per node on a log axis: p0 hollow, p_theta filled. */
function MassPlot({ A }) {
  const lo = -7, hi = -1, W = 460, X0 = 92, X1 = 400;
  const x = (v) => X0 + ((Math.log10(Math.max(v, 1e-7)) - lo) / (hi - lo)) * (X1 - X0);
  const ROW = 38;
  const H = 40 + A.nodes.length * ROW;
  return (
    <svg className="massplot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="illegal mass per node">
      {[-7, -5, -3, -1].map((e) => (
        <g key={e}>
          <line x1={x(10 ** e)} x2={x(10 ** e)} y1={18} y2={H - 8} className="grid" />
          <text x={x(10 ** e)} y={12} textAnchor="middle" className="tick">1e{e}</text>
        </g>
      ))}
      <text x={W - 6} y={12} textAnchor="end" className="tick">KL</text>
      {A.nodes.map((n, i) => {
        const y = 36 + i * ROW;
        return (
          <g key={i}>
            <text x={8} y={y + 4} className={n.kind === "gold" ? "rowlab gold" : "rowlab"}>{nodeName(n.path)}</text>
            <text x={8} y={y + 16} className="rowsub">{n.kind === "gold" ? "gold path" : "off path"}</text>
            <motion.line x1={x(n.r0)} y1={y} y2={y} className="mline" initial={{ x2: x(n.r0) }} animate={{ x2: x(n.r) }} transition={{ duration: 0.8, delay: 0.2 + i * 0.08 }} />
            <circle cx={x(n.r0)} cy={y} r={5} className="dot0" />
            <motion.circle cy={y} r={5} className="dot1" initial={{ cx: x(n.r0) }} animate={{ cx: x(n.r) }} transition={{ duration: 0.8, delay: 0.2 + i * 0.08 }} />
            <text x={W - 6} y={y + 4} textAnchor="end" className="kl">{sci(n.mass)}</text>
          </g>
        );
      })}
      <g transform={`translate(${X0}, ${H - 2})`}>
        <circle cx={0} cy={-4} r={4} className="dot0" />
        <text x={8} y={0} className="tick">p₀</text>
        <circle cx={40} cy={-4} r={4} className="dot1" />
        <text x={48} y={0} className="tick">p_θ (illegal mass 1 − m)</text>
      </g>
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

function NodeChips({ A, node, setNode }) {
  return (
    <div className="node-chips">
      {A.nodes.map((n, i) => (
        <button key={i} className={["nchip", n.kind, i === node ? "on" : ""].join(" ")} onClick={() => setNode(i)}>
          {nodeName(n.path)}
        </button>
      ))}
    </div>
  );
}

function OutView({ A, node, setNode }) {
  const n = A.nodes[node];
  return (
    <div>
      <span className="viz-label">pick a node</span>
      <NodeChips A={A} node={node} setNode={setNode} />
      <span className="viz-label">
        p̄ at node {nodeName(n.path)}: <i className="sw p0" /> p₀ <i className="sw p1" /> p_θ · KL {sci(n.out)}
      </span>
      <PairBars rows={n.illegal} rest={n.illegalRest} />
    </div>
  );
}

function NodeMap({ A }) {
  return (
    <div className="nodemap">
      {["gold", "off"].map((kind) => (
        <div key={kind} className="nm-group">
          <span className="viz-label">{kind === "gold" ? "gold path (always)" : `${a.offPath} sampled off-path nodes`}</span>
          {A.nodes
            .filter((n) => n.kind === kind)
            .map((n, i) => (
              <motion.div key={i} className={`nm-row ${kind}`} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.08 + (kind === "off" ? 0.3 : 0) }}>
                <span className="nm-name">{nodeName(n.path)}</span>
                <span className="nm-legal">
                  legal {n.legal.length > 6 ? `${txt(n.legal[0])} … ${txt(n.legal[n.legal.length - 1])}` : n.legal.map(txt).join(" ")}
                </span>
                <span className="nm-w">w {n.weight.toFixed(3)}</span>
                <span className="nm-v">mass {sci(n.mass)} · out {sci(n.out)}</span>
              </motion.div>
            ))}
        </div>
      ))}
    </div>
  );
}

function PosView({ A }) {
  return (
    <div className="posview">
      {A.positions.map((p, i) => (
        <div key={i}>
          <span className="viz-label">
            {i === 0 ? "reply start (before the prefill)" : "after the closing ]"}: <i className="sw p0" /> p₀ <i className="sw p1" /> p_θ · KL{" "}
            {sci(p.kl)}
          </span>
          <PairBars rows={p.top.slice(0, 5)} rest={p.rest} />
        </div>
      ))}
    </div>
  );
}

/** The three anchors in plain words, each with a before/after from this row (λ = 0.01). */
function Glossary() {
  const A = a.adapters.lam001;
  const B = a.adapters.lam1;
  // the node with the largest illegal mass after training, the root's biggest illegal shift, the post-] shift at λ = 1
  const mNode = A.nodes.reduce((x, y) => (y.r > x.r ? y : x));
  const root = A.nodes[0];
  const moved = [...root.illegal].sort((x, y) => Math.abs(y[2] - y[1]) - Math.abs(x[2] - x[1])).slice(0, 2);
  const after = B.positions[1].top.slice(0, 2);
  const cards = [
    {
      cls: "k-m",
      name: "L_mass",
      plain: "Keeps the share of probability on valid answer tokens where the base model had it.",
      at: `At each trie node only a few tokens continue a valid answer (the legal tokens). Their total probability is the legal mass m.`,
      example: (
        <>
          At node <code>{nodeName(mNode.path)}</code> the base model puts {small(mNode.r0)} on invalid tokens; after training with
          λ = 0.01 it puts <b>{small(mNode.r)}</b> there, {Math.round(mNode.r / mNode.r0)}× more.
        </>
      ),
      without: "The model drifts toward writing something other than a digit or ] mid-answer, so free-text answers break format more often.",
      math: "KL( (m₀, 1−m₀) ‖ (m, 1−m) ): a two-outcome KL, valid vs invalid",
    },
    {
      cls: "k-o",
      name: "L_out",
      plain: "Keeps the ranking of the invalid tokens the same as in the base model.",
      at: "Look only inside the invalid share and renormalise it to 100%: which wrong tokens would the model write if it left the format?",
      example: (
        <>
          At the root, among invalid tokens: {moved.map(([t0, p0, p1], i) => (
            <span key={i}>
              {i > 0 && ", "}
              <code>{show(t0)}</code> {pct(p0)} → <b>{pct(p1)}</b>
            </span>
          ))}
          .
        </>
      ),
      without: "These are ordinary language tokens the model uses everywhere else; reshuffling them is a side effect that leaks into normal generation.",
      math: "KL( p̄₀ ‖ p̄_θ ) over p₀'s top 64 invalid tokens + one bucket",
    },
    {
      cls: "k-p",
      name: "L_pos",
      plain: "Keeps two positions outside the answer exactly as the base model had them.",
      at: "The start of the reply (what the model would write before the forced “Best answer: [”) and the position right after the closing ].",
      example: (
        <>
          After <code>[27]</code>, the base model continues with {after.map(([t0, p0, p1], i) => (
            <span key={i}>
              {i > 0 && " / "}
              <code>{show(t0)}</code> {pct(p0)}
            </span>
          ))}
          ; the λ = 1 model moved to {after.map(([t0, , p1], i) => (
            <span key={i}>
              {i > 0 && " / "}
              <code>{show(t0)}</code> <b>{pct(p1)}</b>
            </span>
          ))}
          .
        </>
      ),
      without: "L_tree never trains these positions, but LoRA changes shared weights, so they drift anyway: the chat behaviour around the answer changes.",
      math: "KL( p₀ ‖ p_θ ) over the full vocabulary (top 64 + bucket), 0.5 per position",
    },
  ];
  return (
    <div className="gloss">
      <p className="gloss-intro">
        All three compare the trained model p<sub>θ</sub> with <b>p₀, the same model with the LoRA adapter switched off</b>{" "}
        (the model before training). Each is a KL divergence: 0 when nothing changed, growing as the distribution moves.
        L_tree decides the answer; these three stop training from changing anything else.
      </p>
      <div className="gloss-grid">
        {cards.map((c, i) => (
          <motion.article
            key={c.name}
            className={`gloss-card ${c.cls}`}
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.1 }}
          >
            <h4>{c.name}</h4>
            <p className="g-plain">{c.plain}</p>
            <p className="g-at">{c.at}</p>
            <p className="g-ex">
              <span className="g-tag">this row</span> {c.example}
            </p>
            <p className="g-without">
              <span className="g-tag">without it</span> {c.without}
            </p>
            <p className="g-math">{c.math}</p>
          </motion.article>
        ))}
      </div>
    </div>
  );
}

import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Stepper, Toggle } from "../components/Controls.jsx";
import d from "../../data/trie_loss.json";
import { Prompt } from "../components/Prompt.jsx";

const key = (path) => path.join(",");
const txt = (id) => d.tokText[String(id)];
const fmt = (x, n = 3) => (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(1) : x.toFixed(n));
const pct = (p) => (p >= 0.995 ? (p * 100).toFixed(1) : p < 0.001 ? "<0.1" : (p * 100).toFixed(1)) + "%";
const legalOf = Object.fromEntries(d.nodes.map((n) => [key(n.path), n.legal]));
const GOLD = d.row.gold;
const goldSuffix = d.suffixes[GOLD];
const goldPath = goldSuffix.map((_, t) => goldSuffix.slice(0, t));

/** Softmax over a node's legal logits: q(token | x, prefix) for every legal token. */
function legalProbs(model, path) {
  const z = d.models[model].nodes[key(path)].z;
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return { z, q: e.map((v) => v / s), lse: m + Math.log(s) };
}

const STEPS = [
  { title: "Options become suffixes", body: `The prompt lists ${d.row.k} options; the gold one is [${GOLD + 1}] ${d.row.options[GOLD]}. Option k is answered by writing the suffix k] after the prefill. Qwen splits numbers into one token per digit, and ] is its own token.` },
  { title: "Insert 1] … 9]", body: "Every suffix is a path from the root. The one-digit ids hang directly off the root." },
  { title: "Insert 10] … 19]", body: "The node 1 now has eleven children: ] closes 1], the digits continue to 10]–19]." },
  { title: "Insert 20] … 27]", body: `${d.row.k} suffixes, ${d.nodes.length} internal nodes. Because every suffix ends with ], no suffix is a prefix of another, so every leaf is a complete answer.` },
  { title: "Legal tokens", body: `At each node only some next tokens stay inside the trie. Those are the legal tokens; everything else in the ${d.vocab.toLocaleString("en-US")}-token vocabulary is illegal.` },
  { title: "The gold path", body: "Training feeds the prompt plus the gold suffix (teacher forcing). The logits at each of its positions predict the next suffix token." },
  ...goldSuffix.map((id, t) => ({
    title: t === 0 ? `Root: predict the first token` : `After ${goldSuffix.slice(0, t).map(txt).join("")}: predict ${txt(id) === "]" ? "the closing ]" : "the next digit"}`,
    body: "",
  })),
  { title: "Sum: L_tree", body: "" },
];
// the row's prompt, in the shape components/Prompt.jsx draws
const PROMPT_EX = { segs: d.row.segs, k: d.row.k, gold: GOLD, suffix: goldSuffix.map(txt) };
const NODE_STEP = 6; // steps 6..8 are the gold-path nodes

export default function TrieLoss() {
  const [step, setStep] = useState(0);
  const [model, setModel] = useState("base");
  return (
    <section className="trie-section" id="trie">
      <p className="eyebrow">2 · The answer trie and L_tree</p>
      <h2>How the tree loss is computed, one node at a time</h2>
      <p className="lede small">
        A real Bitext row from the eval split: {d.row.k} options, gold <b>[{GOLD + 1}] {d.row.options[GOLD]}</b>. Logits
        from {d.model}, before and after training.
      </p>

      <Stepper steps={STEPS} step={step} setStep={setStep} sectionId="trie">
        {step >= 5 && (
          <Toggle id="model" label="Model" options={[["base", "Base"], ["trained", "Trained"]]} value={model} onChange={setModel} />
        )}
      </Stepper>

      <div className="trie-grid">
        <AnimatePresence mode="wait">
          {step === 0 ? (
            <motion.div key="prompt" className="trie-prompt" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, scale: 0.98 }}>
              <span className="trie-legend">the prompt, as the model sees it; the answer is written after [</span>
              <Prompt ex={PROMPT_EX} stagger />
            </motion.div>
          ) : (
            <motion.div key="trie" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ minWidth: 0 }}>
              <Trie step={step} model={model} />
            </motion.div>
          )}
        </AnimatePresence>
        <div className="panel">
          <AnimatePresence mode="wait">
            <motion.div
              key={`${step}-${step >= NODE_STEP ? model : ""}`}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.22 }}
            >
              <h3 className="panel-title">{STEPS[step].title}</h3>
              {STEPS[step].body && <p className="panel-body">{STEPS[step].body}</p>}
              <Panel step={step} model={model} />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

function Panel({ step, model }) {
  if (step === 0) return <SuffixTable />;
  if (step >= 1 && step <= 3) return <InsertNote step={step} />;
  if (step === 4) return <LegalSets />;
  if (step === 5) return <TeacherForcing />;
  if (step >= NODE_STEP && step < NODE_STEP + goldSuffix.length) return <NodeLoss t={step - NODE_STEP} model={model} />;
  return <Total model={model} />;
}

function Tok({ id, gold = false }) {
  return <span className={gold ? "tchip gold" : "tchip"}>{txt(id)}</span>;
}

function SuffixTable() {
  return (
    <div className="suffix-table">
      {d.row.options.map((o, k) => (
        <motion.div
          key={k}
          className={k === GOLD ? "srow gold" : "srow"}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: k * 0.02 }}
        >
          <span className="sopt">
            [{k + 1}] {o}
          </span>
          <span className="stoks">
            {d.suffixes[k].map((id, i) => (
              <Tok key={i} id={id} gold={k === GOLD} />
            ))}
          </span>
        </motion.div>
      ))}
    </div>
  );
}

function InsertNote({ step }) {
  const ranges = [null, [0, 9], [9, 19], [19, 27]];
  const [a, b] = ranges[step];
  return (
    <div className="chips-wrap">
      {d.suffixes.slice(a, b).map((s, i) => (
        <motion.span key={a + i} className="suffix-chip" initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: i * 0.05 }}>
          {s.map(txt).join("")}
        </motion.span>
      ))}
    </div>
  );
}

function LegalSets() {
  const groups = [
    { node: "root", path: [], what: "the first digit" },
    { node: "1", path: [d.suffixes[0][0]], what: "] closes 1], or a second digit" },
    { node: "2", path: [d.suffixes[1][0]], what: "] closes 2], or 0–7 (up to 27)" },
    { node: "3 … 9, 10 … 27", path: [d.suffixes[2][0]], what: "only ]" },
  ];
  const depth = Math.floor(Math.log10(d.row.k)) + 2;
  return (
    <div className="legal-sets">
      {groups.map((g) => (
        <div key={g.node} className="legal-row">
          <span className="legal-node">{g.node}</span>
          <span className="stoks">
            {legalOf[key(g.path)].map((id) => (
              <Tok key={id} id={id} />
            ))}
          </span>
          <span className="legal-what">{g.what}</span>
        </div>
      ))}
      <p className="formula">
        depth = ⌊log<sub>10</sub> {d.row.k}⌋ + 2 = <b>{depth}</b> tokens at most
      </p>
    </div>
  );
}

function TeacherForcing() {
  return (
    <div className="tf">
      <pre className="tf-line">
        <span className="dimtext">… </span>
        <span className="pre-c">Best answer: [</span>
        {goldSuffix.map((id, t) => (
          <motion.span key={t} className="tf-tok" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 + t * 0.25 }}>
            {txt(id)}
          </motion.span>
        ))}
      </pre>
      <div className="tf-arrows">
        {goldSuffix.map((id, t) => (
          <div key={t} className="tf-row">
            <span className="tf-pos">logits after {t === 0 ? "[" : goldSuffix.slice(0, t).map(txt).join("")}</span>
            <span className="tf-arrow">→ predict</span>
            <Tok id={id} gold />
            <span className="tf-n">{legalOf[key(goldPath[t])].length} legal</span>
          </div>
        ))}
      </div>
      <p className="panel-body">One forward pass gives all three rows of logits; each node adds one term to L_tree.</p>
    </div>
  );
}

function NodeLoss({ t, model }) {
  const path = goldPath[t];
  const legal = legalOf[key(path)];
  const gold = goldSuffix[t];
  const { z, q, lse } = legalProbs(model, path);
  const gi = legal.indexOf(gold);
  const node = d.models[model].nodes[key(path)];
  const term = lse - z[gi];
  const done = d.models[model].terms.slice(0, t).reduce((a, b) => a + b, 0);
  return (
    <div className="nodeloss">
      <p className="panel-body">
        Node <code>{t === 0 ? "root" : path.map(txt).join("")}</code>: {legal.length} legal token{legal.length > 1 ? "s" : ""}.
        Softmax over their logits only, then −log of the gold one. x is the prompt; the condition is x plus
        the suffix tokens already written{path.length ? ` (${path.map(txt).join(" ")})` : " (none yet)"}.
      </p>
      <div className="bars">
        {legal.map((id, i) => (
          <div key={id} className={i === gi ? "bar-row gold" : "bar-row"}>
            <span className="bar-tok">{txt(id)}</span>
            <span className="bar-track">
              <motion.span
                className="bar-fill"
                initial={{ width: 0 }}
                animate={{ width: `${Math.max(q[i] * 100, 0.6)}%` }}
                transition={{ duration: 0.6, delay: i * 0.03, ease: [0.2, 0.8, 0.2, 1] }}
              />
            </span>
            <span className="bar-q">{pct(q[i])}</span>
            <span className="bar-z">z {z[i].toFixed(2)}</span>
          </div>
        ))}
      </div>
      <div className="eq">
        <span>
          ℓ<sub>{t + 1}</sub> = logsumexp(z<sub>legal</sub>) − z<sub>gold</sub>
        </span>
        <span>
          = {lse.toFixed(3)} − {z[gi].toFixed(3)} = <b>{fmt(term)}</b>
        </span>
        <span className="eq-note">
          = −log q({txt(gold)} | x{path.length ? `, ${path.map(txt).join(" ")}` : ""}) · running total {fmt(done)} + {fmt(term)} = <b>{fmt(done + term)}</b>
        </span>
      </div>
      {legal.length === 1 && <p className="callout">One legal token: q = 1, so this node adds exactly 0.</p>}
      <div className="vocab">
        <span className="vocab-label">Whole vocabulary at this position (legal mass {pct(node.legalMass)})</span>
        <span className="stoks">
          {node.top.map(([s, p], i) => (
            <span key={i} className={legal.some((id) => txt(id) === s) ? "tchip" : "tchip illegal"}>
              {JSON.stringify(s).slice(1, -1)} <small>{pct(p)}</small>
            </span>
          ))}
        </span>
        <span className="vocab-note">The full-vocabulary normaliser cancels: only the legal logits enter ℓ.</span>
      </div>
    </div>
  );
}

function Total({ model }) {
  const M = d.models[model];
  const argmax = M.q.indexOf(Math.max(...M.q));
  const W = 420, H = 150, pad = 18;
  const bw = (W - pad) / d.row.k;
  return (
    <div className="total">
      <div className="eq">
        <span>
          L<sub>tree</sub> = {M.terms.map((x, i) => (
            <span key={i}>
              {i > 0 && " + "}
              {fmt(x)}
            </span>
          ))}{" "}
          = <b>{M.loss.toFixed(3)}</b>
        </span>
        <span className="eq-note">
          q(gold) = e<sup>−L</sup> = <b>{pct(M.q[GOLD])}</b> · the q over all {d.row.k} answers sums to 1
        </span>
      </div>
      <svg className="qchart" viewBox={`0 0 ${W} ${H + 22}`} role="img" aria-label="q over all answers">
        <line x1={pad} x2={W} y1={H} y2={H} className="axis" />
        {[0.5, 1].map((v) => (
          <g key={v}>
            <line x1={pad} x2={W} y1={H - v * (H - 10)} y2={H - v * (H - 10)} className="grid" />
            <text x={0} y={H - v * (H - 10) + 4} className="tick">
              {v}
            </text>
          </g>
        ))}
        {M.q.map((p, k) => (
          <motion.rect
            key={`${model}-${k}`}
            x={pad + k * bw + 1}
            width={bw - 2}
            initial={{ y: H, height: 0 }}
            animate={{ y: H - p * (H - 10), height: p * (H - 10) }}
            transition={{ duration: 0.6, delay: k * 0.015 }}
            className={k === GOLD ? "qbar gold" : k === argmax ? "qbar top" : "qbar"}
          />
        ))}
        {M.q.map((_, k) =>
          (k + 1) % 5 === 0 || k === 0 || k === GOLD ? (
            <text key={k} x={pad + k * bw + bw / 2} y={H + 15} textAnchor="middle" className={k === GOLD ? "tick gold" : "tick"}>
              {k + 1}
            </text>
          ) : null,
        )}
      </svg>
      <p className="panel-body">
        Top answer: <b>[{argmax + 1}] {d.row.options[argmax]}</b> ({pct(M.q[argmax])}). Toggle Base / Trained to
        see training move mass onto the gold leaf.
      </p>
    </div>
  );
}

/** The trie, laid out left to right with one row per leaf. */
function Trie({ step, model }) {
  const layout = useMemo(buildLayout, []);
  const shown = step === 0 ? 0 : step === 1 ? 9 : step === 2 ? 19 : d.row.k;
  const fresh = step >= 1 && step <= 3 ? [[0, 9], [9, 19], [19, 27]][step - 1] : null;
  const pathKeys = new Set(goldPath.map(key).concat([key(goldSuffix)]));
  const showGold = step >= 5;
  const current = step >= NODE_STEP && step < NODE_STEP + goldSuffix.length ? key(goldPath[step - NODE_STEP]) : null;
  const flow = step >= NODE_STEP;
  // probability of reaching a node: the product of q(token | x, prefix) along its path
  const reach = (path) =>
    path.reduce((acc, tok, t) => {
      const parent = path.slice(0, t);
      return acc * legalProbs(model, parent).q[legalOf[key(parent)].indexOf(tok)];
    }, 1);
  const { nodes, H } = layout;
  const visible = (n) => n.key === "" || n.minLeaf < shown;
  const X0 = 26, DX = 88, LABEL = X0 + 3 * DX + 24;

  return (
    <div className="trie-wrap">
      <svg className="trie" viewBox={`0 0 560 ${H}`} role="img" aria-label="answer trie">
        {nodes.filter((n) => n.parent && visible(n)).map((n) => {
          const p = n.parent;
          const x1 = X0 + p.depth * DX, y1 = p.y, x2 = X0 + n.depth * DX, y2 = n.y;
          const onGold = showGold && pathKeys.has(n.key);
          const w = flow ? 0.8 + 8 * reach(n.path) : 1.4;
          return (
            <motion.path
              key={`e${n.key}`}
              d={`M${x1},${y1} C${x1 + DX * 0.55},${y1} ${x2 - DX * 0.55},${y2} ${x2},${y2}`}
              className={onGold ? "edge gold" : "edge"}
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1, strokeWidth: w }}
              transition={{ duration: 0.5, delay: fresh && n.minLeaf >= fresh[0] ? (n.minLeaf - fresh[0]) * 0.04 : 0 }}
            />
          );
        })}
        {nodes.filter(visible).map((n) => {
          const x = X0 + n.depth * DX;
          const onGold = showGold && pathKeys.has(n.key);
          const isNew = fresh && n.leaf !== null && n.leaf >= fresh[0] && n.leaf < fresh[1];
          const cls = ["tnode", n.leaf !== null ? "leaf" : "", onGold ? "gold" : "", current === n.key ? "current" : "",
            step === 4 && n.leaf === null ? "legal" : ""].join(" ");
          return (
            <motion.g
              key={n.key || "root"}
              className={cls}
              initial={{ opacity: 0, scale: 0.4 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ type: "spring", stiffness: 380, damping: 24, delay: isNew ? (n.leaf - fresh[0]) * 0.04 : 0 }}
              style={{ transformOrigin: `${x}px ${n.y}px` }}
            >
              {current === n.key && (
                <motion.circle cx={x} cy={n.y} className="ring" initial={{ r: 13 }} animate={{ r: [13, 18, 13] }} transition={{ repeat: Infinity, duration: 1.6 }} />
              )}
              <circle cx={x} cy={n.y} r={n.key === "" ? 11 : 9} />
              <text x={x} y={n.y + 4} textAnchor="middle">
                {n.key === "" ? "[" : txt(n.tok)}
              </text>
              {n.leaf !== null && (
                <text x={LABEL} y={n.y + 4} className={onGold ? "lab gold" : "lab"}>
                  [{n.leaf + 1}] {d.row.options[n.leaf]}
                  {flow ? `  ${pct(d.models[model].q[n.leaf])}` : ""}
                </text>
              )}
              {n.leaf !== null && <line x1={x + 10} x2={LABEL - 6} y1={n.y} y2={n.y} className="leader" />}
            </motion.g>
          );
        })}
      </svg>
      {step >= 4 && (
        <div className="trie-legend">
          {flow ? "edge width = probability of reaching that node" : step === 4 ? "filled dots: internal nodes with legal tokens" : "green: gold path"}
        </div>
      )}
    </div>
  );
}

/** Nodes (internal and leaves) with depth, y position and first leaf index. */
export function buildLayout() {
  const leafOf = Object.fromEntries(d.suffixes.map((s, i) => [key(s), i]));
  const ROW = 21, TOP = 16;
  const nodes = [];
  let row = 0;
  const order = (a, b) => (txt(a) === "]" ? -1 : txt(b) === "]" ? 1 : Number(txt(a)) - Number(txt(b)));
  function visit(path, parent) {
    const k = key(path);
    const n = { key: k, path, depth: path.length, parent, tok: path[path.length - 1], leaf: null };
    nodes.push(n);
    if (k in leafOf) {
      n.leaf = leafOf[k];
      n.y = TOP + row++ * ROW;
      n.minLeaf = n.leaf;
      return n;
    }
    const kids = [...legalOf[k]].sort(order).map((t) => visit([...path, t], n));
    n.y = (kids[0].y + kids[kids.length - 1].y) / 2;
    n.minLeaf = Math.min(...kids.map((c) => c.minLeaf));
    return n;
  }
  visit([], null);
  return { nodes, H: TOP * 2 + (row - 1) * ROW };
}

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import c from "../../data/contextcite.json";
import { Stepper } from "../components/Controls.jsx";

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const pct = (p) => `${(p * 100).toFixed(1)}%`;
const D = c.paragraphs.length;
const order = [...c.w.keys()].sort((i, j) => c.w[j] - c.w[i]);
const zeros = c.w.filter((v) => v === 0).length;
const goldIdx = c.paragraphs.flatMap((p, i) => (p.gold ? [i] : []));

const STEPS = [
  { title: "The question and its sources" },
  { title: "Ablate and re-score" },
  { title: "Fit a sparse linear surrogate" },
  { title: "Does the surrogate hold up?" },
  { title: "From attributions to losses" },
];

export default function ContextCite() {
  const [step, setStep] = useState(0);
  const [sel, setSel] = useState(goldIdx[0]);
  return (
    <section className="trie-section" id="contextcite">
      <p className="eyebrow">5 · ContextCite with the Jev scorer</p>
      <h2>Which paragraphs does the answer depend on?</h2>
      <p className="lede small">
        ContextCite (Cohen-Wang et al., 2024) removes random subsets of the context, re-scores the answer, and fits a sparse
        linear model. With Jev the answer is a 2–3 token suffix, so every ablation is one cheap trie scoring. A MuSiQue eval
        row, {c.model} with the λ = 0.01 adapter.
      </p>
      <Stepper steps={STEPS} step={step} setStep={setStep} sectionId="contextcite" />
      <div className="trie-grid">
        <div className="viz-card">
          <AnimatePresence mode="wait">
            <motion.div key={step} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
              {step === 0 && <Sources sel={sel} setSel={setSel} />}
              {step === 1 && <Ablations />}
              {step === 2 && <Weights />}
              {step === 3 && <Check />}
              {step === 4 && <Losses />}
            </motion.div>
          </AnimatePresence>
        </div>
        <div className="panel">
          <AnimatePresence mode="wait">
            <motion.div key={step} initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }} transition={{ duration: 0.22 }}>
              <h3 className="panel-title">{STEPS[step].title}</h3>
              <Text step={step} sel={sel} />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

function Text({ step, sel }) {
  if (step === 0)
    return (
      <>
        <p className="cc-q">{c.row.question}</p>
        <p className="panel-body">
          Gold: <b>{c.row.options[c.row.label]}</b> With all {D} paragraphs the model gives it q = <b className="num">{pct(sigmoid(c.full))}</b>.
        </p>
        <p className="panel-body">
          Each paragraph is one <i>source</i>. The dataset marks {goldIdx.length} as supporting (green); ContextCite never sees
          those marks. Click a paragraph to read it.
        </p>
        <div className="cc-read">
          <b>
            {sel + 1}. {c.paragraphs[sel].title}
          </b>
          <p>{c.paragraphs[sel].text}</p>
        </div>
      </>
    );
  if (step === 1)
    return (
      <>
        <p className="panel-body">
          Draw a mask v ∈ {"{0,1}"}<sup>{D}</sup>, keeping each paragraph with probability ½. Drop the others from the prompt and
          score it with the trie:
        </p>
        <div className="eq">
          <span>y(v) = logit q(gold | x<sub>v</sub>) = log q − log(1 − q)</span>
          <span className="eq-note">q is the tree probability from answers.score_batch, the same q that L_tree trains</span>
        </div>
        <p className="panel-body">
          {c.train} masks to fit, {c.test} more held out. Every mask is one batched prompt pass plus two trie steps: no text is
          generated.
        </p>
      </>
    );
  if (step === 2)
    return (
      <>
        <p className="panel-body">ContextCite's fitting loss: least squares on the masks, with an L1 penalty that zeroes most weights.</p>
        <div className="eq">
          <span>
            min<sub>w,b</sub> &nbsp;1/2n Σ<sub>i</sub> (y(v<sub>i</sub>) − b − w·v<sub>i</sub>)² + α‖w‖<sub>1</sub>
          </span>
          <span className="eq-note">α = {c.alpha}; fitted by coordinate descent on the {c.train} masks</span>
        </div>
        <p className="panel-body">
          w<sub>j</sub> is how much paragraph j adds to the logit of the gold answer. {zeros} of {D} weights are exactly zero. The
          two largest are paragraphs <b>{order[0] + 1}</b> and <b>{order[1] + 1}</b>, the two the dataset marks as supporting.
        </p>
      </>
    );
  if (step === 3)
    return (
      <>
        <p className="panel-body">
          On the {c.test} held-out masks the surrogate's predictions rank the true logits with Spearman ρ ={" "}
          <b className="num">{c.lds.toFixed(2)}</b> (ContextCite's linear datamodeling score, LDS).
        </p>
        <p className="panel-body">
          A high LDS means the linear model is a faithful summary of how this answer reacts to the context, so its weights can
          be trusted as attributions.
        </p>
      </>
    );
  return (
    <>
      <p className="panel-body">Two SelfCite-style scores for a set of paragraphs S, both in nats of log q(gold):</p>
      <div className="eq">
        <span>necessity(S) = log q(gold | x) − log q(gold | x without S)</span>
        <span>sufficiency(S) = log q(gold | S only) − log q(gold | x)</span>
      </div>
      <p className="panel-body">
        Here, keeping only the top-2 paragraphs (the supporting pair) keeps the answer (+{c.evals["top-2"].sufficiency.toFixed(2)}),
        while 2 random ones lose {(-c.evals.random.sufficiency).toFixed(2)} nats. Removing the pair costs only{" "}
        {c.evals["top-2"].necessity.toFixed(2)} nats: the 0.6B model still leans on a prior toward “yes”.
      </p>
      <p className="panel-body">How they could become training losses in this recipe (not implemented yet):</p>
      <ul className="cc-uses">
        <li>
          <b>Distillation:</b> ContextCite weights as targets for an evidence head on the paragraph states, e.g. a
          cross-entropy toward the top-weighted set.
        </li>
        <li>
          <b>Reward:</b> score the head's predicted set with necessity + sufficiency − a size penalty, and train it with
          preferences or policy gradient.
        </li>
        <li>
          <b>Cost:</b> one ablation = one prompt pass + ⌊log<sub>10</sub>K⌋+2 trie steps, so 64 ablations cost about 64
          forward passes.
        </li>
      </ul>
    </>
  );
}

function Sources({ sel, setSel }) {
  const maxT = Math.max(...c.paragraphs.map((p) => p.tokens));
  return (
    <div className="cc-grid">
      {c.paragraphs.map((p, i) => (
        <motion.button
          key={i}
          className={["cc-tile", p.gold ? "gold" : "", i === sel ? "on" : ""].join(" ")}
          onClick={() => setSel(i)}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: i * 0.025 }}
        >
          <span className="cc-n">{i + 1}</span>
          <span className="cc-title">{p.title}</span>
          <span className="cc-len" style={{ width: `${(p.tokens / maxT) * 100}%` }} />
        </motion.button>
      ))}
    </div>
  );
}

/** The training masks: one row per ablation, one column per paragraph, with its target. */
function Ablations() {
  const rows = c.masks.slice(0, c.train);
  const ys = c.y.slice(0, c.train);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  return (
    <div className="cc-abl">
      <div className="cc-head">
        {c.paragraphs.map((p, j) => (
          <span key={j} className={p.gold ? "gold" : ""}>{j + 1}</span>
        ))}
        <span className="ylab">y</span>
      </div>
      {rows.map((m, i) => (
        <motion.div key={i} className="cc-row" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.025 }}>
          {m.map((v, j) => (
            <span key={j} className={v ? (c.paragraphs[j].gold ? "cell on gold" : "cell on") : "cell"} />
          ))}
          <span className="ybar">
            <span style={{ width: `${((ys[i] - lo) / (hi - lo)) * 100}%` }} />
          </span>
        </motion.div>
      ))}
      <div className="viz-label">
        <i className="sw kept" /> kept <i className="sw dropped" /> dropped · y from {lo.toFixed(1)} to {hi.toFixed(1)}
      </div>
    </div>
  );
}

function Weights() {
  const max = Math.max(...c.w.map(Math.abs));
  return (
    <div className="cc-weights">
      {c.paragraphs.map((p, j) => (
        <div key={j} className={p.gold ? "cw-row gold" : "cw-row"}>
          <span className="cw-n">{j + 1}</span>
          <span className="cw-title">{p.title}</span>
          <span className="cw-track">
            <motion.span
              className={c.w[j] < 0 ? "cw-fill neg" : "cw-fill"}
              initial={{ width: 0 }}
              animate={{ width: `${(Math.abs(c.w[j]) / max) * 100}%` }}
              transition={{ duration: 0.7, delay: 0.1 + order.indexOf(j) * 0.04 }}
            />
          </span>
          <span className="cw-v">{c.w[j] === 0 ? "0" : c.w[j].toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function Check() {
  const act = c.y.slice(c.train);
  const all = [...act, ...c.pred];
  const lo = Math.min(...all) - 0.2, hi = Math.max(...all) + 0.2;
  const S = 300, P = 34;
  const s = (v) => P + ((v - lo) / (hi - lo)) * (S - P - 8);
  const sy = (v) => S - s(v) + P - 8;
  return (
    <div>
      <svg className="cc-scatter" viewBox={`0 0 ${S} ${S}`} role="img" aria-label="surrogate vs actual">
        <line x1={s(lo)} y1={sy(lo)} x2={s(hi)} y2={sy(hi)} className="diag" />
        {[Math.ceil(lo), Math.floor(hi)].map((v) => (
          <g key={v}>
            <text x={s(v)} y={S - 12} textAnchor="middle" className="tick">{v}</text>
            <text x={P - 6} y={sy(v) + 3} textAnchor="end" className="tick">{v}</text>
          </g>
        ))}
        <text x={S / 2 + 10} y={S - 1} textAnchor="middle" className="tick">surrogate b + w·v</text>
        <text x={10} y={S / 2} textAnchor="middle" className="tick" transform={`rotate(-90 10 ${S / 2})`}>actual y(v)</text>
        {act.map((y, i) => (
          <motion.circle key={i} cx={s(c.pred[i])} cy={sy(y)} r={4.5} className="pt" initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.03 }} />
        ))}
      </svg>
      <div className="viz-label">{c.test} held-out ablations · LDS ρ = {c.lds.toFixed(2)}</div>
    </div>
  );
}

function Losses() {
  const names = ["top-1", "top-2", "top-3", "gold", "random"];
  const vals = names.flatMap((n) => [c.evals[n].necessity, c.evals[n].sufficiency]);
  const m = Math.max(...vals.map(Math.abs));
  const label = (n) =>
    n === "random" ? "random 2 (mean of 8)" : n === "gold" ? "dataset's supporting" : `${n} by w`;
  return (
    <div className="cc-loss">
      {["necessity", "sufficiency"].map((kind) => (
        <div key={kind} className="cl-block">
          <span className="viz-label">{kind} (nats; higher is better)</span>
          {names.map((n, i) => {
            const v = c.evals[n][kind];
            return (
              <div key={n} className={`cl-row ${n === "random" ? "rand" : ""}`}>
                <span className="cl-name">
                  {label(n)}
                  {c.evals[n].set.length > 0 && <small> {c.evals[n].set.map((j) => j + 1).join(", ")}</small>}
                </span>
                <span className="cl-axis">
                  <motion.span
                    className={v < 0 ? "cl-bar neg" : "cl-bar"}
                    initial={{ width: 0 }}
                    animate={{ width: `${(Math.abs(v) / m) * 50}%` }}
                    transition={{ duration: 0.6, delay: i * 0.07 }}
                    style={v < 0 ? { right: "50%" } : { left: "50%" }}
                  />
                </span>
                <span className="cl-v">{v >= 0 ? "+" : ""}{v.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

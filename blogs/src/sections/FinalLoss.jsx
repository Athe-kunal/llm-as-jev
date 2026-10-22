import { useEffect, useState } from "react";
import { animate, motion } from "framer-motion";
import a from "../../data/anchors.json";
import { Toggle } from "../components/Controls.jsx";
import { ADAPTERS } from "./Anchors.jsx";

const LAMBDA = { lam1: 1, lam001: 0.01 };
const TERMS = [
  ["mass", "L_mass"],
  ["out", "L_out"],
  ["pos", "L_pos"],
];
const RESULTS = [
  ["dev", "Dev (in-domain)"],
  ["banking77", "Banking77"],
  ["jevbench", "JevBench"],
  ["boolq", "BoolQ"],
  ["external_macro", "External macro"],
];

/** A number that counts to its new value. */
function Num({ v, digits = 3 }) {
  const [x, setX] = useState(v);
  useEffect(() => {
    const c = animate(x, v, { duration: 0.6, ease: "easeOut", onUpdate: setX });
    return () => c.stop();
  }, [v]); // eslint-disable-line react-hooks/exhaustive-deps
  return <>{x.toFixed(digits)}</>;
}

export default function FinalLoss() {
  const [ad, setAd] = useState("lam1");
  const T = a.adapters[ad].total;
  const lam = LAMBDA[ad];
  const anchors = T.mass + T.out + T.pos;
  const total = T.tree + lam * anchors;

  return (
    <section className="trie-section" id="final">
      <p className="eyebrow">4 · The final loss</p>
      <h2>One number per example</h2>
      <p className="lede small">
        Tree loss on the gold path, plus λ times the three anchors. This row's values with each trained adapter.
      </p>
      <div className="final-bar">
        <Toggle id="final-lam" label="Adapter" options={ADAPTERS} value={ad} onChange={setAd} />
      </div>

      <div className="big-eq">
        <div className="term t-tree">
          <span className="sym">L_tree</span>
          <span className="val"><Num v={T.tree} /></span>
        </div>
        <span className="op">+</span>
        <div className="term t-lam">
          <span className="sym">λ</span>
          <span className="val">{lam}</span>
        </div>
        <span className="op">·</span>
        <span className="paren">(</span>
        {TERMS.map(([k, name], i) => (
          <div key={k} className="term-group">
            {i > 0 && <span className="op">+</span>}
            <div className={`term t-${k}`}>
              <span className="sym">{name}</span>
              <span className="val"><Num v={T[k]} digits={k === "mass" ? 5 : 3} /></span>
            </div>
          </div>
        ))}
        <span className="paren">)</span>
        <span className="op">=</span>
        <div className="term t-total">
          <span className="sym">L</span>
          <span className="val"><Num v={total} /></span>
        </div>
      </div>
      <motion.p key={ad} className="lede small center" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        The anchors add <b>{(lam * anchors).toFixed(4)}</b> to this row's loss
        {lam < 1 ? ": at λ = 0.01 they barely pull, so p_θ is nearly free to drift." : ": at λ = 1 every nat of drift costs as much as a nat of tree loss."}
      </motion.p>

      <h3 className="sub-h">Over training: tree loss the same, drift 4–10× apart</h3>
      <Curves />

      <h3 className="sub-h">What it bought: Qwen3-0.6B, base vs both adapters</h3>
      <Results />
    </section>
  );
}

/** Small multiples of the logged losses, both runs overlaid. */
function Curves() {
  const W = 260, H = 130, P = { l: 40, r: 8, t: 14, b: 22 };
  const series = ADAPTERS.map(([k, label]) => ({ k, label, rows: a.adapters[k].curve }));
  return (
    <div className="curves">
      {[["tree", 1], ["mass", 2], ["out", 3], ["pos", 4]].map(([name, col]) => {
        const all = series.flatMap((s) => s.rows.map((r) => r[col]));
        const max = Math.max(...all) * 1.05;
        const steps = series[0].rows[series[0].rows.length - 1][0];
        const x = (s) => P.l + (s / steps) * (W - P.l - P.r);
        const y = (v) => H - P.b - (v / max) * (H - P.t - P.b);
        const last = (s) => {
          const tail = s.rows.filter((r) => r[0] > steps - 75);
          return tail.reduce((acc, r) => acc + r[col], 0) / tail.length;
        };
        return (
          <figure key={name} className="curve">
            <figcaption>
              L_{name}
              <span>
                last 75 steps: {series.map((s) => `${s.label.replace("λ = ", "λ ")} ${last(s).toFixed(name === "mass" ? 4 : 3)}`).join(" · ")}
              </span>
            </figcaption>
            <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`L_${name} over training`}>
              {[0, 0.5, 1].map((f) => (
                <g key={f}>
                  <line x1={P.l} x2={W - P.r} y1={y(max * f)} y2={y(max * f)} className="grid" />
                  <text x={P.l - 4} y={y(max * f) + 3} textAnchor="end" className="tick">
                    {(max * f).toFixed(name === "mass" ? 3 : 2)}
                  </text>
                </g>
              ))}
              <text x={P.l} y={H - 6} className="tick">0</text>
              <text x={W - P.r} y={H - 6} textAnchor="end" className="tick">{steps} steps</text>
              {series.map((s) => (
                <motion.path
                  key={s.k}
                  className={`cline ${s.k}`}
                  d={s.rows.map((r, i) => `${i ? "L" : "M"}${x(r[0]).toFixed(1)},${y(r[col]).toFixed(1)}`).join(" ")}
                  initial={{ pathLength: 0 }}
                  whileInView={{ pathLength: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 1.4, ease: "easeInOut" }}
                />
              ))}
            </svg>
          </figure>
        );
      })}
      <div className="curve-key">
        <span><i className="sw lam1" /> λ = 1</span>
        <span><i className="sw lam001" /> λ = 0.01</span>
      </div>
    </div>
  );
}

function Results() {
  const rows = [["base", "Base", a.base], ...ADAPTERS.map(([k, l]) => [k, l, a.adapters[k].results])];
  const best = Object.fromEntries(RESULTS.map(([k]) => [k, Math.max(...rows.map((r) => r[2][k]))]));
  return (
    <div className="results-wrap">
      <table className="results">
        <thead>
          <tr>
            <th />
            {RESULTS.map(([k, l]) => (
              <th key={k}>{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, label, r], i) => (
            <motion.tr key={k} initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }}>
              <th>{label}</th>
              {RESULTS.map(([m]) => {
                const d = r[m] - a.base[m];
                return (
                  <td key={m} className={r[m] === best[m] ? "best" : ""}>
                    {(r[m] * 100).toFixed(1)}
                    {k !== "base" && <span className={d >= 0 ? "delta up" : "delta down"}>{d >= 0 ? "+" : ""}{(d * 100).toFixed(1)}</span>}
                  </td>
                );
              })}
            </motion.tr>
          ))}
        </tbody>
      </table>
      <p className="panel-body">
        A weak anchor (λ = 0.01) buys more in-domain accuracy and Banking77, and pays with BoolQ. Neither run moves JevBench
        or the External average beyond noise.
      </p>
    </div>
  );
}

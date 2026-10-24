import { useRef, useState } from "react";
import { motion } from "framer-motion";
import r from "../../data/results.json";

const MODELS = r.models.map((m, i) => ({ ...m, color: `var(--m${i + 1})` }));
const DATASETS = [
  ["anli", "ANLI"], ["bitext", "Bitext"], ["clinc150", "CLINC150"], ["contractnli", "ContractNLI"],
  ["cosmosqa", "CosmosQA"], ["logiqa2", "LogiQA 2.0"], ["massive", "MASSIVE"], ["musique", "MuSiQue"],
  ["piqa", "PIQA"], ["proofwriter", "ProofWriter"], ["quality", "QuALITY"], ["reclor", "ReClor"],
];
const BENCH = [
  ["arc_challenge", "ARC-Challenge"], ["banking77", "Banking77"], ["boolq", "BoolQ"], ["jevbench", "JevBench"],
  ["mmlu", "MMLU"], ["mmlu_pro", "MMLU-Pro"], ["sciq", "SciQ"], ["winogrande", "Winogrande"],
];
const CURVES = [
  ["tree", "L_tree"], ["mass", "L_mass"], ["out", "L_out"], ["pos", "L_pos"],
  ["grad_norm", "Gradient norm"], ["lr", "Learning rate"],
];

const pct = (v, d = 0) => (v * 100).toFixed(d);
const signed = (v, d = 0) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(d)}`;
const num = (v) => (v !== 0 && Math.abs(v) < 0.01 ? v.toExponential(1) : v.toFixed(3));

/** TensorBoard's smoothing: a debiased exponential moving average. */
function smooth(points, a) {
  let s = 0;
  return points.map(([x, y], i) => {
    s = a * s + (1 - a) * y;
    return [x, s / (1 - Math.pow(a, i + 1))];
  });
}

/** A tooltip that follows the pointer inside a `position: relative` box. */
export function useTip() {
  const box = useRef(null);
  const [tip, setTip] = useState(null);
  const show = (e, content) => {
    const b = box.current.getBoundingClientRect();
    setTip({ x: e.clientX - b.left, y: e.clientY - b.top, content, flip: e.clientX - b.left > b.width * 0.6 });
  };
  const view = tip && (
    <div className="rs-tip" style={{ left: tip.x, top: tip.y, transform: `translate(${tip.flip ? "calc(-100% - 14px)" : "14px"}, -50%)` }}>
      {tip.content}
    </div>
  );
  return { box, show, hide: () => setTip(null), view };
}

function Legend() {
  return (
    <div className="rs-legend">
      {MODELS.map((m) => (
        <span key={m.id}><i style={{ background: m.color }} />{m.label}</span>
      ))}
    </div>
  );
}

function Cards() {
  return (
    <div className="rs-cards">
      {MODELS.map((m, i) => {
        const b = m.base.ALL, f = m.final.ALL;
        return (
          <motion.div key={m.id} className="rs-card" style={{ "--mc": m.color }}
            initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.08 }}>
            <h3><i />{m.label}</h3>
            <p className="rs-big">{pct(b.accuracy, 1)} <span>→</span> {pct(f.accuracy, 1)}<small>{signed(f.accuracy - b.accuracy, 1)} pts</small></p>
            <dl>
              <dt>log p(gold)</dt><dd>{b.log_p_gold.toFixed(2)} → {f.log_p_gold.toFixed(2)}</dd>
              <dt>text-mode accuracy</dt><dd>{pct(b.gen_correct, 1)} → {pct(f.gen_correct, 1)}</dd>
              <dt>training</dt><dd>r{m.lora_rank} · λ {m.lam} · {m.steps} steps</dd>
            </dl>
          </motion.div>
        );
      })}
    </div>
  );
}

/** One model's base → trained accuracy on every dataset: a dumbbell per row. */
function Dumbbells({ model }) {
  const { box, show, hide, view } = useTip();
  const W = 400, L = 104, R = 14, row = 22, top = 8;
  const rows = [...DATASETS, ["ALL", "ALL"]];
  const H = top + rows.length * row + 14 + 22;
  const x = (v) => L + v * (W - L - R);
  const y = (i) => top + i * row + (i === rows.length - 1 ? 8 : 0) + row / 2;
  return (
    <figure className="rs-db" ref={box} onPointerLeave={hide}>
      <figcaption><i style={{ background: model.color }} />{model.label}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${model.label}: accuracy before and after training on each dataset`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top} y2={H - 22} className="rs-grid" />
            <text x={x(t)} y={H - 6} textAnchor="middle" className="rs-tick">{t * 100}</text>
          </g>
        ))}
        {rows.map(([k, label], i) => {
          const b = model.base[k].accuracy, f = model.final[k].accuracy;
          return (
            <g key={k} onPointerMove={(e) => show(e, (
              <>
                <b>{label}</b>
                <span>{pct(b, 1)} → <strong>{pct(f, 1)}</strong> ({signed(f - b, 1)} pts)</span>
              </>
            ))}>
              <rect x={0} y={y(i) - row / 2} width={W} height={row} className="rs-hit" />
              <text x={L - 10} y={y(i) + 4} textAnchor="end" className={k === "ALL" ? "rs-lab all" : "rs-lab"}>{label}</text>
              <line x1={x(b)} x2={x(f)} y1={y(i)} y2={y(i)} stroke={model.color} className="rs-link" />
              <circle cx={x(b)} cy={y(i)} r={4} className="rs-base" stroke={model.color} />
              <circle cx={x(f)} cy={y(i)} r={4.5} className="rs-final" fill={model.color} />
            </g>
          );
        })}
      </svg>
      {view}
    </figure>
  );
}

function Delta({ v, d = 0 }) {
  const tone = Math.min(Math.abs(v) / 0.4, 1) * 55;
  const bg = v === 0 ? "transparent" : `color-mix(in srgb, var(${v > 0 ? "--c6" : "--c5"}) ${tone}%, transparent)`;
  return <td className="rs-delta" style={{ background: bg }}>{signed(v, d)}</td>;
}

function Table({ rows, get, groupLabel, digits = 0 }) {
  return (
    <div className="results-wrap">
      <table className="rs-table">
        <thead>
          <tr>
            <th rowSpan={2}>{groupLabel}</th>
            {MODELS.map((m) => (
              <th key={m.id} colSpan={3} className="rs-grp"><i style={{ background: m.color }} />{m.label}</th>
            ))}
          </tr>
          <tr>{MODELS.map((m) => ["Base", "Trained", "Δ"].map((h) => <th key={m.id + h} className="rs-sub">{h}</th>))}</tr>
        </thead>
        <tbody>
          {rows.map(([k, label]) => (
            <tr key={k} className={k === "ALL" || k === "external_macro" ? "rs-total" : ""}>
              <th>{label}</th>
              {MODELS.map((m) => {
                const pair = get(m, k);
                if (!pair) return <td key={m.id} colSpan={3} className="rs-none">not run</td>;
                const [b, f] = pair;
                return [
                  <td key={m.id + "b"} className="rs-b">{pct(b, digits)}</td>,
                  <td key={m.id + "f"}>{pct(f, digits)}</td>,
                  <Delta key={m.id + "d"} v={f - b} d={digits} />,
                ];
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Small multiple: one metric over steps for every run, with a crosshair readout. */
export function LineChart({ title, series, xmax, a = 0, markers = false, format = num, note, wide = false }) {
  const { box, show, hide, view } = useTip();
  const W = wide ? 560 : 360, H = wide ? 230 : 190, P = { l: 52, r: 12, t: 12, b: 26 };
  const lines = series.map((s) => ({ ...s, shown: a > 0 ? smooth(s.points, a) : s.points }));
  const all = lines.flatMap((s) => [...s.shown.map((p) => p[1]), ...(a > 0 ? s.points.map((p) => p[1]) : [])]);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi === lo) { hi += 1; lo -= 1; }
  const pad = (hi - lo) * 0.06;
  lo = lo >= 0 && lo < hi * 0.5 ? 0 : lo - pad; // rates and losses start at zero
  hi += pad;
  const x = (v) => P.l + (v / xmax) * (W - P.l - P.r);
  const y = (v) => P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b);
  const path = (pts) => pts.map(([px, py], i) => `${i ? "L" : "M"}${x(px).toFixed(1)},${y(py).toFixed(1)}`).join(" ");
  const [cursor, setCursor] = useState(null);

  const onMove = (e) => {
    const svg = e.currentTarget.ownerSVGElement || e.currentTarget;
    const b = svg.getBoundingClientRect();
    const px = ((e.clientX - b.left) / b.width) * W;
    const want = Math.max(0, Math.min(xmax, ((px - P.l) / (W - P.l - P.r)) * xmax));
    const steps = [...new Set(lines.flatMap((s) => s.points.map((p) => p[0])))];
    const step = steps.reduce((best, s) => (Math.abs(s - want) < Math.abs(best - want) ? s : best), steps[0]);
    setCursor(step);
    show(e, (
      <>
        <b>step {step}</b>
        {lines.map((s) => {
          const i = s.points.findIndex((p) => p[0] === step);
          if (i < 0) return null;
          return (
            <span key={s.id} className="rs-row"><i style={{ background: s.color }} /><strong>{format(s.shown[i][1])}</strong>
              <em>{s.label}{a > 0 ? ` · raw ${format(s.points[i][1])}` : ""}</em></span>
          );
        })}
      </>
    ));
  };

  return (
    <figure className="rs-chart" ref={box} onPointerLeave={() => { hide(); setCursor(null); }}>
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} over training steps for each run`}>
        {[0, 0.5, 1].map((t) => {
          const v = lo + (hi - lo) * t;
          return (
            <g key={t}>
              <line x1={P.l} x2={W - P.r} y1={y(v)} y2={y(v)} className="rs-grid" />
              <text x={P.l - 6} y={y(v) + 3.5} textAnchor="end" className="rs-tick">{format(v)}</text>
            </g>
          );
        })}
        <text x={P.l} y={H - 6} className="rs-tick">0</text>
        <text x={W - P.r} y={H - 6} textAnchor="end" className="rs-tick">{xmax} steps</text>
        {lines.map((s) => (
          <g key={s.id}>
            {a > 0 && <path d={path(s.points)} fill="none" stroke={s.color} className="rs-raw" />}
            <path d={path(s.shown)} fill="none" stroke={s.color} className="rs-line" />
            {markers && s.shown.map(([px, py]) => <circle key={px} cx={x(px)} cy={y(py)} r={3.5} fill={s.color} className="rs-dot" />)}
          </g>
        ))}
        {cursor !== null && <line x1={x(cursor)} x2={x(cursor)} y1={P.t} y2={H - P.b} className="rs-cursor" />}
        <rect x={P.l} y={0} width={W - P.l - P.r} height={H - P.b} fill="transparent" onPointerMove={onMove} />
      </svg>
      {note && <p className="rs-note">{note}</p>}
      {view}
    </figure>
  );
}

export default function Results() {
  const [a, setA] = useState(0.8);
  const series = (get) => MODELS.map((m) => ({ id: m.id, label: m.label, color: m.color, points: get(m) }));
  return (
    <section className="trie-section" id="results">
      <p className="eyebrow">5 · Results</p>
      <h2>Three models, one recipe</h2>
      <p className="lede small">
        The same LoRA recipe (rank 16, λ = 1, batch 32, learning rate 1e-4) on the 12,000-row Reason+Long mix for one epoch,
        {" "}{MODELS[0].steps} steps, on Qwen3-0.6B, Qwen3-4B and Qwen3.5-4B. Each model is scored on 100 held-out rows per
        dataset, before and after training. These runs predate Nimble, so the mix has the original 12 datasets.
      </p>
      <Legend />
      <Cards />

      <h3 className="sub-h">Accuracy on every dataset, before and after</h3>
      <p className="lede small">Hollow dot: the base model. Filled dot: after training. ALL pools the 1,200 held-out rows.</p>
      <div className="rs-dbs">
        {MODELS.map((m) => <Dumbbells key={m.id} model={m} />)}
      </div>
      <Table rows={[...DATASETS, ["ALL", "ALL"]]} groupLabel="Accuracy, %" get={(m, k) => [m.base[k].accuracy, m.final[k].accuracy]} />
      <p className="rs-note">With 100 rows per dataset one standard error is about 5 points; for ALL (1,200 rows) it is about 1.4 points. Differences of a few points on a single dataset are within noise.</p>

      <h3 className="sub-h">What it did to the external benchmarks</h3>
      <Table rows={[...BENCH, ["external_macro", "External macro"]]} groupLabel="Accuracy, %" digits={1}
        get={(m, k) => (m.bench_base ? [m.bench_base[k].accuracy, m.bench_final[k].accuracy] : null)} />
      <p className="rs-note">Each benchmark is a seeded sample of the pinned download (BoolQ 500, MMLU 1,200, MMLU-Pro 800, ARC-Challenge 600, Winogrande 500, SciQ 400, Banking77 3,080, JevBench 231), so a one-point difference on a single benchmark is noise. External macro is the mean over the six external sets.</p>

      <h3 className="sub-h">Training curves from TensorBoard</h3>
      <div className="rs-ctl">
        <label htmlFor="rs-smooth">Smoothing</label>
        <input id="rs-smooth" type="range" min="0" max="0.98" step="0.02" value={a} onChange={(e) => setA(+e.target.value)} />
        <span>{a.toFixed(2)}</span>
        <Legend />
      </div>
      <div className="rs-grid-charts">
        {CURVES.map(([k, title]) => (
          <LineChart key={k} title={title} xmax={MODELS[0].steps} a={a} series={series((m) => m.curves.train[k])}
            note={k === "lr" ? "The schedule is the same for all runs: 100 warm-up steps, then linear decay to 10%." : undefined} />
        ))}
      </div>

      <h3 className="sub-h">Held-out accuracy and log p(gold) during training</h3>
      <div className="rs-grid-charts two">
        <LineChart wide title="ALL accuracy" xmax={MODELS[0].steps} markers series={series((m) => m.curves.eval.accuracy)}
          format={(v) => `${(v * 100).toFixed(1)}%`} />
        <LineChart wide title="ALL log p(gold)" xmax={MODELS[0].steps} markers series={series((m) => m.curves.eval.log_p_gold)}
          format={(v) => v.toFixed(2)}
          note="Qwen3-0.6B and Qwen3.5-4B were evaluated only before and after training; Qwen3-4B also every 75 steps." />
      </div>
    </section>
  );
}

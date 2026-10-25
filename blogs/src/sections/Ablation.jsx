import { useState } from "react";
import { motion } from "framer-motion";
import a from "../../data/ablation.json";
import { LineChart, useTip } from "./Results.jsx";

// Run colours follow the terms of section 3: L_mass cyan, L_out pink, L_pos purple.
const COLORS = {
  tree_only: "var(--c0)", mass_only: "var(--c1)", out_only: "var(--c5)", pos_only: "var(--c4)",
  no_mass: "var(--c1)", no_out: "var(--c5)", no_pos: "var(--c4)", full: "var(--fg)",
};
const RUNS = a.runs.map((r) => ({ ...r, color: COLORS[r.id], done: r.final !== null }));
const DONE = RUNS.filter((r) => r.done);
const BY = Object.fromEntries(RUNS.map((r) => [r.id, r]));
const TERMS = [["mass", "L_mass", "k-m"], ["out", "L_out", "k-o"], ["pos", "L_pos", "k-p"]];
const LABELS = {
  anli: "ANLI", bitext: "Bitext", clinc150: "CLINC150", contractnli: "ContractNLI", cosmosqa: "CosmosQA",
  logiqa2: "LogiQA 2.0", massive: "MASSIVE", musique: "MuSiQue", piqa: "PIQA", proofwriter: "ProofWriter",
  quality: "QuALITY", reclor: "ReClor", nimble: "Nimble",
};
const BENCH = [
  ["arc_challenge", "ARC-Challenge"], ["banking77", "Banking77"], ["boolq", "BoolQ"], ["jevbench", "JevBench"],
  ["mmlu", "MMLU"], ["mmlu_pro", "MMLU-Pro"], ["sciq", "SciQ"], ["winogrande", "Winogrande"],
  ["external_macro", "External macro"],
];

const pct = (v, d = 1) => (v * 100).toFixed(d);
const signed = (v, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(d)}`;
const datasets = (a.base ?? DONE[0]?.final ?? {});
const NAMES = Object.keys(datasets).filter((k) => k !== "ALL");
const N_EVAL = NAMES.length * a.eval_per;

function Legend({ ids }) {
  return (
    <div className="rs-legend">
      {ids.map((id) => (
        <span key={id}><i style={{ background: COLORS[id] }} />{BY[id].label}</span>
      ))}
    </div>
  );
}

function Status() {
  const running = RUNS.filter((r) => !r.done);
  if (!running.length) return null;
  return (
    <p className="ab-status">
      {DONE.length} of {RUNS.length} runs have finished. Still training or evaluating:{" "}
      {running.map((r) => `${r.label}${r.steps_done ? ` (step ${r.steps_done}/${a.total_steps})` : ""}`).join(", ")}.
    </p>
  );
}

/** Which terms each run optimises, next to its headline numbers. */
function Scoreboard() {
  const base = a.base?.ALL;
  const macro = (r) => r.bench_final?.external_macro?.accuracy;
  const baseMacro = a.bench_base?.external_macro?.accuracy;
  return (
    <div className="results-wrap">
      <table className="rs-table ab-score">
        <thead>
          <tr>
            <th rowSpan={2}>Run</th>
            <th colSpan={4} className="rs-grp">terms optimised</th>
            <th colSpan={3} className="rs-grp">held-out</th>
            <th colSpan={2} className="rs-grp">format</th>
            <th rowSpan={2} className="rs-sub">External macro</th>
          </tr>
          <tr>
            <th className="rs-sub k-q">L_tree</th>
            {TERMS.map(([, n, c]) => <th key={n} className={`rs-sub ${c}`}>{n}</th>)}
            <th className="rs-sub">Acc, %</th><th className="rs-sub">Δ base</th><th className="rs-sub">log p(gold)</th>
            <th className="rs-sub">root legal mass</th><th className="rs-sub">text-mode acc, %</th>
          </tr>
        </thead>
        <tbody>
          <tr className="ab-base">
            <th>base model</th>
            <td colSpan={4} className="rs-none">no training</td>
            <td>{base ? pct(base.accuracy) : "…"}</td><td />
            <td>{base ? base.log_p_gold.toFixed(3) : "…"}</td>
            <td>{base ? base.root_legal_mass.toFixed(4) : "…"}</td>
            <td>{base ? pct(base.gen_correct) : "…"}</td>
            <td>{baseMacro ? pct(baseMacro) : "…"}</td>
          </tr>
          {RUNS.map((r) => {
            const f = r.final?.ALL;
            return (
              <tr key={r.id} className={r.id === "full" ? "rs-total" : ""}>
                <th><i className="ab-dot" style={{ background: r.color }} />{r.label}</th>
                <td className="ab-on k-q">●</td>
                {TERMS.map(([k, , c]) => <td key={k} className={r.anchors.includes(k) ? `ab-on ${c}` : "ab-off"}>{r.anchors.includes(k) ? "●" : "○"}</td>)}
                {f ? (
                  <>
                    <td>{pct(f.accuracy)}</td>
                    <td className="rs-delta" style={{ background: tint(base && f.accuracy - base.accuracy) }}>{base ? signed(f.accuracy - base.accuracy) : "…"}</td>
                    <td>{f.log_p_gold.toFixed(3)}</td>
                    <td>{f.root_legal_mass.toFixed(4)}</td>
                    <td>{pct(f.gen_correct)}</td>
                    <td>{macro(r) ? pct(macro(r)) : "…"}</td>
                  </>
                ) : (
                  <td colSpan={6} className="rs-none">{r.started ? `training · step ${r.steps_done}/${a.total_steps}` : "queued"}</td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function tint(v) {
  if (!v) return "transparent";
  return `color-mix(in srgb, var(${v > 0 ? "--c6" : "--c5"}) ${Math.min(Math.abs(v) / 0.2, 1) * 55}%, transparent)`;
}

/** One value per run with an optional ±1 standard-error whisker; a dashed line marks the base model. */
function DotPlot({ title, rows, base, format, note }) {
  const { box, show, hide, view } = useTip();
  const W = 380, L = 104, R = 16, row = 25, top = 22;
  const vals = rows.flatMap((r) => [r.v - (r.se ?? 0), r.v + (r.se ?? 0)]).concat(base ?? []);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.12 || 0.01;
  lo -= pad; hi += pad;
  const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  const H = top + rows.length * row + 24;
  const ticks = [0, 0.5, 1].map((t) => lo + (hi - lo) * t);
  return (
    <figure className="rs-chart ab-dots" ref={box} onPointerLeave={hide}>
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top - 8} y2={H - 22} className="rs-grid" />
            <text x={x(t)} y={H - 6} textAnchor="middle" className="rs-tick">{format(t)}</text>
          </g>
        ))}
        {base !== undefined && base !== null && (
          <g>
            <line x1={x(base)} x2={x(base)} y1={top - 8} y2={H - 22} className="ab-base-line" />
            <text x={x(base)} y={top - 11} textAnchor="middle" className="rs-tick">base</text>
          </g>
        )}
        {rows.map((r, i) => {
          const y = top + i * row + row / 2;
          return (
            <g key={r.id} onPointerMove={(e) => show(e, <><b>{r.label}</b><span>{format(r.v)}{r.se ? ` ± ${format(r.se).replace(/^[+−]/, "")}` : ""}</span></>)}>
              <rect x={0} y={y - row / 2} width={W} height={row} className="rs-hit" />
              <text x={L - 10} y={y + 4} textAnchor="end" className={r.id === "full" ? "rs-lab all" : "rs-lab"}>{r.label}</text>
              {r.se ? <line x1={x(r.v - r.se)} x2={x(r.v + r.se)} y1={y} y2={y} stroke={r.color} className="ab-whisker" /> : null}
              <circle cx={x(r.v)} cy={y} r={5} fill={r.color} className="rs-dot" />
            </g>
          );
        })}
      </svg>
      {note && <p className="rs-note">{note}</p>}
      {view}
    </figure>
  );
}

function Summary() {
  const acc = DONE.map((r) => ({ id: r.id, label: r.label, color: r.color, v: r.final.ALL.accuracy, se: Math.sqrt(r.final.ALL.accuracy * (1 - r.final.ALL.accuracy) / N_EVAL) }));
  const lp = DONE.map((r) => ({ id: r.id, label: r.label, color: r.color, v: r.final.ALL.log_p_gold }));
  const ext = DONE.filter((r) => r.bench_final).map((r) => ({ id: r.id, label: r.label, color: r.color, v: r.bench_final.external_macro.accuracy }));
  if (!DONE.length) return <p className="ab-status">No run has finished yet, so there is nothing to plot.</p>;
  return (
    <div className="rs-grid-charts three">
      <DotPlot title="Held-out accuracy" rows={acc} base={a.base?.ALL.accuracy} format={(v) => `${(v * 100).toFixed(1)}%`} note={`Whiskers: ±1 standard error over the held-out rows.`} />
      <DotPlot title="Held-out log p(gold)" rows={lp} base={a.base?.ALL.log_p_gold} format={(v) => v.toFixed(2)} note="Higher is better; it is not subject to the 0/1 rounding of accuracy." />
      {ext.length > 0 && <DotPlot title="External macro accuracy" rows={ext} base={a.bench_base?.external_macro.accuracy} format={(v) => `${(v * 100).toFixed(1)}%`} note="Mean of the eight outside benchmarks; the base score comes from the earlier Qwen3-0.6B evaluation." />}
    </div>
  );
}

function Curves() {
  const [s, setS] = useState(0.8);
  const series = (ids, k) => ids.filter((id) => BY[id].curves[k]).map((id) => ({ id, label: BY[id].label, color: COLORS[id], points: BY[id].curves[k] }));
  const xmax = Math.max(a.steps, 1);
  return (
    <>
      <div className="rs-ctl">
        <label htmlFor="ab-smooth">Smoothing</label>
        <input id="ab-smooth" type="range" min="0" max="0.98" step="0.02" value={s} onChange={(e) => setS(+e.target.value)} />
        <span>{s.toFixed(2)}</span>
      </div>
      <div className="rs-grid-charts two">
        {TERMS.map(([k, name]) => {
          const ids = ["tree_only", `${k}_only`, "full"];
          return (
            <div key={k}>
              <LineChart title={`${name} during training`} xmax={xmax} a={s} series={series(ids, k)}
                note={k === "mass" ? "Per example, weighted over six trie nodes." : undefined} />
              <Legend ids={ids} />
            </div>
          );
        })}
        <div>
          <LineChart title="L_tree during training" xmax={xmax} a={s} series={series(["tree_only", "mass_only", "out_only", "pos_only", "full"], "tree")} />
          <Legend ids={["tree_only", "mass_only", "out_only", "pos_only", "full"]} />
        </div>
      </div>
      <p className="rs-note">
        Every run logs all four terms, so the curve of a term that a run does not optimise shows how far that term drifts
        when nothing holds it. Compare the grey line (L_tree only) with the coloured one.
      </p>
    </>
  );
}

function DatasetTable({ title, rows, get, digits = 0 }) {
  const ref = BY.tree_only;
  return (
    <div className="results-wrap">
      <table className="rs-table ab-table">
        <thead>
          <tr>
            <th>{title}</th>
            <th className="rs-sub">Base</th>
            {RUNS.map((r) => <th key={r.id} className="rs-sub"><i className="ab-dot" style={{ background: r.color }} />{r.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, label]) => {
            const b = get(null, k), r0 = ref.done ? get(ref, k) : undefined;
            return (
              <tr key={k} className={k === "ALL" || k === "external_macro" ? "rs-total" : ""}>
                <th>{label}</th>
                <td className="rs-b">{b === undefined ? "…" : pct(b, digits)}</td>
                {RUNS.map((r) => {
                  const v = r.done ? get(r, k) : undefined;
                  if (v === undefined) return <td key={r.id} className="rs-none">…</td>;
                  const d = r.id !== "tree_only" && r0 !== undefined ? v - r0 : 0;
                  return <td key={r.id} className="rs-delta" style={{ background: tint(d) }}>{pct(v, digits)}</td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function Ablation() {
  return (
    <section className="trie-section" id="ablation">
      <p className="eyebrow">6 · Loss ablation</p>
      <h2>Do the three extra terms earn their keep?</h2>
      <p className="lede small">
        {a.model}, one epoch ({a.total_steps} steps, batch {a.batch_size}, rank {a.lora_rank}, learning rate {a.lr}, λ = {a.lam}) on the
        same training set, the same data order and seed for every run. Only the loss changes: L_tree alone, each anchor
        added alone, and the full loss with each anchor left out. Held-out and benchmark numbers come from the final adapter.
      </p>
      <Status />
      <Scoreboard />
      <p className="rs-note">● the run optimises that term; ○ it only logs it. Shaded cells are changes from the base model.</p>

      <p className="rs-note">
        Reading: the anchors do not move accuracy. Held-out accuracy spans 64.5–66.9% across all eight runs, inside about two
        standard errors (±1.4 points each, one seed), and external macro accuracy stays at 55.6–57.1% against 56.8% for the base
        model. What they do change is legality: L_tree alone leaves 97.5% of the root mass on legal tokens, and every run with
        any anchor reaches 99.3–99.9%. The full loss has the highest legal mass and a slightly lower log p(gold) than L_tree alone.
      </p>

      <h3 className="sub-h">Headline metrics, one dot per run</h3>
      <Summary />

      <h3 className="sub-h">What each term does while it trains</h3>
      <Curves />

      <h3 className="sub-h">Held-out accuracy by dataset</h3>
      <DatasetTable title="Accuracy, %" rows={[...NAMES.map((k) => [k, LABELS[k] ?? k]), ["ALL", "ALL"]]} get={(r, k) => (r ? r.final?.[k]?.accuracy : a.base?.[k]?.accuracy)} />
      <p className="rs-note">Shading compares each run with the L_tree-only run. One standard error is about 5 points on a single dataset, so single-dataset differences are mostly noise.</p>

      <h3 className="sub-h">External benchmarks</h3>
      <DatasetTable title="Accuracy, %" digits={1} rows={BENCH} get={(r, k) => (r ? r.bench_final?.[k]?.accuracy : a.bench_base?.[k]?.accuracy)} />
    </section>
  );
}

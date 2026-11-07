import { useState } from "react";
import { motion } from "framer-motion";
import { LineChart } from "./Results.jsx";
import d from "../../data/encoder.json";

const NAMES = {
  anli: "ANLI", bitext: "Bitext", clinc150: "CLINC150", contractnli: "ContractNLI", cosmosqa: "CosmosQA",
  logiqa2: "LogiQA 2.0", massive: "MASSIVE", musique: "MuSiQue", nimble: "Nimble", piqa: "PIQA",
  proofwriter: "ProofWriter", quality: "QuALITY", reclor: "ReClor", arc_challenge: "ARC-Challenge",
  banking77: "Banking77", boolq: "BoolQ", jevbench: "JevBench", mmlu: "MMLU", mmlu_pro: "MMLU-Pro",
  sciq: "SciQ", winogrande: "Winogrande", ALL: "ALL", external_macro: "External macro",
};
// Sweep runs: the best run takes --e1 (its color everywhere in this section), the rest follow by learning rate.
// Only runs with the featured run's epoch count share an x-axis and are plotted (at most the four
// validated --e colors); longer runs are listed in the sweep table without a color.
const FEATURED = d.encoders.find((m) => m.best);
const PLOTTED = d.encoders.filter((m) => m.epochs === FEATURED.epochs)
  .sort((a, b) => (b.best - a.best) || (a.lr - b.lr)).slice(0, 4)
  .map((m, i) => ({ ...m, color: `var(--e${i + 1})` }));
const RUNS = [...PLOTTED, ...d.encoders.filter((m) => !PLOTTED.some((p) => p.id === m.id))
  .sort((a, b) => a.epochs - b.epochs || a.lr - b.lr).map((m) => ({ ...m, color: null }))];
const BEST = RUNS[0];
const LONGER = RUNS.filter((m) => m.epochs > BEST.epochs);
const [Q06, Q4] = d.baselines;

const pct = (v, k = 1) => (v == null ? "–" : (v * 100).toFixed(k));
const signed = (v, k = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(k)}`;
const lrText = (lr) => lr.toExponential(1).replace("e-", "e−");
const runText = (m) => `lr ${lrText(m.lr)}${m.epochs > 1 ? ` · ${m.epochs} epochs` : ""}`;

function Delta({ v }) {
  if (v == null || Number.isNaN(v)) return <td className="rs-none">–</td>;
  const tone = Math.min(Math.abs(v) / 0.3, 1) * 55;
  const bg = v === 0 ? "transparent" : `color-mix(in srgb, var(${v > 0 ? "--c6" : "--c5"}) ${tone}%, transparent)`;
  return <td className="rs-delta" style={{ background: bg }}>{signed(v)}</td>;
}

function Cards() {
  const cards = [
    { id: "enc", label: "UltraBERT-380M", color: BEST.color, m: BEST, note: `full fine-tune · ${runText(BEST)} · 1 forward pass` },
    { id: Q06.id, label: Q06.label, color: "var(--m1)", m: Q06, note: "LoRA r16 + KL anchors · trie scoring" },
    { id: Q4.id, label: Q4.label, color: "var(--m2)", m: Q4, note: "LoRA r16 + KL anchors · trie scoring" },
  ];
  return (
    <div className="rs-cards">
      {cards.map((c, i) => (
        <motion.div key={c.id} className="rs-card" style={{ "--mc": c.color }}
          initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.08 }}>
          <h3><i />{c.label}</h3>
          <p className="rs-big">{pct(c.m.final.ALL)}<small>held-out ALL, %</small></p>
          <dl>
            <dt>external macro</dt><dd>{pct(c.m.bench?.external_macro)}%</dd>
            <dt>Banking77 / JevBench</dt><dd>{pct(c.m.bench?.banking77)} / {pct(c.m.bench?.jevbench)}</dd>
            <dt>recipe</dt><dd>{c.note}</dd>
          </dl>
        </motion.div>
      ))}
    </div>
  );
}

/** UltraBERT next to each decoder, with the signed gap (UltraBERT − decoder) per row. */
function Compare({ rows, get, label }) {
  const cols = [Q06, Q4];
  return (
    <div className="results-wrap">
      <table className="rs-table">
        <thead>
          <tr>
            <th rowSpan={2}>{label}</th>
            <th className="rs-grp"><i style={{ background: BEST.color }} />UltraBERT-380M</th>
            {cols.map((q, i) => (
              <th key={q.id} colSpan={2} className="rs-grp"><i style={{ background: `var(--m${i + 1})` }} />{q.label}</th>
            ))}
          </tr>
          <tr>
            <th className="rs-sub">Acc</th>
            {cols.map((q) => ["Acc", "UltraBERT −"].map((h) => <th key={q.id + h} className="rs-sub">{h}</th>))}
          </tr>
        </thead>
        <tbody>
          {rows.map((k) => {
            const u = get(BEST, k);
            return (
              <tr key={k} className={k === "ALL" || k === "external_macro" ? "rs-total" : ""}>
                <th>{NAMES[k] ?? k}</th>
                <td>{pct(u)}</td>
                {cols.map((q) => {
                  const v = get(q, k);
                  return [
                    <td key={q.id + "a"} className="rs-b">{pct(v)}</td>,
                    <Delta key={q.id + "d"} v={u != null && v != null ? u - v : null} />,
                  ];
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Sweep() {
  return (
    <div className="results-wrap">
      <table className="rs-table">
        <thead>
          <tr>
            <th>Run</th><th className="rs-sub">lr</th><th className="rs-sub">epochs</th><th className="rs-sub">steps</th>
            <th className="rs-sub">ALL before</th><th className="rs-sub">ALL after</th><th className="rs-sub">External macro</th>
          </tr>
        </thead>
        <tbody>
          {RUNS.map((m) => (
            <tr key={m.id} className={m.best ? "rs-total" : ""}>
              <th>{m.color ? <i className="rs-key" style={{ background: m.color }} /> : <i className="rs-key rs-key-none" />}{m.id}{m.best ? " (featured)" : ""}</th>
              <td>{lrText(m.lr)}</td><td>{m.epochs}</td><td>{m.steps ?? "–"}</td>
              <td className="rs-b">{pct(m.base?.ALL)}</td><td>{pct(m.final.ALL)}</td><td>{pct(m.bench?.external_macro)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Encoder() {
  const [a, setA] = useState(0.9);
  const xmax = Math.max(...PLOTTED.map((m) => m.steps ?? 1));
  const series = (k) => PLOTTED.filter((m) => m.curves[k]).map((m) => ({
    id: m.id, label: runText(m), color: m.color, points: m.curves[k],
  }));
  return (
    <section className="trie-section" id="encoder">
      <p className="eyebrow">9 · Encoder</p>
      <h2>A 380M encoder against the decoders</h2>
      <p className="lede small">
        UltraBERT-380M is a bidirectional encoder with a native 32K context. It reads a decision with one mask per
        option, <code>[k] &lt;|mask|&gt; option</code>, and scores option k by the masked-LM logit of “ yes” at its mask;
        a softmax over the K masks is the decision, so one forward pass scores every option (no trie, no generation).
        Training is plain cross-entropy over the K options (L_tree with one level, no KL anchors), full fine-tuning with
        fp32 weights under bf16 autocast, one epoch of the same {d.datasets.length}-dataset mix. The decoders are the
        released checkpoints, re-scored on the same held-out rows and benchmark samples.
      </p>
      <Cards />

      <h3 className="sub-h">Held-out accuracy per dataset</h3>
      <p className="lede small">
        Accuracy in %. The shaded column is UltraBERT minus that decoder, in points: green where the encoder is ahead,
        pink where it is behind. ALL is the mean over the {d.datasets.length} datasets for every model.
      </p>
      <Compare rows={[...d.datasets, "ALL"]} label="Dataset" get={(m, k) => m.final?.[k]} />
      <p className="rs-note">About 100 rows per dataset, so one standard error is about 5 points on a single dataset and about 1.4 points for ALL.</p>

      {d.benchmarks.length > 0 && (
        <>
          <h3 className="sub-h">External benchmarks</h3>
          <Compare rows={[...d.benchmarks, "external_macro"]} label="Benchmark" get={(m, k) => m.bench?.[k]} />
          <p className="rs-note">The same seeded samples as section 5. External macro is the mean over the six external sets.</p>
        </>
      )}

      <h3 className="sub-h">Learning-rate sweep</h3>
      <p className="lede small">
        The encoder card recommends lower learning rates than other encoders. Each run below changes only{" "}
        <code>optim.lr</code> or <code>run.epochs</code> in <code>configs/train_mlm/ultrabert_380m.yaml</code>. The featured run is the best
        {" "}{BEST.epochs}-epoch run on held-out ALL (the decoders also trained for {BEST.epochs} epoch), chosen on the same
        rows reported above, so its ALL is slightly optimistic; the benchmarks are untouched by the choice.
        {LONGER.map((m) => (
          <span key={m.id}>
            {" "}Training longer ({runText(m)}) lifts held-out ALL to {pct(m.final.ALL)} but lowers the external macro to{" "}
            {pct(m.bench?.external_macro)}: the extra epoch fits the training mix more than it generalizes. It is listed, not
            plotted (its x-axis is {m.epochs}× longer).
          </span>
        ))}
      </p>
      <Sweep />
      <div className="rs-ctl">
        <label htmlFor="enc-smooth">Smoothing</label>
        <input id="enc-smooth" type="range" min="0" max="0.98" step="0.02" value={a} onChange={(e) => setA(+e.target.value)} />
        <span>{a.toFixed(2)}</span>
        <div className="rs-legend">
          {PLOTTED.map((m) => <span key={m.id}><i style={{ background: m.color }} />{runText(m)}</span>)}
        </div>
      </div>
      <div className="rs-grid-charts two">
        <LineChart wide title="Training loss (cross-entropy over the K options)" xmax={xmax} a={a} series={series("loss")}
          format={(v) => v.toFixed(2)} />
        <LineChart wide title="Training accuracy (batch)" xmax={xmax} a={a} series={series("accuracy")}
          format={(v) => `${(v * 100).toFixed(1)}%`} />
      </div>
    </section>
  );
}

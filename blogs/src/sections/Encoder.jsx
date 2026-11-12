import { useState } from "react";
import { motion } from "framer-motion";
import { LineChart } from "./Results.jsx";
import { Toggle } from "../components/Controls.jsx";
import { Tex } from "../components/Math.jsx";
import d from "../../data/encoder.json";

const pct = (v, k = 1) => (v == null ? "–" : (v * 100).toFixed(k));
const signed = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)}`;
const Q = d.decoder;

function Delta({ v }) {
  const tone = Math.min(Math.abs(v) / 0.15, 1) * 55;
  const bg = `color-mix(in srgb, var(${v > 0 ? "--c6" : "--c5"}) ${tone}%, transparent)`;
  return <td className="rs-delta" style={{ background: bg }}>{signed(v)}</td>;
}

/** Test, dev and external accuracy of one recipe next to the decoder. */
function Cards({ r }) {
  const cards = [
    { label: `UltraBERT-380M · ${r.label}`, color: "var(--e1)", m: r },
    { label: Q.label, color: "var(--m1)", m: Q },
  ];
  return (
    <div className="rs-cards">
      {cards.map((c, i) => (
        <motion.div key={c.label} className="rs-card" style={{ "--mc": c.color }}
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }}>
          <h3>{c.label}</h3>
          <p className="rs-big">{pct(c.m.test)}<small>test accuracy, %</small></p>
          <dl>
            <dt>dev</dt><dd>{pct(c.m.dev)}%</dd>
            <dt>external benchmarks</dt><dd>{pct(c.m.ext)}%</dd>
          </dl>
        </motion.div>
      ))}
    </div>
  );
}

function PerDataset({ r }) {
  return (
    <div className="results-wrap">
      <table className="rs-table">
        <thead>
          <tr><th>Dataset</th><th className="rs-sub">UltraBERT</th><th className="rs-sub">Qwen3-0.6B</th><th className="rs-sub">Difference</th></tr>
        </thead>
        <tbody>
          {d.datasets.map(([k, name]) => (
            <tr key={k}>
              <th>{name}</th><td>{pct(r.per_dataset[k])}</td><td className="rs-b">{pct(Q.per_dataset[k])}</td>
              <Delta v={r.per_dataset[k] - Q.per_dataset[k]} />
            </tr>
          ))}
          <tr className="rs-total">
            <th>ALL</th><td>{pct(r.test)}</td><td className="rs-b">{pct(Q.test)}</td><Delta v={r.test - Q.test} />
          </tr>
        </tbody>
      </table>
    </div>
  );
}

/** The selected recipe's option-sensitivity numbers, with the other recipes for reference. */
function Sensitivity({ sel }) {
  const rows = [
    ["reorder", "Answer changes when the options are reordered", (v) => `${pct(v)}%`],
    ["remove", "Answer changes when one wrong option is removed", (v) => `${pct(v)}%`],
    ["label_tv", "Label bias (TV between predicted and gold label frequencies)", (v) => v.toFixed(3)],
    ["ece", "Calibration error of the top option", (v) => v.toFixed(3)],
  ];
  return (
    <div className="results-wrap">
      <table className="rs-table">
        <thead>
          <tr><th>Dev rows</th>{d.recipes.map((r) => <th key={r.key} className={`rs-sub${r.key === sel ? " on" : ""}`}>{r.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map(([k, name, fmt]) => (
            <tr key={k}>
              <th>{name}</th>
              {d.recipes.map((r) => (
                <td key={r.key} className={r.key === sel ? "rs-sel" : "rs-b"}>{fmt(r.coupling[k])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Cost() {
  return (
    <div className="results-wrap">
      <table className="rs-table">
        <thead><tr><th>Model</th><th className="rs-sub">ms per decision</th><th className="rs-sub">Peak memory, GiB</th></tr></thead>
        <tbody>
          {d.cost.map((c) => (
            <tr key={c.label}><th>{c.label}</th><td>{c.ms.toFixed(0)}</td><td>{c.peak_gib.toFixed(1)}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Encoder() {
  const [sel, setSel] = useState("ub_sig");
  const r = d.recipes.find((x) => x.key === sel);
  const xmax = Math.max(...r.loss.map(([s]) => s));
  return (
    <section className="trie-section" id="encoder">
      <p className="eyebrow">9 · Encoder</p>
      <h2>A 380M encoder against the decoders</h2>
      <p className="lede small">
        UltraBERT-380M reads the same row with a mask in front of each option, <code>[k] &lt;mask&gt; option</code>,
        and scores every option in one forward pass, with no trie and no generation. It is fully fine-tuned for one
        epoch on the same data as Qwen3-0.6B. The four recipes below differ only in how an option is scored and how
        the scores become probabilities.
      </p>

      <div className="rs-ctl">
        <Toggle id="enc-recipe" label="Recipe" value={sel} onChange={setSel}
          options={d.recipes.map((x) => [x.key, x.label])} />
      </div>

      <div className="enc-recipe">
        <p className="lede small">{r.text}</p>
        <Tex block>{r.formula}</Tex>
        <p className="rs-note">Released at <a href={r.hf}>{r.hf.replace("https://huggingface.co/", "")}</a>.</p>
      </div>

      <Cards r={r} />

      <h3 className="sub-h">Test accuracy per dataset</h3>
      <PerDataset r={r} />
      <p className="rs-note">
        Averaged over training seeds. For reference, the reproduced Qwen3-4B LoRA model reaches {pct(d.decoder4b.test)}%.
      </p>

      <h3 className="sub-h">Does the answer depend on the other options?</h3>
      <p className="lede small">
        Each dev row is scored in four random option orders and once with a wrong option removed. The selected
        recipe is highlighted.
      </p>
      <Sensitivity sel={sel} />
      <p className="rs-note">
        Label bias: on the three datasets whose options are the same labels in every row (ContractNLI, MuSiQue,
        ProofWriter), we compare how often the model predicts each label with how often it is correct. The total
        variation is the share of predictions that would have to change label for the two to match, so 0 means no
        bias. For example, “Yes” is correct on 48% of MuSiQue dev rows, but the softmax models predict it for 64–69%
        of them; the sigmoid models for 47–60%. This measures bias, not accuracy: a model could predict each label
        exactly as often as it is correct and still put it on the wrong rows.
      </p>

      <h3 className="sub-h">Training loss</h3>
      <div className="rs-grid-charts">
        <LineChart wide title={`${r.label}, released run`} xmax={xmax} a={0.9}
          series={[{ id: r.key, label: r.label, color: "var(--e1)", points: r.loss }]}
          format={(v) => v.toFixed(2)} />
      </div>

      <h3 className="sub-h">Time and memory</h3>
      <p className="lede small">
        One row at a time on one A100. The decoders read the input, then all option numbers in one more pass on
        the shared cache, their fastest path.
      </p>
      <Cost />
    </section>
  );
}

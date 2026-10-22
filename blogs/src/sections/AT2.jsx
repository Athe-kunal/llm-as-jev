import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import t from "../../data/at2.json";
import c from "../../data/contextcite.json";
import { Stepper, Toggle } from "../components/Controls.jsx";
import { Doc, Part, Tex, TokenDoc } from "../components/Math.jsx";

const S = c.sentences;
const d = S.length;
const L = t.layers, H = t.heads, F = L * H;
const gold = t.gold[0];
// the largest feature over every sentence: the shared colour scale
const PHI_MAX = t.phi.reduce((m, row) => row.reduce((a, v) => Math.max(a, v), m), 0);
// both files describe the same sentences only if export_at2 ran after export_contextcite
const CONSISTENT = t.phi.length === d && t.ranges.length === d;
const f2 = (x) => (Math.abs(x) < 0.005 ? 0 : x).toFixed(2);
const pearson = (a, b) => {
  const ma = a.reduce((x, y) => x + y, 0) / a.length, mb = b.reduce((x, y) => x + y, 0) / b.length;
  let n = 0, da = 0, db = 0;
  a.forEach((v, i) => { n += (v - ma) * (b[i] - mb); da += (v - ma) ** 2; db += (b[i] - mb) ** 2; });
  return n / Math.sqrt(da * db);
};

const STEPS = [
  { title: "Why learn attribution" },
  { title: "Attention features" },
  { title: "A score per sentence" },
  { title: "The surrogate and the loss" },
  { title: "Training data and optimisation" },
  { title: "Learned weights" },
];

export default function AT2() {
  const [step, setStep] = useState(0);
  const [sel, setSel] = useState(gold);
  const [scale, setScale] = useState("own");
  if (!CONSISTENT)
    return (
      <section className="trie-section" id="at2">
        <p className="eyebrow">6 · AT2</p>
        <h2>Learning to attribute from the model's own attention</h2>
        <div className="pending">
          The AT2 data is being regenerated: it describes {t.phi.length} sentences, the ContextCite data {d}. Re-run{" "}
          <code>npm run data:at2</code> after <code>npm run data:contextcite</code>; this section returns when both match.
        </div>
      </section>
    );
  return (
    <section className="trie-section" id="at2">
      <p className="eyebrow">6 · AT2</p>
      <h2>Learning to attribute from the model's own attention</h2>
      <p className="lede small">
        Cohen-Wang et al., “Learning to Attribute with Attention” (2025), as implemented in <code>src/modeling/at2.py</code>.
        Same row and model as section 5: {d} sentences, {t.tokens.toLocaleString("en-US")} tokens, {L} layers × {H} heads.
      </p>
      <Stepper steps={STEPS} step={step} setStep={setStep} sectionId="at2" />
      <div className="trie-grid">
        <div className="viz-card">
          <AnimatePresence mode="wait">
            <motion.div key={step} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
              <Visual step={step} sel={sel} setSel={setSel} scale={scale} setScale={setScale} />
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
        <Part label="Problem">
          ContextCite needs n ablated forward passes per row at inference: here n = {c.n}, {c.seconds.toFixed(1)} s. AT2 moves
          the ablations to training time and, at inference, scores sentences from <b>one</b> forward pass.
        </Part>
        <Part label="Idea">
          Attention from the answer to a sentence is a cheap signal of its influence, but heads differ: some track content,
          some track position or formatting. AT2 learns one weight per (layer, head) so that the weighted attention predicts
          the effect of ablations, using exactly ContextCite's targets f(v).
        </Part>
        <Part label="In code">
          <code>modeling/at2.py</code>; training data from <code>dataloader/at2_data.py</code>.
        </Part>
      </>
    );
  if (step === 1)
    return (
      <>
        <Part label="Definition">
          Let <Tex>{"A^{\\ell h}_{t,i}"}</Tex> be the attention probability from position t to token i in layer ℓ, head h,
          and T the positions whose logits predict the answer's suffix tokens. Sentence j occupies tokens{" "}
          <Tex>{"I_j"}</Tex>. Its feature vector <Tex>{`\\phi_j\\in\\mathbb R^{${F}}`}</Tex> has entries
          <Tex block>{"\\phi_j^{\\ell h}=\\frac{1}{|T|}\\sum_{t\\in T}\\;\\sum_{i\\in I_j}A^{\\ell h}_{t,i}."}</Tex>
        </Part>
        <Part label="In code">
          <code>attention_weights</code> (one sdpa forward pass; hooks on q_norm/k_norm, then RoPE and a causal softmax for
          the T rows only), <code>source_token_ranges</code> (char → token), <code>source_features</code>.
        </Part>
        <Part label="On this row">
          The answer is [{c.row.answer + 1}], suffix tokens {t.suffix.map((x) => `“${x}”`).join(" ")}, so |T| ={" "}
          {t.queries.length} (positions {t.queries.join(", ")}). Sentence {sel} spans tokens {t.ranges[sel][0]}–
          {t.ranges[sel][1] - 1}. Click a sentence on the left to see its {F} features.
        </Part>
      </>
    );
  if (step === 2)
    return (
      <>
        <Part label="Definition">
          A parameter vector <Tex>{`\\theta\\in\\mathbb R^{${F}}`}</Tex> gives each sentence the score{" "}
          <Tex>{"w_j(\\theta)=\\theta^\\top\\phi_j=\\sum_{\\ell,h}\\theta_{\\ell h}\\,\\phi_j^{\\ell h}"}</Tex>. Before
          training, <Tex>{`\\theta=\\tfrac{1}{${F}}\\mathbf 1`}</Tex>: the score is the attention averaged over all heads (the
          paper's “average attention” baseline).
        </Part>
        <Part label="In code">
          The reference's <code>LinearScoreEstimator</code>: <code>nn.Linear({F}, 1, bias=False)</code> initialised to 1/{F}.
        </Part>
        <Part label="On this row">
          With uniform θ the annotated sentence ranks{" "}
          {1 + t.uniform.filter((v) => v > t.uniform[gold]).length} of {d}.
          {t.eval && (
            <>
              {" "}On 20 eval rows per dataset its AP is{" "}
              {Object.entries(t.eval).map(([k, m]) => `${f2(m.average_attention.ap)} (${k})`).join(", ")}
              {Object.values(t.eval)[0].contextcite && (
                <>, against ContextCite's {Object.values(t.eval).map((m) => f2(m.contextcite.ap)).join(" / ")}</>
              )}
              .
            </>
          )}
        </Part>
      </>
    );
  if (step === 3)
    return (
      <>
        <Part label="Definition">
          For a mask v the surrogate sums the scores of kept sentences minus dropped ones:
          <Tex block>{"\\hat f_\\theta(v)=\\sum_j(2v_j-1)\\,w_j(\\theta)=\\theta^\\top x_v,\\qquad x_v=\\Phi^\\top(2v-1)."}</Tex>
          For each row r with masks <Tex>{"v_1,\\dots,v_m"}</Tex> (m = {t.y.length}) and targets{" "}
          <Tex>{"y_i=f(v_i)"}</Tex>, the loss is the negative Pearson correlation, averaged over rows:
          <Tex block>{"\\mathcal L(\\theta)=-\\frac1R\\sum_{r}\\rho\\big(\\{\\theta^\\top x_{v_i}\\}_i,\\{y_i\\}_i\\big)."}</Tex>
        </Part>
        <Part label="Why Pearson">
          ρ is invariant to <Tex>{"\\hat f\\mapsto a\\hat f+c"}</Tex> with a &gt; 0, so θ needs no per-row intercept or scale
          (each row has its own baseline logit), and the ±1 coding of v is equivalent to 0/1 coding:{" "}
          <Tex>{"\\sum_j(2v_j-1)w_j=2\\,w^\\top v-\\sum_j w_j"}</Tex> is affine in <Tex>{"w^\\top v"}</Tex>.
        </Part>
        <Part label="On this row">
          With uniform θ, ρ = <b className="num">{f2(t.pearsonUniform)}</b> over this row's {t.y.length} masks
          {t.pearsonTrained !== undefined && (
            <>
              ; with the trained θ, ρ = <b className="num">{f2(t.pearsonTrained)}</b>
            </>
          )}
          .
        </Part>
      </>
    );
  if (step === 4)
    return (
      <>
        <Part label="Data">
          {t.trainRows ? (
            <>
              {t.trainRows.toLocaleString("en-US")} training rows ({Object.entries(t.trainSources).map(([k, v]) => `${k} ${v}`).join(", ")}), each with m ={" "}
              {t.y.length} masks. Stored per row: <Tex>{`X_r\\in\\mathbb R^{${t.y.length}\\times ${F}}`}</Tex> (rows{" "}
              <Tex>{"x_{v_i}"}</Tex>) and <Tex>{`y_r\\in\\mathbb R^{${t.y.length}}`}</Tex>. Mean ρ with uniform θ over these rows:{" "}
              <b className="num">{f2(t.trainPearsonUniform)}</b>.
            </>
          ) : (
            <>2,000 training rows (250 from each of the 8 multi-sentence datasets), each with 32 masks; being computed.</>
          )}
        </Part>
        <Part label="Cost">
          32 scored ablations + 1 attention pass per row, once: about 66,000 prompt passes. After that, training touches only
          the stored <Tex>{"X_r, y_r"}</Tex>, never the LM.
        </Part>
        <Part label="Optimisation">
          As the reference: Adam, learning rate 10⁻³ with a cosine schedule, 1,000 steps of 512 rows, gradient clipping at 1,
          then <Tex>{"\\theta\\leftarrow\\theta/\\lVert\\theta\\rVert_1"}</Tex> (scores' scale is irrelevant under ρ).
        </Part>
      </>
    );
  return t.theta ? (
    <>
      <Part label="Learned θ">
        Each cell is one head's weight (L1-normalised; red cells are negative). On this row ρ goes from{" "}
        {f2(t.pearsonUniform)} (uniform) to {f2(t.pearsonTrained)} (trained).
      </Part>
      {t.eval && (
        <Part label="Against human evidence">
          <table className="results small-table">
            <thead>
              <tr><th>AP</th><th>average attention</th><th>AT2</th>{Object.values(t.eval)[0].contextcite && <th>ContextCite</th>}<th>AT2 s/row</th></tr>
            </thead>
            <tbody>
              {Object.entries(t.eval).map(([k, m]) => (
                <tr key={k}>
                  <th>{k}</th>
                  <td>{f2(m.average_attention.ap)}</td>
                  <td className="best">{f2(m.at2.ap)}</td>
                  {m.contextcite && <td>{f2(m.contextcite.ap)}</td>}
                  <td>{m.at2.seconds_per_row.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          20 eval rows per dataset, scored with <code>span_metrics</code> (§5 step 7).
        </Part>
      )}
    </>
  ) : (
    <Part label="Pending">
      θ is trained in AT2 steps 5–6 (<code>ScoreEstimator</code>, <code>pearson_loss</code>, the training script), after the
      training data finishes. This step then shows the learned weights per layer and head and the attribution they give this
      row.
    </Part>
  );
}

function Visual({ step, sel, setSel, scale, setScale }) {
  if (step === 0) return <Costs />;
  if (step === 1)
    return (
      <div className="cc2">
        <div className="heat-head">
          <span className="viz-label">φ for sentence {sel}{S[sel].gold ? " (annotated)" : ""}: rows = layers 0–{L - 1}, columns = heads 0–{H - 1}</span>
          <Toggle id="phi-scale" label="Colour scale" options={[["own", "this sentence"], ["shared", "all sentences"]]} value={scale} onChange={setScale} />
        </div>
        <Heat values={t.phi[sel]} max={scale === "shared" ? PHI_MAX : null}
              unit="share of the head's attention" />
        <Doc sentences={S} selected={sel} onSelect={setSel} maxHeight={230} />
      </div>
    );
  if (step === 2)
    return (
      <div className="cc2">
        <span className="viz-label">uniform-θ scores (average attention) over the document</span>
        <Doc sentences={S} value={t.uniform} maxHeight={500} />
      </div>
    );
  if (step === 3) return <Scatter pred={t.predTrained ?? t.predUniform} label={t.predTrained ? "trained θ" : "uniform θ"} />;
  if (step === 4) return <Pipeline />;
  return t.theta ? (
    <div className="cc2">
      <span className="viz-label">learned θ (layers × heads)</span>
      <Heat values={t.theta} diverging unit="weight" />
      <TrainedView />
    </div>
  ) : (
    <div className="pending">training data in progress; the learned weights appear here after training</div>
  );
}

/** The trained θ's scores on the worked row: per token (as the paper) or summed per sentence. */
function TrainedView() {
  const [view, setView] = useState("tokens");
  const tokens = t.tokenScores ?? [];
  const top = new Set([...tokens].sort((a, b) => b.w - a.w).slice(0, 10).map((x) => x.index));
  const gold = S.flatMap((s, j) => (s.gold ? [j] : []));
  // gold char ranges for the token view, rebuilt from the sentences' order in the part
  const ranges = [];
  let at = 0;
  S.forEach((s, j) => {
    const start = t.partText.indexOf(s.text, at);
    if (start >= 0) {
      if (gold.includes(j)) ranges.push({ start, end: start + s.text.length });
      at = start + s.text.length;
    }
  });
  return (
    <>
      <div className="heat-head">
        <span className="viz-label">trained scores over the document</span>
        {t.tokenScores && (
          <div className="ex-tabs view-toggle" role="tablist" aria-label="Highlight">
            {["tokens", "sentences"].map((v) => (
              <button key={v} role="tab" aria-selected={v === view} onClick={() => setView(v)}>{v}</button>
            ))}
          </div>
        )}
      </div>
      {t.tokenScores && view === "tokens" ? (
        <TokenDoc text={t.partText} tokens={tokens} top={top} gold={ranges} maxHeight={260} />
      ) : (
        <Doc sentences={S} value={t.trained} maxHeight={260} />
      )}
    </>
  );
}

/** A layers × heads heatmap of one feature vector (index l*H + h), with its
 *  colour scale: `max` fixes the top of the scale (else the vector's own
 *  largest |value|); `diverging` colours negative values red. */
function Heat({ values, max = null, diverging = false, unit = "" }) {
  const ref = useRef(null);
  const cw = 18, ch = 9;
  const top = max ?? Math.max(...values.map(Math.abs), 1e-9);
  const own = Math.max(...values.map(Math.abs));
  useEffect(() => {
    const cv = ref.current, css = getComputedStyle(cv), ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    values.forEach((v, i) => {
      const l = Math.floor(i / H), h = i % H;
      ctx.fillStyle = css.getPropertyValue(diverging && v < 0 ? "--c5" : "--accent");
      ctx.globalAlpha = 0.08 + 0.92 * Math.min(Math.abs(v) / top, 1);
      ctx.fillRect(28 + h * cw, l * ch, cw - 1, ch - 1);
    });
    ctx.globalAlpha = 1;
    ctx.fillStyle = css.getPropertyValue("--muted");
    ctx.font = "9px monospace";
    for (let l = 0; l < L; l += 4) ctx.fillText(String(l), 2, l * ch + 8);
    for (let h = 0; h < H; h += 4) ctx.fillText(String(h), 28 + h * cw + 4, L * ch + 10);
  }, [values, diverging, top]);
  const fmt = (v) => (Math.abs(v) >= 0.01 ? v.toFixed(3) : v.toExponential(1));
  return (
    <div className="heat-wrap">
      <canvas ref={ref} width={28 + H * cw} height={L * ch + 14} className="heat" />
      <div className="heat-legend">
        <span>{diverging ? `−${fmt(top)}` : "0"}</span>
        <span className={diverging ? "bar diverging" : "bar"} />
        <span>{fmt(top)}</span>
        <span className="unit">
          {unit}
          {max !== null && ` · this sentence's largest: ${fmt(own)}`}
        </span>
      </div>
    </div>
  );
}

function Scatter({ pred, label }) {
  const W = 300, P = 34;
  const lx = Math.min(...pred), hx = Math.max(...pred), ly = Math.min(...t.y), hy = Math.max(...t.y);
  const sx = (v) => P + ((v - lx) / (hx - lx || 1)) * (W - P - 8);
  const sy = (v) => W - P - ((v - ly) / (hy - ly || 1)) * (W - P - 8);
  return (
    <div>
      <svg className="cc-scatter" viewBox={`0 0 ${W} ${W}`} role="img" aria-label="surrogate vs targets">
        <text x={W / 2 + 10} y={W - 4} textAnchor="middle" className="tick">θᵀx_v ({label}, scale-free)</text>
        <text x={10} y={W / 2} textAnchor="middle" className="tick" transform={`rotate(-90 10 ${W / 2})`}>y = f(v)</text>
        {t.y.map((y, i) => (
          <motion.circle key={i} cx={sx(pred[i])} cy={sy(y)} r={4.5} className="pt" initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.03 }} />
        ))}
      </svg>
      <div className="viz-label">{t.y.length} masks of this row · ρ = {f2(pearson(pred, t.y))}</div>
    </div>
  );
}

function Costs() {
  const rows = [
    ["ContextCite", `${c.n} ablated passes + Lasso, per row`, c.seconds],
    ["AT2", "1 pass + a dot product, per row", t.seconds],
  ];
  return (
    <div className="costs">
      {rows.map(([name, what, s], i) => (
        <div key={name} className="cost-row">
          <span className="cost-name">{name}</span>
          <span className="cost-what">{what}</span>
          <span className="cost-bar">
            <motion.span initial={{ width: 0 }} animate={{ width: `${Math.max((s / c.seconds) * 100, 1.5)}%` }} transition={{ duration: 0.8, delay: i * 0.2 }} className={i ? "fast" : ""} />
          </span>
          <span className="cost-s">{s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(1)} s`}</span>
        </div>
      ))}
      <p className="panel-body">
        Both measured on this {t.tokens.toLocaleString("en-US")}-token row, one GPU; AT2 is the attention pass and features, mean of 5 runs.
      </p>
    </div>
  );
}

function Pipeline() {
  const stages = [
    ["training row", "prompt + predicted answer"],
    ["Φ", `${d} × ${F} attention features (1 pass)`],
    ["32 masks", "v ~ Bernoulli(½)^d"],
    ["32 targets", "y = logit q(a | x_v)"],
    ["store", `X = (2V − 1)Φ  (32 × ${F}),  y (32)`],
    ["fit θ", "−Pearson, Adam, 1,000 × 512"],
  ];
  return (
    <div className="pipeline">
      {stages.map(([h, b], i) => (
        <motion.div key={h} className="stage" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.12 }}>
          <b>{h}</b>
          <span>{b}</span>
        </motion.div>
      ))}
    </div>
  );
}

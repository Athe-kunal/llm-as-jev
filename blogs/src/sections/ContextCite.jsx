import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import c from "../../data/contextcite.json";
import { Stepper } from "../components/Controls.jsx";
import { Doc, Part, Tex } from "../components/Math.jsx";

const S = c.sentences;
const d = S.length;
const w = S.map((s) => s.w);
const order = [...w.keys()].sort((i, j) => w[j] - w[i]);
const nonzero = w.filter((v) => v !== 0).length;
const goldIdx = S.flatMap((s, j) => (s.gold ? [j] : []));
const A = c.row.answer;
const f3 = (x) => x.toFixed(3);
const f2 = (x) => (Math.abs(x) < 0.005 ? 0 : x).toFixed(2);

const STEPS = [
  { title: "What is being explained" },
  { title: "An ablation" },
  { title: "Sampling ablations" },
  { title: "The surrogate and its loss" },
  { title: "Reading the weights" },
  { title: "Is the surrogate faithful?" },
  { title: "Against human evidence" },
];

export default function ContextCite() {
  const [step, setStep] = useState(0);
  return (
    <section className="trie-section" id="contextcite">
      <p className="eyebrow">6 · ContextCite</p>
      <h2>Attributing the answer to sentences by ablation</h2>
      <p className="lede small">
        Cohen-Wang et al., NeurIPS 2024, as implemented in <code>src/modeling/context_cite.py</code>. One ContractNLI eval
        row ({d} sentences), {c.model} with the λ = 0.01 adapter. Each step gives the definition, the function that computes
        it, and its value on this row.
      </p>
      <Stepper steps={STEPS} step={step} setStep={setStep} sectionId="contextcite" />
      <div className="trie-grid">
        <div className="viz-card">
          <AnimatePresence mode="wait">
            <motion.div key={step} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
              <Visual step={step} />
            </motion.div>
          </AnimatePresence>
        </div>
        <div className="panel">
          <AnimatePresence mode="wait">
            <motion.div key={step} initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }} transition={{ duration: 0.22 }}>
              <h3 className="panel-title">{STEPS[step].title}</h3>
              <Text step={step} />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

function Text({ step }) {
  if (step === 0)
    return (
      <>
        <Part label="Definition">
          The context is split into sources <Tex>{"s_1,\\dots,s_d"}</Tex>. The model answers with the option of highest
          tree probability, <Tex>{"a=\\arg\\max_k q(k\\mid x)"}</Tex>, with{" "}
          <Tex>{"q(k\\mid x)=e^{s_k(x)}/\\sum_j e^{s_j(x)}"}</Tex> and <Tex>{"s_k"}</Tex> the summed log-probability of
          suffix <Tex>{"k]"}</Tex> (§2). The quantity explained is the logit of that answer:
          <Tex block>{"f(x)=\\operatorname{logit} q(a\\mid x)=s_a(x)-\\log\\sum_{k\\ne a}e^{s_k(x)}."}</Tex>
          The answer is the model's own prediction, not the gold label: ContextCite explains what the model decided.
        </Part>
        <Part label="In code">
          <code>split_sources</code> (nltk sentences per line, char offsets kept; a sentence without content, e.g. a blank form field “Agent:____”, joins its neighbour), <code>predict</code> (argmax of{" "}
          <code>answers.score</code>), <code>logit_q</code>.
        </Part>
        <Part label="On this row">
          d = {d} sentences in one part ({c.row.chars.toLocaleString("en-US")} characters). q = (
          {c.row.q.map(f3).join(", ")}) over the three options; a = [{A + 1}] “{c.row.options[A]}” (gold [{c.row.gold + 1}]),
          so <Tex>{`f(x)=${f3(c.row.logit)}`}</Tex>. Underlined: the sentence the annotators marked (#{goldIdx.join(', ')}).
        </Part>
      </>
    );
  if (step === 1)
    return (
      <>
        <Part label="Definition">
          A mask <Tex>{"v\\in\\{0,1\\}^d"}</Tex> keeps source <Tex>{"j"}</Tex> iff <Tex>{"v_j=1"}</Tex>.{" "}
          <Tex>{"x_v"}</Tex> is the prompt rebuilt from the kept sentences only, and{" "}
          <Tex>{"f(v)=\\operatorname{logit} q(a\\mid x_v)"}</Tex> scores the <i>same</i> option a on it.
        </Part>
        <Part label="In code">
          <code>ablate(row, sources, keep)</code>: each kept sentence is preceded by its own separator (except the first kept
          one of its part); whitespace before the first and after the last sentence is always kept, so{" "}
          <Tex>{"x_{\\mathbf 1}=x"}</Tex> byte for byte (checked on all 5,991 eval rows). Options stay in release order, so a
          means the same option under every mask.
        </Part>
        <Part label="On this row">
          Mask <Tex>{"v_1"}</Tex> keeps {c.example.mask.reduce((a, b) => a + b, 0)} of {d} sentences (
          {c.example.chars.toLocaleString("en-US")} of {c.row.chars.toLocaleString("en-US")} characters); it drops the marked
          sentence, and <Tex>{`f(v_1)=${f2(c.example.y)}`}</Tex> against <Tex>{`f(\\mathbf 1)=${f2(c.row.logit)}`}</Tex>.
        </Part>
      </>
    );
  if (step === 2)
    return (
      <>
        <Part label="Definition">
          <Tex>{`n=${c.n}`}</Tex> masks with independent entries <Tex>{"v_{ij}\\sim\\text{Bernoulli}(1/2)"}</Tex>, then{" "}
          <Tex>{"y_i=f(v_i)"}</Tex>. Each row of the matrix is one prompt scored by the model.
        </Part>
        <Part label="In code">
          <code>ablation_masks(n, d, seed)</code>: mask i from <code>np.random.RandomState(seed + i)</code>, identical to the
          reference. <code>ablation_targets</code> scores all <Tex>{"x_{v_i}"}</Tex> with <code>answers.score_batch</code> in
          length-sorted batches; no text is generated.
        </Part>
        <Part label="On this row">
          {c.n} prompts in {c.seconds.toFixed(1)} s on one GPU. <Tex>{"y"}</Tex> ranges from {f2(Math.min(...c.y))} to{" "}
          {f2(Math.max(...c.y))} (mean {f2(c.y.reduce((a, b) => a + b, 0) / c.n)}). Green cells mark the annotated sentence's
          column: the low-y rows tend to have it dropped.
        </Part>
      </>
    );
  if (step === 3)
    return (
      <>
        <Part label="Definition">
          A linear surrogate <Tex>{"\\hat f(v)=b+w^\\top v"}</Tex>. Columns are standardised,{" "}
          <Tex>{"z_{ij}=(v_{ij}-\\mu_j)/\\sigma_j"}</Tex>, and the Lasso is fitted:
          <Tex block>{"\\min_{\\beta,\\beta_0}\\;\\frac{1}{2n}\\sum_{i=1}^{n}\\big(y_i-\\beta_0-z_i^\\top\\beta\\big)^2+\\alpha\\lVert\\beta\\rVert_1,"}</Tex>
          then mapped back to the 0/1 scale: <Tex>{"w_j=\\beta_j/\\sigma_j"}</Tex>,{" "}
          <Tex>{"b=\\beta_0-\\sum_j\\mu_j\\beta_j/\\sigma_j"}</Tex>. The L1 term sets most <Tex>{"\\beta_j"}</Tex> exactly to 0:
          with n = {c.n} masks and d = {d} unknowns, least squares alone is underdetermined.
        </Part>
        <Part label="In code">
          <code>fit_surrogate</code>: scikit-learn <code>StandardScaler</code> + <code>Lasso(alpha={c.alpha})</code>, the
          reference's <code>LassoRegression.fit</code>.
        </Part>
        <Part label="On this row">
          {nonzero} of {d} weights are non-zero; <Tex>{`b=${f2(c.bias)}`}</Tex>. The table lists the largest five.
        </Part>
      </>
    );
  if (step === 4)
    return (
      <>
        <Part label="Definition">
          In the surrogate, adding sentence j to any context changes the logit of a by <Tex>{"w_j"}</Tex>:{" "}
          <Tex>{"\\hat f(v+e_j)-\\hat f(v)=w_j"}</Tex>. Positive weights support the answer, negative ones argue against it.
          The attribution is the vector w; a span selection keeps the top-k sentences with <Tex>{"w_j>0"}</Tex>.
        </Part>
        <Part label="In code">
          <code>attribute(model, tok, row)</code> returns every sentence with its w; <code>select(row, result, k)</code>{" "}
          returns the top-k in the row's span format.
        </Part>
        <Part label="On this row">
          The largest weight, <Tex>{`w_{${order[0]}}=${f2(w[order[0]])}`}</Tex>, is on the annotated sentence (#{order[0]}).
          The next two, <Tex>{`w_{${order[1]}}=${f2(w[order[1]])}`}</Tex> and <Tex>{`w_{${order[2]}}=${f2(w[order[2]])}`}</Tex>,
          are the two sentences right after it in the same “Return of Information” clause. Green is positive, red negative;
          hover a sentence for its value.
        </Part>
      </>
    );
  if (step === 5)
    return (
      <>
        <Part label="Definition">
          <b>LDS</b> (linear datamodeling score): on fresh masks <Tex>{"v'_1,\\dots,v'_m"}</Tex>, the Spearman correlation
          between <Tex>{"\\hat f(v'_i)"}</Tex> and <Tex>{"f(v'_i)"}</Tex>. <b>Top-k drop</b>: the fall in{" "}
          <Tex>{"\\log q(a\\mid\\cdot)"}</Tex> when the k highest-weighted sentences are removed, against k random ones.
        </Part>
        <Part label="In code">
          Held-out masks are <code>ablation_masks(m, d, seed=n)</code>, i.e. seeds {c.n}–{c.n + c.test - 1}, never used in the
          fit; the random baseline averages 8 random k-subsets.
        </Part>
        <Part label="On this row">
          LDS = <b className="num">{f2(c.lds)}</b> on m = {c.test}. Removing the top sentence lowers log q(a) by{" "}
          <b className="num">{f2(c.drops[0].top)}</b> nats (q × {Math.exp(-c.drops[0].top).toFixed(2)}); one random sentence
          moves it by {f2(c.drops[0].random)}.
        </Part>
      </>
    );
  return (
    <>
      <Part label="Definition">
        A sentence is <i>gold</i> if it overlaps a gold span (ContractNLI's spans are often clauses inside a sentence). For
        the top-k selection: precision P, recall R and{" "}
        <Tex>{"F_\\beta=\\frac{(1+\\beta^2)PR}{\\beta^2P+R}"}</Tex> (β &lt; 1 favours precision). AP is the average precision
        of the full ranking by w (threshold-free); a random ranking scores the gold fraction in expectation.
      </Part>
      <Part label="In code">
        <code>gold_sentences</code>, <code>span_metrics(row, result, k=3, beta=1)</code>; 20 eval rows per span dataset,
        64 ablations each.
      </Part>
      <Part label="Caveat">
        ContextCite explains the predicted answer. When the prediction is wrong, the sentences that drove it need not be the
        gold evidence, so these scores mix attribution quality with accuracy.
      </Part>
    </>
  );
}

function Visual({ step }) {
  if (step === 0)
    return (
      <div className="cc2">
        <div className="cc2-q">{c.row.question}</div>
        <div className="qbars">
          {c.row.options.map((o, k) => (
            <div key={k} className={k === A ? "qrow on" : "qrow"}>
              <span className="qlab">[{k + 1}] {o}</span>
              <span className="qtrack">
                <motion.span className="qfill" initial={{ width: 0 }} animate={{ width: `${c.row.q[k] * 100}%` }} transition={{ duration: 0.6, delay: k * 0.1 }} />
              </span>
              <span className="qv">{f3(c.row.q[k])}</span>
            </div>
          ))}
        </div>
        <span className="viz-label">the context: {d} sentences, the annotated one underlined</span>
        <Doc sentences={S} maxHeight={330} />
      </div>
    );
  if (step === 1)
    return (
      <div className="cc2">
        <span className="viz-label">x with mask v₁: struck sentences are removed from the prompt</span>
        <Doc sentences={S} keep={c.example.mask.map(Boolean)} maxHeight={470} />
        <div className="eqline">
          <Tex>{`f(\\mathbf 1)=${f2(c.row.logit)}\\quad\\longrightarrow\\quad f(v_1)=${f2(c.example.y)}`}</Tex>
        </div>
      </div>
    );
  if (step === 2) return <MaskMatrix />;
  if (step === 3)
    return (
      <div>
        <span className="viz-label">the five largest weights (columns as defined left)</span>
        <table className="results small-table">
          <thead>
            <tr>
              <th>j</th><th>μⱼ</th><th>σⱼ</th><th>βⱼ</th><th>wⱼ = βⱼ/σⱼ</th><th>gold</th>
            </tr>
          </thead>
          <tbody>
            {order.slice(0, 5).map((j) => (
              <tr key={j}>
                <th>{j}</th>
                <td>{f3(S[j].mu)}</td>
                <td>{f3(S[j].sigma)}</td>
                <td>{f3(S[j].beta)}</td>
                <td className={S[j].gold ? "best" : ""}>{f3(S[j].w)}</td>
                <td>{S[j].gold ? "yes" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <WeightBars />
      </div>
    );
  if (step === 4)
    return (
      <div className="cc2">
        <span className="viz-label">every sentence tinted by its weight w (green +, red −)</span>
        <Doc sentences={S} value={w} maxHeight={500} />
      </div>
    );
  if (step === 5) return <Faithful />;
  return <MetricTable />;
}

/** The n × d masks as pixels (kept = filled), with y beside each row. */
function MaskMatrix() {
  const ref = useRef(null);
  const cell = 3, H = c.n * (cell + 1);
  useEffect(() => {
    const cv = ref.current;
    const css = getComputedStyle(cv);
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    const lo = Math.min(...c.y), hi = Math.max(...c.y);
    c.masks.forEach((m, i) => {
      m.forEach((v, j) => {
        ctx.fillStyle = v ? (S[j].gold ? css.getPropertyValue("--c6") : css.getPropertyValue("--accent")) : css.getPropertyValue("--raised");
        ctx.globalAlpha = v ? 0.85 : 1;
        ctx.fillRect(j * cell, i * (cell + 1), cell - 0.5, cell);
      });
      ctx.globalAlpha = 1;
      ctx.fillStyle = css.getPropertyValue("--c2");
      ctx.fillRect(d * cell + 8, i * (cell + 1), ((c.y[i] - lo) / (hi - lo)) * 80, cell);
    });
  }, []);
  return (
    <div>
      <span className="viz-label">
        {c.n} masks (rows) × {d} sentences (columns); right: y = f(v)
      </span>
      <canvas ref={ref} width={d * cell + 92} height={H} className="maskcanvas" />
    </div>
  );
}

function WeightBars() {
  const max = Math.max(...w.map(Math.abs));
  return (
    <div className="wstrip" aria-label="all weights">
      {w.map((v, j) => (
        <span key={j} className={["wb", S[j].gold ? "gold" : "", v < 0 ? "neg" : ""].join(" ")} style={{ height: `${(Math.abs(v) / max) * 100}%` }} title={`#${j} ${f3(v)}`} />
      ))}
      <span className="viz-label wcap">all {d} weights by sentence index; {d - nonzero} are exactly 0</span>
    </div>
  );
}

function Faithful() {
  const all = [...c.yt, ...c.predt];
  const lo = Math.min(...all) - 0.2, hi = Math.max(...all) + 0.2;
  const W = 300, P = 34;
  const sx = (v) => P + ((v - lo) / (hi - lo)) * (W - P - 8);
  const sy = (v) => W - sx(v) + P - 8;
  const maxDrop = Math.max(...c.drops.map((x) => x.top));
  return (
    <div className="faith">
      <svg className="cc-scatter" viewBox={`0 0 ${W} ${W}`} role="img" aria-label="surrogate vs actual on held-out masks">
        <line x1={sx(lo)} y1={sy(lo)} x2={sx(hi)} y2={sy(hi)} className="diag" />
        <text x={W / 2 + 10} y={W - 1} textAnchor="middle" className="tick">predicted f̂(v′)</text>
        <text x={10} y={W / 2} textAnchor="middle" className="tick" transform={`rotate(-90 10 ${W / 2})`}>actual f(v′)</text>
        {c.yt.map((y, i) => (
          <motion.circle key={i} cx={sx(c.predt[i])} cy={sy(y)} r={4.5} className="pt" initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.03 }} />
        ))}
      </svg>
      <span className="viz-label">top-k drop in log q(a), nats</span>
      {c.drops.map((x) => (
        <div key={x.k} className="drop-row">
          <span className="drop-k">k = {x.k}</span>
          <span className="drop-bars">
            <motion.span className="drop top" initial={{ width: 0 }} animate={{ width: `${(x.top / maxDrop) * 100}%` }} transition={{ duration: 0.6 }} />
            <motion.span className="drop rand" initial={{ width: 0 }} animate={{ width: `${(Math.max(x.random, 0) / maxDrop) * 100}%` }} transition={{ duration: 0.6 }} />
          </span>
          <span className="drop-v">
            top {f2(x.top)} · random {f2(x.random)}
          </span>
        </div>
      ))}
    </div>
  );
}

function MetricTable() {
  const names = { contractnli: "ContractNLI", musique: "MuSiQue", proofwriter: "ProofWriter" };
  return (
    <div className="results-wrap">
      <table className="results small-table">
        <thead>
          <tr>
            <th />
            <th>sentences</th><th>answer correct</th><th>P@3</th><th>R@3</th><th>F1</th><th>AP</th><th>AP random</th><th>top-1 gold</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(c.table).map(([k, m]) => (
            <tr key={k}>
              <th>{names[k] ?? k}</th>
              <td>{m.sentences.toFixed(0)}</td>
              <td>{f2(m.correct)}</td>
              <td>{f2(m.precision)}</td>
              <td>{f2(m.recall)}</td>
              <td>{f2(m.f_beta)}</td>
              <td className="best">{f2(m.ap)}</td>
              <td>{f2(m.random_ap)}</td>
              <td>{f2(m.top1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="panel-body">
        AP is 15× random on ContractNLI and 3× on ProofWriter. On MuSiQue the gold evidence is two whole paragraphs (≈ 9
        sentences), so 3 sentences recall at most a third of it, and 40% of the attributed answers are wrong.
      </p>
    </div>
  );
}

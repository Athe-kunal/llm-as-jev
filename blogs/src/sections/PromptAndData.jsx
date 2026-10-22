import { useEffect, useState } from "react";
import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import data from "../../data/prompt_and_data.json";
import { PositionStrip, Prompt, STEPS } from "../components/Prompt.jsx";

const fmt = (n) => n.toLocaleString("en-US");
const reduced =
  typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const FACTS = [
  { big: "12", text: "datasets: 3 intent, 6 short reasoning, 3 long input" },
  { big: "1,000", text: "training rows from each one (Reason+Long-12k)" },
  { big: "[k]", text: "options are reshuffled for every training example" },
  { big: "10,240", text: "token cap; longer rows are dropped, never truncated" },
];

export default function PromptAndData() {
  return (
    <main>
      <Hero />
      <section className="facts">
        {FACTS.map((f, i) => (
          <motion.div
            key={f.big}
            className="fact"
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.08 }}
          >
            <b>{f.big}</b>
            <span>{f.text}</span>
          </motion.div>
        ))}
      </section>
      <Browser />
    </main>
  );
}

/** The prompt of one PIQA row, assembled step by step, then answered. */
function Hero() {
  const ex = data.datasets.find((d) => d.id === "piqa").examples[0];
  const [step, setStep] = useState(reduced ? 6 : -1);
  const [run, setRun] = useState(0);
  const [focus, setFocus] = useState(null);

  useEffect(() => {
    if (reduced) return undefined;
    setStep(-1);
    const timers = STEPS.map((_, i) => setTimeout(() => setStep(i), 500 + i * 750));
    return () => timers.forEach(clearTimeout);
  }, [run]);

  return (
    <section className="hero">
      <div className="hero-copy">
        <p className="eyebrow">1 · Prompt and data</p>
        <h1>Every task becomes one prompt and a bracketed number.</h1>
        <p className="lede">
          Intent routing, logic puzzles, 20-paragraph retrieval: all rendered by{" "}
          <code>answers.build_prompt</code> into the same shape. The model answers by writing{" "}
          <code>k]</code>.
        </p>
      </div>
      <div className="hero-grid">
        <div className="hero-prompt">
          <Prompt ex={ex} upTo={step} focus={focus} />
        </div>
        <ol className="steps" onMouseLeave={() => setFocus(null)}>
          {STEPS.map((s, i) => (
            <li
              key={s.name}
              className={[`s${i}`, i === step ? "now" : "", i <= step ? "done" : ""].join(" ")}
              onMouseEnter={() => step >= 6 && setFocus(i)}
            >
              <span className="dot" />
              <div>
                <b>{s.name}</b>
                <span>{s.note}</span>
              </div>
            </li>
          ))}
          <li className="replay-row">
            <button className="replay" onClick={() => setRun((r) => r + 1)} disabled={step < 6}>
              ↻ Replay
            </button>
          </li>
        </ol>
      </div>
    </section>
  );
}

/** Two rendered training prompts per dataset. */
function Browser() {
  const [dsId, setDsId] = useState(data.datasets[0].id);
  const [exIdx, setExIdx] = useState(0);
  const ds = data.datasets.find((d) => d.id === dsId);
  const ex = ds.examples[exIdx];
  const groups = [...new Set(data.datasets.map((d) => d.group))];

  return (
    <section className="browser">
      <h2>The twelve datasets</h2>
      <p className="lede small">
        Two training rows from each, exactly as the model sees them: options in their shuffled
        order, gold highlighted.
      </p>
      <div className="browser-grid">
        <LayoutGroup id="rail">
          <nav className="rail" aria-label="Datasets">
            {groups.map((g) => (
              <div key={g} className="rail-group">
                <span className="rail-label">{g}</span>
                {data.datasets
                  .filter((d) => d.group === g)
                  .map((d) => (
                    <button
                      key={d.id}
                      className={d.id === dsId ? "rail-item on" : "rail-item"}
                      onClick={() => {
                        setDsId(d.id);
                        setExIdx(0);
                      }}
                    >
                      {d.id === dsId && (
                        <motion.span layoutId="rail-pill" className="pill" transition={{ type: "spring", stiffness: 500, damping: 38 }} />
                      )}
                      <span className="rail-name">{d.title}</span>
                      <span className="rail-count">{fmt(d.rows)}</span>
                    </button>
                  ))}
              </div>
            ))}
          </nav>
        </LayoutGroup>

        <div className="viewer">
          <AnimatePresence mode="wait">
            <motion.div
              key={`${dsId}-${exIdx}`}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
            >
              <div className="viewer-head">
                <div>
                  <h3>{ds.title}</h3>
                  <p className="desc">{ds.desc}</p>
                </div>
                <LayoutGroup id="ex">
                  <div className="ex-tabs" role="tablist">
                    {ds.examples.map((_, i) => (
                      <button key={i} role="tab" aria-selected={i === exIdx} onClick={() => setExIdx(i)}>
                        {i === exIdx && <motion.span layoutId="ex-under" className="under" />}
                        Example {i + 1}
                      </button>
                    ))}
                  </div>
                </LayoutGroup>
              </div>
              <div className="meta">
                <span><b>{ex.k}</b> options</span>
                <span><b>{fmt(ex.tokens)}</b> prompt tokens</span>
                <span>gold at <b>[{ex.gold + 1}]</b></span>
                <span>answer = <b>{ex.suffix.length}</b> tokens</span>
                <code className="rowid">{ex.id}</code>
              </div>
              <PositionStrip ex={ex} />
              <Prompt ex={ex} stagger />
              <p className="gold-line">
                <span className="tag">gold</span> [{ex.gold + 1}] {ex.goldText}
              </p>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

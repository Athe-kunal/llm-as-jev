import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Stepper } from "../components/Controls.jsx";
import d from "../../data/loss_code.json";

const KEYWORDS = /\b(def|return|for|in|with|if|else|and|or|not|None|True|False)\b/g;

/** `code` spans inside a sentence. */
function Rich({ text }) {
  return text.split("`").map((part, i) => (i % 2 ? <code key={i}>{part}</code> : <span key={i}>{part}</span>));
}

/** One source line with comments dimmed and keywords tinted. */
function Source({ text }) {
  const hash = text.indexOf("#");
  const code = hash < 0 ? text : text.slice(0, hash);
  const comment = hash < 0 ? "" : text.slice(hash);
  const parts = code.split(KEYWORDS);
  return (
    <>
      {parts.map((p, i) => (i % 2 ? <span key={i} className="lc-kw">{p}</span> : p))}
      {comment && <span className="lc-cm">{comment}</span>}
    </>
  );
}

/** The function being executed, with the line that just ran highlighted. */
function CodeCard({ step }) {
  const lines = d.funcs[step.fn].lines;
  const box = useRef(null);
  useEffect(() => {
    // keep the active line in view inside the card, not the page
    const el = box.current?.querySelector(".lc-line.on");
    if (!el || !box.current) return;
    const top = el.offsetTop - box.current.clientHeight / 2 + el.clientHeight / 2;
    box.current.scrollTo({ top: Math.max(top, 0), behavior: "smooth" });
  }, [step]);
  return (
    <div className="lc-card">
      <div className="lc-stack" aria-label="Call stack">
        {step.stack.map((f, i) => (
          <span key={i} className={i === step.stack.length - 1 ? "lc-frame on" : "lc-frame"}>
            {f}
            {i < step.stack.length - 1 && <i>›</i>}
          </span>
        ))}
      </div>
      <div className="lc-code" ref={box}>
        {lines.map((text, i) => (
          <div key={`${step.fn}-${i}`} className={i === step.line ? "lc-line on" : "lc-line"}>
            <span className="lc-ln">{i + 1}</span>
            <span className="lc-src"><Source text={text} /></span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Variables after the line ran: the ones it changed are open, the rest fold away. */
function Variables({ vars }) {
  const ordered = [...vars.filter((v) => v.changed), ...vars.filter((v) => !v.changed)];
  return (
    <div className="lc-vars">
      {ordered.map((v) => (
        <details key={v.name} className={v.changed ? "lc-var changed" : "lc-var"} open={v.changed}>
          <summary>
            <b>{v.name}</b>
            <span className="lc-shape">{v.shape}</span>
            {v.changed && <em>{v.name === "return" ? "returned" : "new"}</em>}
          </summary>
          {v.text && <pre>{v.text}</pre>}
        </details>
      ))}
    </div>
  );
}

function Example() {
  const r = d.row;
  return (
    <div className="lc-example">
      <p className="lc-q"><b>{r.source}</b> · {r.tokens} tokens · {r.passage} <span>{r.question}</span></p>
      <ol>
        {r.options.map((o, i) => (
          <li key={i} className={i === r.gold ? "gold" : ""}>{o}</li>
        ))}
      </ol>
    </div>
  );
}

export default function LossCode() {
  const [step, setStep] = useState(0);
  const s = d.steps[step];
  return (
    <section className="trie-section" id="losscode">
      <p className="eyebrow">5 · The loss, line by line</p>
      <h2>Stepping through <code>losses.compute</code></h2>
      <p className="lede small">
        The real code run once on one training row ({d.row.options.length} options, shuffled, gold in slot {d.row.gold + 1}), with
        {" "}{d.model} and a trained adapter. Every value below is what the line produced. A bold name with “new” was set or
        changed by the highlighted line; the other variables are folded.
      </p>
      <Example />
      <Stepper steps={d.steps} step={step} setStep={setStep} sectionId="losscode" />
      <div className="trie-grid lc-grid">
        <CodeCard step={s} />
        <div className="panel lc-panel">
          <AnimatePresence mode="wait">
            <motion.div key={step} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }}>
              <h3 className="panel-title">{s.title}</h3>
              <p className="panel-body"><Rich text={s.body} /></p>
              <Variables vars={s.vars} />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

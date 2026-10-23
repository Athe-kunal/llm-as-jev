import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Stepper } from "../components/Controls.jsx";
import mermaid from "mermaid";
import d from "../../data/loss_code.json";

mermaid.initialize({
  startOnLoad: false,
  theme: "base",
  // measure the labels in the font the CSS draws them in, so the boxes fit
  themeVariables: { fontFamily: '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: "11px" },
  flowchart: { htmlLabels: true, curve: "basis", nodeSpacing: 22, rankSpacing: 26, padding: 6 },
});

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

/** The data flow of compute as a Mermaid diagram. The step's node is lit and the
 *  nodes visited so far are tinted, so the line-by-line view stays tied to the whole. */
function Flow({ node, seen }) {
  const box = useRef(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let live = true;
    mermaid.render("lc-flow-svg", d.flow).then(({ svg }) => {
      if (!live || !box.current) return;
      box.current.innerHTML = svg;
      setReady(true);
    });
    return () => { live = false; };
  }, []);
  useEffect(() => {
    if (!ready) return;
    box.current.querySelectorAll("g.node").forEach((g) => {
      const id = g.id.replace(/^flowchart-/, "").replace(/-\d+$/, "");
      g.classList.toggle("on", id === node);
      g.classList.toggle("seen", id !== node && seen.has(id));
    });
  }, [ready, node, seen]);
  return <div className="lc-flow" ref={box} aria-label="Data flow of losses.compute" />;
}

/** The function being executed, with the line that just ran highlighted. */
function CodeCard({ step }) {
  const { lines, doc } = d.funcs[step.fn];
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
          <div key={`${step.fn}-${i}`} className={[
            "lc-line",
            i >= step.line && i <= step.line_end ? (step.call ? "on call" : "on") : "",
            doc && i >= doc[0] && i <= doc[1] ? "lc-doc" : "",
          ].join(" ")}>
            <span className="lc-ln">{i + 1}</span>
            <span className="lc-src">{doc && i >= doc[0] && i <= doc[1] ? text : <Source text={text} />}</span>
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
      {!ordered.some((v) => v.changed) && <p className="lc-same">This line changes no value.</p>}
      {ordered.map((v) => (
        <details key={v.name} className={v.changed ? "lc-var changed" : "lc-var"} open={v.changed}>
          <summary>
            <b>{v.name}</b>
            <span className="lc-shape">{v.shape}</span>
            {v.changed && <em>{v.tag || (v.name === "return" ? "returned" : "new")}</em>}
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
  const [showFlow, setShowFlow] = useState(true);
  const s = d.steps[step];
  const seen = new Set(d.steps.slice(0, step + 1).map((x) => x.node));
  return (
    <section className="trie-section" id="losscode">
      <p className="eyebrow">4 · The loss, line by line</p>
      <h2>Stepping through <code>losses.compute</code></h2>
      <p className="lede small">
        The real code run once on one training row ({d.row.options.length} options, shuffled, gold in slot {d.row.gold + 1}), with
        {" "}{d.model} and a trained adapter. Every value below is what the line produced. A bold name with “new” was set or
        changed by the highlighted line; the other variables are folded. The highlighted line is the one that just ran; when it calls
        a function, the steps inside that function come first and the calling line is highlighted once the call returns.
        The diagram shows where you are in the data flow.
      </p>
      <Example />
      <div className="lc-top">
        <div hidden={!showFlow}><Flow node={s.node} seen={seen} /></div>
        <Stepper steps={d.steps} step={step} setStep={setStep} sectionId="losscode">
          <button className="lc-toggle" onClick={() => setShowFlow(!showFlow)}>{showFlow ? "Hide diagram" : "Show diagram"}</button>
        </Stepper>
      </div>
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

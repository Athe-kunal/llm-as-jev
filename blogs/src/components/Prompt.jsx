import { useState } from "react";
import { motion } from "framer-motion";

// The prompt is assembled in this order; each segment kind maps to one step.
export const STEPS = [
  { name: "Chat template", note: "Qwen's own template, thinking off" },
  { name: "State", note: "the document parts, unnamed, a blank line apart" },
  { name: "Question", note: "always right before Options:" },
  { name: "Options", note: "[1] … [K], shuffled per training example" },
  { name: "Instruction", note: "one fixed line" },
  { name: "Prefill", note: "the assistant turn starts with Best answer: [" },
  { name: "Answer", note: "the model writes k] — digits, then ]" },
];

// Option lists longer than FOLD show their first SHOW lines and the gold one.
const FOLD = 20;
const SHOW = 5;

/** Step index of every segment ([kind, text, k?]) of a prompt. */
export function stepsOf(segs) {
  let sys = 0;
  let last = 0;
  return segs.map(([kind, text]) => {
    const step = {
      sys: sys++ === 0 ? 0 : 5,
      label: text === "State:" ? 1 : 3,
      part: 1,
      q: 2,
      opt: 3,
      ins: 4,
      pre: 5,
    }[kind] ?? last;
    last = step;
    return step;
  });
}

/** The rendered prompt; segments past `upTo` are hidden, `focus` dims the rest. */
export function Prompt({ ex, upTo = 6, focus = null, stagger = false }) {
  const [open, setOpen] = useState(false);
  const steps = stepsOf(ex.segs);
  const fold = ex.k > FOLD && !open;
  const out = [];
  ex.segs.forEach(([kind, text, k], i) => {
    if (kind === "opt" && fold && k >= SHOW && k !== ex.gold) {
      if (k === SHOW) {
        out.push(
          <button key="fold" className="fold" onClick={() => setOpen(true)}>
            + {ex.k - SHOW - (ex.gold >= SHOW ? 1 : 0)} more options
          </button>,
        );
      }
      return;
    }
    const cls = [
      "seg",
      kind,
      `s${steps[i]}`,
      steps[i] > upTo ? "off" : "",
      focus !== null && steps[i] !== focus ? "dim" : "",
      kind === "opt" && k === ex.gold ? "gold" : "",
    ].join(" ");
    const style = stagger ? { "--d": `${Math.min(out.length * 16, 720)}ms` } : undefined;
    // a document part is drawn as a block, which already ends its line
    const shown = kind === "sep" && ex.segs[i - 1]?.[0] === "part" ? text.replace(/^\n/, "") : text;
    out.push(
      <span key={i} className={cls} style={style}>
        {shown}
      </span>,
    );
  });
  return (
    <pre className={stagger ? "prompt rise" : "prompt"}>
      {out}
      {upTo >= 6 && <AnswerTokens ex={ex} dim={focus !== null && focus !== 6} />}
    </pre>
  );
}

/** The gold suffix, token by token, typed after the prefill. */
function AnswerTokens({ ex, dim }) {
  return (
    <span className={dim ? "answer dim" : "answer"}>
      {ex.suffix.map((t, i) => (
        <motion.span
          key={i}
          className="tok"
          initial={{ opacity: 0, y: -10, scale: 0.6 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ delay: 0.15 + i * 0.28, type: "spring", stiffness: 420, damping: 22 }}
        >
          {t}
        </motion.span>
      ))}
    </span>
  );
}

/** One cell per option; the gold one is lit. */
export function PositionStrip({ ex }) {
  const wide = ex.k <= 12;
  return (
    <div className={wide ? "strip wide" : "strip"} aria-label={`gold is option ${ex.gold + 1} of ${ex.k}`}>
      {Array.from({ length: ex.k }, (_, i) => (
        <span
          key={i}
          className={i === ex.gold ? "cell gold" : "cell"}
          style={{ "--d": `${Math.min(i * 5, 600)}ms` }}
        >
          {wide ? i + 1 : ""}
        </span>
      ))}
    </div>
  );
}

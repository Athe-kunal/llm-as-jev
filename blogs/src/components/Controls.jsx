import { useEffect } from "react";
import { LayoutGroup, motion } from "framer-motion";

/** Prev / next buttons and step dots. Arrow keys work while `sectionId` is
 *  the section in the middle of the screen. `children` sit at the right. */
export function Stepper({ steps, step, setStep, sectionId, children }) {
  const last = steps.length - 1;
  const go = (s) => setStep(Math.max(0, Math.min(last, s)));

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest("input, textarea")) return;
      const box = document.getElementById(sectionId)?.getBoundingClientRect();
      const mid = window.innerHeight / 2;
      if (!box || box.top > mid || box.bottom < mid) return;
      if (e.key === "ArrowRight") go(step + 1);
      if (e.key === "ArrowLeft") go(step - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div className="stepper">
      <button className="nav-btn" onClick={() => go(step - 1)} disabled={step === 0} aria-label="Previous step">
        ←
      </button>
      <div className="dots" role="tablist">
        {steps.map((s, i) => (
          <button
            key={i}
            role="tab"
            aria-selected={i === step}
            aria-label={s.title}
            title={s.title}
            className={i === step ? "sdot on" : i < step ? "sdot past" : "sdot"}
            onClick={() => go(i)}
          />
        ))}
      </div>
      <button className="nav-btn" onClick={() => go(step + 1)} disabled={step === last} aria-label="Next step">
        →
      </button>
      <span className="step-count">
        {step + 1} / {steps.length}
      </span>
      {children}
    </div>
  );
}

/** A segmented control; options are [value, label] pairs. */
export function Toggle({ id, options, value, onChange, label }) {
  return (
    <LayoutGroup id={id}>
      <div className="ex-tabs model-toggle" role="tablist" aria-label={label}>
        {options.map(([v, text]) => (
          <button key={v} role="tab" aria-selected={v === value} onClick={() => onChange(v)}>
            {v === value && <motion.span layoutId={`${id}-under`} className="under" />}
            {text}
          </button>
        ))}
      </div>
    </LayoutGroup>
  );
}

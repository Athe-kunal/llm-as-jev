import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

/** TeX rendered by KaTeX, inline or as a centred display block. */
export function Tex({ children, block = false }) {
  const html = katex.renderToString(children, { displayMode: block, throwOnError: false });
  return <span className={block ? "tex block" : "tex"} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** One labelled part of a step: "Definition", "In code" or "On this row". */
export function Part({ label, children }) {
  return (
    <div className="part-block">
      <span className="part-label">{label}</span>
      <div className="part-body">{children}</div>
    </div>
  );
}

/** The row's sentences as running text, each tinted by value[j] (diverging
 *  around 0), dropped sentences struck through, gold sentences underlined,
 *  sentences in `marked` (a Set) outlined; scrollToMarked scrolls the box to
 *  the first outlined one. */
export function Doc({ sentences, value = null, keep = null, selected = null, marked = null, onSelect = null, maxHeight = 520, scrollToMarked = false }) {
  const max = value ? Math.max(...value.map(Math.abs), 1e-9) : 1;
  const box = useRef(null);
  useEffect(() => {
    // bring the first outlined sentence into view inside the box (not the page)
    const el = scrollToMarked && box.current?.querySelector(".dsent.marked");
    if (el) box.current.scrollTo({ top: Math.max(el.offsetTop - 40, 0), behavior: "smooth" });
  }, [scrollToMarked, marked]);
  return (
    <div className="doc" style={{ maxHeight }} ref={box}>
      {sentences.map((s, j) => {
        const v = value ? value[j] / max : 0;
        const style = value
          ? { background: `color-mix(in srgb, ${v >= 0 ? "var(--c6)" : "var(--c5)"} ${Math.round(Math.abs(v) * 70)}%, transparent)` }
          : undefined;
        const cls = ["dsent", s.gold ? "gold" : "", keep && !keep[j] ? "dropped" : "", selected === j ? "sel" : "", marked?.has(j) ? "marked" : "", onSelect ? "click" : ""].join(" ");
        return (
          <span key={j}>
            {j > 0 && s.sep}
            <span className={cls} style={style} onClick={onSelect ? () => onSelect(j) : undefined} title={value ? `#${j} · ${value[j].toFixed(3)}` : `#${j}`}>
              {s.text}
            </span>
          </span>
        );
      })}
    </div>
  );
}

/** One part's text with every token tinted by its score w (green +, red −),
 *  scaled to the 99th percentile of |w| so a few attention-sink tokens do not
 *  wash out the rest; tokens inside a gold range get the gold underline, the
 *  `top` token indices are outlined, and the box scrolls to the first of them. */
export function TokenDoc({ text, tokens, gold = [], top = new Set(), maxHeight = 380 }) {
  const box = useRef(null);
  const abs = tokens.map((t) => Math.abs(t.w)).sort((a, b) => a - b);
  const scale = abs[Math.floor(0.99 * (abs.length - 1))] || 1e-12;
  useEffect(() => {
    const el = box.current?.querySelector(".tk.marked");
    if (el) box.current.scrollTo({ top: Math.max(el.offsetTop - 40, 0), behavior: "smooth" });
  }, [tokens]);
  const out = [];
  let cur = 0;
  [...tokens].sort((a, b) => a.start - b.start).forEach((t, i) => {
    if (t.start > cur) out.push(text.slice(cur, t.start));
    const v = Math.max(-1, Math.min(1, t.w / scale));
    const isGold = gold.some((g) => t.start < g.end && t.end > g.start);
    out.push(
      <span key={i} className={["tk", isGold ? "gold" : "", top.has(t.index) ? "marked" : ""].join(" ")}
            title={`${t.w.toExponential(2)}`}
            style={{ background: `color-mix(in srgb, ${v >= 0 ? "var(--c6)" : "var(--c5)"} ${Math.round(Math.abs(v) * 75)}%, transparent)` }}>
        {text.slice(Math.max(t.start, cur), t.end)}
      </span>,
    );
    cur = Math.max(cur, t.end);
  });
  if (cur < text.length) out.push(text.slice(cur));
  return (
    <div className="doc tokdoc" style={{ maxHeight }} ref={box}>
      {out}
    </div>
  );
}

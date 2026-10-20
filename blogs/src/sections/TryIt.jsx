import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Doc, TokenDoc } from "../components/Math.jsx";

// The model behind /api (blogs/server.py): the 0.6B λ = 0.01 adapter AT2's θ was trained for.
const MODEL = "Qwen3-0.6B, λ = 0.01 adapter";
const METHODS = [
  { id: "at2", name: "AT2", note: "one forward pass, learned head weights" },
  { id: "contextcite", name: "ContextCite", note: "64 ablated passes + Lasso" },
];
const SHOW_OPTIONS = 8; // options listed: the most probable ones, plus gold and the answer

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `${r.status} ${r.statusText}`);
  return r.json();
}

export default function TryIt() {
  const [datasets, setDatasets] = useState([]);
  const [source, setSource] = useState("contractnli");
  const [spansOnly, setSpansOnly] = useState(true);
  const [row, setRow] = useState(null);
  const [attr, setAttr] = useState({});
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api("/api/datasets").then(setDatasets).catch((e) => setError(e.message));
  }, []);

  const next = useCallback(async () => {
    setBusy(true);
    setError(null);
    setAttr({});
    try {
      const { id } = await api(`/api/sample?source=${source}&spans=${spansOnly ? 1 : 0}`);
      setRow(await api(`/api/row?id=${encodeURIComponent(id)}`));
      // AT2 first (≈0.2 s), then ContextCite (seconds); the server runs one at a time
      for (const m of METHODS) {
        setAttr((a) => ({ ...a, [m.id]: "loading" }));
        const res = await api(`/api/attribute?id=${encodeURIComponent(id)}&method=${m.id}`);
        setAttr((a) => ({ ...a, [m.id]: res }));
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }, [source, spansOnly]);

  useEffect(() => {
    if (datasets.length) next();
  }, [datasets.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const chosen = datasets.find((d) => d.id === source);
  return (
    <section className="trie-section" id="try">
      <p className="eyebrow">7 · Try it</p>
      <h2>Random eval rows, live</h2>
      <p className="lede small">
        {MODEL}, running on this machine. Each row is answered with the guided decoder, then explained by both attribution
        methods. Amber underline: the dataset's gold evidence; outlined: each method's top 3 sentences, scrolled into view; green and red tint: positive and negative weights.
      </p>
      <div className="try-bar">
        <label className="try-field">
          <span>Dataset</span>
          <select id="try-source" value={source} onChange={(e) => setSource(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.id} ({d.rows}
                {d.withSpans ? `, ${d.withSpans} with evidence` : ""})
              </option>
            ))}
          </select>
        </label>
        <label className="try-check">
          <input id="try-spans" type="checkbox" checked={spansOnly} disabled={!chosen?.withSpans}
                 onChange={(e) => setSpansOnly(e.target.checked)} />
          only rows with gold evidence
        </label>
        <button className="try-go" onClick={next} disabled={busy}>
          {busy ? "Running…" : "Random row"}
        </button>
        {row && <code className="rowid">{row.id}</code>}
      </div>
      {error && (
        <div className="pending">
          {error}. Is the model API running? Start it with <code>npm run api</code> in <code>blogs/</code>.
        </div>
      )}
      <AnimatePresence mode="wait">
        {row && (
          <motion.div key={row.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
            <Answer row={row} />
            <div className="try-grid">
              {METHODS.map((m) => (
                <Explain key={m.id} method={m} row={row} res={attr[m.id]} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}

function Answer({ row }) {
  const order = [...row.q.keys()].sort((a, b) => row.q[b] - row.q[a]);
  const shown = [...new Set([...order.slice(0, SHOW_OPTIONS), row.gold, row.answer])].sort((a, b) => row.q[b] - row.q[a]);
  const right = row.answer === row.gold;
  return (
    <div className="try-answer">
      <div className="try-q">{row.question}</div>
      <div className="try-opts">
        {shown.map((k) => (
          <div key={k} className={["qrow", k === row.answer ? "on" : "", k === row.gold ? "goldopt" : ""].join(" ")}>
            <span className="qlab">
              [{k + 1}] {row.options[k]}
              {k === row.answer && <b className="tag">model</b>}
              {k === row.gold && <b className="tag gold">gold</b>}
            </span>
            <span className="qtrack">
              <motion.span className="qfill" initial={{ width: 0 }} animate={{ width: `${row.q[k] * 100}%` }} transition={{ duration: 0.5 }} />
            </span>
            <span className="qv">{row.q[k].toFixed(3)}</span>
          </div>
        ))}
        {row.options.length > shown.length && <span className="viz-label">+ {row.options.length - shown.length} less likely options</span>}
      </div>
      <div className={right ? "verdict ok" : "verdict bad"}>
        {right ? "correct" : "wrong"} · answer {row.answer + 1}] · {row.seconds.toFixed(2)} s · {row.sentences.length} sentences
      </div>
    </div>
  );
}

const TOP_TOKENS = 10; // tokens outlined in the token view

function Explain({ method, row, res }) {
  const loading = res === "loading" || res === undefined;
  const [view, setView] = useState("tokens"); // AT2 only: tokens (as the paper) or sentences
  const tokens = !loading && res.tokens;
  return (
    <div className="try-panel">
      <div className="try-head">
        <div>
          <h3>{method.name}</h3>
          <span className="viz-label">{method.note}</span>
        </div>
        <div className="try-head-right">
          {tokens && (
            <div className="ex-tabs view-toggle" role="tablist" aria-label="Highlight">
              {["tokens", "sentences"].map((v) => (
                <button key={v} role="tab" aria-selected={v === view} onClick={() => setView(v)}>{v}</button>
              ))}
            </div>
          )}
          {res && res !== "loading" && res.w && <span className="try-time">{res.seconds.toFixed(2)} s</span>}
        </div>
      </div>
      {loading ? (
        <div className="try-loading">
          <span className="spinner" /> {method.id === "contextcite" ? "scoring 64 ablated prompts…" : "reading attention…"}
        </div>
      ) : res.w === null ? (
        <div className="pending">{res.reason}</div>
      ) : (
        <>
          <Metrics m={res.metrics} />
          {tokens && view === "tokens" ? (
            <TokenView row={row} tokens={tokens} />
          ) : row.parts.map((name, p) => {
            const idx = row.sentences.flatMap((s, j) => (s.part === p ? [j] : []));
            if (!idx.length) return null;
            const local = new Set(idx.flatMap((j, i) => (res.top.includes(j) ? [i] : [])));
            return (
              <div key={p} className="try-part">
                {row.parts.length > 1 && <span className="viz-label">{name}</span>}
                <Doc sentences={idx.map((j) => row.sentences[j])} value={idx.map((j) => res.w[j])} marked={local} maxHeight={380} scrollToMarked />
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}

/** AT2's score for every document token, part by part; the top tokens outlined. */
function TokenView({ row, tokens }) {
  const top = new Set([...tokens].sort((a, b) => b.w - a.w).slice(0, TOP_TOKENS).map((t) => t.index));
  return (
    <>
      <span className="viz-label">token scores θ·φᵢ; a sentence's score is the sum of its tokens'; top {TOP_TOKENS} outlined</span>
      {row.partTexts.map((text, p) => (
        <div key={p} className="try-part">
          {row.parts.length > 1 && <span className="viz-label">{row.parts[p]}</span>}
          <TokenDoc text={text} tokens={tokens.filter((t) => t.part === p)} top={top}
                    gold={row.sentences.filter((s) => s.part === p && s.gold)} />
        </div>
      ))}
    </>
  );
}

function Metrics({ m }) {
  if (!m) return <span className="viz-label">no gold evidence on this row</span>;
  const items = [
    ["AP", m.ap],
    ["P@3", m.precision],
    ["R@3", m.recall],
    ["top-1 gold", m.top1],
  ];
  return (
    <div className="try-metrics">
      {items.map(([k, v]) => (
        <span key={k} className={v >= 0.5 ? "good" : ""}>
          {k} <b>{v.toFixed(2)}</b>
        </span>
      ))}
    </div>
  );
}

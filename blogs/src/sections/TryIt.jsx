import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Doc } from "../components/Math.jsx";
import { Toggle } from "../components/Controls.jsx";

// auto: about two per sentence, a power of two in [32, 256] (masked_cite.ablation_count); or fixed
const ABLATIONS = [["auto", "auto"], ["64", "64"], ["256", "256"]];
// reader: a dropped sentence is hidden from the question and answer only (the document is shared, one pass);
// full: hidden from every later token, the document included (the document recomputed per ablation)
const REMOVAL = [["reader", "reader-only"], ["full", "full masking"]];
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
  const [attr, setAttr] = useState();
  const [ablations, setAblations] = useState("auto");
  const [removal, setRemoval] = useState("reader");
  const query = (id, a, m) => `/api/attribute?id=${encodeURIComponent(id)}&ablations=${a}&removal=${m}`;
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api("/api/datasets").then(setDatasets).catch((e) => setError(e.message));
  }, []);

  const chosen = datasets.find((d) => d.id === source);
  // the evidence filter applies only where the dataset's eval rows have gold spans
  const withSpans = spansOnly && Boolean(chosen?.withSpans);

  const next = useCallback(async () => {
    setBusy(true);
    setError(null);
    setAttr(undefined);
    try {
      const { id } = await api(`/api/sample?source=${source}&spans=${withSpans ? 1 : 0}`);
      setRow(await api(`/api/row?id=${encodeURIComponent(id)}`));
      setAttr("loading");
      setAttr(await api(query(id, ablations, removal)));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }, [source, withSpans, ablations, removal]);

  // re-run ContextCite on the current row with another ablation count or removal
  const rerun = useCallback(async (n, m) => {
    setAblations(n);
    setRemoval(m);
    if (!row) return;
    setBusy(true);
    setError(null);
    setAttr("loading");
    try {
      setAttr(await api(query(row.id, n, m)));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }, [row]);

  useEffect(() => {
    if (datasets.length) next();
  }, [datasets.length]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="trie-section" id="try">
      <p className="eyebrow">9 · Try it</p>
      <h2>Random eval rows, live</h2>
      <p className="lede small">
        {row ? row.model : "The trained decision model"}, running on this machine. Each row is answered with the guided decoder, then explained by ContextCite in its
        one-pass form: the prompt is packed with many copies of the question (about two per sentence, 32 to 256), each blind to a
        random half of the sentences. Amber
        underline: the dataset's gold evidence; outlined: the top 3 sentences, scrolled into view; green and red tint: sentences
        that raise and lower the answer's logit.
      </p>
      <div className="try-bar">
        <label className="try-field">
          <span>Dataset</span>
          <select id="try-source" value={source} onChange={(e) => setSource(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.id}{d.withSpans ? " (with evidence)" : ""}
              </option>
            ))}
          </select>
        </label>
        <label className="try-check">
          <input id="try-spans" type="checkbox" checked={withSpans} disabled={!chosen?.withSpans}
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
          {error}
          {/fetch|NetworkError|50[234]/.test(error) && (
            <>. Is the model API running? Start it with <code>npm run api</code> in <code>blogs/</code>.</>
          )}
        </div>
      )}
      <AnimatePresence mode="wait">
        {row && (
          <motion.div key={row.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
            <Answer row={row} />
            <Explain row={row} res={attr} ablations={ablations} removal={removal} onRerun={rerun} busy={busy} />
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

function Explain({ row, res, ablations, removal, onRerun, busy }) {
  const loading = res === "loading" || res === undefined;
  return (
    <div className="try-panel">
      <div className="try-head">
        <div>
          <h3>ContextCite</h3>
          <span className="viz-label">
            one packed forward: the decision + {loading ? (ablations === "auto" ? "auto" : ablations) : res.ablations} ablations
            {ablations === "auto" && !loading && res.w && ` (auto, ${row.sentences.length} sentences)`}
            {removal === "full"
              ? " (sentences hidden from every later token: each ablation's kept tokens, FA2, shared prefixes once)"
              : " (sentences hidden from the question and answer; the document shared)"}, then Lasso
          </span>
        </div>
        <div className="try-head-right">
          <Toggle id="removal" label="Removal" value={removal} options={REMOVAL}
                  onChange={(m) => !busy && m !== removal && onRerun(ablations, m)} />
          <Toggle id="ablations" label="Ablations" value={ablations} options={ABLATIONS}
                  onChange={(n) => !busy && n !== ablations && onRerun(n, removal)} />
          {!loading && res.w && <span className="try-time">{res.seconds.toFixed(2)} s</span>}
        </div>
      </div>
      {loading ? (
        <div className="try-loading">
          <span className="spinner" /> scoring the decision and {ablations === "auto" ? "its" : ablations} ablations
          {removal === "full" ? " (full masking: the kept tokens of each ablation, FA2 with shared prefixes; long rows take 5–15 s)…" : " in one pass…"}
        </div>
      ) : res.w === null ? (
        <>
          <div className="pending">{res.reason}</div>
          {row.partTexts.map((text, p) => (
            <div key={p} className="try-part">
              <span className="viz-label">{row.parts[p]}</span>
              <div className="try-plain">{text}</div>
            </div>
          ))}
        </>
      ) : (
        <>
          <Metrics m={res.metrics} />
          {res.answer !== row.answer && (
            <span className="viz-label">
              the packed pass picks answer {res.answer + 1}] (the decoder above: {row.answer + 1}]): a near tie, bf16 rounding
            </span>
          )}
          {row.parts.map((name, p) => {
            const idx = row.sentences.flatMap((s, j) => (s.part === p ? [j] : []));
            if (!idx.length) return null;
            const local = new Set(idx.flatMap((j, i) => (res.top.includes(j) ? [i] : [])));
            return (
              <div key={p} className="try-part">
                {row.parts.length > 1 && <span className="viz-label">{name}</span>}
                <Doc sentences={idx.map((j) => row.sentences[j])} value={idx.map((j) => res.w[j])} marked={local} maxHeight={520} scrollToMarked />
              </div>
            );
          })}
        </>
      )}
    </div>
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

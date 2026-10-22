import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

// One row's path through src/span_generation/: propose on the top lane,
// score on the bottom lane, both through the same vLLM server.
const W = 1230, H = 560;
const N = {
  row:    { x: 20,   y: 60,  w: 150, h: 64, title: "Reason+Long-12k row", sub: "question + gold", kind: "data" },
  split:  { x: 200,  y: 60,  w: 140, h: 64, title: "split_sources", sub: "nltk · merged", kind: "step" },
  prompt: { x: 370,  y: 60,  w: 140, h: 64, title: "cite.j2", sub: "+ gold answer", kind: "step" },
  server: { x: 590,  y: 36,  w: 190, h: 336, title: "vLLM · Qwen3.6-27B", sub: "one server per GPU", kind: "server" },
  sampled:{ x: 860,  y: 60,  w: 150, h: 64, title: "10 citation sets", sub: "T 1.2 · top-p 0.9", kind: "step" },
  filter: { x: 1040, y: 60,  w: 170, h: 64, title: "dedupe · ≤ 384 tok", sub: "m candidate sets", kind: "step" },
  ablate: { x: 1040, y: 284, w: 170, h: 64, title: "ablate each S", sub: "x · x∖S · S only", kind: "step" },
  q:      { x: 370,  y: 284, w: 140, h: 64, title: "q(gold | ·)", sub: "softmax of sₖ", kind: "step" },
  reward: { x: 200,  y: 284, w: 140, h: 64, title: "reward(S)", sub: "suff. + nec.", kind: "reward" },
  best:   { x: 20,   y: 284, w: 150, h: 64, title: "argmax S", sub: "→ row spans", kind: "step" },
  out:    { x: 20,   y: 456, w: 230, h: 64, title: "labels.jsonl", sub: "candidates · q(gold | x)", kind: "data" },
  gate:   { x: 330,  y: 456, w: 260, h: 64, title: "validate on human spans", sub: "ContractNLI · MuSiQue · ProofWriter", kind: "gate" },
  head:   { x: 700,  y: 456, w: 200, h: 64, title: "citation head", sub: "trained on the labels", kind: "data" },
};
const PORT_TOP = 92, PORT_BOT = 316; // y of the server's two endpoints

const r = (n) => N[n].x + N[n].w, l = (n) => N[n].x, cy = (n) => N[n].y + N[n].h / 2;
const cx = (n) => N[n].x + N[n].w / 2, top = (n) => N[n].y, bot = (n) => N[n].y + N[n].h;
const mid = (a, b) => (a + b) / 2;
// labels only where the hop carries information the boxes do not
const EDGES = [
  { d: `M${r("row")},${cy("row")} H${l("split")}` },
  { d: `M${r("split")},${cy("split")} H${l("prompt")}` },
  { d: `M${r("prompt")},${PORT_TOP} H${l("server")}`, label: "chat", at: [mid(r("prompt"), l("server")), PORT_TOP - 9], hot: true },
  { d: `M${r("server")},${PORT_TOP} H${l("sampled")}`, label: "n = 10", at: [mid(r("server"), l("sampled")), PORT_TOP - 9], hot: true },
  { d: `M${r("sampled")},${cy("sampled")} H${l("filter")}` },
  { d: `M${cx("filter")},${bot("filter")} V${top("ablate")}`, label: "m sets", at: [cx("filter") + 34, mid(bot("filter"), top("ablate")) + 4] },
  { d: `M${l("ablate")},${PORT_BOT} H${r("server")}`, label: "1 + 2m prompts × K options", at: [mid(l("ablate"), r("server")), PORT_BOT - 9], hot: true },
  { d: `M${l("server")},${PORT_BOT} H${r("q")}`, label: "log p", at: [mid(l("server"), r("q")), PORT_BOT - 9], hot: true },
  { d: `M${l("q")},${cy("q")} H${r("reward")}` },
  { d: `M${l("reward")},${cy("reward")} H${r("best")}` },
  { d: `M${cx("best")},${bot("best")} V${top("out")}`, label: "write", at: [cx("best") + 26, mid(bot("best"), top("out")) + 4] },
  { d: `M${r("out")},${cy("out")} H${l("gate")}` },
  { d: `M${r("gate")},${cy("gate")} H${l("head")}`, label: "if it wins", at: [mid(r("gate"), l("head")), cy("gate") - 9] },
];

const DETAILS = {
  row: ["The input", "A training row: its document parts, the question, the options and the gold label. Only rows with at least two sentences are labelled.", "data.load_rows", "multi-sentence rows of the 12 datasets"],
  split: ["Sentences", "The same sentences ContextCite and AT2 use: nltk per line, with content-free lines (blank form fields, checkboxes) merged into a neighbour, so every candidate is a real sentence.", "context_cite.split_sources", "min_words = 2, min_alnum = 0.5"],
  prompt: ["The citation prompt", "Numbered sentences, the question and the gold answer; the model is asked for the smallest set that lets a reader reach the answer and without which it is no longer supported, as JSON.", "cite.j2 · best_of_n.propose", "thinking off"],
  server: ["One server, two jobs", "The labelling model both proposes citation sets (chat completions) and scores them (completions with the prompt's log-probabilities echoed back), as in SelfCite where the model judges its own citations. Prefix caching lets the K option suffixes of one prompt share its prefix.", "serve.sh", "vLLM 0.29 · one server per GPU · --enable-prefix-caching"],
  sampled: ["Best-of-N samples", "N citation sets per row, sampled for diversity; replies that are not valid JSON with in-range sentence numbers are dropped.", "best_of_n.propose", "n = 10, temperature 1.2, top-p 0.9 (the paper)"],
  filter: ["Candidates", "Duplicate sets are merged (their vote count is kept); sets citing more than 384 tokens are dropped unless they are a single sentence, the paper's length control.", "best_of_n.label", "--max-cited-tokens 384"],
  ablate: ["Ablated prompts", "For each candidate S: the decision prompt without S and with S only, plus the full context once per row: 1 + 2m prompts, each followed by every option's suffix k].", "context_cite.ablate · answers.encode", "options in release order"],
  q: ["Tree probability of the gold answer", "sₖ is the sum of the suffix tokens' log-probabilities (Eq. 1, the same scores as the trie decoder); q = softmax(s). Checked against the local trie scorer: |Δq| ≤ 0.035.", "best_of_n.option_scores", "echo=True, logprobs=0, token ids sent"],
  reward: ["SelfCite reward", "reward(S) = log q(gold | S only) − log q(gold | x∖S): sufficiency (S alone keeps the answer) plus necessity (removing S loses it), with equal weight.", "best_of_n.rewards", "no size penalty; the 384-token cap instead"],
  best: ["The label", "The highest-reward set becomes the row's spans, in the dataset's own span format.", "context_cite.to_spans", "—"],
  out: ["What is kept", "Every candidate with votes, necessity, sufficiency and reward, plus q(gold | x) and the model's own answer: the reward is only meaningful where the model believes the gold answer, so the cut-off is chosen on the validation rows.", "best_of_n.run", "async, rows spread over servers"],
  gate: ["Validate before labelling everything", "On rows with human spans, compare the winning sets with the human evidence, and with plain voting over the samples. Only if best-of-N wins do the other datasets get labelled.", "(to do)", "span_metrics: P, R, F_β, AP"],
  head: ["The consumer", "The labels train the fast citation head (one forward pass at inference), which so far learned from ablation correlations only (held-out ρ 0.23).", "modeling.citation · train.train_citation", "—"],
};

export default function SpanPipeline() {
  const [sel, setSel] = useState("server");
  const d = DETAILS[sel];
  return (
    <section className="trie-section" id="spans">
      <p className="eyebrow">8 · Span data</p>
      <h2>Labelling evidence with best-of-N</h2>
      <p className="lede small">
        <code>src/span_generation/</code>: a 27B model proposes citation sets and then scores them on our decision prompt;
        the best set becomes the row's evidence. Click a step for what it does and where it lives.
      </p>
      <figure className="pipe-fig">
        <svg viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label="Span generation: sentences are numbered into a prompt, the vLLM server samples ten citation sets, the sets are filtered and ablated, the same server scores the ablated prompts, the reward picks the best set, labels are validated against human spans and train the citation head.">
          <defs>
            <marker id="pipe-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" className="pipe-head" />
            </marker>
          </defs>
          <text x={l("prompt")} y={24} className="lane">propose</text>
          <text x={l("q")} y={bot("q") + 24} className="lane">score</text>
          {EDGES.map((e, i) => (
            <g key={i}>
              <motion.path d={e.d} className={e.hot ? "pipe-edge hot" : "pipe-edge"} markerEnd="url(#pipe-arrow)"
                           initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.25 + i * 0.12, duration: 0.5 }} />
              {e.hot && <path d={e.d} className="pipe-flow" />}
              {e.label && <text x={e.at[0]} y={e.at[1]} textAnchor="middle" className="pipe-label">{e.label}</text>}
            </g>
          ))}
          {Object.entries(N).map(([k, n], i) => (
            <motion.g key={k} className={`pipe-node ${n.kind} ${sel === k ? "on" : ""}`} onClick={() => setSel(k)}
                      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }}
                      role="button" tabIndex={0} onKeyDown={(ev) => ev.key === "Enter" && setSel(k)}>
              <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={12} />
              {n.kind === "server" ? (
                <>
                  <text x={n.x + n.w / 2} y={n.y + 26} textAnchor="middle" className="t1">{n.title}</text>
                  <text x={n.x + n.w / 2} y={n.y + 44} textAnchor="middle" className="t2">{n.sub}</text>
                  <rect x={n.x + 14} y={PORT_TOP - 20} width={n.w - 28} height={40} rx={8} className="port" />
                  <text x={n.x + n.w / 2} y={PORT_TOP + 4} textAnchor="middle" className="t2">chat completions</text>
                  <rect x={n.x + 14} y={PORT_BOT - 20} width={n.w - 28} height={40} rx={8} className="port" />
                  <text x={n.x + n.w / 2} y={PORT_BOT + 4} textAnchor="middle" className="t2">echoed log-probs</text>
                  <text x={n.x + n.w / 2} y={(PORT_TOP + PORT_BOT) / 2 - 6} textAnchor="middle" className="t3">same model</text>
                  <text x={n.x + n.w / 2} y={(PORT_TOP + PORT_BOT) / 2 + 12} textAnchor="middle" className="t3">proposes and scores</text>
                </>
              ) : (
                <>
                  <text x={n.x + n.w / 2} y={n.y + 28} textAnchor="middle" className="t1">{n.title}</text>
                  <text x={n.x + n.w / 2} y={n.y + 47} textAnchor="middle" className="t2">{n.sub}</text>
                </>
              )}
            </motion.g>
          ))}
        </svg>
        <figcaption>
          One row's path. Top lane: the server proposes ten citation sets. Bottom lane: every set is ablated and the same
          server scores the gold answer on each prompt; the reward picks the label.
        </figcaption>
      </figure>
      <AnimatePresence mode="wait">
        <motion.div key={sel} className="pipe-detail" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
          <h3>{d[0]}</h3>
          <p>{d[1]}</p>
          <div className="pipe-meta">
            <span><b>In code</b> <code>{d[2]}</code></span>
            <span><b>Settings</b> {d[3]}</span>
          </div>
        </motion.div>
      </AnimatePresence>
    </section>
  );
}

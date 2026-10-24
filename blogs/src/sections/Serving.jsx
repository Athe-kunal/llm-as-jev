import { useState } from "react";
import { motion } from "framer-motion";

// PrefillOnly (SOSP '25), Fig. 5: A and B arrive, then C and D while A runs.
// A and D share a prefix, B and C share another; the cache holds one prefix.
const REQ = {
  A: { len: 4, group: "x", arrive: 0 },
  B: { len: 6, group: "y", arrive: 0 },
  C: { len: 5, group: "y", arrive: 1 },
  D: { len: 7, group: "x", arrive: 1 },
};
const SHARED = 3;

function simulate(policy) {
  let waiting = Object.keys(REQ).filter((k) => REQ[k].arrive === 0);
  const later = Object.keys(REQ).filter((k) => REQ[k].arrive === 1);
  let cache = null, step = 0;
  const out = [];
  while (waiting.length) {
    const work = (k) => REQ[k].len - (cache === REQ[k].group ? SHARED : 0);
    const order = {
      fifo: (a, b) => "ABCD".indexOf(a) - "ABCD".indexOf(b),
      spf: (a, b) => REQ[a].len - REQ[b].len,
      cont: (a, b) => work(a) - work(b),
    }[policy];
    const next = [...waiting].sort(order)[0];
    const hit = cache === REQ[next].group;
    out.push({ k: next, hit, work: work(next) });
    cache = REQ[next].group;
    waiting = waiting.filter((k) => k !== next);
    if (++step === 1) waiting.push(...later);
  }
  return out;
}

const POLICIES = [
  ["fifo", "FIFO", "SGLang's default"],
  ["spf", "SPF at arrival", "estimate once"],
  ["cont", "SPF, re-estimated", "--schedule-policy shortest-prefill-first"],
];

const CHANGES = [
  ["Scheduling", "shortest uncached prefill first, re-estimated at every step", "fcfs", "PrefillOnly §6"],
  ["Chunked prefill", "off: a prompt is one attention call", "32k chunks", "PrefillOnly §2.6"],
  ["Prefill batch", "32k tokens (CONTEXT)", "64k tokens", "PrefillOnly §6.1"],
  ["Decode CUDA graphs", "off: score requests never decode", "captured", "prefill-only"],
  ["Full prompt", "scored with the ablations", "one serial call first", "fewer round trips"],
  ["Trie nodes", "per chunk, right after its prompts", "after all of the row's prompts", "cache hits before eviction"],
  ["vLLM labeller", "logprob_token_ids per trie node", "echo=True (skips the prefix cache)", "LMSYS: explicit labels"],
];

export default function Serving() {
  const [policy, setPolicy] = useState("cont");
  const run = simulate(policy);
  const total = run.reduce((a, r) => a + r.work, 0);
  return (
    <section className="trie-section" id="serving">
      <p className="eyebrow">11 · Faster inference</p>
      <h2>Serving a prefill-only workload</h2>
      <p className="lede small">
        A decision or a ContextCite ablation needs one forward pass and a few label log-probabilities, never a
        decode. A row's 33–257 ablated prompts share prefixes, so the order in which the server runs them sets how
        much of each prompt comes from the radix cache.
      </p>
      <div className="viz-card sched-card">
        <div className="ex-tabs" role="tablist">
          {POLICIES.map(([k, name]) => (
            <button key={k} role="tab" aria-selected={policy === k} onClick={() => setPolicy(k)}>
              {policy === k && <motion.span layoutId="sched-under" className="under" />}
              {name}
            </button>
          ))}
        </div>
        <p className="viz-label">{POLICIES.find(([k]) => k === policy)[2]}</p>
        <div className="sched">
          {run.map((r, i) => (
            <motion.div key={`${policy}${r.k}`} className={`sched-job g${REQ[r.k].group}`} layout
                        initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.08 }}
                        style={{ flexGrow: REQ[r.k].len }}>
              {r.hit && <span className="sched-hit" style={{ width: `${(100 * SHARED) / REQ[r.k].len}%` }} />}
              <b>{r.k}</b>
            </motion.div>
          ))}
        </div>
        <p className="formula">
          cache hits <b>{run.filter((r) => r.hit).length}</b> · tokens prefilled <b>{total}</b> of{" "}
          {Object.values(REQ).reduce((a, r) => a + r.len, 0)}
        </p>
        <p className="eq-note">
          Lengths A &lt; C &lt; B &lt; D; colour is the shared prefix, hatched is read from the cache. Only the
          re-estimating scheduler sees that D became short once A's prefix was cached.
        </p>
      </div>
      <div className="change-table" role="table">
        <div className="change-row head" role="row">
          <span>What</span><span>Now</span><span>Before</span><span>Why</span>
        </div>
        {CHANGES.map(([what, now, before, why]) => (
          <div key={what} className="change-row" role="row">
            <b>{what}</b><span>{now}</span><span className="dimtext">{before}</span><span className="dimtext">{why}</span>
          </div>
        ))}
      </div>
      <p className="eq-note">
        Details, numbers and how to measure: <code>docs/inference_performance.md</code>.
      </p>
    </section>
  );
}

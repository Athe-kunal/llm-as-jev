import PromptAndData from "./sections/PromptAndData.jsx";
import TrieLoss from "./sections/TrieLoss.jsx";
import Anchors from "./sections/Anchors.jsx";
import LossCode from "./sections/LossCode.jsx";
import Results from "./sections/Results.jsx";
import Ablation from "./sections/Ablation.jsx";
import ContextCite from "./sections/ContextCite.jsx";
import AT2 from "./sections/AT2.jsx";
import TryIt from "./sections/TryIt.jsx";
import SpanPipeline from "./sections/SpanPipeline.jsx";
import Serving from "./sections/Serving.jsx";

// Sections in reading order; later ones are added as the walkthrough grows.
const SECTIONS = [
  { n: 1, title: "Prompt and data", id: "top", ready: true },
  { n: 2, title: "Trie and tree loss", id: "trie", ready: true },
  { n: 3, title: "KL anchors", id: "anchors", ready: true },
  { n: 4, title: "Loss in code", id: "losscode", ready: true },
  { n: 5, title: "Results", id: "results", ready: true },
  { n: 6, title: "Loss ablation", id: "ablation", ready: true },
  { n: 7, title: "ContextCite", id: "contextcite", ready: true },
  { n: 8, title: "AT2", id: "at2", ready: true },
  { n: 9, title: "Try it", id: "try", ready: true },
  { n: 10, title: "Span data", id: "spans", ready: true },
  { n: 11, title: "Faster inference", id: "serving", ready: true },
];

export default function App() {
  return (
    <div className="page" id="top">
      <header className="masthead">
        <span className="brand">Inside LLM-as-Jev</span>
        <nav className="sections" aria-label="Sections">
          {SECTIONS.map((s) => (
            s.ready ? (
              <a key={s.n} className="sec on" href={`#${s.id}`}>
                <b>{s.n}</b> {s.title}
              </a>
            ) : (
              <span key={s.n} className="sec">
                <b>{s.n}</b> {s.title}
              </span>
            )
          ))}
        </nav>
      </header>
      <PromptAndData />
      <TrieLoss />
      <Anchors />
      <LossCode />
      <Results />
      <Ablation />
      <ContextCite />
      <AT2 />
      <TryIt />
      <SpanPipeline />
      <Serving />
    </div>
  );
}

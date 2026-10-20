import PromptAndData from "./sections/PromptAndData.jsx";
import TrieLoss from "./sections/TrieLoss.jsx";
import Anchors from "./sections/Anchors.jsx";
import FinalLoss from "./sections/FinalLoss.jsx";
import ContextCite from "./sections/ContextCite.jsx";
import AT2 from "./sections/AT2.jsx";
import TryIt from "./sections/TryIt.jsx";
import SpanPipeline from "./sections/SpanPipeline.jsx";

// Sections in reading order; later ones are added as the walkthrough grows.
const SECTIONS = [
  { n: 1, title: "Prompt and data", id: "top", ready: true },
  { n: 2, title: "Trie and tree loss", id: "trie", ready: true },
  { n: 3, title: "KL anchors", id: "anchors", ready: true },
  { n: 4, title: "Final loss", id: "final", ready: true },
  { n: 5, title: "ContextCite", id: "contextcite", ready: true },
  { n: 6, title: "AT2", id: "at2", ready: true },
  { n: 7, title: "Try it", id: "try", ready: true },
  { n: 8, title: "Span data", id: "spans", ready: true },
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
      <FinalLoss />
      <ContextCite />
      <AT2 />
      <TryIt />
      <SpanPipeline />
    </div>
  );
}

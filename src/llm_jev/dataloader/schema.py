"""The uniform example every dataset is converted to.

A dataset source reads its raw files and emits `Example`s; everything
downstream (prompt building, training, evaluation, span generation) consumes
only this class. `Example.to_row` / `Example.from_row` convert it to and from
a row of a `datasets.Dataset` (see `storage.FEATURES`).
"""

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

_EXAMPLE_KEYS = frozenset(
    {"id", "source", "parts", "question", "options", "gold", "spans", "group",
     "subset", "meta", "overlap_text"})
_REQUIRED_KEYS = frozenset(
    {"id", "source", "parts", "question", "options", "gold"})


@dataclasses.dataclass(frozen=True)
class Part:
    """A named piece of the input text (a passage, a document, a goal).

    Attributes:
      name: Section name, e.g. "Passage".
      text: The section text.
    """

    name: str
    text: str


@dataclasses.dataclass(frozen=True)
class Span:
    """A character range of one part that supports the answer.

    Attributes:
      part: Index into `Example.parts`.
      start: First character of the span in that part's text.
      end: One past the last character of the span.
    """

    part: int
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class Example:
    """One decision: input parts, a question, ordered options and the gold.

    Attributes:
      id: Unique within a dataset, e.g. "reclor:3f9a...".
      source: Name of the dataset the example comes from.
      parts: The input text, in prompt order. Empty when the question is
        self-contained (e.g. MMLU).
      question: The question or instruction.
      options: The answer options in prompt order; option i is shown as
        number i + 1. At least two.
      gold: 0-based index of the correct option.
      spans: Evidence in `parts` supporting the gold option; empty when the
        dataset has none or the answer has no evidence.
      group: Leakage unit (document, story, theory, ...). Examples of one
        group must stay in one split.
      subset: Key to stratify by (label name, hypothesis, question type).
      meta: Extra provenance or annotation output, JSON-serializable.
      overlap_text: Texts to check against evaluation benchmarks when they
        should differ from parts + question + options. A working column of
        the build (`to_row(with_overlap_text=True)`); not part of saved rows.
    """

    id: str
    source: str
    parts: tuple[Part, ...]
    question: str
    options: tuple[str, ...]
    gold: int
    spans: tuple[Span, ...] = ()
    group: str = ""
    subset: str = ""
    meta: Mapping[str, Any] = dataclasses.field(
        default_factory=dict, hash=False)
    overlap_text: tuple[str, ...] | None = dataclasses.field(
        default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Checks the invariants.

        Raises:
          ValueError: If there are fewer than two options, `gold` is out of
            range, or a span does not fit inside its part.
        """
        if len(self.options) < 2:
            raise ValueError(f"{self.id}: needs at least 2 options")
        if not 0 <= self.gold < len(self.options):
            raise ValueError(
                f"{self.id}: gold {self.gold} not in 0..{len(self.options)-1}")
        for span in self.spans:
            if not 0 <= span.part < len(self.parts):
                raise ValueError(
                    f"{self.id}: span part {span.part} not in "
                    f"0..{len(self.parts) - 1}")
            length = len(self.parts[span.part].text)
            if not 0 <= span.start < span.end <= length:
                raise ValueError(
                    f"{self.id}: span {span.start}:{span.end} is outside "
                    f"part {span.part} (length {length})")

    @property
    def answer(self) -> str:
        """The text of the gold option."""
        return self.options[self.gold]

    @property
    def has_spans(self) -> bool:
        """Whether the example carries evidence spans."""
        return bool(self.spans)

    def to_row(self, with_overlap_text: bool = False) -> dict[str, Any]:
        """The row of a `datasets.Dataset` (see `storage.FEATURES`).

        Args:
          with_overlap_text: Whether to add the `overlap_text` working column
            (empty when it is None), so a `map` over the rows sees it.
        """
        row = {
            "id": self.id,
            "source": self.source,
            "parts": [{"name": p.name, "text": p.text} for p in self.parts],
            "question": self.question,
            "options": list(self.options),
            "gold": self.gold,
            "spans": [
                {"part": s.part, "start": s.start, "end": s.end}
                for s in self.spans
            ],
            "group": self.group,
            "subset": self.subset,
            "meta": json.dumps(dict(self.meta), ensure_ascii=False),
        }
        if with_overlap_text:
            row["overlap_text"] = list(self.overlap_text or ())
        return row

    @classmethod
    def from_row(cls, data: Mapping[str, Any]) -> "Example":
        """Builds an example from a dataset row.

        Args:
          data: A mapping with the keys `to_row` writes (`meta` is a JSON
            string); `spans`, `group`, `subset` and `meta` may be absent.

        Returns:
          The validated example.

        Raises:
          ValueError: If a required key is missing, an unknown key is present
            (for instance the retired `labels` / `label` format), or the
            example breaks an invariant.
        """
        keys = set(data)
        if missing := _REQUIRED_KEYS - keys:
            raise ValueError(
                f"{data.get('id', '?')}: missing keys {sorted(missing)}")
        if unknown := keys - _EXAMPLE_KEYS:
            raise ValueError(
                f"{data['id']}: unknown keys {sorted(unknown)}; rows in "
                "the old labels/label format must be rebuilt")
        return cls(
            id=data["id"],
            source=data["source"],
            parts=tuple(Part(p["name"], p["text"]) for p in data["parts"]),
            question=data["question"],
            options=tuple(data["options"]),
            gold=data["gold"],
            spans=tuple(
                Span(s["part"], s["start"], s["end"])
                for s in data.get("spans", ())),
            group=data.get("group", ""),
            subset=data.get("subset", ""),
            meta=json.loads(data.get("meta") or "{}"),
            overlap_text=tuple(data["overlap_text"])
            if data.get("overlap_text") else None,
        )


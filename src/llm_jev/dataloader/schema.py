"""The uniform example every dataset is converted to.

Sources emit `Example`s; everything downstream consumes only this class.
`Example.to_row` / `Example.from_row` convert it to and from a row of a
`datasets.Dataset` (see `storage.FEATURES`).
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
    """A named piece of the input text (a passage, a document, a goal)."""

    name: str
    text: str


@dataclasses.dataclass(frozen=True)
class Span:
    """A character range [start, end) of one part (an index into `Example.parts`) that supports the answer."""

    part: int
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class Example:
    """One decision: input parts, a question, ordered options and the gold.

    Attributes:
      id: Unique within a dataset.
      parts: The input text, in prompt order; empty if the question is self-contained.
      options: Option i is shown as number i + 1. At least two.
      gold: 0-based index of the correct option.
      spans: Evidence in `parts` for the gold option; may be empty.
      group: Leakage unit (document, story, ...); one group stays in one split.
      subset: Key to stratify by.
      meta: Extra JSON-serializable provenance.
      overlap_text: Texts to check against benchmarks, when they differ from
        parts + question + options. A build working column, not in saved rows.
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
        """Checks the invariants; raises ValueError on a violation."""
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
          with_overlap_text: Also add the `overlap_text` working column.
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
        """Builds a validated example from a dataset row.

        Raises:
          ValueError: If a required key is missing, an unknown key is present
            (e.g. the retired `labels` / `label` format), or the example
            breaks an invariant.
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


"""Prompt templates (Jinja, llm_jev/prompts/*.j2) and how examples fill them."""

from collections.abc import Sequence
import functools
import pathlib
from typing import Any

import jinja2

from llm_jev.data import schema

TEMPLATES = pathlib.Path(__file__).parent.parent / "prompts"


@functools.cache
def _environment(directory: pathlib.Path) -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(directory),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )


def render(name: str, **values: Any) -> str:
    """Renders llm_jev/prompts/<name>.j2.

    Args:
      name: Template name without the extension.
      **values: Template variables; a missing one raises an error.

    Returns:
      The prompt text, without a trailing newline.
    """
    template = _environment(TEMPLATES).get_template(f"{name}.j2")
    return template.render(**values).rstrip("\n")


def render_example(
    example: schema.Example,
    order: Sequence[int],
    template: str = "decision",
    **values: Any,
) -> tuple[str, list[tuple[int, int]]]:
    """An example's prompt and each part's (start, end) character range in it.

    The template is rendered with markers in place of the part texts, which
    are then substituted: searching for the texts could match the template's
    own words.

    Args:
      example: The decision to render.
      order: Option indices in display order; option order[i] is shown as
        number i + 1.
      template: Template name (e.g. "decision" or "decision_mlm").
      **values: The template's extra variables (e.g. `mask`).

    Returns:
      (prompt text, [(start, end) of each part's text in it]).
    """
    marks = [f"\x00{i}\x00" for i in range(len(example.parts))]
    marked = render(
        template,
        parts=[
            schema.Part(p.name, m)
            for p, m in zip(example.parts, marks, strict=True)
        ],
        question=example.question,
        options=[example.options[i] for i in order],
        **values,
    )
    pieces: list[str] = []
    where: list[tuple[int, int]] = []
    cursor = length = 0
    for part, mark in zip(example.parts, marks, strict=True):
        found = marked.index(mark, cursor)
        pieces += [marked[cursor:found], part.text]
        length += found - cursor
        where.append((length, length + len(part.text)))
        length += len(part.text)
        cursor = found + len(mark)
    pieces.append(marked[cursor:])
    return "".join(pieces), where

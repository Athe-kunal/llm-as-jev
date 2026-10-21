"""Prompt templates (Jinja, prompts/*.j2)."""

import functools
import pathlib
from typing import Any

import jinja2

_PROMPTS = pathlib.Path(__file__).parent.parent / "prompts"


@functools.cache
def _env(directory: pathlib.Path) -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(directory),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )


def render(name: str, *, directory: pathlib.Path = _PROMPTS, **values: Any) -> str:
    """Renders <directory>/<name>.j2 with the given values.

    Args:
      name: Template name without the extension.
      directory: The template folder (default: prompts/).
      **values: Template variables; a missing one raises an error.

    Returns:
      The prompt text, without a trailing newline.
    """
    return _env(pathlib.Path(directory)).get_template(f"{name}.j2").render(**values).rstrip("\n")


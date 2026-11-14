"""Dataset sources: each reads one dataset's raw files into `Example`s.

Importing this package registers every source in `base.SOURCES`.
"""

from llm_jev.data.sources import intent  # noqa: F401 (registers)
from llm_jev.data.sources import multiple_choice  # noqa: F401 (registers)
from llm_jev.data.sources import nimble  # noqa: F401 (registers)
from llm_jev.data.sources import reasoning  # noqa: F401 (registers)
from llm_jev.data.sources.base import get_source

__all__ = ["get_source"]

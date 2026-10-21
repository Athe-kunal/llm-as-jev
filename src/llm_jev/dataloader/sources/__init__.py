"""Dataset sources: each reads one dataset's raw files into `Example`s.

Importing this package registers every source in `base.SOURCES`.
"""

# The imports register the sources.
from llm_jev.dataloader.sources import intent  # pylint: disable=unused-import
from llm_jev.dataloader.sources import multiple_choice  # pylint: disable=unused-import
from llm_jev.dataloader.sources import nimble  # pylint: disable=unused-import
from llm_jev.dataloader.sources import reasoning  # pylint: disable=unused-import
from llm_jev.dataloader.sources.base import SOURCES
from llm_jev.dataloader.sources.base import Source
from llm_jev.dataloader.sources.base import get_source

"""Define what counts as an entity. Run it locally.

A 0.8B model that extracts and labels exact spans using your plain-English rulebook.
"""

from .api import LabelResult, Span, Tagger
from .rulebook import to_text

__version__ = "1.0.7"
__all__ = ["LabelResult", "Span", "Tagger", "to_text", "__version__"]

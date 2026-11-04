"""Pull out the entities you define, using a plain-English rulebook, on your own CPU.

A 0.8B model that extracts and labels exact spans using your plain-English rulebook.
"""

from .api import LabelResult, Span, Tagger
from .rulebook import to_text

__version__ = "1.0.8"
__all__ = ["LabelResult", "Span", "Tagger", "to_text", "__version__"]

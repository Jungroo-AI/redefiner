"""Entity extraction and span labeling, guided by the rulebook you write. Define what to extract. In plain English."""

from .api import LabelResult, Span, Tagger
from .rulebook import to_text

__version__ = "1.0.0"
__all__ = ["LabelResult", "Span", "Tagger", "to_text", "__version__"]

"""Input checks in the Python API that need no weights."""
import pytest

from redefiner import Tagger
from redefiner.api import _offsets


def test_span_offsets():
    assert _offsets((1, 2)) == (1, 2) and _offsets({"start": 1, "end": 2}) == (1, 2)
    for bad in ([0], [0, 1, 2], {"start": 1}, None, "ab"):
        with pytest.raises(ValueError, match="a span must be"):
            _offsets(bad)


@pytest.mark.parametrize("path", ["./weigths", "/no/such/folder", "~/no-such-folder"])
def test_missing_local_folder_is_not_sent_to_the_hub(path):
    with pytest.raises(FileNotFoundError, match="no folder at"):
        Tagger.from_pretrained(path)

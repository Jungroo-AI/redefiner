import os
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
requires_weights = pytest.mark.skipif(not (WEIGHTS / "model.safetensors").exists(),
                                      reason="needs the release weights in ./weights")


@pytest.fixture(scope="session")
def tagger():
    from redefiner import Tagger
    return Tagger.from_pretrained(WEIGHTS, device="cpu")

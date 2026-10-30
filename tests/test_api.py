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


@pytest.mark.parametrize("cuda_build", [None, "12.8"])
def test_warns_when_an_nvidia_gpu_goes_unused(monkeypatch, cuda_build):
    import torch

    from redefiner import api
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", cuda_build)
    monkeypatch.setattr(api, "_nvidia_driver_present", lambda: True)
    with pytest.warns(RuntimeWarning, match="the model runs on CPU"):
        api._warn_if_gpu_unused()


def test_no_warning_without_an_nvidia_gpu(monkeypatch, recwarn):
    import torch

    from redefiner import api
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(api, "_nvidia_driver_present", lambda: False)
    api._warn_if_gpu_unused()
    assert not recwarn.list

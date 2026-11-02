"""Input checks in the Python API that need no weights."""
import pytest

from redefiner import Tagger
from redefiner.api import _offsets, _span_lists, _text_list


def test_span_offsets():
    assert _offsets((1, 2)) == (1, 2) and _offsets({"start": 1, "end": 2}) == (1, 2)
    for bad in ([0], [0, 1, 2], {"start": 1}, None, "ab"):
        with pytest.raises(ValueError, match="a span must be"):
            _offsets(bad)


def test_texts_must_be_strings():
    assert _text_list("a") == (["a"], True) and _text_list(("a", "b")) == (["a", "b"], False)
    for bad, message in [(None, "not NoneType"), (b"hi", "not bytes"), (["a", None], "item 1 is NoneType")]:
        with pytest.raises(TypeError, match=message):
            _text_list(bad)


def test_spans_need_one_list_per_text():
    assert _span_lists([(0, 1)], 1, single=True) == [[(0, 1)]]
    assert _span_lists([[(0, 1)], ()], 2, single=False) == [[(0, 1)], []]
    for bad, single, message in [(None, True, "not NoneType"), ({"start": 0, "end": 1}, True, "not dict"),
                                 (None, False, "not NoneType"), ([[(0, 1)], None], False, "item 1 is NoneType")]:
        with pytest.raises(TypeError, match=message):
            _span_lists(bad, 1 if single else 2, single)
    with pytest.raises(ValueError, match="one list of spans per text"):
        _span_lists([[(0, 1)]], 2, single=False)


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

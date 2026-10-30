"""The HTTP server's request handling, with a stand-in for the model."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from redefiner.serve import Server


class FakeTagger:
    config = {"format": "int8"}

    def extract(self, texts, rulebook, threshold=None, none_bias=None):
        return [[{"start": 0, "end": len(t), "text": t, "label": "X", "confidence": 1.0}] for t in texts]

    def label(self, texts, spans, rulebook, allow_none=False, snap=False, none_bias=None):
        if "[TYPE]" not in rulebook:
            raise ValueError("no types found")
        return [[{"start": s, "end": e, "label": "X"} for s, e in sp] for sp in spans]


@pytest.fixture(scope="module")
def url():
    server = Server(("127.0.0.1", 0), FakeTagger())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def call(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url + path, data=data)) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_extract(url):
    code, res = call(url, "/extract", {"rulebook": "[TYPE] X", "texts": ["ab", "cde"]})
    assert code == 200 and [r[0]["end"] for r in res["results"]] == [2, 3]


def test_texts_must_be_a_list(url):
    code, res = call(url, "/extract", {"rulebook": "[TYPE] X", "texts": "hello"})
    assert code == 400 and "texts" in res["error"]


def test_missing_rulebook(url):
    code, res = call(url, "/extract", {"texts": ["a"]})
    assert code == 400 and "rulebook" in res["error"]


def test_label_needs_one_span_list_per_text(url):
    code, res = call(url, "/label", {"rulebook": "[TYPE] X", "texts": ["a", "b"], "spans": [[[0, 1]]]})
    assert code == 400 and "spans" in res["error"]


def test_model_errors_are_bad_requests(url):
    code, res = call(url, "/label", {"rulebook": "no types", "texts": ["a"], "spans": [[[0, 1]]]})
    assert code == 400 and res["error"] == "no types found"


def test_bad_json_and_unknown_paths(url):
    req = urllib.request.Request(url + "/extract", data=b"{not json")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 400
    assert call(url, "/nope", {})[0] == 404
    assert call(url, "/info") == (200, {"format": "int8"})


def test_malformed_span_is_a_bad_request(url):
    for span in ([0], [0, 1, 2], ["a", "b"], {"start": 0}, [True, 1]):
        code, res = call(url, "/label", {"rulebook": "[TYPE] X", "texts": ["ab"], "spans": [[span]]})
        assert code == 400 and "span" in res["error"], span


@pytest.mark.parametrize("field", ["allow_none", "snap"])
def test_flags_must_be_true_or_false(url, field):
    code, res = call(url, "/label", {"rulebook": "[TYPE] X", "texts": ["ab"], "spans": [[[0, 1]]], field: "false"})
    assert code == 400 and field in res["error"]


def test_numbers_must_be_numbers(url):
    code, res = call(url, "/extract", {"rulebook": "[TYPE] X", "texts": ["ab"], "threshold": "0.5"})
    assert code == 400 and "threshold" in res["error"]

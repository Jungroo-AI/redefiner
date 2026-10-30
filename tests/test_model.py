"""End-to-end checks with the release weights (skipped without ./weights). About a minute on a laptop CPU."""
import pytest

from .conftest import WEIGHTS, requires_weights

pytestmark = requires_weights

RULEBOOK = {"general": ["Extract every entity of the supplied types. Return exact spans from the text."],
            "classes": [{"name": "PERSON", "description": "A person's name."},
                        {"name": "CITY", "description": "A city."}]}
TEXT = "Ada Lovelace was born in London."


def test_extract_offsets_are_exact(tagger):
    spans = tagger.extract(TEXT, RULEBOOK)
    assert {(s["text"], s["label"]) for s in spans} == {("Ada Lovelace", "PERSON"), ("London", "CITY")}
    for s in spans:
        assert TEXT[s["start"]:s["end"]] == s["text"]


def test_extract_batch_matches_single(tagger):
    texts = [TEXT, "Marie Curie moved to Paris.", "", "No names here."]
    assert tagger.extract(texts, RULEBOOK) == [tagger.extract(t, RULEBOOK) for t in texts]


def test_higher_threshold_keeps_fewer_spans(tagger):
    loose = tagger.extract(TEXT, RULEBOOK, threshold=0.1)
    strict = tagger.extract(TEXT, RULEBOOK, threshold=0.999)
    assert {(s["start"], s["end"]) for s in strict} <= {(s["start"], s["end"]) for s in loose}


def test_label_contract(tagger):
    res = tagger.label(TEXT, [(0, 12), (25, 31), (0, 2), (5, 99), {"start": 25, "end": 31}], RULEBOOK)
    assert [r.get("label") for r in res] == ["PERSON", "CITY", None, None, "CITY"]
    assert res[2]["unsupported"].startswith("would be snapped")
    assert res[3]["unsupported"] == "invalid offsets"
    snapped = tagger.label(TEXT, [(0, 2)], RULEBOOK, snap=True)[0]
    assert snapped["snapped"] and snapped["text"] == "Ada" and snapped["label"] == "PERSON"


def test_label_none(tagger):
    forced = tagger.label(TEXT, [(13, 16)], RULEBOOK)[0]   # "was"
    assert forced["label"] in ("PERSON", "CITY")
    assert all(name != "NONE" for name, _ in forced["alternatives"])
    assert tagger.label(TEXT, [(13, 16)], RULEBOOK, allow_none=True)[0]["label"] == "NONE"


def test_label_rejects_mismatched_spans(tagger):
    with pytest.raises(ValueError):
        tagger.label([TEXT, TEXT], [[(0, 12)]], RULEBOOK)


def test_rulebook_without_types_is_an_error(tagger):
    with pytest.raises(ValueError):
        tagger.extract(TEXT, "[RULES]\nExtract everything.")


def test_long_text_is_chunked(tagger):
    text = " ".join(f"In year {1900 + i} the mayor of Paris met Marie Curie." for i in range(40))
    assert len(tagger.encoder.tokens(text)[0]) > tagger.max_q
    spans = tagger.extract(text, RULEBOOK)
    curie = [s for s in spans if s["text"] == "Marie Curie"]
    paris = [s for s in spans if s["text"] == "Paris"]
    assert len(curie) >= 38 and len(paris) >= 38, "every window is read; the tail of the text is not lost"
    for a, b in zip(spans, spans[1:]):
        assert a["end"] <= b["start"], "spans are sorted and do not overlap after merging windows"
    last = text.rindex("Marie Curie")
    assert any(s["start"] == last for s in curie)
    assert tagger.label(text, [(last, last + len("Marie Curie"))], RULEBOOK)[0]["label"] == "PERSON"


def test_release_files_are_enough(tagger, tmp_path):
    """The files from_pretrained downloads from the Hub load the model on their own; ./weights may hold more."""
    from redefiner import Tagger
    from redefiner.api import RELEASE_FILES
    assert not [name for name in RELEASE_FILES if not (WEIGHTS / name).exists()]
    for name in RELEASE_FILES:
        (tmp_path / name).symlink_to(WEIGHTS / name)
    assert Tagger.from_pretrained(tmp_path, device="cpu").extract(TEXT, RULEBOOK) == tagger.extract(TEXT, RULEBOOK)


def test_windows_line_endings_give_the_same_output(tagger):
    from redefiner import to_text
    windows = "\ufeff" + to_text(RULEBOOK).replace("\n", "\r\n")
    assert tagger.extract(TEXT, windows) == tagger.extract(TEXT, RULEBOOK)

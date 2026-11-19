"""1.0.9: the span limit, pointer folders with a default rulebook and settings, and `redefiner redact`."""
import json
import subprocess
import sys

import pytest

from .conftest import WEIGHTS, requires_weights

RULEBOOK = """[RULES]
G1 Extract every piece of personal data of the supplied types. Return exact spans from the text.

[TYPE] PERSON
[DESCRIPTION] A person's name.

[TYPE] EMAIL
[DESCRIPTION] An email address.

[TYPE] PHONE
[DESCRIPTION] A phone number.

[TYPE] CARD_NUMBER
[DESCRIPTION] A credit or debit card number, every digit with its spaces.
"""
CARD = "Please refund the card 4556 7375 8689 9855 by Friday."


def pointer(tmp_path, **extra):
    (tmp_path / "pii.txt").write_text(RULEBOOK)
    config = {"format": "pointer", "base_model": str(WEIGHTS), "rulebook": "pii.txt", **extra}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@requires_weights
def test_release_config_naming_its_base_model_is_not_a_pointer(tagger):
    # the release's own config.json names Qwen as base_model; only "format": "pointer" makes a pointer
    assert tagger.default_rulebook is None and tagger.max_width == 16


@requires_weights
def test_span_limit_finds_a_spaced_card_number(tagger):
    before = [s["text"] for s in tagger.extract(CARD, RULEBOOK)]
    assert "4556 7375 8689 9855" not in before   # 19 tokens: longer than the trained 16
    tagger.set_max_span_tokens(32)
    try:
        after = [s["text"] for s in tagger.extract(CARD, RULEBOOK)]
    finally:
        tagger.set_max_span_tokens(16)
    assert "4556 7375 8689 9855" in after


@requires_weights
def test_span_limit_16_changes_nothing(tagger):
    text = "Ada Lovelace wrote to charles@example.org and called +44 20 7946 0321 on Monday."
    before = tagger.extract(text, RULEBOOK)
    tagger.set_max_span_tokens(16)
    assert tagger.extract(text, RULEBOOK) == before


@pytest.mark.parametrize("width", [0, -3, 2.5, True, 10_000])
@requires_weights
def test_span_limit_must_be_a_sensible_whole_number(tagger, width):
    with pytest.raises(ValueError):
        tagger.set_max_span_tokens(width)


@requires_weights
def test_pointer_folder_loads_the_base_weights_with_its_rulebook_and_limit(tmp_path):
    from redefiner import Tagger
    t = Tagger.from_pretrained(pointer(tmp_path, max_span_tokens=48), device="cpu")
    assert t.max_width == 48 and t.default_rulebook == RULEBOOK
    assert t.config["pointer"] == str(tmp_path) and t.config["max_span_tokens"] == 48
    assert "4556 7375 8689 9855" in [s["text"] for s in t.extract(CARD)]   # no rulebook: the default is used


@requires_weights
def test_an_argument_overrides_the_pointer_span_limit(tmp_path):
    from redefiner import Tagger
    t = Tagger.from_pretrained(pointer(tmp_path, max_span_tokens=48), device="cpu", max_span_tokens=20)
    assert t.max_width == 20


@requires_weights
def test_pointer_none_bias_and_threshold_become_the_defaults(tmp_path):
    from redefiner import Tagger
    t = Tagger.from_pretrained(pointer(tmp_path, none_bias=3.5, threshold=0.4), device="cpu")
    assert t.config["none_bias"] == 3.5 and t.config["threshold"] == 0.4
    text = "Ada Lovelace wrote to charles@example.org and called +44 20 7946 0321 on Monday."
    assert t.extract(text) == t.extract(text, RULEBOOK, none_bias=3.5, threshold=0.4)
    # an argument still wins over the pointer's setting
    assert t.extract(text, none_bias=0) == t.extract(text, RULEBOOK, none_bias=0, threshold=0.4)


@pytest.mark.parametrize("key,value", [("none_bias", "high"), ("threshold", True), ("none_bias", None)])
def test_pointer_settings_must_be_numbers(tmp_path, key, value):
    from redefiner import Tagger
    with pytest.raises(ValueError, match=key):
        Tagger.from_pretrained(pointer(tmp_path, **{key: value}), device="cpu")


def test_pointer_needs_a_base_model(tmp_path):
    from redefiner import Tagger
    (tmp_path / "config.json").write_text(json.dumps({"format": "pointer"}))
    with pytest.raises(ValueError, match="base_model"):
        Tagger.from_pretrained(tmp_path, device="cpu")


@requires_weights
def test_extract_without_a_rulebook_and_no_default_is_a_type_error(tagger):
    with pytest.raises(TypeError):
        tagger.extract("Ada Lovelace")


def test_redact_text_replaces_each_span_with_its_type():
    from redefiner.redact import redact_text
    text = "Mail rosa@example.org or call 555-0101."
    spans = [{"start": 5, "end": 21, "label": "EMAIL"}, {"start": 30, "end": 38, "label": "PHONE"}]
    assert redact_text(text, spans) == "Mail [EMAIL] or call [PHONE]."
    assert redact_text(text, []) == text


@requires_weights
def test_redact_command_writes_the_redacted_file_and_the_audit(tmp_path):
    (tmp_path / "ptr").mkdir()
    folder = pointer(tmp_path / "ptr")
    note = tmp_path / "note.txt"
    note.write_text("Hi, this is Rosa Diaz. Write to rosa.diaz@example.org.\n")
    out = subprocess.run([sys.executable, "-m", "redefiner", "redact", str(note), "--model", str(folder),
                          "--device", "cpu"], capture_output=True, text=True, check=True)
    redacted = (tmp_path / "note.redacted.txt").read_text()
    audit = json.loads((tmp_path / "note.audit.json").read_text())
    assert "rosa.diaz@example.org" not in redacted and "[EMAIL]" in redacted
    assert {s["label"] for s in audit["spans"]} >= {"EMAIL"}
    assert all(note.read_text()[s["start"]:s["end"]] == s["text"] for s in audit["spans"])
    assert "only safeguard" in out.stdout


@requires_weights
def test_redact_none_bias_is_passed_on_and_recorded(tmp_path):
    (tmp_path / "ptr").mkdir()
    folder = pointer(tmp_path / "ptr", none_bias=2.0)
    note = tmp_path / "note.txt"
    note.write_text("Hi, this is Rosa Diaz. Write to rosa.diaz@example.org.\n")
    base = [sys.executable, "-m", "redefiner", "redact", str(note), "--model", str(folder), "--device", "cpu"]
    subprocess.run(base, capture_output=True, text=True, check=True)
    assert json.loads((tmp_path / "note.audit.json").read_text())["none_bias"] == 2.0   # the pointer's setting
    subprocess.run([*base, "--none-bias", "4"], capture_output=True, text=True, check=True)
    assert json.loads((tmp_path / "note.audit.json").read_text())["none_bias"] == 4.0


def test_redact_refuses_files_that_are_not_text(tmp_path):
    blob = tmp_path / "scan.pdf"
    blob.write_bytes(b"%PDF-1.7\n\xff\xfe\x00\x81")
    out = subprocess.run([sys.executable, "-m", "redefiner", "redact", str(blob), "--model", str(tmp_path)],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "not a UTF-8 text file" in out.stderr

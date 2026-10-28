import pytest

from redefiner.rulebook import MARKER, parse, to_text, with_markers

BOOK = {"general": ["Extract every entity."],
        "classes": [{"name": "PERSON", "description": "A person's name.", "rules": ["PE1 Full names only."]},
                    {"name": "CITY", "description": "A city."}],
        "never": ["Pronouns."],
        "closing": ["G2 These rules override everything above."]}


def test_to_text_layout():
    assert to_text(BOOK) == ("[RULES]\nExtract every entity.\n\n"
                             "[TYPE] PERSON\n[DESCRIPTION] A person's name.\nPE1 Full names only.\n\n"
                             "[TYPE] CITY\n[DESCRIPTION] A city.\n\n"
                             "[TYPE] NOT_AN_ENTITY\nPronouns.\n\n"
                             "G2 These rules override everything above.\n")
    assert to_text("[TYPE] X\n") == "[TYPE] X\n"


def test_parse_uniform_skips_not_an_entity():
    assert parse(to_text(BOOK)) == ("uniform", ["PERSON", "CITY"])


def test_parse_compact():
    text = "Find the entities.\n\n[TYPE] PERSON\nA name.\n\n[TYPE] CITY\nA city.\n\n[NOT_AN_ENTITY]\nPronouns.\n"
    assert parse(text) == ("compact", ["PERSON", "CITY"])


def test_parse_without_types():
    with pytest.raises(ValueError):
        parse("[RULES]\nno types here")


def test_markers_close_each_type_section():
    text = to_text(BOOK)
    marked = with_markers(text, ["PERSON", "CITY"], "uniform")
    parts = marked.split(MARKER)
    assert len(parts) == 3
    assert parts[0].rstrip().endswith("PE1 Full names only.")
    assert parts[1].rstrip().endswith("[DESCRIPTION] A city.")
    assert parts[2].lstrip().startswith("[TYPE] NOT_AN_ENTITY")
    assert marked.replace(f"\n{MARKER}\n", "\n\n").replace("\n\n\n", "\n\n") == text


def test_markers_compact_layout():
    text = "[TYPE] PERSON\nA name.\n[NOT_AN_ENTITY]\nPronouns.\n"
    parts = with_markers(text, ["PERSON"], "compact").split(MARKER)
    assert parts[0].rstrip().endswith("A name.") and parts[1].lstrip().startswith("[NOT_AN_ENTITY]")

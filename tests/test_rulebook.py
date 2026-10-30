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


@pytest.mark.parametrize("line", ["[TYPE] DATE-OF-BIRTH", "[TYPE]EMAIL", "[TYPE] Job Title", "[type] CITY",
                                  "[TYPE] CITY  (towns too)", "  [TYPE] CITY", "[TYPE]"])
def test_unreadable_type_line_is_an_error(line):
    text = f"[RULES]\nExtract every entity.\n\n[TYPE] PERSON\n[DESCRIPTION] A name.\n\n{line}\n[DESCRIPTION] More.\n"
    with pytest.raises(ValueError, match="unreadable type line"):
        parse(text)


def test_windows_line_endings_and_bom_read_the_same():
    compact = "Find the entities.\n\n[TYPE] PERSON\nA name.\n\n[NOT_AN_ENTITY]\nPronouns.\n"
    for text in (to_text(BOOK), compact):
        windows = "\ufeff" + text.replace("\n", "\r\n")
        assert to_text(windows) == text
        assert parse(windows) == parse(text)


def test_mixed_layouts_are_an_error():
    with pytest.raises(ValueError, match="mixes two layouts"):
        parse("[RULES]\nFind the entities.\n\n[TYPE] PERSON\nA name.\n\n[NOT_AN_ENTITY]\nPronouns.\n")


def test_rulebook_object_needs_classes_with_names():
    with pytest.raises(ValueError, match='"classes"'):
        to_text({"general": ["Extract every entity."]})
    with pytest.raises(ValueError, match='"name"'):
        to_text({"classes": [{"description": "A city."}]})

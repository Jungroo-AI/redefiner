"""Window logic for long texts. The window functions only need token offsets, so most tests use whitespace tokens."""
import re

import pytest

from redefiner.windows import edge_distance, label_windows, text_windows, token_range

from .conftest import requires_weights

LONG = " ".join(f"Sentence number {i} mentions Ada Lovelace and the Analytical Engine in London." for i in range(80))


def words(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def test_short_text_is_one_window():
    offs = words("Ada Lovelace met Charles Babbage.")
    assert text_windows("Ada Lovelace met Charles Babbage.", offs, 256) == [(0, len(offs))]


def test_windows_cover_every_token_overlap_and_fit():
    offs = words(LONG)
    wins = text_windows(LONG, offs, 256)
    assert len(wins) > 1 and wins[0][0] == 0 and wins[-1][1] == len(offs)
    for (_, e1), (s2, _) in zip(wins, wins[1:]):
        assert s2 < e1, "consecutive windows overlap"
    assert all(e - s <= 256 for s, e in wins)


def test_windows_cut_at_sentence_ends():
    offs = words(LONG)
    for _, e in text_windows(LONG, offs, 256)[:-1]:
        assert LONG[offs[e - 1][0]:offs[e - 1][1]].endswith(".")


def test_token_range():
    offs = words("Ada Lovelace met Charles Babbage.")
    assert token_range(offs, 0, 12) == (0, 1)
    assert token_range(offs, 17, 24) == (3, 3)
    assert token_range(offs, 2, 6) == (0, 1), "partial overlap counts"


def test_label_windows_contain_their_spans():
    offs = words(LONG)
    todo = [(i, m.start(), m.end()) for i, m in enumerate(re.finditer("Ada Lovelace", LONG))]
    groups = label_windows(offs, todo, 256)
    assert sorted(si for _, _, g in groups for si, _, _ in g) == [si for si, _, _ in todo]
    for s_tok, e_tok, group in groups:
        assert e_tok - s_tok <= 256
        for _, s, e in group:
            assert offs[s_tok][0] <= s and e <= offs[e_tok - 1][1]


def test_edge_distance():
    assert edge_distance(3, 4, 100, at_text_start=False, at_text_end=False) == 3
    assert edge_distance(3, 4, 100, at_text_start=True, at_text_end=False) == 95
    assert edge_distance(3, 4, 100, at_text_start=True, at_text_end=True) == 100


@requires_weights
@pytest.mark.parametrize("max_tokens", [64, 256])
def test_window_queries_fit_and_point_into_the_text(tagger, max_tokens, monkeypatch):
    monkeypatch.setattr(tagger, "max_q", max_tokens)
    _, offs = tagger.encoder.tokens(LONG)
    for s, e in text_windows(LONG, offs, max_tokens):
        q, c0 = tagger._window_query(LONG, offs, s, e)
        assert len(q.offsets) <= max_tokens
        assert LONG[c0:c0 + len(q.text)] == q.text

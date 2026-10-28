"""Splitting long texts into windows the model can read.

The model reads at most `max_tokens` tokens of text per query. These functions work on token offsets only (one
(char start, char end) pair per token), so they do not need the model or the tokenizer.
"""

import re

SENTENCE_END = re.compile(r"[.!?;:]['\")\]]*$|\n")


def text_windows(text: str, offsets: list[tuple[int, int]], max_tokens: int) -> list[tuple[int, int]]:
    """Token ranges (start, end) that cover every token of `text`.

    Each window holds at most `max_tokens` tokens. It is cut after a sentence end in its last quarter when there is
    one, and consecutive windows overlap by an eighth of a window.
    """
    n, w = len(offsets), max_tokens
    if n <= w:
        return [(0, n)]
    out, s = [], 0
    while True:
        e = min(s + w, n)
        if e < n:
            for c in range(e - 1, s + 3 * w // 4, -1):
                if SENTENCE_END.search(text[offsets[c][0]:offsets[c][1]]):
                    e = c + 1
                    break
        out.append((s, e))
        if e >= n:
            return out
        s = max(e - w // 8, s + 1)


def token_range(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    """The first and last token (inclusive) that overlap characters start..end."""
    first = next((t for t, (_, b) in enumerate(offsets) if b > start), len(offsets) - 1)
    last = max((t for t, (a, _) in enumerate(offsets) if a < end), default=first)
    return first, max(last, first)


def label_windows(offsets: list[tuple[int, int]], spans: list[tuple[int, int, int]],
                  max_tokens: int) -> list[tuple[int, int, list[tuple[int, int, int]]]]:
    """Windows for labeling the given spans of one long text.

    Args:
        offsets: (char start, char end) of each token of the text.
        spans: (span index, char start, char end) of each span to label.
        max_tokens: the most tokens a window may hold.

    Returns:
        (first token, end token, spans) per window. A window starts a quarter window before its first span and takes
        every later span that ends at least an eighth of a window before the window's end. A span that runs past the
        end of its window gets a window of its own.
    """
    n, w = len(offsets), max_tokens
    ranged = sorted(((token_range(offsets, s, e), (i, s, e)) for i, s, e in spans), key=lambda x: x[0])
    out, i = [], 0
    while i < len(ranged):
        (first, _), _ = ranged[i]
        s_tok = max(0, min(first - w // 4, n - w))
        e_tok = min(n, s_tok + w)
        group = []
        while i < len(ranged):
            (_, last), span = ranged[i]
            if group and last >= e_tok - w // 8 and e_tok < n:
                break
            group.append(span)
            i += 1
            if last >= e_tok:
                break
        out.append((s_tok, e_tok, group))
    return out


def edge_distance(start: int, end: int, n: int, at_text_start: bool, at_text_end: bool) -> int:
    """How many tokens separate span start..end (inclusive) from the nearest window edge that cuts the text.

    Edges that are also the start or end of the whole text don't count, so a span in a single-window text is n tokens
    from any edge. Where two windows disagree, the span further from a cut is kept.
    """
    left = n if at_text_start else start
    right = n if at_text_end else n - 1 - end
    return min(left, right)

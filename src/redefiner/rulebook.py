"""Rulebooks: the plain-text format the model reads, and how its input is laid out.

A rulebook is text with one section per type:

    [RULES]
    Extract every entity of the supplied types. Return exact spans from the query.

    [TYPE] PERSON
    [DESCRIPTION] A person's name.
    PE1 Label the full name, without titles such as "Dr".

    [TYPE] COMPANY
    [DESCRIPTION] A company or brand name.

The [RULES] section holds rules for every type; each [TYPE] section has a one-line [DESCRIPTION] and any number of
rule lines. An optional "[TYPE] NOT_AN_ENTITY" section lists what is never tagged, and a second [RULES] section after
the types holds closing rules. The same rulebook can be given as a dict (see `to_text`).

Two layouts are read:
  uniform  the layout above; a type's section ends at the next [TYPE] or [RULES] line.
  compact  no [RULES] line: free text, then the [TYPE] sections, optionally followed by a "[NOT_AN_ENTITY]" section;
           a type's section ends at the next [TYPE] or [NOT_AN_ENTITY] line.
A rulebook with a "[RULES]" line of its own (or "[TYPE] NOT_AN_ENTITY") is read as uniform; one that also has a
"[NOT_AN_ENTITY]" line mixes the two and is an error. Windows line endings and a byte-order mark are removed first.

Input layout (one rulebook pass, then one short pass per query):
  prefix  <|im_start|>system\\n RULES <|im_end|>\\n            RULES has a marker after each type's section
  query   <|im_start|>user\\n Q <|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n Q
The query is written twice: the second copy sees the whole first copy, so its token states know the full query.
"""

import hashlib
import re
from dataclasses import dataclass

TYPE_HEADER = re.compile(r"^\[TYPE\][ \t]+([A-Za-z0-9_]+)[ \t]*$", re.M)
TYPE_LINE = re.compile(r"^[ \t]*\[type\].*$", re.M | re.I)   # any line written as a type header
UNIFORM_MARK = re.compile(r"^(\[RULES\]|\[TYPE\][ \t]+NOT_AN_ENTITY)[ \t]*$", re.M)
COMPACT_NEVER = re.compile(r"^\[NOT_AN_ENTITY\]", re.M)
SECTION_END = {"uniform": re.compile(r"^\[(TYPE|RULES)\]", re.M),
               "compact": re.compile(r"^\[(TYPE|NOT_AN_ENTITY)\]", re.M)}
MARKER = "\x00MARKER\x00"
PREFIX_HEAD = "<|im_start|>system\n"
PREFIX_TAIL = "<|im_end|>\n"
QUERY_HEAD = "<|im_start|>user\n"
QUERY_MID = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def normalize(text: str) -> str:
    """Rulebook text with Unix line endings and no byte-order mark: a file saved on Windows reads the same."""
    return text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def to_text(book: str | dict) -> str:
    """A rulebook dict as rulebook text (text is returned normalized, see `normalize`).

    {"general": ["rule for every type", ...],
     "classes": [{"name": "PERSON", "description": "A person's name.", "rules": ["PE1 ...", ...]}, ...],
     "never": ["what is never tagged", ...],      # optional
     "closing": ["rules after the classes", ...]} # optional

    Raises:
        ValueError: a dict without a "classes" list, or a class without a "name".
    """
    if isinstance(book, str):
        return normalize(book)
    classes = book.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError('a rulebook object needs a "classes" list, e.g. {"classes": [{"name": "PERSON"}]}')
    if not all(isinstance(c, dict) and c.get("name") for c in classes):
        raise ValueError('each entry of "classes" needs a "name"')
    sections = ["\n".join(["[RULES]", *book.get("general", [])])]
    for c in classes:
        lines = [f"[TYPE] {c['name']}"]
        if c.get("description"):
            lines.append(f"[DESCRIPTION] {c['description']}")
        lines.extend(c.get("rules", []))
        sections.append("\n".join(lines))
    if book.get("never"):
        sections.append("\n".join(["[TYPE] NOT_AN_ENTITY", *book["never"]]))
    if book.get("closing"):
        sections.append("\n".join(book["closing"]))
    return "\n\n".join(sections) + "\n"


def parse(text: str) -> tuple[str, list[str]]:
    """The rulebook's layout ("uniform" or "compact") and its type names in rulebook order.

    Raises:
        ValueError: the rulebook has no [TYPE] sections; a line written as a type header is not "[TYPE] NAME" with a
            name of letters, digits and underscores; or the rulebook mixes the two layouts.
    """
    text = normalize(text)
    bad = [m.group(0).strip() for m in TYPE_LINE.finditer(text) if not TYPE_HEADER.fullmatch(m.group(0))]
    if bad:
        more = f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""
        raise ValueError(f'unreadable type line {bad[0]!r}{more}: write "[TYPE] NAME" at the start of the line, with '
                         f'a name of letters, digits and underscores, e.g. "[TYPE] DATE_OF_BIRTH"')
    layout = "uniform" if UNIFORM_MARK.search(text) else "compact"
    if layout == "uniform" and COMPACT_NEVER.search(text):
        raise ValueError('this rulebook mixes two layouts: with a "[RULES]" line of its own, write the never-extract '
                         'section as "[TYPE] NOT_AN_ENTITY", not "[NOT_AN_ENTITY]"')
    names = []
    for m in TYPE_HEADER.finditer(text):
        name = m.group(1)
        if name not in names and not (layout == "uniform" and name == "NOT_AN_ENTITY"):
            names.append(name)
    if not names:
        raise ValueError('no types found: the rulebook needs a "[TYPE] NAME" line per type')
    return layout, names


def with_markers(text: str, types: list[str], layout: str) -> str:
    """The rulebook with MARKER after each type's section, in the order of `types`."""
    end = SECTION_END[layout]
    found = {m.group(1): m for m in TYPE_HEADER.finditer(text)}
    cuts = []
    for t in types:
        nxt = end.search(text, found[t].end())
        cuts.append(nxt.start() if nxt else len(text))
    out, last = [], 0
    for c in sorted(cuts):
        out += [text[last:c].rstrip("\n"), "\n", MARKER, "\n"]
        last = c
    out.append(text[last:].lstrip("\n"))
    return "".join(out)


@dataclass
class EncodedRulebook:
    key: str                # hash of the rulebook text
    types: list[str]
    ids: list[int]          # token ids of the prefix, with the marker id at each marker position
    marker_pos: list[int]   # position of each type's marker, in `types` order


@dataclass
class Query:
    text: str
    ids: list[int]                     # token ids of the query part
    q2_start: int                      # where the second copy of the query starts in `ids`
    offsets: list[tuple[int, int]]     # (char start, char end) of each token of the second copy


class Encoder:
    """Tokenizes rulebooks and queries into the model's input layout."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.marker_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.head = self.encode(QUERY_HEAD)
        self.mid = self.encode(QUERY_MID)

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def rulebook(self, text: str) -> EncodedRulebook:
        text = normalize(text)
        layout, types = parse(text)
        full = PREFIX_HEAD + with_markers(text, types, layout) + PREFIX_TAIL
        ids, pos = [], []
        for i, piece in enumerate(full.split(MARKER)):
            if i:
                pos.append(len(ids))
                ids.append(self.marker_id)
            ids += self.encode(piece)
        return EncodedRulebook(hashlib.sha1(text.encode()).hexdigest(), types, ids, pos)

    def tokens(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Token ids of a text and the (char start, char end) of each token, untruncated."""
        t = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return t["input_ids"], [tuple(o) for o in t["offset_mapping"]]

    def query(self, text: str, ids: list[int] | None = None, offsets: list[tuple[int, int]] | None = None) -> Query:
        if ids is None:
            ids, offsets = self.tokens(text)
        full = self.head + ids + self.mid
        return Query(text, full + ids, len(full), offsets)

    @staticmethod
    def char_span(q: Query, first: int, last: int) -> tuple[int, int]:
        """Character span of tokens first..last (inclusive) of the second copy, without surrounding whitespace."""
        s, e = q.offsets[first][0], q.offsets[last][1]
        while s < e and q.text[s].isspace():
            s += 1
        while e > s and q.text[e - 1].isspace():
            e -= 1
        return s, e

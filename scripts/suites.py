"""Reading the evaluation suites the demo examples come from (benchmark test data, not part of this repository).

A suite is a folder with test.jsonl (one row per test example: id, text, entities, book_id) and books/ (one rulebook
per book_id, as JSON).
"""

import json
from pathlib import Path

from redefiner import to_text


def load_row(suites: Path, suite: str, row_id: str) -> dict:
    with open(suites / suite / "test.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] == row_id:
                return row
    raise KeyError(f"{row_id} is not in {suite}")


def load_book(suites: Path, suite: str, book_id: str) -> str:
    """The suite's rulebook for book_id as text (a suite with a single rulebook uses it for every row)."""
    books = {b.stem: b for b in (suites / suite / "books").glob("*.json")}
    path = books.get(book_id) or (next(iter(books.values())) if len(books) == 1 else None)
    if path is None:
        raise KeyError(f"no rulebook {book_id} in {suite}: {sorted(books)}")
    return to_text(json.loads(path.read_text()))


def spans_of(results: list[dict]) -> list[list]:
    """[start, end, label] per extracted span, the form the demo stores."""
    return [[r["start"], r["end"], r["label"]] for r in results]

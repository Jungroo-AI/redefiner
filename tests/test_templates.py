"""The rulebook files in docs/ parse, and the domain templates match the demo's templates exactly."""
import sys
from pathlib import Path

import pytest

from redefiner.rulebook import parse

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DEMO = {"pii.txt": "PII", "news.txt": "News", "product-reviews.txt": "Product reviews",
        "scientific-text.txt": "Scientific text"}


@pytest.mark.parametrize("path", sorted(DOCS.glob("templates/*.txt")) + sorted(DOCS.glob("examples/*.txt")),
                         ids=lambda p: p.name)
def test_rulebook_files_parse(path):
    layout, types = parse(path.read_text())
    assert types and "NOT_AN_ENTITY" not in types


@pytest.mark.parametrize("filename", sorted(DEMO))
def test_domain_templates_match_the_demo(filename, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "space"))
    sys.modules.pop("templates", None)
    from templates import text
    assert (DOCS / "templates" / filename).read_text() == text(DEMO[filename])

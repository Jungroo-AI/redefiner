"""README.md is also the PyPI page, where relative links don't resolve: keep its links absolute and working."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_FILE = re.compile(r"https://(?:github\.com/jungroo-ai/redefiner/(?:blob|tree)/main|"
                       r"raw\.githubusercontent\.com/jungroo-ai/redefiner/main)/([^#?]+)", re.I)


def readme_text():
    return re.sub(r"```.*?```", "", (ROOT / "README.md").read_text(), flags=re.S)


def readme_links():
    text = readme_text()
    return (re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:src|href)="([^"]+)"', text)
            + re.findall(r"^\[[^\]]+\]:\s*(\S+)", text, flags=re.M))


def test_readme_full_urls_are_reference_style():
    # Tools that read the raw text take a URL up to the next space, so an inline link ending "...md)." becomes a 404.
    assert not re.findall(r"\]\((https?://[^)\s]+)\)", readme_text())


def test_readme_links_are_absolute():
    relative = [link for link in readme_links() if not link.startswith(("https://", "http://", "#", "mailto:"))]
    assert not relative


def test_readme_links_into_the_repo_point_to_files_that_exist():
    missing = [link for link in readme_links()
               if (m := REPO_FILE.match(link)) and not (ROOT / m.group(1)).exists()]
    assert not missing

"""Never extract and Closing rules for the three benchmark examples in the Edit a rule tab.

Each of these benchmark rulebooks is shown with one Never extract rule and one Closing rule added for the demo. The
rules fit the example's text (it contains something they cover), quote only strings that are not in the text, and
must leave the output unchanged: the script runs the rulebook with the added rules and fails if the output differs
from the stored one. The published test F1 was measured without these rules; the demo's caption says so.

Writes the rulebook with the added rules and its output into space/examples.json ("demo_rules", "demo_b").

usage: build_demo_rules.py WEIGHTS_DIR [--check]
"""

import argparse
import json
import sys
from pathlib import Path

from redefiner import Tagger

SPACE = Path(__file__).resolve().parent.parent / "space"
ADDED = {   # dataset: (never extract, closing rule)
    "CrossNER Music": ('Years and decades, such as "1999" or "the 1980s".',
                       'In "X of Y", a person named before a band is MusicalArtist and the band is Band ("Q-Tip of A '
                       'Tribe Called Quest"); this overrides everything above.'),
    "MIT Movie": ('Generic words for the medium, such as "movie", "show" or "picture".',
                  'A story theme next to a genre is Plot and the genre stays Genre: "heist" in "heist thriller", '
                  '"betrayal" in "betrayal drama"; this overrides everything above.'),
    "MultiNERD": ('Seasons and repeated times, such as "every winter" or "each spring".',
                  'An organisation that founded or runs an event is Organization, not Event ("Major League Baseball"); '
                  'this overrides everything above.'),
}


def quoted(rule: str) -> list[str]:
    """The quoted examples in a rule that could be answers (symbols such as "@" are not)."""
    return [q for q in rule.split('"')[1::2] if len(q) >= 3 and any(ch.isalnum() for ch in q)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("weights", help="release weights folder")
    ap.add_argument("--check", action="store_true", help="report differences without writing examples.json")
    args = ap.parse_args()
    sys.path.insert(0, str(SPACE))   # the demo's form module
    from form import form_to_text, text_to_form

    path = SPACE / "examples.json"
    examples = json.loads(path.read_text())
    tagger = Tagger.from_pretrained(args.weights, device="cpu")
    changed = []
    for e in examples:
        if e["dataset"] not in ADDED:   # in the Benchmarks tab only
            e.pop("demo_rules", None)
            e.pop("demo_b", None)
            continue
        never, closing = ADDED[e["dataset"]]
        copied = [q for q in quoted(never) + quoted(closing) if q in e["text"]]
        if copied:
            raise SystemExit(f"{e['dataset']}: rule examples {copied} appear in the text")
        form = text_to_form(e["rules"])
        if form["never"] or form["closing"]:
            raise SystemExit(f"{e['dataset']}: the rulebook already has Never extract or Closing rules")
        form["never"], form["closing"] = never, closing
        rules = form_to_text(form)
        got = [[x["start"], x["end"], x["label"]] for x in tagger.extract(e["text"], rules)]
        same = sorted(map(tuple, got)) == sorted(map(tuple, e["b"]))
        print(f"{e['dataset']:16s} output unchanged: {same}")
        if not same:
            print(f"   stored {sorted(map(tuple, e['b']))}\n   now    {sorted(map(tuple, got))}")
            changed.append(e["dataset"])
        e.update(demo_rules=rules, demo_b=got)
    if changed:
        raise SystemExit(f"the added rules change the output of {changed}")
    if not args.check:
        path.write_text(json.dumps(examples, indent=1, ensure_ascii=False) + "\n")
        print("->", path)


if __name__ == "__main__":
    main()

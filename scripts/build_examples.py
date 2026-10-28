"""space/examples.json: the benchmark examples of the Edit a rule and Benchmarks tabs, with real outputs.

Each example is one benchmark test sentence, run with GoLLIE's guidelines (seed-0 type order) and with the same
guidelines plus our dev-tuned rules. Outputs are computed here with the release weights, so the demo shows what it
will compute. Run build_demo_rules.py afterwards.

usage: build_examples.py SUITES_DIR WEIGHTS_DIR
"""

import argparse
import json
from pathlib import Path

from suites import load_book, load_row, spans_of

from redefiner import Tagger

DEST = Path(__file__).resolve().parent.parent / "space" / "examples.json"
# entry name, dataset, test id, suite with the gold labels and GoLLIE's guidelines, suite with our rules, test set size
EXAMPLES = [
    ("Social media: where an @username span starts", "BroadTwitter", "BroadTwitter/test/01157",
     "ner-gollie-more-v1-s0", "ner-broadtwitter-rules-v2-s0", "2,001 tweets"),
    ("Disaster tweets: skip places too broad to map", "HarveyNER", "HarveyNER/test/00005",
     "ner-gollie-more-v1-s0", "ner-harvey-rules-v3-s0", "1,303 tweets"),
    ("Science: scientists, not just people", "CrossNER Science", "CrossNER_science/test/00059",
     "ner-gollie-v1-s0", "ner-r9-science-v2-s0", "543 sentences"),
    ("Music: artists, bands, songs and albums", "CrossNER Music", "CrossNER_music/test/00137",
     "ner-gollie-v1-s0", "ner-r9-music-v3-s0", "465 sentences"),
    ("Movie search: ratings, plots and genres", "MIT Movie", "mit-movie/test/02005",
     "ner-gollie-v1-s0", "ner-r9-movie-v3-s0", "2,443 queries"),
    ("Wikipedia: leave out bare years such as 1959", "MultiNERD", "MultiNERD/test/02532",
     "ner-gollie-more-v1-s0", "ner-r9-multinerd-v3b-s0", "16,454 sentences"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("suites", type=Path, help="folder of evaluation suites")
    ap.add_argument("weights", help="release weights folder")
    args = ap.parse_args()

    tagger = Tagger.from_pretrained(args.weights, device="cpu")
    out = []
    for name, dataset, row_id, gold_suite, rules_suite, size in EXAMPLES:
        row = load_row(args.suites, gold_suite, row_id)
        base = load_book(args.suites, gold_suite, row["book_id"])
        rules = load_book(args.suites, rules_suite, row["book_id"])
        gold = [[e["start"], e["end"], e["label"]] for e in row["entities"]]
        a, b = spans_of(tagger.extract(row["text"], base)), spans_of(tagger.extract(row["text"], rules))
        out.append({"name": name, "dataset": dataset, "id": row_id, "text": row["text"], "base": base,
                    "rules": rules, "gold": gold, "a": a, "b": b, "size": size})
        g, ga, gb = ({tuple(x) for x in xs} for xs in (gold, a, b))
        print(f"{dataset:16s} gold {len(g):2d} | guidelines {len(ga & g)}/{len(ga)} correct | "
              f"+ rules {len(gb & g)}/{len(gb)} correct")
    DEST.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("->", DEST)


if __name__ == "__main__":
    main()

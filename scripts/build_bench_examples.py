"""space/bench_examples.json: one real test example for each Benchmarks-tab dataset not in examples.json.

GoLLIE datasets: gold, GoLLIE's guidelines (seed-0 order), and guidelines + our dev-tuned rules.
S1MB sets (label mode): every target span of one document, labelled with that span's own benchmark rulebook, inputs
unchanged, allow_none false.

usage: build_bench_examples.py SUITES_DIR S1MB_CASES WEIGHTS_DIR
"""

import argparse
import collections
import json
from pathlib import Path

from suites import load_book, load_row, spans_of

from redefiner import Tagger

DEST = Path(__file__).resolve().parent.parent / "space" / "bench_examples.json"
NER = [  # dataset, test id, suite with the gold labels and GoLLIE's guidelines, suite with our rules
    ("CrossNER AI", "CrossNER_AI/test/00385", "ner-gollie-v1-s0", "ner-r9-ai-v3b-s0"),
    ("CrossNER Literature", "CrossNER_literature/test/00194", "ner-gollie-v1-s0", "ner-r9-literature-v1-s0"),
    ("CrossNER Politics", "CrossNER_politics/test/00521", "ner-gollie-v1-s0", "ner-r9-politics-v2-s0"),
    ("MIT Restaurant", "mit-restaurant/test/01185", "ner-gollie-v1-s0", "ner-r9-restaurant-v3b-s0"),
    ("E3C", "E3C/test/00317", "ner-gollie-more-v1-s0", "ner-r9-e3c-v5-s0"),
    ("WikiEvents", "WikiEvents_NER/test/00440", "ner-gollie-more-v1-s0", "ner-r9-wikievents-v3-s0"),
    ("FabNER", "FabNER/test/00292", "ner-gollie-more-v1-s0", "ner-r9-fabner-v3b-s0"),
]
S1MB = [("S1MB Few-NERD (label mode)", "few_nerd:supervised:test:31050"),
        ("S1MB Gretel PII (label mode)", "gretel_pii:default:test:3168")]


def ner_example(tagger: Tagger, suites: Path, row_id: str, gold_suite: str, rules_suite: str) -> dict:
    row = load_row(suites, gold_suite, row_id)
    base = load_book(suites, gold_suite, row["book_id"])
    rules = load_book(suites, rules_suite, row["book_id"])
    return {"mode": "extract", "id": row_id, "text": row["text"],
            "gold": [[e["start"], e["end"], e["label"]] for e in row["entities"]],
            "a": spans_of(tagger.extract(row["text"], base)), "b": spans_of(tagger.extract(row["text"], rules)),
            "base": base, "rules_book": rules}


def label_example(tagger: Tagger, case_id: str, rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda c: c["start"])
    text = rows[0]["query"]
    targets = []
    for c in rows:
        res = tagger.label(text, [(c["start"], c["end"])], c["rules"], allow_none=False)[0]
        targets.append({"start": c["start"], "end": c["end"], "gold": c["expected"], "label": res.get("label"),
                        "confidence": res.get("confidence"), "unsupported": res.get("unsupported")})
    return {"mode": "label", "id": case_id, "text": text, "targets": targets, "types": rows[0]["types"],
            "descriptions": rows[0].get("descriptions", [])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("suites", type=Path, help="folder of evaluation suites")
    ap.add_argument("s1mb_cases", type=Path, help="S1MB label-mode cases (JSONL, one row per target span)")
    ap.add_argument("weights", help="release weights folder")
    args = ap.parse_args()

    tagger = Tagger.from_pretrained(args.weights, device="cpu")
    out = {}
    for dataset, row_id, gold_suite, rules_suite in NER:
        e = out[dataset] = ner_example(tagger, args.suites, row_id, gold_suite, rules_suite)
        g, ga, gb = ({tuple(x) for x in e[k]} for k in ("gold", "a", "b"))
        print(f"{dataset:22s} gold {len(g):2d} | guidelines {len(ga & g)}/{len(ga)} | + rules {len(gb & g)}/{len(gb)}")

    cases = collections.defaultdict(list)
    with open(args.s1mb_cases) as f:
        for line in f:
            c = json.loads(line)
            cases[c["case_id"]].append(c)
    for dataset, case_id in S1MB:
        e = out[dataset] = label_example(tagger, case_id, cases[case_id])
        ok = sum(t["label"] == t["gold"] for t in e["targets"])
        print(f"{dataset:30s} {ok}/{len(e['targets'])} spans labelled as gold, {len(e['types'])} candidate types")
    DEST.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("->", DEST)


if __name__ == "__main__":
    main()

"""Reproduce one number per mode on your machine and compare with the published A10 run.

  extract  CrossNER AI test (431 sentences), GoLLIE guidelines, seed-0 type order
  label    S1MB Few-NERD (358 spans), allow_none false

    python benchmarks/reproduce/reproduce.py --model PATH_OR_HF_REPO [--device cpu] [--only extract|label]

Prints F1 / accuracy for this run and the A10 run, and how many predictions differ. Small differences are expected
across hardware and torch versions (different float summation order); the script reports them rather than hiding them.
Scoring here is exact character offsets and type; the published GoLLIE numbers use GoLLIE's own scorer (case-insensitive
span text), which gives the same means within 0.03.
"""
import argparse
import json
import time
from pathlib import Path

from redefiner import Tagger

HERE = Path(__file__).parent


def load(name):
    with open(HERE / name) as f:
        return [json.loads(line) for line in f]


def f1(rows, pred):
    tp = fp = fn = 0
    for r in rows:
        gold = {(e["start"], e["end"], e["label"]) for e in r["entities"]}
        p = pred[r["id"]]
        tp, fp, fn = tp + len(p & gold), fp + len(p - gold), fn + len(gold - p)
    return 200 * tp / (2 * tp + fp + fn)


def run_extract(tagger):
    rows = load("crossner_ai_test.jsonl")
    book = json.loads((HERE / "crossner_ai_rulebook_seed0.json").read_text())
    a10 = {p["id"]: {(e["start"], e["end"], e["label"]) for e in p["entities"]}
           for p in load("crossner_ai_a10_predictions.jsonl")}
    t0 = time.perf_counter()
    out = tagger.extract([r["text"] for r in rows], book)
    secs = time.perf_counter() - t0
    mine = {r["id"]: {(s["start"], s["end"], s["label"]) for s in o} for r, o in zip(rows, out)}
    same = sum(mine[r["id"]] == a10[r["id"]] for r in rows)
    res = {"sentences": len(rows), "f1": round(f1(rows, mine), 2), "f1_a10": round(f1(rows, a10), 2),
           "identical_sentences": same, "seconds": round(secs, 1)}
    print("extract", res, flush=True)
    return res


def run_label(tagger):
    cases = load("s1mb_fewnerd_cases.jsonl")
    a10 = {p["id"]: (p["predicted"] if p["status"] == "ok" else None)
           for p in load("s1mb_fewnerd_a10_predictions.jsonl")}
    t0 = time.perf_counter()
    hit = hit_a10 = agree = unsupported = 0
    for c in cases:
        r = tagger.label(c["query"], [(c["start"], c["end"])], c["rules"], allow_none=False)[0]
        mine = r.get("label")
        unsupported += "unsupported" in r
        hit += mine == c["expected"]
        hit_a10 += a10[c["id"]] == c["expected"]
        agree += mine == a10[c["id"]]
    res = {"spans": len(cases), "accuracy": round(100 * hit / len(cases), 1),
           "accuracy_a10": round(100 * hit_a10 / len(cases), 1), "same_label_as_a10": agree,
           "unsupported": unsupported, "seconds": round(time.perf_counter() - t0, 1)}
    print("label", res, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description="Reproduce one published number per mode and compare with the A10 run.")
    ap.add_argument("--model", required=True, help="release folder or Hugging Face repo")
    ap.add_argument("--device", default=None, help="cpu, cuda or cuda:N (default: cuda if available)")
    ap.add_argument("--only", choices=["extract", "label"], help="run one mode only")
    ap.add_argument("--out", type=Path, default=None, help="also write the results as JSON here")
    args = ap.parse_args()
    tagger = Tagger.from_pretrained(args.model, device=args.device)
    res = {}
    if args.only in (None, "extract"):
        res["extract"] = run_extract(tagger)
    if args.only in (None, "label"):
        res["label"] = run_label(tagger)
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

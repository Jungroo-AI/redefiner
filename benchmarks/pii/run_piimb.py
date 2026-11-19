"""Run ReDefiNER on one English task of the PII Masking Benchmark (PIIMB) and score it with PIIMB's masking metric.

    python run_piimb.py gretel                                        # the redefiner-pii default rulebook
    python run_piimb.py gretel --rulebook rulebooks/identifiers.txt --none-bias 4
    python run_piimb.py gretel --rulebook rulebooks/gretel.txt --max-span-tokens 48
    python run_piimb.py gretel --limit 200                            # a quick look, not a result
    python run_piimb.py gretel --score-only predictions.jsonl         # rescore saved predictions

Needs `pip install "redefiner>=1.0.9" huggingface_hub`. The test rows are downloaded from the PIIMB dataset repo
(CC BY-NC 4.0) at the revision we used; --data takes a local copy of its data/test_sentences.jsonl instead.
Predictions are matched to rows by position, not by uid: Nemotron reuses some uids for different rows.
"""
import argparse
import json
import time
from pathlib import Path

TASKS = ("privy", "gretel", "ai4privacy-en", "nemotron-pii")
DATASET, FILE = "piimb/pii-masking-benchmark", "data/test_sentences.jsonl"
REVISION = "72ea3393522295a680dd46f2f37caf9856d01a7c"
HERE = Path(__file__).resolve().parent


def merged(spans):
    """Sorted, non-overlapping (start, end) intervals covering the same characters."""
    out = []
    for start, end in sorted((s["start"], s["end"]) for s in spans if s["end"] > s["start"]):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def overlap(a, b):
    total, i, j = 0, 0, 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def masking_scores(rows, preds):
    """PIIMB's character-level masking metric (piimb.metrics.compute_masking_metrics): labels are ignored, spans are
    merged per text, and characters are pooled over all texts before precision, recall and F2 are taken."""
    gold_chars = pred_chars = both = 0
    for row, spans in zip(rows, preds, strict=True):
        g, p = merged(row["entities"]), merged(spans)
        gold_chars += sum(e - s for s, e in g)
        pred_chars += sum(e - s for s, e in p)
        both += overlap(p, g)
    precision = both / pred_chars if pred_chars else 0.0
    recall = both / gold_chars if gold_chars else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f2": f2, "gold_chars": gold_chars}


def load_rows(task, data, limit):
    if data is None:
        from huggingface_hub import hf_hub_download
        data = hf_hub_download(DATASET, FILE, repo_type="dataset", revision=REVISION)
    with open(data, encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["task_name"] == task]
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=TASKS)
    ap.add_argument("--rulebook", default=str(HERE / "rulebooks" / "default.txt"))
    ap.add_argument("--none-bias", type=float, default=2.0)
    ap.add_argument("--max-span-tokens", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (a quick look, not a result)")
    ap.add_argument("--data", default=None, help="a local copy of PIIMB's data/test_sentences.jsonl")
    ap.add_argument("--out", default=None, help="where to save the predictions (default: predictions_TASK.jsonl)")
    ap.add_argument("--score-only", default=None, metavar="PREDICTIONS", help="score a saved predictions file")
    a = ap.parse_args()
    rows = load_rows(a.task, a.data, a.limit)
    if a.score_only:
        with open(a.score_only, encoding="utf-8") as f:
            body = [d for d in map(json.loads, f) if "_meta" not in d]
        if len(body) != len(rows) or any(b["uid"] != r["uid"] for b, r in zip(body, rows)):
            raise SystemExit(f"{a.score_only} does not line up with the {len(rows)} rows of {a.task}")
        preds = [b["pred"] for b in body]
    else:
        from redefiner import Tagger
        rulebook = Path(a.rulebook).read_text(encoding="utf-8")
        tagger = Tagger.from_pretrained("jungroo-ai/redefiner-0.8b", device=a.device, max_span_tokens=a.max_span_tokens)
        texts, preds, started = [r["text"] for r in rows], [], time.time()
        for i in range(0, len(texts), 512):
            found = tagger.extract(texts[i:i + 512], rulebook, threshold=a.threshold, none_bias=a.none_bias,
                                   batch_size=a.batch_size)
            preds += [[{k: s[k] for k in ("start", "end", "label", "confidence")} for s in spans] for spans in found]
            print(f"{len(preds)}/{len(texts)} texts, {time.time() - started:.0f} s", flush=True)
        meta = {"task": a.task, "rulebook": a.rulebook, "none_bias": a.none_bias, "threshold": a.threshold,
                "max_span_tokens": a.max_span_tokens, "rows": len(rows), "device": str(tagger.device)}
        with open(a.out or f"predictions_{a.task}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": meta}) + "\n")
            for r, p in zip(rows, preds):
                f.write(json.dumps({"uid": r["uid"], "pred": p}) + "\n")
    s = masking_scores(rows, preds)
    print(f"{a.task}: {len(rows)} texts  F2 {100 * s['f2']:.2f}  precision {100 * s['precision']:.1f}  "
          f"recall {100 * s['recall']:.1f}")


if __name__ == "__main__":
    main()

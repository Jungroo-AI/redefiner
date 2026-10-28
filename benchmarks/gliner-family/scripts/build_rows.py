"""Rows for the GLiNER-family runs: the 7 CrossNER/MIT dev and test sets of the descriptions suite, each with its
type names and one-line descriptions, read through the benchmark harness (which checks the suite digests).

The suite and its scorer come from our evaluation harness, which is not part of this repository.

usage: build_rows.py --harness BENCHMARK_PY --suite SUITE_DIR   (writes inputs/rows.jsonl)
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_harness(path):
    """The harness module that loads suites (checking their digests) and scores predictions."""
    spec = importlib.util.spec_from_file_location('benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def harness_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--harness', required=True, help="path to our evaluation harness's benchmark.py")
    ap.add_argument('--suite', required=True, type=Path, help='the descriptions suite folder')
    return ap


def main():
    args = harness_args('Build the rows for the GLiNER-family runs.').parse_args()
    benchmark = load_harness(args.harness)
    out = []
    for split in ['dev', 'test']:
        rows, books = benchmark.load_suite(args.suite, split)
        for r in rows:
            out.append({'id': r['id'], 'split': split, 'book': r['book_id'], 'text': r['text'],
                        'classes': [[c['name'], c['description']] for c in books[r['book_id']]['classes']]})
    (ROOT / 'inputs').mkdir(parents=True, exist_ok=True)
    (ROOT / 'inputs/rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    print(len(out), 'rows')


if __name__ == '__main__':
    main()

"""Decode and score the Policy-Linter runs; ReDefiNER's numbers come from the saved predictions of its reported runs.

Decoding: a word is flagged with its best rule when that rule's probability is above the threshold (strictly, as in
Liquid's demo); matches of the never-extract rule are dropped. The demo highlights words one by one, so for entity
spans consecutive flagged words with the same type are joined into one span, and punctuation at a span's outer edges
is trimmed. The rule framing and the threshold (0.20 to 0.90 in steps of 0.05, the demo slider's range) are chosen
on the CrossNER/MIT dev sets by mean span F1.

Scoring: CrossNER/MIT with the benchmark harness's exact-span scorer (the scorer of ReDefiNER's 63.15); BUSTER with
SLIMER's document-level scorer (score_buster.py from our BUSTER runs), next to ReDefiNER's saved BUSTER scores.

usage: score_linter.py --harness PATH dev | test
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reqs', ROOT / 'scripts/build_linter_requests.py')
reqs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reqs)
build = benchmark = None   # the harness module and its scorer, set by main()
THRESHOLDS = [round(0.2 + 0.05 * i, 2) for i in range(15)]
LEAD, TRAIL = '([{"\'', '.,;:!?)]}"\''
BUSTER_RUNS = {'buster_slimer': 'a10/runs/FOLD_5', 'buster_v5': 'a10/v5/runs/FOLD_5_v5_bias2'}   # under build.BUSTER


def decode(words, labels, text, threshold):
    """[(start, end, label)] from one request's saved words."""
    spans = []  # [start, end, label, index of its last word]; start/end stay None while only punctuation is covered
    for k, (ws, we, idx, p) in enumerate(words):
        if p <= threshold or labels[idx] == reqs.NEVER:
            continue
        label, cs, ce = labels[idx], ws, we
        while cs < ce and text[cs] in LEAD:
            cs += 1
        while ce > cs and text[ce - 1] in TRAIL:
            ce -= 1
        if spans and spans[-1][3] == k - 1 and spans[-1][2] == label:
            spans[-1][3] = k
            if cs < ce:
                spans[-1][0] = cs if spans[-1][0] is None else spans[-1][0]
                spans[-1][1] = ce
        else:
            spans.append([cs, ce, label, k] if cs < ce else [None, None, label, k])
    return [(s, e, label) for s, e, label, _ in spans if s is not None]


def load(phase):
    with open(ROOT / f'outputs/linter_{phase}.jsonl') as f:
        return {(r['set'], r['framing'], r['id']): r for r in map(json.loads, f)}


def labels_of(rulebook):
    return reqs.rules_for(rulebook, 'desc')[1]


def ner_score(out, split, framing, threshold):
    rows, books = benchmark.load_suite(build.NER, split)
    labels = {b: labels_of(benchmark.serialize_book(books[b])) for b in books}
    preds = []
    for r in rows:
        o = out[(f'ner_{split}', framing, r['id'])]
        spans = decode(o['words'], labels[r['book_id']], r['text'], threshold)
        preds.append({'id': r['id'], 'entities': [{'start': s, 'end': e, 'label': label} for s, e, label in spans]})
    return benchmark.score_rows(rows, preds)


def dev():
    out = load('dev')
    grid = {}
    for framing in reqs.FRAMINGS:
        for t in THRESHOLDS:
            grid[f'{framing}@{t:.2f}'] = round(100 * ner_score(out, 'dev', framing, t)['mean_dataset_f1'], 2)
    best = max(grid, key=grid.get)
    (ROOT / 'scores').mkdir(exist_ok=True)
    (ROOT / 'scores/linter_dev.json').write_text(json.dumps({'grid': grid, 'chosen': best}, indent=2) + '\n')
    for framing in reqs.FRAMINGS:
        top = max((k for k in grid if k.startswith(framing + '@')), key=grid.get)
        print(framing, 'best', top, grid[top])
    print('chosen', best, grid[best])


def test():
    chosen = json.loads((ROOT / 'scores/linter_dev.json').read_text())['chosen']
    framing, t = chosen.split('@')[0], float(chosen.split('@')[1])
    out = load('test')
    ner = ner_score(out, 'test', framing, t)
    result = {'chosen_on_dev': chosen, 'ner_test': {d: {k: round(100 * v[k], 2) for k in ['precision', 'recall', 'f1']}
                                                     for d, v in ner['datasets'].items()},
              'ner_mean_f1': round(100 * ner['mean_dataset_f1'], 2)}
    texts = {f"{d['document_id']}#{k}": w for d in build.read_jsonl(build.BUSTER / 'sources/buster/FOLD_5.jsonl')
             for k, w in enumerate(build.chunk_document_with_sliding_window(' '.join(d['tokens'])))}
    for name, file in [('buster_slimer', 'rulebook.txt'), ('buster_v5', 'rulebook_v5.txt')]:
        labels = labels_of((build.BUSTER / 'inputs' / file).read_text())
        run = ROOT / f'scores/buster/linter_{name}'
        run.mkdir(parents=True, exist_ok=True)
        with open(run / 'predictions.jsonl', 'w') as f:
            for wid, w in texts.items():
                spans = decode(out[(name, framing, wid)]['words'], labels, w, t)
                f.write(json.dumps({'id': wid, 'doc': wid.rsplit('#', 1)[0],
                                    'spans': [{'text': w[s:e], 'label': label} for s, e, label in spans]}) + '\n')
        subprocess.run([sys.executable, str(build.BUSTER / 'scripts/score_buster.py'), 'FOLD_5', str(run)], check=True,
                       capture_output=True)
        result[name] = json.loads((run / 'scores.json').read_text())['micro']
        ours = build.BUSTER / BUSTER_RUNS[name] / 'scores.json'
        result[name + '_redefiner'] = json.loads(ours.read_text())['micro']
    result['too_long'] = sum(1 for o in out.values() if o.get('too_long') and o['framing'] == framing)
    (ROOT / 'scores/linter_test.json').write_text(json.dumps(result, indent=2) + '\n')
    print('chosen', chosen, '| NER mean F1', result['ner_mean_f1'], '(ReDefiNER 63.15)')
    for name in ['buster_slimer', 'buster_v5']:
        print(name, result[name]['f1'], '| ReDefiNER', result[name + '_redefiner']['f1'])
    print('too long', result['too_long'])


def main():
    global build, benchmark
    ap = argparse.ArgumentParser(description='Decode and score the Policy-Linter runs.')
    ap.add_argument('--harness', required=True, help="path to our evaluation harness's build_requests.py")
    ap.add_argument('phase', choices=['dev', 'test'])
    args = ap.parse_args()
    build = reqs.build = reqs.load_harness(args.harness)
    benchmark = build.benchmark
    {'dev': dev, 'test': test}[args.phase]()


if __name__ == '__main__':
    main()

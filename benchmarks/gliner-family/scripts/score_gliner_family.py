"""Score the GLiNER-family runs with the benchmark harness's exact-span scorer (the scorer of ReDefiNER's 63.15).

A span is kept when its score is at least the threshold. For NuNER Zero, which tags tokens, adjacent kept spans with
the same label are then joined, as its model card's merge_entities does. Thresholds from 0.10 to 0.90 are tried on
the dev sets; test is reported at the dev-chosen threshold and at the gliner default, 0.5.

usage: score_gliner_family.py --harness BENCHMARK_PY --suite SUITE_DIR   (writes scores.json)
"""
import gzip
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rows', ROOT / 'scripts/build_rows.py')
rows_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rows_mod)
THRESHOLDS = [round(0.1 + 0.05 * i, 2) for i in range(17)]
TOKEN_LEVEL = {'NuNER_Zero'}


def decode(spans, threshold, merge, text):
    kept = sorted([s for s in spans if s[3] >= threshold], key=lambda s: (s[0], s[1]))
    if merge:
        out = []
        for s in kept:
            touching = s[0] in (out[-1][1], out[-1][1] + 1) if out else False
            if touching and out[-1][2] == s[2] and text[out[-1][1]:s[0]].strip() == '':
                out[-1] = [out[-1][0], s[1], s[2], min(out[-1][3], s[3])]
            else:
                out.append(list(s))
        kept = out
    return [{'start': s[0], 'end': s[1], 'label': s[2]} for s in kept]


def score(benchmark, suite, run, split, threshold, merge):
    rows, _ = benchmark.load_suite(suite, split)
    preds = [{'id': r['id'], 'entities': decode(run[r['id']], threshold, merge, r['text'])} for r in rows]
    return benchmark.score_rows(rows, preds)


def main():
    args = rows_mod.harness_args('Score the GLiNER-family runs.').parse_args()
    benchmark = rows_mod.load_harness(args.harness)
    report = {}
    for path in sorted((ROOT / 'outputs').glob('*.jsonl.gz')):
        name = path.name.removesuffix('.jsonl.gz')
        model, mode = name.split('__')
        with gzip.open(path, 'rt') as f:
            recs = [json.loads(line) for line in f]
        if len(recs) != 8691:  # run still being written
            print(f'{name}: incomplete ({len(recs)} rows), skipped')
            continue
        run = {r['id']: r['spans'] for r in recs}
        merge = model in TOKEN_LEVEL
        dev = {t: round(100 * score(benchmark, args.suite, run, 'dev', t, merge)['mean_dataset_f1'], 2)
               for t in THRESHOLDS}
        best = max(dev, key=dev.get)
        test_best = score(benchmark, args.suite, run, 'test', best, merge)
        test_05 = score(benchmark, args.suite, run, 'test', 0.5, merge)
        report[name] = {'dev_by_threshold': dev, 'chosen_threshold': best,
                             'test_mean_f1_at_chosen': round(100 * test_best['mean_dataset_f1'], 2),
                             'test_mean_f1_at_0.5': round(100 * test_05['mean_dataset_f1'], 2),
                             'test_at_chosen': {d: round(100 * v['f1'], 2) for d, v in test_best['datasets'].items()},
                        'raw_labels_sample': sorted({lab for r in recs[:200] for lab in r['raw_labels']})[:6]}
        print(f"{name:36s} dev-chosen {best:.2f} -> test {report[name]['test_mean_f1_at_chosen']:6.2f} | "
              f"test @0.5 {report[name]['test_mean_f1_at_0.5']:6.2f} | dev best {dev[best]:.2f}")
    (ROOT / 'scores.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

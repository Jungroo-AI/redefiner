"""Run a GLiNER-family model on the CrossNER/MIT rows and keep every predicted span with its score.

Label modes:
- names: the type names;
- desc_list: one "name: description" string per type, so the description reaches the model;
- desc_dict: {name: "name: description"}, exactly what the benchmark harness passed for the descriptions condition
  (to check what that did).
NuNER Zero requires lower-cased labels (--lower); predicted labels are mapped back to the type names. Predictions are
made at threshold 0.1 and saved with their scores, so any higher threshold can be applied when scoring (with flat,
greedy decoding, filtering a 0.1 run at t gives the same spans as running at t).

usage: run_gliner_family.py --model REPO --mode MODE [--lower] --rows FILE --out FILE
"""
import argparse
import json
import time
from collections import defaultdict

import torch
from gliner import GLiNER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--mode', required=True, choices=['names', 'desc_list', 'desc_dict'])
    ap.add_argument('--lower', action='store_true')
    ap.add_argument('--rows', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()
    model = GLiNER.from_pretrained(args.model).to('cuda').eval()
    with open(args.rows) as f:
        rows = [json.loads(s) for s in f]
    groups = defaultdict(list)
    for r in rows:
        groups[(r['split'], r['book'])].append(r)
    start = time.time()
    with open(args.out, 'w') as f, torch.inference_mode():
        for (split, _book), members in groups.items():
            classes = members[0]['classes']
            names = [n for n, _ in classes]
            prompts = names if args.mode == 'names' else [f'{n}: {d}' for n, d in classes]
            back = {}
            for n, p in zip(names, prompts):
                back[n] = n
                back[p] = n
            if args.lower:
                back = {k.lower(): v for k, v in back.items()}
                prompts = [p.lower() for p in prompts]
                names_for_dict = [n.lower() for n in names]
            else:
                names_for_dict = names
            labels = dict(zip(names_for_dict, prompts)) if args.mode == 'desc_dict' else prompts
            for i in range(0, len(members), args.batch):
                chunk = members[i:i + args.batch]
                texts = [r['text'] for r in chunk]
                if hasattr(model, 'batch_predict_entities'):
                    preds = model.batch_predict_entities(texts, labels, threshold=0.1)
                else:  # newer gliner releases
                    preds = model.inference(texts, labels, threshold=0.1, batch_size=args.batch)
                for r, ents in zip(chunk, preds):
                    spans = [[e['start'], e['end'], back.get(e['label'], back.get(e['label'].lower(), e['label'])),
                              round(float(e['score']), 4)] for e in ents]
                    f.write(json.dumps({'id': r['id'], 'split': split, 'spans': spans,
                                        'raw_labels': sorted({e['label'] for e in ents})}) + '\n')
    print(f'{args.model} {args.mode}: {len(rows)} rows in {time.time() - start:.0f} s', flush=True)


if __name__ == '__main__':
    main()

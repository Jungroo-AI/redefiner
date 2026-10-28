"""Run LFM2.5-Encoder-350M-Policy-Linter over requests; save each word's best rule and its probability.

Input as Liquid's demo server builds it (bizlint.py): "Policy:\n- rule\n...\n\nText:\n" + text, each rule pooled as
the mean of its tokens, one sigmoid probability per text token and rule, float32, one text per forward pass. Words are
maximal runs of non-space characters, and a word's probability for a rule is the highest of its tokens', as the
demo's page computes it. Spans are decoded later from the saved words at any threshold.

usage: run_linter.py --model DIR --requests FILE --framing NAME [NAME ...] --out FILE
"""
import argparse
import json
import re
import time

import torch
from transformers import AutoModel, AutoTokenizer

MAX_TOKENS = 4096  # the demo's RoPE table size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--requests', required=True)
    ap.add_argument('--framing', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).eval().to('cuda')
    with open(args.requests) as f:
        reqs = [r for r in map(json.loads, f) if r['framing'] in args.framing]
    start, too_long = time.time(), 0
    with open(args.out, 'w') as f, torch.inference_mode():
        for n, r in enumerate(reqs):
            rules, text = r['rules'], r['text']
            prefix = 'Policy:\n' + '\n'.join(f'- {x}' for x in rules) + '\n\nText:\n'
            enc = tok(prefix + text, return_offsets_mapping=True, return_tensors='pt')
            offsets = enc.pop('offset_mapping')[0].tolist()
            words = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
            if len(offsets) > MAX_TOKENS:
                too_long += 1
                f.write(json.dumps({'id': r['id'], 'set': r['set'], 'framing': r['framing'], 'words': [],
                                    'tokens': len(offsets), 'too_long': True}) + '\n')
                continue
            pool = torch.zeros(1, len(rules), len(offsets))
            pos = len('Policy:\n')
            for i, rule in enumerate(rules):
                s, e = pos + 2, pos + 2 + len(rule)
                idxs = [j for j, (a, b) in enumerate(offsets) if a < e and b > s and a != b]
                pool[0, i, idxs] = 1 / len(idxs)
                pos = e + 1
            enc = {k: v.to('cuda') for k, v in enc.items()}
            probs = model(**enc, rule_pool=pool.to('cuda'))['logits'].sigmoid()[0].float().cpu()
            best = torch.zeros(len(words), len(rules))
            p0 = len(prefix)
            w = 0
            for j, (a, b) in enumerate(offsets):
                if b <= p0 or a == b:
                    continue
                a, b = max(a, p0) - p0, b - p0
                while w < len(words) and words[w][1] <= a:
                    w += 1
                k = w
                while k < len(words) and words[k][0] < b:
                    best[k] = torch.maximum(best[k], probs[j])
                    k += 1
            top = best.max(dim=1) if len(words) else None
            out = [[ws, we, int(i), round(float(p), 4)]
                   for (ws, we), p, i in zip(words, top.values, top.indices)] if top is not None else []
            f.write(json.dumps({'id': r['id'], 'set': r['set'], 'framing': r['framing'], 'words': out,
                                'tokens': len(offsets)}) + '\n')
            if n % 2000 == 0:
                print(f'{n}/{len(reqs)} {time.time() - start:.0f} s', flush=True)
    print(f'done {len(reqs)} requests in {time.time() - start:.0f} s, {too_long} over {MAX_TOKENS} tokens', flush=True)


if __name__ == '__main__':
    main()

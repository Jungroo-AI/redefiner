"""Run a Liquid d1 decision model on S1MB's Few-NERD and Gretel PII span-labeling cases, in its native format.

S1MB records already have d1's shape: a state and decisions, each with instructions and named criteria. Each case is
one system_one call: state = the case's state_json, questions = its decisions as "choice" questions whose criteria map
each candidate id to its description, in S1MB's order. The prediction is the returned choice. The cases are the 100
per task that ReDefiNER was scored on, read from the pinned S1MB parquet files (sha256 checked).

usage: run_d1.py --model REPO --dtype float16|bfloat16 --case-ids FILE --out FILE
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel

REPO, REV = 'hotchpotch/S1MB-english-v1', '10bcb8c6916b98b966cc1f03aab1781a7c354b4c'
FILES = {'few_nerd': ('data/few_nerd/test/00000-of-00001.parquet',
                      '0a338690aeaba64c348c69afe0280a3882a84f5ea88f3bcb12cc17dfbbbed85b'),
         'gretel_pii': ('data/gretel_pii/test/00000-of-00001.parquet',
                        'bda981a52790c185c044e02561a83730b982acf2581ce4a876fbdcd594be9ffc')}


def cases(task, wanted):
    path, sha = FILES[task]
    local = hf_hub_download(REPO, path, repo_type='dataset', revision=REV)
    assert hashlib.sha256(Path(local).read_bytes()).hexdigest() == sha
    rows = [r for r in pq.read_table(local).to_pylist() if r['case_id'] in wanted]
    assert len(rows) == len(wanted), (task, len(rows), len(wanted))
    return rows


def request(case):
    state = json.loads(case['input']['state_json'])
    questions = {}
    for d in case['input']['decisions']:
        questions[d['id']] = {'type': 'choice', 'instructions': json.loads(d['instructions_json']),
                              'criteria': {c['id']: json.loads(c['description_json']) for c in d['criteria']}}
    gold = {}
    for t in case['targets']:
        assert t['annotation_kind'] == 'hard_label' and t['probabilities'].count(1.0) == 1
        gold[t['decision_id']] = t['ids'][t['probabilities'].index(1.0)]
    return state, questions, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--dtype', required=True, choices=['float16', 'bfloat16', 'float32'])
    ap.add_argument('--case-ids', required=True, help='JSON {task: [case_id, ...]}')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    wanted = json.loads(Path(args.case_ids).read_text())
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=getattr(torch, args.dtype)).to('cuda')
    model.eval()
    start = time.time()
    with open(args.out, 'w') as f:
        for task in ['few_nerd', 'gretel_pii']:
            for case in cases(task, set(wanted[task])):
                state, questions, gold = request(case)
                result = model.system_one(state, questions)
                for qid, answer in result['answers'].items():
                    f.write(json.dumps({'task': task, 'case_id': case['case_id'], 'decision_id': qid,
                                        'gold': gold[qid], 'answer': answer}, ensure_ascii=False) + '\n')
    print(f'{args.model}: done in {time.time() - start:.0f} s', flush=True)


if __name__ == '__main__':
    main()

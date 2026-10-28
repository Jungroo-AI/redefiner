"""Build LFM2.5-Encoder-350M-Policy-Linter requests: one text plus one rule per entity type.

Same rows and rulebooks as the ReDefiNER runs on the benchmark page: the 7 CrossNER/MIT sets with one-line descriptions
(dev and test), and BUSTER fold 5 in SLIMER's 900-character windows with both rulebooks. The linter takes free-text
rules, so each [TYPE] becomes one rule: its description followed by its rule lines. NOT_AN_ENTITY lines become one
extra rule whose matches are dropped. General [RULES] lines are not type rules and are left out.

Three rule framings, chosen on the CrossNER/MIT dev sets only:
- filter: "Filter " + the description, the phrasing of Liquid's demo ("Filter all person names");
- name_desc: "TYPE: description";
- desc: the description alone.

The datasets and rulebook parsing come from our evaluation harness, which is not part of this repository; pass its
request-building module with --harness.

usage: build_linter_requests.py --harness PATH   (writes inputs/linter_{dev,test}.jsonl)
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEVER = '__never__'
FRAMINGS = ['filter', 'name_desc', 'desc']


def load_harness(path):
    spec = importlib.util.spec_from_file_location('harness', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = None   # the harness module, set by main() or by score_linter.py


def rule_text(name, definition, framing):
    if framing == 'filter':
        return 'Filter ' + definition[:1].lower() + definition[1:]
    if framing == 'name_desc':
        return f'{name}: {definition}'
    return definition


def rules_for(rulebook, framing):
    general, types, never, closing = build.parse_rulebook(rulebook)
    rules = [rule_text(t['name'], build.definition(t) or t['name'], framing) for t in types]
    labels = [t['name'] for t in types]
    if never:
        rules.append(rule_text('NOT_AN_ENTITY', 'Never extract: ' + ' '.join(never), framing))
        labels.append(NEVER)
    return rules, labels


def requests(phase):
    out = []
    split = 'dev' if phase == 'dev' else 'test'
    rows, books = build.benchmark.load_suite(build.NER, split)
    texts = {}
    for framing in FRAMINGS:
        cache = {}
        for r in rows:
            if r['book_id'] not in cache:
                cache[r['book_id']] = rules_for(build.benchmark.serialize_book(books[r['book_id']]), framing)
            rules, labels = cache[r['book_id']]
            out.append({'id': r['id'], 'set': f'ner_{split}', 'framing': framing, 'text': r['text'], 'rules': rules,
                        'labels': labels})
        if phase == 'dev':
            continue
        if not texts:
            for d in build.read_jsonl(build.BUSTER / 'sources/buster/FOLD_5.jsonl'):
                for k, w in enumerate(build.chunk_document_with_sliding_window(' '.join(d['tokens']))):
                    texts[f"{d['document_id']}#{k}"] = w
        for name, file in [('buster_slimer', 'rulebook.txt'), ('buster_v5', 'rulebook_v5.txt')]:
            rules, labels = rules_for((build.BUSTER / 'inputs' / file).read_text(), framing)
            for wid, w in texts.items():
                out.append({'id': wid, 'set': name, 'framing': framing, 'text': w, 'rules': rules, 'labels': labels})
    return out


def main():
    global build
    ap = argparse.ArgumentParser(description='Build Policy-Linter requests for the benchmark sets.')
    ap.add_argument('--harness', required=True, help="path to our evaluation harness's build_requests.py")
    build = load_harness(ap.parse_args().harness)
    (ROOT / 'inputs').mkdir(parents=True, exist_ok=True)
    for phase in ['dev', 'test']:
        reqs = requests(phase)
        path = ROOT / f'inputs/linter_{phase}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in reqs))
        counts = {}
        for r in reqs:
            counts[(r['set'], r['framing'])] = counts.get((r['set'], r['framing']), 0) + 1
        print(path.name, len(reqs), counts)


if __name__ == '__main__':
    main()

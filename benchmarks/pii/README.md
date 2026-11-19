# Masking personal data: PIIMB and RedactionBench

ReDefiNER was not trained on any personal-data dataset. These runs measure how well it masks personal data when it
is given a PII rulebook, on two public benchmarks, next to the models on their leaderboards.

The rulebooks are in [`rulebooks/`](rulebooks/), every score is in [`results.json`](results.json), and
[`run_piimb.py`](run_piimb.py) reruns and scores any row of the PIIMB table.

## PII Masking Benchmark (PIIMB), English tasks

[PIIMB][piimb] scores masking by character: a character counts as right when it is masked and lies inside a gold
span, whatever the label. The ranking metric is F2, which weights recall twice as much as precision. We ran the four
English tasks on their full test splits.

| Setup | Privy | Gretel | AI4Privacy EN | Nemotron-PII | Mean | Place among 37 |
|---|---:|---:|---:|---:|---:|---:|
| ReDefiNER, `redefiner-pii` default rulebook (20 types) | 73.4 | 94.1 | 96.4 | 84.8 | 87.2 | 8th |
| ReDefiNER, `identifiers.txt` (10 types) | 51.9 | 88.3 | 90.5 | 58.8 | 72.4 | 23rd |
| ReDefiNER, a rulebook written for each dataset | 80.5 | 94.5 | 97.1 | 86.0 | 89.5 | 6th |
| OpenMed privacy-filter-multilingual-v2 (best mean) | 81.8 | 97.3 | 97.5 | 90.7 | 91.8 | 1st |
| Knowledgator gliner-stream-pii-v1.0 | 90.4 | 93.9 | 95.8 | 83.3 | 90.8 | 2nd |
| NVIDIA gliner-PII | 71.4 | 92.0 | 94.5 | 77.2 | 83.8 | |
| Fastino gliner2-privacy-filter-PII-multi | 64.6 | 93.6 | 91.4 | 85.4 | 83.7 | |
| OpenAI Privacy Filter | 53.6 | 86.9 | 84.3 | 60.7 | 71.4 | |
| GLiNER multi PII v1 | 55.5 | 79.5 | 73.5 | 73.1 | 70.4 | |
| Presidio (en_core_web_lg) | 51.8 | 75.8 | 73.4 | 72.9 | 68.5 | |

How to read this table:

- **"Place among 37"** is where a ReDefiNER row would stand among the 36 models with published PIIMB results, ordered
  by the mean of these four tasks. It is not a leaderboard rank: PIIMB's leaderboard averages six tasks, and we did
  not run its two multilingual tasks, because ReDefiNER is English only.
- **Other models' scores** are PIIMB's published results (`piimb/pii-masking-benchmark-results`, read on
  2026-11-17). Each uses one setting for every task. We did not rerun them.
- **Training data.** ReDefiNER was not trained on any PIIMB source dataset. Of the seven models above the default
  row, five declare training on at least one of them (Nemotron-PII most often), and the other two declare nothing.
- **The default row is the like-for-like row**: one rulebook and one setting for all four tasks, run once each.
- **The per-dataset row** shows what writing rules for your own data buys. Its rulebooks were written for each
  dataset from dev documents, so it is not a general-purpose setup.

### Settings

| Setup | Rulebook | Span limit | `none_bias` | Threshold |
|---|---|---:|---:|---:|
| Default | `rulebooks/default.txt` | 64 | 2 | 0.5 |
| Identifiers | `rulebooks/identifiers.txt` | 64 | 4 | 0.5 |
| Per dataset | `rulebooks/privy.txt`, `gretel.txt`, `ai4privacy-en.txt`, `nemotron-pii.txt` | 48 | 2 | 0.5 |

No confidence cutoff and no filter of any kind is applied to the model's output.

### How the rulebooks were written

- **Dev documents.** For each task we took 1,000 documents from the source dataset's train or validation split.
  PIIMB's test rows come from the test split; for AI4Privacy, where PIIMB also uses validation documents, the ones in
  PIIMB's test set were left out.
- **Writing and choosing.** Rules were written while reading dev documents 0 to 399 only. Candidates were then
  compared on dev documents 400 to 999, which nobody had read, and the best one by F2 was kept. Each task had at
  most six candidates. Test documents were not used to write or choose rules.
- **Default rulebook.** Written from dev documents 0 to 399 of all four tasks together, and chosen among five
  candidates by the mean F2 of the four tasks on documents 400 to 999 (86.3 there, 87.2 on test). The same step chose
  `none_bias` 2 from 1.5, 2, 2.5 and 3.
- **Identifiers rulebook.** The first rulebook we wrote, before any tuning: ten identifier types and no dates,
  places, organisations or job titles. Its `none_bias` (4) was chosen from 2, 3 and 4 on the same dev documents.
- **Span limit.** The model was trained on spans of up to 16 tokens. Longer limits reuse that training and need
  package 1.0.9. On Gretel dev documents 48 found more than 32 (recall 96.6% to 97.4%) and 64 matched 48.

### Test runs

Every row above is a single test run per task, with these additions:

- **Privy and Gretel, per-dataset row:** these had an earlier test run with a first rulebook at span limit 32
  (74.2 and 89.5). The table shows the second run.
- **Nemotron, per-dataset row:** one more run exists at `none_bias` 3. It was started before we settled on one
  setting for all tasks and was never scored.
- **Nemotron scoring.** Nemotron's test rows reuse 1,221 ids for different texts, so predictions must be matched to
  rows by position. `run_piimb.py` does this.

### Where it loses

- **Privy** is made of API records (JSON, XML, SQL, HTML). The default rulebook masks many filler values and
  technical ids there, so precision is 43%.
- **Nemotron** has long URLs and cookies that are only partly covered, and its documents mention the names of data
  types in running text ("medical record number"), which are sometimes masked.
- **Field labels.** A label next to its value ("Customer ID: 4471") is sometimes masked along with the value, at
  low confidence. Rules reduce this and do not remove it.

## RedactionBench

[RedactionBench][redactionbench] has 200 documents and labels what must be masked and what must stay under
contextual integrity. Its R-Score rewards masking the first and charges for masking the second. It has no dev split.
Its leaderboard tunes one confidence cutoff per model on all 200 documents, and we did the same.

| Setup | No cutoff | At the tuned cutoff | Place among 45 |
|---|---:|---:|---:|
| ReDefiNER, `identifiers.txt`, `none_bias` 4 | 0.329 | 0.520 (cutoff 0.35) | 11th |
| ReDefiNER, `redefiner-pii` default rulebook | 0.312 | 0.393 (cutoff 0.40) | 15th |
| Perplexity PII-Tracer | | 0.601 | 6th |
| CobuildX 17M / 32M / 68M | | 0.536 / 0.535 / 0.532 | 7th, 8th, 10th |
| OpenAI Privacy Filter | | 0.533 | 9th |
| Fastino GLiNER2 PII | | 0.445 | 12th |

- **The two rulebooks swap places here.** The default rulebook masks dates, places, organisations and job titles,
  which PIIMB counts as personal data and RedactionBench mostly counts as over-masking. The identifiers rulebook
  masks less and scores higher.
- **Kind of document matters more than the rulebook.** With the identifiers rulebook the score is 0.54 to 0.72 on
  prose (medical, legal, academic, operations, financial, emails, government) and 0.29 to 0.39 on logs, files, code
  and terminal output. The model was not trained on machine-written text.
- **Six runs in total.** Besides these two, we ran four configurations with rulebooks written for RedactionBench
  from its dataset card and from public PII label lists. They scored 0.42 to 0.49, below the identifiers rulebook, and are not shipped. The two
  rulebooks shown were written without looking at RedactionBench.
- The other scores are from the leaderboard on the dataset card, read on 2026-11-18. General-purpose LLM agents
  hold its top five places.

## Reproduce

```bash
pip install "redefiner>=1.0.9" huggingface_hub
python run_piimb.py gretel                                                      # default rulebook: 94.07
python run_piimb.py gretel --rulebook rulebooks/identifiers.txt --none-bias 4   # 88.25
python run_piimb.py gretel --rulebook rulebooks/gretel.txt --max-span-tokens 48 # 94.49
```

The script downloads PIIMB's test rows (CC BY-NC 4.0) at the revision we used, saves the predictions, and prints F2,
precision and recall. A GPU is advisable: with the default rulebook the full Gretel test split took 25 minutes on an
NVIDIA A10 and Nemotron took 67 minutes, with two other runs sharing the GPU. Our scores come from that GPU; a CPU run can differ in the second
decimal. `--limit 200` gives a quick look on any machine.

## Limits

- English only. The two multilingual PIIMB tasks were not run.
- These are synthetic or curated benchmark texts. Check the model on your own data before relying on it, and do not
  use it as the only safeguard.
- The default rulebook contains a few example strings taken from the dev documents it was written from.

[piimb]: https://huggingface.co/datasets/piimb/pii-masking-benchmark
[redactionbench]: https://huggingface.co/datasets/RedactionBench/RedactionBench

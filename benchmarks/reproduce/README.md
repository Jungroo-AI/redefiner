# Reproduce

`reproduce.py` reruns one published number per mode on your machine and compares every prediction with the A10 run:

```bash
python benchmarks/reproduce/reproduce.py --model jungroo-ai/redefiner-0.8b [--device cpu] [--only extract|label]
```

| File | What it is | Licence |
|---|---|---|
| `crossner_ai_test.jsonl` | The CrossNER AI test set (431 sentences with gold spans) | [MIT](https://github.com/zliucr/CrossNER/blob/main/LICENSE), from [CrossNER](https://github.com/zliucr/CrossNER) |
| `crossner_ai_rulebook_seed0.json` | GoLLIE's CrossNER AI guidelines as a rulebook (seed-0 type order) | [Apache-2.0](https://github.com/hitz-zentroa/GoLLIE/blob/main/LICENSE), from [GoLLIE](https://github.com/hitz-zentroa/GoLLIE) |
| `crossner_ai_a10_predictions.jsonl` | ReDefiNER's predictions on the A10 | Apache-2.0 (contains spans of the MIT-licensed test text) |
| `s1mb_fewnerd_cases.jsonl` | The 358 S1MB Few-NERD labeling cases, inputs unchanged | **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**, from [Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd) via [S1MB](https://huggingface.co/datasets/hotchpotch/S1MB-english-v1); share-alike: redistributions of this file must keep this licence |
| `s1mb_fewnerd_a10_predictions.jsonl` | ReDefiNER's predicted type for each case | Apache-2.0 |
| `reproduce.py` | The script | Apache-2.0 |

See [Datasets and licences](../README.md#datasets-and-licences) for every dataset used on the benchmark page.

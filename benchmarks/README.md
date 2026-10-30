# ReDefiNER benchmarks

ReDefiNER is a 0.8B rule-following model for entity extraction and span labeling. You give it a rulebook (entity types, descriptions and rules) and text. In extract mode it returns labeled spans; in label mode it assigns a type to spans you supply. To change what it extracts, you edit the rulebook; the model is not retrained. This page reports both modes: extraction against GoLLIE, SLIMER, KnowCoder, B2NER, GLiNER, GNER, UniNER, InstructUIE and Liquid AI's Policy-Linter, and span labeling against the S1MB leaderboard and Liquid AI's d1 models.

**How to read the tables.** Every NER score is entity-level F1: a prediction counts only when its boundary and type both match the gold span exactly, micro-averaged within each dataset. S1MB scores are accuracy, the share of supplied spans given the right type. Scores are shown to one decimal. Bold marks the best score among the models compared: across the row when the columns are models, down the column when the rows are models. Baseline numbers are published results at their authors' own settings unless a row says "our run".

## Summary

| Benchmark | Metric | ReDefiNER 0.8B | Comparison |
|---|---|---:|---|
| 13 GoLLIE NER sets, GoLLIE's own guidelines | F1 | **59.8** | GoLLIE 34B 58.1 |
| 7 CrossNER and MIT sets, other instruction-following NER models | F1 | 69.9 | B2NER 20B **72.1**, B2NER 7B 69.8 |
| 13 GoLLIE sets, our rules tuned on dev (not equal-information) | F1 | **67.9** | GoLLIE 34B 58.1 |
| BUSTER, SLIMER's type wording | micro-F1 | 22.1 | SLIMER 7B **45.3**; Liquid AI Policy-Linter 0.2 (our run) |
| BUSTER, our rules tuned on dev | micro-F1 | 35.8 | SLIMER 7B **45.3**; Liquid AI Policy-Linter 2.2 (our run) |
| Open NER, 7 CrossNER and MIT sets, one-line type descriptions | F1 | **63.1** | GLiNER large 61.7, 62.2 with a dev-tuned threshold (our run) |
| Open NER, same sets, type names only | F1 | 56.2 | GNER-T5-large **63.5**; GLiNER large 61.6, 63.1 tuned, and NuNER Zero-span 62.6 (our runs) |
| S1MB span labeling, Few-NERD and Gretel PII | accuracy | 72.0 | best entry at 1B or less **92.3**; ReDefiNER ranks 87 of 208, 3 of 60 at 1B or less; Liquid AI's d1-2B 74.3 (our run) |

ReDefiNER benefits from descriptions and rules in the rulebook. Remaining weaknesses include pronoun and common-noun mentions, clinical text, transaction roles and extraction with type names only.

## Size and speed

| | |
|---|---|
| Parameters | 0.8B: Qwen3.5-0.8B fine-tuned with LoRA (rank 32, merged into the base) plus span and label heads. |
| Download | One 0.77 GB int8 file with the LoRA already merged. It is bit-identical to the int8 weights every benchmark on this page ran with. |
| Benchmark hardware | One NVIDIA A10, int8 weight-only with fp32 compute. |
| CPU, Apple M2 (8 cores), int8 | First text after loading or editing a rulebook about 3.5 s. Each further text 0.5 to 0.7 s. 16 texts in one call about 6.8 s. Measured with a 4,300-character rulebook of 14 types. |
| CPU parity | The package on the M2 reproduces the A10 runs: CrossNER AI 68.1 vs 68.1 F1 (428 of 431 sentences identical), S1MB Few-NERD 71.8 vs 71.8 accuracy (358 of 358 labels identical). |

Against the baselines, 0.8B is about 1/40 of GoLLIE 34B and 1/9 of the 7B models.

## Settings

Every ReDefiNER number on this page uses one fixed setting: int8 weight-only with fp32 compute, span threshold 0.5 (a span is kept when its span probability is above 0.5; its reported confidence also includes the type probability), NONE bias 2 (a constant that shifts the choice between "no entity" and a type; bias 2 labels more spans than bias 0), a 256-token query cap and a 16-token span cap. The Open NER comparison ran with a 160-token query cap; every one of its queries fits in 160 tokens, so its scores are unaffected. Nothing was tuned per benchmark except where a section says "rules tuned on dev". Runs date from 2026-10-23 to 2026-10-27, with the training checkpoint the release was exported from (sha256 9212dde5…, recorded as `source_checkpoint_sha256` in config.json). The released `model.safetensors` has sha256 8650567b860d59f01b1b930097a176cd4c99764e34e9d89446425d08954cf111.

Reproduce: `reproduce/reproduce.py` reruns CrossNER AI (extract mode) and S1MB Few-NERD (label mode) on your machine and compares every prediction with the A10 run. Every rulebook on this page is in `rulebooks/`; the two spot checks, with their inputs, predictions and scoring, are in `reproduce/`.

## Extraction with published GoLLIE guidelines

These are the 13 entity datasets GoLLIE (Sainz et al., ICLR 2024) reports as zero-shot. GoLLIE's own guideline text and five examples per type are converted into our rulebook format: the class name is the type, the first guideline sentence is the description, and the remaining sentences plus the examples are rule lines. Nothing was added. Loading and scoring use GoLLIE's code at commit 164c6117. ReDefiNER scores are the mean over 3 guideline orders (seeds 0, 24, 42 shuffle the order of the types in the rulebook). GoLLIE scores are taken from its paper, not rerun.

| Dataset | ReDefiNER 0.8B | GoLLIE 7B | GoLLIE 13B | GoLLIE 34B |
|---|---:|---:|---:|---:|
| FabNER | **32.7** | 24.8 | 25.4 | 26.3 |
| MultiNERD | **82.0** | 76.0 | 77.5 | 77.2 |
| BroadTwitter | 45.5 | 49.5 | **51.4** | 50.3 |
| HarveyNER | 33.5 | 37.3 | **41.3** | 38.9 |
| E3C | 47.7 | 59.0 | 59.0 | **60.0** |
| WikiEvents | 46.4 | 80.7 | 80.2 | **81.3** |
| CrossNER AI | **66.7** | 59.1 | 56.7 | 61.6 |
| CrossNER Literature | **70.1** | 62.7 | 59.7 | 59.1 |
| CrossNER Music | **73.7** | 67.8 | 65.5 | 68.4 |
| CrossNER Politics | **78.0** | 57.2 | 54.4 | 60.2 |
| CrossNER Science | **66.0** | 55.5 | 56.2 | 56.3 |
| MIT Movie | **71.3** | 63.0 | 62.5 | 62.4 |
| MIT Restaurant | **63.8** | 43.4 | 49.8 | 52.7 |
| **Mean, all 13** | **59.8** | 56.6 | 56.9 | 58.1 |

The spread over the 3 guideline orders (standard deviation) is at most 2.4 F1 (BroadTwitter) and under 1 for 10 of the 13 sets.

Across all 13 datasets, ReDefiNER averages 59.8 F1 versus 58.1 for GoLLIE 34B and leads it on 9 datasets. The largest loss is WikiEvents, which labels pronouns and common nouns ("his", "officials", "bomb") under the ACE05 scheme GoLLIE was trained on, while ReDefiNER mostly labels names. On E3C, HarveyNER and BroadTwitter its precision is low.

Reproduce: `reproduce/reproduce.py --only extract` reruns CrossNER AI with the seed-0 rulebook. The converted guidelines for every set and seed are in `rulebooks/gollie/<dataset>/gollie-guidelines-s{0,24,42}.json`.

### Other instruction-following NER models on the 7 CrossNER and MIT sets

SLIMER, KnowCoder and B2NER report zero-shot results on the same 7 test sets. Their numbers are copied from their papers, not rerun: SLIMER (Zamai et al., arXiv:2407.01272v3, Table 3), KnowCoder (Li et al., arXiv:2403.07969v2, Table 2, the variant without refinement on human-annotated data) and B2NER (Yang et al., arXiv:2406.11192v3, Table 3). Each model was scored with its own prompt and its own scorer; all count an entity only when its boundary and type are exact. ReDefiNER scores are the GoLLIE-guideline runs from the table above.

| Dataset | ReDefiNER 0.8B | SLIMER 7B | KnowCoder 7B | B2NER 7B | B2NER 20B |
|---|---:|---:|---:|---:|---:|
| CrossNER AI | **66.7** | 50.1 | 60.3 | 64.1 | 64.7 |
| CrossNER Literature | 70.1 | 58.7 | 61.1 | 69.0 | **71.6** |
| CrossNER Music | 73.7 | 60.0 | 70.0 | 76.4 | **82.4** |
| CrossNER Politics | 78.0 | 63.9 | 72.2 | 75.5 | **78.2** |
| CrossNER Science | 66.0 | 56.3 | 59.1 | 76.7 | **79.4** |
| MIT Movie | 71.3 | 50.9 | 50.0 | 70.2 | **71.4** |
| MIT Restaurant | **63.8** | 38.2 | 48.2 | 56.8 | 57.1 |
| **Mean** | 69.9 | 54.0 | 60.1 | 69.8 | **72.1** |

What each model sees per type: ReDefiNER gets GoLLIE's guideline and five examples; SLIMER a definition and guidelines; KnowCoder a class docstring with description and examples; B2NER the label name only (its paper tests guidelines only on Chinese data). SLIMER reports that 7% of these test types appear in its training data.

On the mean, ReDefiNER is ahead of SLIMER by 15.9 and KnowCoder by 9.8, level with B2NER 7B (69.9 vs 69.8) and 2.2 behind B2NER 20B. It is the best of the five on CrossNER AI and MIT Restaurant.

## Same model, better rules

Here the rulebook is changed and the weights are not. These are **not** equal-information comparisons: the baselines did not get tuned guidelines, and the rules were written by us (drafted with Claude) from dev errors only.

- **Dev splits:** each dataset's own dev split, except three. E3C uses its train split, since it has no public dev. MIT uses 500 rows from its train split, since it has no dev split. BUSTER uses fold 1 as dev and fold 5 as test.
- **How rules are added:** each GoLLIE guideline book from the section above is kept, and rule lines are added to it. Label names, label sets and descriptions are never changed.
- **Selection:** each dataset got between 2 and 7 rule versions, and the version with the best dev score was kept. For the CrossNER, MIT, FabNER and MultiNERD sets, only the kept version was run on test. For BroadTwitter, HarveyNER, E3C, WikiEvents and BUSTER, earlier versions had also been run on test.

Scores on the 13 GoLLIE datasets are the mean over the same 3 guideline orders as above. Their spread is at most 1.0 F1 (CrossNER AI) and under 1 for the other 12. "Best GoLLIE" is the best of the three GoLLIE sizes on that row, with its own guidelines. "Lines added" counts the rule lines added to GoLLIE's book; the link opens the kept rulebook (seed 0).

| Dataset | GoLLIE guidelines | Our rules, tuned on dev | Gain | Best GoLLIE | Lines added | Rulebook |
|---|---:|---:|---:|---:|---:|---|
| FabNER | 32.7 | **35.4** | +2.7 | 26.3 (34B) | 24 | [v3b](rulebooks/dev-tuned/fabner/tuned-rules-s0.json) |
| MultiNERD | 82.0 | **87.6** | +5.6 | 77.5 (13B) | 15 | [v3b](rulebooks/dev-tuned/multinerd/tuned-rules-s0.json) |
| BroadTwitter | 45.5 | **70.6** | +25.1 | 51.4 (13B) | 7 | [v2](rulebooks/dev-tuned/broadtwitter/tuned-rules-s0.json) |
| HarveyNER | 33.5 | **48.1** | +14.6 | 41.3 (13B) | 11 | [v3](rulebooks/dev-tuned/harveyner/tuned-rules-s0.json) |
| E3C | 47.7 | 59.4 | +11.7 | **60.0** (34B) | 7 | [v5](rulebooks/dev-tuned/e3c/tuned-rules-s0.json) |
| WikiEvents | 46.4 | 53.6 | +7.2 | **81.3** (34B) | 22 | [v3](rulebooks/dev-tuned/wikievents/tuned-rules-s0.json) |
| CrossNER AI | 66.7 | **74.4** | +7.7 | 61.6 (34B) | 22 | [v3b](rulebooks/dev-tuned/crossner-ai/tuned-rules-s0.json) |
| CrossNER Literature | 70.1 | **71.5** | +1.4 | 62.7 (7B) | 10 | [v1](rulebooks/dev-tuned/crossner-literature/tuned-rules-s0.json) |
| CrossNER Music | 73.7 | **79.5** | +5.8 | 68.4 (34B) | 13 | [v3](rulebooks/dev-tuned/crossner-music/tuned-rules-s0.json) |
| CrossNER Politics | 78.0 | **80.7** | +2.7 | 60.2 (34B) | 7 | [v2](rulebooks/dev-tuned/crossner-politics/tuned-rules-s0.json) |
| CrossNER Science | 66.0 | **75.4** | +9.4 | 56.3 (34B) | 9 | [v2](rulebooks/dev-tuned/crossner-science/tuned-rules-s0.json) |
| MIT Movie | 71.3 | **78.5** | +7.2 | 63.0 (7B) | 15 | [v3](rulebooks/dev-tuned/mit-movie/tuned-rules-s0.json) |
| MIT Restaurant | 63.8 | **68.1** | +4.3 | 52.7 (34B) | 11 | [v3b](rulebooks/dev-tuned/mit-restaurant/tuned-rules-s0.json) |
| **Mean, all 13** | 59.8 | **67.9** | +8.1 | 58.1 (34B) | | |

Rules improved every one of the 13 datasets on test. With them, ReDefiNER beats every GoLLIE size on 11 of 13 sets and is behind on E3C (by 0.6) and WikiEvents. The cost is small: each kept rulebook adds 7 to 24 rule lines (median 11, 1.1 to 3.7 thousand characters) to GoLLIE's book, written from dev errors in at most three rounds (five for E3C).

Reproduce: the kept rulebooks for all 3 seeds are in `rulebooks/dev-tuned/<dataset>/tuned-rules-s{0,24,42}.json`; the GoLLIE-only books they extend are in `rulebooks/gollie/<dataset>/`.

### BUSTER

BUSTER is a NER set of M&A documents with six transaction-role types (buyer, acquired company, seller, legal and generic advisors, annual revenues). We follow SLIMER's protocol: fold 5 is the test split, fold 1 is dev, documents are cut into 900-character windows, and the score is micro-F1 over the six types with UniNER's evaluator. The one departure is that windows over 256 tokens were split into pieces of at most 250 tokens. Two rulebooks were run, both at the fixed settings above.

| Rulebook or model | micro-F1 | Precision | Recall |
|---|---:|---:|---:|
| ReDefiNER, SLIMER's definitions and guidelines as the rulebook (16 rule lines) | 22.1 | 13.8 | 55.1 |
| ReDefiNER, our rules written on fold 1 (38 rule lines, a rewrite rather than additions) | 35.8 | 23.9 | 71.1 |
| SLIMER 7B, reported | **45.3** | | |
| UniNER-7B-type, reported | 34.8 | | |
| GoLLIE, reported | 27.7 | | |
| GLiNER-L, reported | 26.6 | | |
| GNER-LLaMA, reported | 23.6 | | |

Reported rows are from SLIMER (Zamai et al., arXiv:2407.01272v3, Table 5), with each model's own prompt. With SLIMER's wording ReDefiNER is last: it labels every money amount as revenue and gives most companies a role. Our rules raise it to 35.8, still 9.5 below SLIMER, and the result moves 22 to 36 on wording alone. The remaining errors, sellers and revenue amounts, are rules that go against a strong prior, which the model ignores (see What worked and what didn't).

Reproduce: `rulebooks/buster/slimer-wording.txt` and `rulebooks/dev-tuned/buster/tuned-rules-v5.txt`.

## Open NER comparisons

Mean F1 over the same 7 CrossNER and MIT test sets (6,470 sentences), in two conditions: types given with one-line descriptions, and type names only. This run used a 160-token query cap; the longest query is 138 tokens.

Our runs of other models show two numbers: at the model's default threshold, then at a threshold chosen on the CrossNER and MIT dev sets (0.10 to 0.90). ReDefiNER is shown at its one fixed release setting.

| Model | With descriptions | Names only |
|---|---:|---:|
| ReDefiNER 0.8B | **63.1** | 56.2 |
| GLiNER large v2.5, our run | 61.7 / 62.2 | 61.6 / 63.1 |
| GLiNER2.5 base, our run | 58.5 / – | 55.5 / – |
| NuNER Zero-span 0.4B, our run | 26.5 / 55.5 | 62.6 / 62.6 |
| NuNER Zero 0.4B, our run | 25.8 / 46.6 | 56.8 / 56.9 |
| Liquid AI Policy-Linter 350M, our run | 47.5 / 48.5 | – |
| GNER-T5-large 0.78B, reported | – | **63.5** |
| UniNER-7B-definition, reported | 45.0 | – |
| UniNER-7B-type, reported | – | 53.4 |
| InstructUIE 11B, reported | – | 47.3 |

ReDefiNER leads when types are described, also against every baseline with its threshold tuned on dev. With type names only it trails GLiNER large by 5.4 points (6.9 with GLiNER's threshold tuned) and NuNER Zero-span by 6.4. Descriptions hurt both NuNER models, which were trained on short type names. GNER-T5-large, at a similar size, reports 63.5 with type names only, 7.3 above ReDefiNER. UniNER-7B-type (53.4) and InstructUIE 11B (47.3) are both below ReDefiNER with type names only; UniNER-7B-definition is reported at 45.0 with its own generated type definitions.

The GLiNER and NuNER rows are our runs on the same 6,470 sentences, with each type's description (or its name) as the label, through the gliner library (0.2.29 for the reruns; rerunning GLiNER large at its default reproduces the earlier 61.67 and 61.58 exactly). NuNER labels are lower-cased, as its model card requires, and NuNER Zero's token predictions are joined into spans as its card's merge_entities does. NuNER Zero's own reported 68.0 on these sets is token-level F1; scored that way, our run gives 67.9. The GLiNER2.5 base outputs were not kept, so it has no tuned number. The Policy-Linter row is our run too; see the next section. The "reported" rows are copied from papers, not rerun: GNER-T5-large from Ding et al. (arXiv:2402.16602, Table 2), UniNER-7B-definition from SLIMER (Zamai et al., arXiv:2407.01272v3, Table 3; the UniNER paper gives no per-dataset scores for this variant), UniNER-7B-type from Zhou et al. (ICLR 2024, Table 3) and InstructUIE from Wang et al. (arXiv:2304.08085v1, Table 4; mean of its 7 per-dataset scores, which the paper does not average). Each used its own prompt and scorer, counting an entity only when boundary and type are exact. GNER and both UniNER rows use UniNER's test setup, which drops CrossNER's catch-all "misc" type; InstructUIE, ReDefiNER, GLiNER and NuNER keep it.

Reproduce: the rulebooks for both conditions are in `rulebooks/open-ner/descriptions/` and `rulebooks/open-ner/names/`.

### Liquid AI's Policy-Linter, our run

[LFM2.5-Encoder-350M-Policy-Linter](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Policy-Linter) takes free-text rules at inference time and scores every token of a text against every rule, so, like ReDefiNER, it is steered by rules rather than retrained. Liquid reports no scores for it, so we ran it on the sentences and rulebooks ReDefiNER got: the 7 CrossNER and MIT test sets with one-line descriptions, and BUSTER fold 5 with both rulebooks.

- **Input:** as in the model card and Liquid's demo: "Policy:", one line per rule, then "Text:". Each type becomes one rule, its description followed by its rule lines; NOT_AN_ENTITY lines become one extra rule whose matches are dropped. General [RULES] lines have no place in its input, which takes one rule per label, so they are left out.
- **Spans:** as in Liquid's demo, each word takes the rule with the highest probability above the threshold. The demo highlights words one at a time; for entity spans we join neighbouring words that get the same type and trim punctuation at the edges, which helps it.
- **Settings chosen on the CrossNER and MIT dev sets:** three rule phrasings, each at thresholds from 0.20 to 0.90 (table below). The demo's phrasing ("Filter …") at 0.65 was kept.
- **Scoring:** it returns positions, so CrossNER and MIT use the same exact-span scorer as ReDefiNER's 63.1, and BUSTER uses SLIMER's scorer, as above.

| Dataset | ReDefiNER 0.8B | Policy-Linter 350M |
|---|---:|---:|
| CrossNER AI | **55.6** | 50.5 |
| CrossNER Literature | **58.4** | 48.9 |
| CrossNER Music | **72.4** | 63.2 |
| CrossNER Politics | **70.2** | 42.5 |
| CrossNER Science | **63.9** | 49.5 |
| MIT Movie | **68.4** | 39.7 |
| MIT Restaurant | **53.2** | 45.1 |
| **Mean, 7 sets** | **63.1** | 48.5 |
| BUSTER, SLIMER's wording ¹ | **22.1** | 0.2 |
| BUSTER, our rules ¹ | **35.8** | 2.2 |

¹ At the model card's default threshold, 0.5, where the Policy-Linter scores higher than at the dev-chosen 0.65 (0.2 and 0.1). With BUSTER's long rules its probabilities stay low (the highest is 0.73), and no threshold from 0.20 to 0.90 takes it above 3.5.

| Rule phrasing (CrossNER and MIT dev) | Best threshold | Mean F1 |
|---|---:|---:|
| "Filter " + the description (kept) | 0.65 | **48.3** |
| The description alone | 0.60 | 46.9 |
| "TYPE: description" | 0.65 | 45.6 |

At the model card's 0.5 the kept phrasing scores 47.1 on dev. The Policy-Linter scores each word on its own, so multi-word names break when an inner word matches another rule. In the dev sentence "The Conservative Party of Canada took back this historically New Democratic Party ( NDP ) seat in 2004 Canadian federal election", it returns country "Canada", political party "New Democratic Party ( NDP", election "2004", country "Canadian" and election "federal election". Its precision on the test sets is 38 to 62%.

Reproduce: the scripts, outputs, scores and threshold sweeps are in `liquid/`.

## Span labeling on S1MB

[S1MB](https://huggingface.co/datasets/hotchpotch/S1MB-english-v1) gives the model a sentence, a span and a fixed list of candidate types with descriptions, and asks for the type. We used the benchmark's inputs unchanged and report only its two entity-labeling tasks. Other models' scores are from the S1MB leaderboard (results revision 5142118d). Means are computed from unrounded scores. Bold marks the best score in each column.

| Model | Params | Few-NERD | Gretel PII | Mean |
|---|---:|---:|---:|---:|
| bekko-system-one-v0-400m ¹ | 0.39B | **89.4** | 95.2 | **92.3** |
| Superion S1 | n/a ² | 88.5 | **95.5** | 92.0 |
| bekko-system-one-v0-68m ¹ | 0.07B | 87.2 | 90.1 | 88.6 |
| Jev 1.13.0 | n/a ² | 80.2 | 88.1 | 84.1 |
| Fastino GLiDE (hosted API) | n/a ² | 83.2 | 74.2 | 78.7 |
| d1-2B (Liquid AI), our run | 1.92B | 70.1 | 78.5 | 74.3 |
| ReDefiNER (this page) | 0.8B | 71.8 | 72.2 | 72.0 |
| bekko-system-one-v0-17m ¹ | 0.02B | 76.0 | 66.9 | 71.4 |
| Saihai v1 310M | 0.31B | 74.9 | 65.7 | 70.3 |
| Mapika/decider-0.8b | 0.75B | 59.5 | 68.6 | 64.0 |
| GLiNER2.5-Decide | 0.49B | 49.7 | 27.2 | 38.5 |
| gliner2-large-v1 | 0.49B | 42.2 | 27.2 | 34.7 |
| d1-omni-600M (Liquid AI), our run | 0.6B | 31.8 | 30.6 | 31.2 |
| laya | 0.42B | 15.4 | 22.7 | 19.0 |

On the mean of these two tasks, ReDefiNER ranks 87th of 208 (the 207 leaderboard entries plus ReDefiNER) and 3rd of the 60 with a reported size of 1B or less. This is not a rank on the overall S1MB benchmark, which covers 137 tasks.

The two d1 rows are our runs of Liquid AI's decision models, not leaderboard entries, and are left out of the ranks. S1MB's records were passed to d1's `system_one` as they are: the case's state, and each decision as a choice question with its candidate types and descriptions, on the same 711 decisions ReDefiNER answered (float16 for d1-omni-600M and bfloat16 for d1-2B, as their model cards recommend). d1-2B, at 2.4 times ReDefiNER's size, is ahead of it on Gretel PII and on the mean.

¹ Trained on bekko-system-one-dataset-v0, which includes the Few-NERD and Gretel PII train splits; its split audit reports 0 exact matches with the test splits.
² Size not reported on the leaderboard.

Reproduce: `reproduce/reproduce.py --only label` reruns Few-NERD; its inputs, with the exact rulebook for each case, are in `reproduce/s1mb_fewnerd_cases.jsonl`.

## What worked and what didn't

Notes from writing the dev-tuned rulebooks above (counts are from dev).

- **Rules that fix boundaries or remove a common false positive work.** On BroadTwitter, the rule that the span includes the "@" raised @-mention recall from 17% to 61%. On HarveyNER, a rule never to label "Houston", "Texas" or "Harvey" cut those predictions from 389 to 5.
- **Rules that separate two confusable types give the biggest gains.** "Scientist, not Person" cut Scientist spans labeled Person from 212 to 23 on CrossNER Science. Similar rules worked for MusicalArtist vs Person (100 to 25) and PoliticalParty vs Organization (76 to 19).
- **A quoted example counts as a positive for the type it is listed under, even in a "never" sentence.** On MultiNERD, a rule under the Time type saying years are never Time, with "2014" as its example, raised years labeled Time from 2,630 to 4,331. The same exclusion moved to the general rules, still quoting "2014" and "19th century", cut Time false positives from 2,268 to 443.
- **Pronouns quoted in a short list are followed; common nouns mostly aren't.** On WikiEvents, quoting "his" and "their" raised them from 6 and 1 correct to 24 each, and pronouns left out of the list fell back. Group nouns were almost never labeled correctly, even when quoted: "authorities" at most 1 of 10, "forces" 0 of 14. WikiEvents' remaining gap is these nominal mentions.
- **On E3C, rule text first made the model label less overall** (746 to about 390 predictions). Listing example disorders under the type itself brought recall back, and a short-span line then cut long false positives.
- **Dev gains shrink on test after several rounds.** CrossNER AI gained 12.8 on dev but 7.7 on test, so revision stopped at three rounds for every dataset except E3C, which got five.
- **On BUSTER, the model ignores rules that go against a strong prior.** Rules saying "only revenue amounts, not every money amount" left revenue precision at about 9%. Results are also sensitive to wording: 22 to 36 from rewording alone.

## Datasets and licences

The benchmark datasets are not part of this release and keep their own licences. The test sentences, gold spans and
short quoted examples that appear on this page, in `rulebooks/`, in `reproduce/` and in the demo are excerpts from
these datasets for evaluation and are covered by the dataset's licence below, not by this repository's Apache-2.0
licence. Please cite the original papers when you use them.

| Dataset | Licence | Excerpts in this release |
|---|---|---|
| [CrossNER](https://github.com/zliucr/CrossNER) (AI, Literature, Music, Politics, Science) | [MIT](https://github.com/zliucr/CrossNER/blob/main/LICENSE) | the CrossNER AI test set in `reproduce/`; one demo sentence per domain; quoted spans in rulebooks |
| [Broad Twitter Corpus](https://github.com/GateNLP/broad_twitter_corpus) | [CC BY 4.0](https://github.com/GateNLP/broad_twitter_corpus/blob/master/LICENSE) | one demo tweet; quoted spans in rulebooks |
| [HarveyNER](https://github.com/brickee/HarveyNER) | not stated by the authors | one demo tweet; quoted spans in rulebooks |
| [MultiNERD](https://github.com/Babelscape/multinerd) | [CC BY-NC-SA 4.0](https://github.com/Babelscape/multinerd/blob/master/LICENSE) | one demo sentence; quoted spans in rulebooks |
| [E3C](https://github.com/hltfbk/E3C-Corpus) | CC BY-NC (per the [corpus README](https://github.com/hltfbk/E3C-Corpus)) | one demo sentence; quoted spans in rulebooks |
| [WikiEvents](https://github.com/raspberryice/gen-arg) | [MIT](https://github.com/raspberryice/gen-arg/blob/main/LICENSE) | one demo sentence; quoted spans in rulebooks |
| [FabNER](https://huggingface.co/datasets/DFKI-SLT/fabner) | not stated by the authors | one demo sentence; quoted spans in rulebooks |
| MIT Movie and MIT Restaurant ([MIT SLS](https://groups.csail.mit.edu/sls/downloads/)) | not stated by the authors | one demo query each; quoted spans in rulebooks |
| [BUSTER](https://huggingface.co/datasets/expertai/BUSTER) | [Apache-2.0](https://huggingface.co/datasets/expertai/BUSTER) | quoted spans in rulebooks |
| [S1MB](https://huggingface.co/datasets/hotchpotch/S1MB-english-v1): [Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd) | [CC BY-SA 4.0](https://huggingface.co/datasets/DFKI-SLT/few-nerd) | the 358 Few-NERD cases in `reproduce/` (shared under CC BY-SA 4.0); one demo sentence |
| S1MB: [Gretel PII](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1) | [Apache-2.0](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1) | one demo document |
| [GoLLIE](https://github.com/hitz-zentroa/GoLLIE) guidelines | [Apache-2.0](https://github.com/hitz-zentroa/GoLLIE/blob/main/LICENSE) | the guideline text in `rulebooks/gollie/` and in the tuned rulebooks built on it |
| [SLIMER](https://github.com/andrewzamai/SLIMER) BUSTER wording | [Apache-2.0](https://github.com/andrewzamai/SLIMER/blob/main/LICENSE) | `rulebooks/buster/slimer-wording.txt` |

If you are a dataset author and want an excerpt removed, open an issue and we will take it out.

## Disclosures

- **Model selection:** CrossNER and MIT test scores were tracked during development alongside internal benchmarks; the final checkpoint was chosen on dev sets only.
- **Baselines:** GoLLIE, SLIMER, KnowCoder, B2NER, GNER, UniNER, InstructUIE and S1MB numbers are published results, not reruns; the GLiNER, NuNER and Policy-Linter rows in the Open NER table, and the two d1 rows in the S1MB table, are our runs. GoLLIE's guideline errors (CrossNER Music Song/Band reusing the genre definition, Award reusing the instrument definition, MIT "Tittle") are kept verbatim, as GoLLIE was scored with them.
- **Limits:** spans over 16 tokens can't be predicted; text is read 256 tokens at a time.
- **Reproducibility:** every rulebook on this page is in `rulebooks/`, and the two spot checks, with their inputs, predictions and scoring, are in `reproduce/`. The Policy-Linter and d1 runs are in `liquid/`, the NuNER and GLiNER reruns in `gliner-family/`.

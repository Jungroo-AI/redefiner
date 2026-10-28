# Liquid AI comparisons

Files behind the Policy-Linter section and the two d1 rows of the [benchmark page](../README.md).

**Policy-Linter** (`LiquidAI/LFM2.5-Encoder-350M-Policy-Linter`, revision 2a56cb94a7083a90263ea5270523ccc10e7ce8af,
float32, one NVIDIA A10, 2026-10-29):

- `outputs/policy_linter_dev.jsonl.gz`: CrossNER and MIT dev, all three rule phrasings. `policy_linter_test.jsonl.gz`:
  the 7 test sets and BUSTER fold 5 (both rulebooks), with the phrasing chosen on dev. Each line keeps every word of
  the text with its best rule and that rule's probability (`[start, end, rule index, probability]`), so spans can be
  decoded at any threshold.
- `policy_scores_dev.json`: mean dev F1 for every phrasing and threshold. `policy_scores_test.json`: the test scores
  at the chosen setting. `policy_buster_by_threshold.json`: BUSTER at every threshold from 0.20 to 0.60.
- `scripts/run_linter.py` runs the model; `build_linter_requests.py` and `score_linter.py` build the rules and score
  the runs. The last two load the datasets through our evaluation harness, which is not in this repository (pass it
  with `--harness`), so they show exactly what was done; `score_linter.py test` reproduces `policy_scores_test.json`.

**d1** (`LiquidAI/d1-omni-600M` at 02b55d7076f15129e59ab3f94783f32c4b088674 in float16, also rerun in float32 with the
same answers, and `LiquidAI/d1-2B` at cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41 in bfloat16):

- `outputs/d1-*.jsonl.gz`: every S1MB decision with the gold type and d1's full answer (choice, confidence and the
  probability of each candidate).
- `scripts/run_d1.py` reads the pinned S1MB files (revision 10bcb8c6916b98b966cc1f03aab1781a7c354b4c, sha256-checked)
  and passes each case to `system_one` unchanged.

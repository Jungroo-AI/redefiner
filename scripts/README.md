# Scripts

Release:

- `export_weights.py CHECKPOINT OUT_DIR` turns a training checkpoint into the int8 release folder (weights, config,
  tokenizer, LICENSE, NOTICE). Needs `pip install -e .[export]`.

Demo data (the stored outputs the demo shows; each script runs the release weights on CPU):

1. `build_examples.py SUITES_DIR WEIGHTS_DIR` writes `space/examples.json`: the benchmark test sentences of the Edit
   a rule and Benchmarks tabs.
2. `build_demo_rules.py WEIGHTS_DIR` adds a Never extract and a Closing rule to the three benchmark examples in the
   Edit a rule tab, and fails if that changes their output.
3. `build_demo_examples.py WEIGHTS_DIR` writes `space/demo_examples.json`: the examples written for the demo (support
   ticket, job application, clinical note) and their rulebooks, and fails if a kind of rule stops having the effect
   its caption describes.
4. `build_bench_examples.py SUITES_DIR S1MB_CASES WEIGHTS_DIR` writes `space/bench_examples.json`: one example for each
   remaining benchmark dataset.

Scripts 1 and 4 read the evaluation suites (`suites.py`), which hold benchmark test data and are not part of this
repository.

Docs:

- `build_rule_cards.py WEIGHTS_DIR` draws the "See a rule at work" cards (`docs/assets/rules/`) from real model
  output, and fails if any output differs from what the README shows. Needs Chrome for the rendering.

# Changelog

Changes to the `redefiner` package and its documentation. The model weights are the same in every release
(1.0, October 2026).

## 1.0.8 (2026-11-05)

Docs and package metadata only. Same weights, and the same outputs as 1.0.7 for every valid input.

- A one-sentence description on PyPI, GitHub and the model card.
- Links to jungroo.com/redefiner, for help with your own data, in the README, the model card and the demo.
- A training-data section in the README and the model card.
- PyPI's project links include the company page.
- The citation names the authors.

## 1.0.7 (2026-11-02)

Same weights, and the same outputs as 1.0.6 for every valid input.

- Works with transformers 5.19 to 5.22 (1.0.6 needed 5.19). transformers 5.20 changed how the rotary embedding is
  created and 5.21 how the Gated DeltaNet layers read their cache. Outputs are identical to 1.0.6 on each version.
- Wrong input types raise a `TypeError` that says what was expected: texts that aren't a string or a list of strings,
  a rulebook that is neither text nor a dict, and `label` spans that aren't lists. Before, `extract(None, rulebook)`
  and `extract(text, None)` failed with Python's own errors from inside the package.
- An empty rulebook is reported as empty, not as "no types found".
- `python -m redefiner` runs the command line, like `redefiner`.
- The Quickstart explains `threshold` and `confidence` up front, outside the collapsed section, and suggests choosing
  a confidence cutoff on 20 to 50 labelled texts of your own.
- The weekly model tests run with the lowest and the highest transformers the package allows, and with the newest
  release.
- The Development section's download command works. In 1.0.6 it gave two patterns to one `--exclude`, and
  `hf download` read the second one as a filename.

## 1.0.6 (2026-11-02)

- `label(..., snap=True)` trims spaces from both ends of a span before widening it to whole tokens. Before, a span
  with a trailing space was widened into the next token: "Ada Lovelace " became "Ada Lovelace correspond". A span of
  only whitespace now comes back unsupported.
- Label mode docs say what snapping does: it widens a span to whole tokens, not words, so a snapped span can still end
  inside a word, and part of a word that starts and ends on token boundaries is labeled as given, with no flag.
- The PyPI page shows its images, and its links work.
- The headline gives each gap: about 5 F1 behind Claude Haiku 5.5 and 4.3 behind Gemini 3.1 Pro on CrossNER AI.
- Limits say what long texts mean: each part is read on its own, so labels can differ across a document.
- Python 3.14 is tested in CI. The weekly model tests also run with the newest transformers, to show early when the
  pinned version range needs testing.
- The Development section downloads the model files without the benchmark outputs and images.
- This changelog.

## 1.0.5 (2026-11-01)

Docs and package metadata only.

- A frontier comparison (`benchmarks/frontier/`): ReDefiNER against Claude Opus 5.5, Claude Opus 4.8, GPT-5.4,
  Claude Haiku 5.5 and Gemini 3.1 Pro, given the same rulebooks, with the time and cost of each request.
- New headline and description; a shorter "At a glance" and Quickstart; benchmark notes that say what each number
  covers; 2-vCPU server timings.
- PyPI summary, authors and keywords.

## 1.0.4 (2026-11-01)

Every rulebook with one section per type gives the same output as 1.0.3.

- A type defined twice is a `ValueError`. Before, it was accepted, and with another type between the two sections
  their labels were swapped.
- A dict's `"closing"` rules are written under a second `[RULES]` line, as the docs and the demo already did.
- `tagger.device` says where the model runs. A `RuntimeWarning` when the model falls back to CPU on a machine with an
  NVIDIA driver. `redefiner serve` prints the device at startup, and `GET /info` includes it.
- Docs: a "Rule lines" section, and `llms.txt` for AI assistants and coding agents.

## 1.0.3 (2026-10-30)

Every well-formed rulebook gives the same output as 1.0.2.

- A malformed type header, such as `[TYPE] DATE-OF-BIRTH`, is an error that quotes the line. Before, the type was
  dropped silently.
- Windows line endings and a byte-order mark are removed before a rulebook is read.
- A rulebook that mixes the two layouts, a dict without `"classes"` and a class without `"name"` each get a clear
  error.
- A malformed span is a `ValueError`, so the server answers 400 instead of 500. `allow_none` and `snap` must be JSON
  booleans; `threshold` and `none_bias` must be numbers.
- A local path that doesn't exist raises `FileNotFoundError` instead of being tried as a Hugging Face repo.
- `extract` never returns a whitespace-only span.
- The source package leaves out tests that can't run outside the repository. Python 3.13 is tested in CI.

## 1.0.2 (2026-10-29)

Docs only: how `confidence` relates to `threshold`, the device default, limits and results up front, install
requirements, and Jev 1.13.0 and Fastino GLiDE in the S1MB comparison.

## 1.0.1 (2026-10-29)

- `Tagger.from_pretrained` downloads only the 9 files it needs from the Hub.
- No "much slower" warning from transformers on CPU.
- README: an HTTP server section.

## 1.0.0 (2026-10-28)

First release.

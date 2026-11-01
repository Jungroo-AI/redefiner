# ReDefiNER against frontier models

Five frontier models, given the same rulebooks as ReDefiNER, on the two sets in [`../reproduce/`](../reproduce/).
ReDefiNER trails all five on accuracy, by 4.3 to 11.3 F1 on extraction and 9.5 to 15.4 points on labeling. For one
extraction request with a 2,000-token prompt, on one GPU, it answers in 0.05 seconds with the rulebook already loaded
(0.39 seconds with a new one), against 2 to 17 seconds for the frontier APIs. On an RTX A5000 GPU, rented at $0.16 an
hour, it costs about 85 times less than Claude Haiku 5.5 and 1,100 to 7,800 times less than the larger models when the
rulebook is cached on both sides, and about 20 times and 540 to 1,200 times less when every request brings a new
rulebook. With no GPU, on a 2 vCPU server, it takes 1.3 seconds with the rulebook loaded and 16 seconds with a new
one, and still costs about 9 times less than Claude Haiku 5.5 and 110 to 790 times less than the larger models with
the rulebook cached, and 1.4 times and 35 to 79 times less with a new one. Run on 2026-11-01.

| Model | CrossNER AI extract, F1 | S1MB Few-NERD label, accuracy | Cost per 1,000 requests, rulebook cached | Cost per 1,000 requests, new rulebook | Seconds per request |
|---|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | **79.41** | **87.2** | $2.42 | $10.02 | 2.2 |
| Claude Opus 4.8 | 77.13 | 84.4 | $2.92 | $10.65 | 4.2 |
| GPT-5.4 | 73.54 | 81.3 | $5.32 | $9.41 | 8.6 |
| Claude Haiku 5.5 | 73.09 | 81.8 | $0.19 | $0.37 | 2.3 |
| Gemini 3.1 Pro | 72.42 | 81.6 | $17.27 | $20.90 | 16.8 |
| ReDefiNER 0.8B, int8, RTX A5000 GPU | 68.08 | 71.8 | $0.0022 | $0.017 | 0.05 cached, 0.39 new |
| ReDefiNER 0.8B, int8, 2 vCPU CPU server, no GPU | 68.14 | 71.8 | $0.022 | $0.27 | 1.3 cached, 16.3 new |

Cost and time are for one CrossNER AI extraction request: the rulebook, instructions and one sentence, about 2,000
tokens; see Setup.

## Setup

- **Data:** CrossNER AI test, 431 sentences, find the spans (micro-F1 on exact start, end and type); S1MB Few-NERD,
  358 spans, label each span with one of the case's 8 types (accuracy). The same files and gold as
  `reproduce/reproduce.py`.
- **Rulebooks:** each model gets the rulebook text ReDefiNER gets: GoLLIE's guidelines for CrossNER AI, the
  names-only S1MB type lists for Few-NERD, with the sentence (or the span to label) and the JSON shape to return.
  The CrossNER AI rulebook lists its types in seed-0 order, the one in `reproduce/`; the GoLLIE table on the
  [benchmark page](../README.md) reports the mean over three type orders, 66.7 for ReDefiNER against 68.1 here.
- **Answers:** JSON. The frontier models return span text, which we place at its next occurrence in the sentence.
  Every answer parsed, and every returned span but one (Claude Haiku 5.5) was found in its sentence, so next to no
  score is lost to formatting.
- **Settings:** Claude Opus 5.5 and Haiku 5.5 at their API defaults (adaptive thinking, effort medium), with the JSON
  constrained to the rulebook's types by structured outputs. GPT-5.4, Gemini 3.1 Pro and Claude Opus 4.8 with web
  search off and reasoning at the model's usual level: GPT-5.4 reasoning effort medium, Gemini 3.1 Pro thinking high,
  Claude Opus 4.8 effort high.
- **Cost:** 1,000 extraction requests with a 2,000-token prompt. Frontier models: 2,000 input tokens plus the model's
  measured output tokens per CrossNER AI request (reasoning included), at the per-token prices we paid on 2026-11-01.
  *Rulebook cached:* the prompt at the cache-read price; for GPT-5.4, Gemini 3.1 Pro and Claude Opus 4.8, where the
  API we used listed none, 10% of the input price, the providers' usual cache discount. *New rulebook:* the prompt at
  the full input price. ReDefiNER on GPU: its seconds per request times $0.16 per hour, the cheapest on-demand price
  we found for a cloud GPU (an NVIDIA RTX A5000, October 2026), with the GPU kept busy: 0.05 s, or $0.0022 per 1,000,
  with the rulebook loaded; 0.39 s, or $0.017, with a new one (1,000 x 0.39 s = 0.108 hours). ReDefiNER on CPU: a 2
  vCPU server (Azure Standard_D2as_v6, $0.0586 per hour), one request at a time: 1.34 s, or $0.022 per 1,000, with
  the rulebook loaded; 16.3 s, or $0.27, with a new one. On that server it scored 68.14 F1 and 71.8%, the same as the
  A10 run in `reproduce/`.
- **Time:** seconds per request. Frontier models: wall time per CrossNER AI API request, measured with 8 requests in
  flight and taken here as single-request latency. ReDefiNER on GPU: the README's measurement on an NVIDIA A10 with a
  2,000-token rulebook (README, Speed), 0.05 seconds with the rulebook already loaded and 0.39 seconds for the first
  text with a new one. The same timing on an RTX A5000 gave the same times.
  ReDefiNER on CPU: measured on that server, one request at a time with the CrossNER AI rulebook the frontier models
  got (1,074 tokens in ReDefiNER's tokenizer): the mean over 20 sentences with the rulebook loaded, and over 10 with
  it cleared from the cache before each request.

## Limits of this comparison

- GPT-5.4, Gemini 3.1 Pro and Claude Opus 4.8 ran through a third-party API gateway, not each provider's own API. It
  does not enforce a JSON schema, so those models got the JSON shape and the allowed types in the prompt instead.
- One run per model, no seeds; frontier answers can vary between runs.
- ReDefiNER's GPU times come from timing a 2,000-token rulebook on an NVIDIA A10 and an RTX A5000 (the same on
  both), not from a rerun on these two sets; label mode was not timed on the GPU.
- The 2,000-token costs are computed from per-token prices and measured output tokens, not billed; the cached column
  assumes a 10% cache-read price for GPT-5.4, Gemini 3.1 Pro and Claude Opus 4.8.
- GPU prices change: $0.16 per hour is the cheapest on-demand price we found in October 2026, on machines hosted by
  third parties.
- Two test sets, both English: an extraction set with long guideline rulebooks and a labeling set with type names
  only. The GoLLIE comparisons on [the benchmark page](../README.md) cover more datasets.

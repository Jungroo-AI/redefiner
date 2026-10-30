# Rulebook templates

Copy a file, change the types, descriptions and rules to fit your task, and pass the text as the rulebook. The
templates are in this repository, not in the pip package: read one from a clone (`Path("docs/templates/pii.txt")`), or
download it:

```python
import urllib.request
from redefiner import Tagger

tagger = Tagger.from_pretrained("jungroo-ai/redefiner-0.8b")
url = "https://raw.githubusercontent.com/Jungroo-AI/redefiner/main/docs/templates/pii.txt"
rulebook = urllib.request.urlopen(url).read().decode()
tagger.extract("Please update the record for Maria Gonzalez, born 14 May 1987.", rulebook)
```

## Starting points

| File | For | What it shows |
|---|---|---|
| [`blank.txt`](blank.txt) | Any task | The layout: general rules, one section per type, never-extract rules, closing rules |
| [`hard-rules.txt`](hard-rules.txt) | Types that depend on context | The structure for conflicts between types: precedence rules and numbered rules with worked examples. Replace every `<...>`. See [Structure for hard rules](../writing-rules.md#7-structure-for-hard-rules) |
| [`../examples/travel-desk-rulebook.txt`](../examples/travel-desk-rulebook.txt) | A filled-in `hard-rules.txt` | 11 travel types: who books vs who flies, origin vs destination, outbound vs return dates, places to avoid |

## Domain templates

These are the templates of the demo's "Your text" tab. Each has a never-extract rule and a closing rule, and each was
run on the sample text below with the released model; the output is shown as it came out.

| File | Types | Sample text and output |
|---|---|---|
| [`pii.txt`](pii.txt) | NAME, EMAIL, PHONE, STREET_ADDRESS, CITY, POSTCODE, DATE_OF_BIRTH, ACCOUNT_NUMBER | A support message with a customer's details and the help desk's own address. All 8 personal details found; the help-desk address is left out (never-extract rule) |
| [`news.txt`](news.txt) | PERSON, ORGANIZATION, LOCATION, EVENT, PRODUCT | "Speaking at the Web Summit in Lisbon, Dr Fei-Fei Li said Stanford University…". 9 entities; names without "Dr" or "President" |
| [`product-reviews.txt`](product-reviews.txt) | PRODUCT, BRAND, FEATURE, PROBLEM | "I switched from the iPhone 14 to the Pixel 9…". 9 mentions, with problems as short phrases ("drains overnight") and app names as products ("Google Photos") |
| [`scientific-text.txt`](scientific-text.txt) | METHOD, DATASET, METRIC, TASK | "We fine-tune BERT and RoBERTa on 8 GPUs for extractive question answering on SQuAD 2.0…". 8 entities; "8 GPUs" left out, "exact match" kept as a metric |

The same layout can be written as a Python dict; see [Rulebooks](../../README.md#rulebooks) in the README. For what to
put in the rules, see [How to write rules that work](../writing-rules.md).

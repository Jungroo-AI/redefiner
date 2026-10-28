# How to write rules that work

These patterns come from the benchmark rulebooks in `benchmarks/` and from the example rulebooks in this folder. The
full BroadTwitter and HarveyNER rulebooks are in `benchmarks/`, before and after the rules were added; templates to
start from are in [`templates/`](templates/).

## 1. Always write a description

Every `[TYPE]` should have a one-line `[DESCRIPTION]`. On our open-NER sets, descriptions are worth about 7 F1 over
type names alone (63.1 vs 56.2). If a type name is ambiguous ("Other", "Point", "Area"), the description is the only
thing that tells the model what you mean.

## 2. Say where a span starts and ends

Most errors on a new dataset are boundary conventions, not wrong types. State them with examples written the way they
appear in your text.

```
B1 An @username is an entity. Its span always starts at the "@" and covers the whole name: label "@ Calum5SOS",
   never "Calum5SOS" alone. This holds anywhere in the tweet: after RT ("RT @ BenHannington :"), at the start
   ("@ DMorifeu What is that ?") and in a list ("@ Luke5SOS @ Calum5SOS @ Ashton5SOS").
```

This one rule took @-mention recall on BroadTwitter from 17% to 61%, and the dataset from 45.5 to 70.6 F1.

For multi-part names, say that the whole phrase is one span and that its parts are not labelled again:

```
G4 A Point is the whole location phrase as one span, with its cross street, address, city, state and zip:
   "corner of Richey St and W Harris Ave in Pasadena", "900 Portway Drive , Houston , TX 77024".
   Do not also label the roads, towns or buildings inside it.
```

## 3. Name the false positives you see most

A short list of the exact strings the model keeps labelling works better than a general principle.

```
G2 Never label "Houston", "Texas" or "Harris County" on their own, a US state, a country, or the storm
   ("Harvey", "Hurricane Harvey"), including their hashtags ("# Houston", "# Harvey").
B3 "RT", links, emoticons, zodiac signs ("Aries", "Gemini", "Virgo") and words that are not names are never entities.
```

On HarveyNER, G2 cut predictions of those strings from 389 to 5.

## 4. Settle overlaps between types explicitly

When two types could claim the same span, say which one wins and when:

```
RO1 A road or highway named without an intersection, address or exit is a Road ("I-45", "Memorial Drive");
    when it has a cross street or address, the whole phrase is one Point instead.
```

## 5. Give positive cues for rare types

A type the model rarely predicts gets more recall from cue words than from more definition text:
"X acted as legal counsel to Y" → LEGAL_CONSULTING_COMPANY lifted that type on BUSTER from 25.7 to 61.4 F1.

## 6. Put exclusions in `[RULES]`, and examples under their type

Under a `[TYPE]`, every quoted string reads as an example of that type, even inside a "never" sentence. This rule
under MultiNERD's Time type made the model label *more* years as Time (2,630 to 4,331 on dev):

```
[TYPE] Time
TI2 Years, decades, centuries and dates are never Time: "2014", "1980 s", "19th century", "February 2014", "3000 BC".
```

The same exclusion in `[RULES]`, which applies to every type, worked. It cut Time false positives from 2,268 to 443:

```
[RULES]
G1 Years, decades, centuries and calendar dates are never entities: "2010", "2014", "1980 s", "19th century", "February 2014".
```

The flip side: examples you want labelled belong under their type. An E3C version that quoted its disorder examples
in `[RULES]` lost every "fever". Listing common disorder words ("fever", "tumor", "cyst") under the type instead took
"fever" from 0 to 5 of 5 correct and recall from 54% to 64% on dev.

## 7. Structure for hard rules

Some types depend on context: who books a trip and who takes it, a date going out or coming back, a city to avoid
rather than a destination. For these, a few short rule lines are often not enough. What works better is a rulebook
laid out like [`templates/hard-rules.txt`](templates/hard-rules.txt):

```text
[RULES] Travel desk rulebook. Label each entity in the traveller's message with one of the types below.
G1 Label all entities in the message, not only the first one.
G3 When two types fit, the precedence rules below decide.
[PRECEDENCE]
P3 The person who will fly is TRAVELLER; the person who arranges or pays is BOOKER.
[TYPE] BOOKER
[DESCRIPTION] A person who arranges or pays for someone else's trip.
BK1 "Petra is booking for her son Leon" -> "Petra" is BOOKER, "Leon" is TRAVELLER.
[NOT_AN_ENTITY]
NE1 Places the traveller wants to avoid ("not via Doha", "no stop in Frankfurt").
```

- **A `[PRECEDENCE]` section**, one line per conflict between two types, naming the cue in the text and the type it
  gives.
- **A general rule that sends conflicts there** ("When two types fit, the precedence rules below decide").
- **Numbered rules** under each type (a short prefix and a number: BK1, NE1).
- **Worked examples** written as `"text" -> "span"`, made up so they don't repeat your real data, including one where
  the other type looks right.
- **Never-extract cases** under `[NOT_AN_ENTITY]`.

With the full [travel-desk rulebook](examples/travel-desk-rulebook.txt), 19 of 21 test messages came out exactly
right, against 11 with the same types and descriptions but no rules. In four other hard cases (negated symptoms,
deadlines among other dates, cities never visited, a company's own contact details), short rule lines changed nothing,
and the same rules written in this structure fixed about half of the errors, not all.

Make the cue something the text shows: words next to the entity ("not via", "back on") or a visible format (a `#` or
`INV-` prefix). Rules that need counting ("thirteen digits") or judging what didn't happen ("cities they only plan
to visit") are still mostly ignored.

## What doesn't work (yet)

- **Retyping against a strong prior.** A rule that relabels a person's name as some other type is mostly ignored: a
  person's name stays PERSON.
- **Asking for common nouns as mentions.** "Label common nouns that refer to an entity" barely moves recall; the
  model labels names. Pronouns are the exception: quoted in a short list under the type ("he", "they", "his",
  "their"), they are followed word by word. On WikiEvents, "their" went from 1 to 24 correct.
- **"Only X, not every Y" when Y is very common.** "Only annual revenue amounts, not every money amount" left revenue
  precision on BUSTER at about 9%.
- **Abstaining when a condition isn't met.** "Leave a company unlabelled unless its role in the deal is clear" makes
  little difference; the model tends to pick a role anyway.
- **Piling on rules.** Each rule moves errors between types; past a point, more rules stop helping. Check every new
  rule on examples it wasn't written from.

## A workflow

1. Write types with descriptions; run on 20–50 examples you have answers for.
2. Group the errors: boundary, false positive, confusion between two types, missing type.
3. Add one rule per group, quoting the exact strings from your errors.
4. Re-run on *different* examples. Keep a rule only if it helps there.

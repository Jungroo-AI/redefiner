"""space/demo_examples.json: examples written for the Edit a rule tab (support ticket, job application, clinical note).

Each example is one text and one rulebook with class rules, a Never extract section and a closing rule. The texts and
rulebooks were written for the demo (not a benchmark), and no quoted example in a rule appears in its text (apart from
cue words such as "plan" or "no history of"), so each fix comes from the rule, not from a copied answer.

The script runs each full rulebook and the rulebook without each kind of rule, and asserts the differences the caption
describes: the spans that appear and disappear when that kind of rule is deleted.

usage: build_demo_examples.py WEIGHTS_DIR
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import NamedTuple

from redefiner import Tagger

SPACE = Path(__file__).resolve().parent.parent / "space"

SUPPORT = {"general": "Extract every entity of the supplied types from a customer support ticket. Return exact spans "
                      "from the text.",
           "classes": [
               {"name": "PRODUCT", "description": "A named product or app."},
               {"name": "VERSION", "description": "A software version number."},
               {"name": "PLAN", "description": "A subscription plan or tier.",
                "rules": 'Label the plan name only, without the word "plan", "account" or "tier": "Business" in '
                         '"our Business account".'},
               {"name": "ORDER_ID", "description": "An order number.",
                "rules": 'The span starts at the # and leaves out the word "order": "#B-11872" in "order #B-11872".'},
               {"name": "INVOICE_ID", "description": "An invoice number."},
               {"name": "REFUND_REF", "description": "A refund reference."},
               {"name": "ERROR_CODE", "description": "An error code shown by the product."},
               {"name": "AMOUNT", "description": "A sum of money with its currency."},
               {"name": "DATE", "description": "A calendar date."}],
           "never": 'A "Plan" followed by a letter is a fallback strategy, never PLAN: "Plan A", "Plan C", "Plan D".\n'
                    'The word plan in "backup plan" or "migration plan".',
           "closing": 'The format decides, whatever the customer calls it: a reference that starts with INV- is '
                      'INVOICE_ID ("INV-2025-0099"), even when the customer calls it an order; a reference that '
                      'starts with # is ORDER_ID ("#B-11872"), even when the customer calls it an invoice. This '
                      'overrides everything above.'}

JOB = {"general": "Extract every entity of the supplied types from a job application. Return exact spans from the "
                  "text.",
       "classes": [
           {"name": "PERSON", "description": "The candidate's name."},
           {"name": "JOB_TITLE", "description": "A job title the candidate has held or applies for."},
           {"name": "COMPANY", "description": "A company the candidate has worked for."},
           {"name": "DEGREE", "description": "A degree or diploma."},
           {"name": "SCHOOL", "description": "A university, college or school."},
           {"name": "SKILL", "description": "A technical skill, tool, language or platform."},
           {"name": "CERTIFICATION", "description": "A professional certification."},
           {"name": "EXPERIENCE", "description": "A length of work experience.",
            "rules": 'Label the number with its unit: "3 years", "18 months".'}],
       "never": 'Spoken languages, such as "French", "Hindi" or "Mandarin".',
       "closing": 'A job title named after a tool is JOB_TITLE as a whole, never SKILL: "SAP consultant", "Tableau '
                  'developer". This overrides everything above.'}

CLINICAL = {"general": "Extract every entity of the supplied types from a clinical note. Return exact spans from the "
                       "text.",
            "classes": [
                {"name": "CONDITION", "description": "A disease, symptom or diagnosis the patient has."},
                {"name": "MEDICATION", "description": "A drug the patient takes or is prescribed.",
                 "rules": "Drug name only; dose and frequency are separate."},
                {"name": "DOSE", "description": "An amount of a drug, such as 250 mg."},
                {"name": "FREQUENCY", "description": "How often a drug is taken, such as three times a day.",
                 "rules": 'Times of day count as frequency: "in the morning", "at bedtime".'},
                {"name": "ALLERGY", "description": "A substance the patient is allergic to."}],
            "never": 'Conditions the patient does not have: a condition after "denies", "no", "no history of" or '
                     '"negative for" ("denies fever", "no rash").',
            "closing": 'A drug the patient is allergic to is ALLERGY, never MEDICATION ("allergic to sulfa"). This '
                       'overrides everything above.'}


class Example(NamedTuple):
    name: str
    book: dict
    text: str
    effects: dict      # kind of rule -> (spans that appear when it is deleted, spans that disappear when it is deleted)
    cues: list[str]    # quoted words a rule may share with the text
    caption: str


EXAMPLES = [
    Example(
        name="Support tickets: class rules, never extract and a closing rule",
        book=SUPPORT,
        text=(
            "Hi, after updating Fleetview to version 4.12.3 on 3 March, our Pro account was billed twice for order "
            "#A-20931 (2 x $49.00) and the app shows ERR_PAYMENT_DUPLICATE. Plan B in our rollout plan was to use "
            "the Fleetview web app instead, but it fails too. The invoice #C-48120 for the Team plan never arrived, "
            "refund reference RF-7781 hasn't shown up, and invoice INV-2026-0412 still says unpaid."
        ),
        effects={
            "class rules": ({("Pro account", "PLAN")}, {("Pro", "PLAN")}),
            "never": ({("Plan B", "PLAN")}, set()),
            "closing": ({("invoice #C-48120", "INVOICE_ID")}, {("#C-48120", "ORDER_ID")}),
        },
        cues=["Plan", "plan", "account", "order"],
        caption=(
            'A support-ticket rulebook written for this demo, not a benchmark. It uses all three kinds of rules. '
            'Class rule: the PLAN card labels "Pro", not "Pro account". Never extract: "Plan B" is a fallback, not a '
            'subscription plan. Closing rule: the reference format decides, so "#C-48120" is an order number even '
            'though the customer calls it an invoice. Delete any of them and Run to see the difference.'
        ),
    ),
    Example(
        name="Job application: what counts as a skill",
        book=JOB,
        text=(
            "Priya Raman is a senior data engineer with 7 years of experience, most recently at Shopify in Toronto. "
            "She started as a Power BI developer at Deloitte. She holds an MSc in Computer Science from the "
            "University of Waterloo, works daily in Python, SQL and Spark, and speaks Spanish and Tamil."
        ),
        effects={
            "class rules": ({("7 years of experience", "EXPERIENCE")}, {("7 years", "EXPERIENCE")}),
            "never": ({("Spanish", "SKILL"), ("Tamil", "SKILL")}, set()),
            "closing": ({("Power BI", "SKILL")}, {("Power BI developer", "JOB_TITLE")}),
        },
        cues=[],
        caption=(
            'A job-application rulebook written for this demo, not a benchmark. Class rule: EXPERIENCE labels "7 '
            'years", not "7 years of experience". Never extract: spoken languages are not skills, so Spanish and '
            'Tamil are left out. Closing rule: a job title named after a tool stays one JOB_TITLE, so "Power BI '
            'developer" does not lose "Power BI" to SKILL. Delete any of them and Run to see the difference.'
        ),
    ),
    Example(
        name="Clinical note: skip conditions the patient doesn't have",
        book=CLINICAL,
        text=(
            "58-year-old man seen for chest pain on exertion. History of type 2 diabetes and hypertension. Denies "
            "shortness of breath; no history of stroke. Currently taking metformin 500 mg twice daily and lisinopril "
            "10 mg once daily. Allergic to penicillin. Started atorvastatin 20 mg at night."
        ),
        effects={
            "class rules": (set(), {("at night", "FREQUENCY")}),
            "never": ({("shortness of breath", "CONDITION"), ("stroke", "CONDITION")}, set()),
        },
        cues=["no history of"],
        caption=(
            'A clinical-note rulebook written for this demo, not a benchmark, and not for clinical use. Never '
            'extract: conditions the patient does not have, so "shortness of breath" and "stroke" are left out. '
            'Class rule: times of day count as FREQUENCY, so "at night" is found. Delete either of them and Run to '
            'see the difference.'
        ),
    ),
]


def without(book: dict, kind: str) -> dict:
    """The rulebook with one kind of rule deleted: "class rules", "never" or "closing"."""
    b = copy.deepcopy(book)
    if kind == "class rules":
        for c in b["classes"]:
            c.pop("rules", None)
    else:
        b[kind] = ""
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("weights", help="release weights folder")
    args = ap.parse_args()
    sys.path.insert(0, str(SPACE))   # the demo's form module
    from form import form_to_text

    tagger = Tagger.from_pretrained(args.weights, device="cpu")
    out = []
    for ex in EXAMPLES:
        rules = [ex.book["never"], ex.book["closing"]] + [c.get("rules", "") for c in ex.book["classes"]]
        quotes = [q for rule in rules for q in rule.split('"')[1::2] if len(q) >= 3 and any(ch.isalnum() for ch in q)]
        copied = [q for q in quotes if q in ex.text and q not in ex.cues]
        if copied:
            raise SystemExit(f"{ex.name}: rule examples {copied} appear in the text")
        spans = tagger.extract(ex.text, form_to_text(ex.book))
        full = {(x["text"], x["label"]) for x in spans}
        for kind, (appear, disappear) in ex.effects.items():
            bare = {(x["text"], x["label"]) for x in tagger.extract(ex.text, form_to_text(without(ex.book, kind)))}
            if not (appear <= bare - full and disappear <= full - bare):
                raise SystemExit(f"{ex.name}, without {kind}: expected +{sorted(appear)} -{sorted(disappear)}, "
                                 f"got +{sorted(bare - full)} -{sorted(full - bare)}")
            print(f"{ex.name.split(':')[0]:16s} without {kind:11s}: +{sorted(appear)} -{sorted(disappear)}")
        print("   full:", sorted(full, key=lambda x: ex.text.find(x[0])))
        out.append({"name": ex.name, "text": ex.text, "rules": form_to_text(ex.book),
                    "b": [[x["start"], x["end"], x["label"]] for x in spans], "caption": ex.caption})
    dest = SPACE / "demo_examples.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("->", dest)


if __name__ == "__main__":
    main()

"""Starter rulebooks for the "Your text" tab, as forms (see form.py).

Every template has Never extract and Closing rules, and its sample text contains something they cover; no rule quotes
its own sample text, apart from cue words.
"""
from typing import NamedTuple

from form import form_to_text


class Template(NamedTuple):
    form: dict
    sample: str        # sample text
    marks: list[str]   # spans pre-marked in label mode


GENERAL = "Extract every entity of the supplied types. Return exact spans from the text."

PII = Template(
    form={
        "general": "Extract every piece of personal information of the supplied types. Return exact spans from the "
                   "text.",
        "classes": [
            {"name": "NAME", "description": "A person's first, last or full name."},
            {"name": "EMAIL", "description": "An email address."},
            {"name": "PHONE", "description": "A phone or fax number."},
            {"name": "STREET_ADDRESS", "description": "A house number and street name."},
            {"name": "CITY", "description": "A city or town."},
            {"name": "POSTCODE", "description": "A postcode or ZIP code."},
            {"name": "DATE_OF_BIRTH", "description": "A person's date of birth."},
            {"name": "ACCOUNT_NUMBER", "description": "A bank account, card or customer account number."},
        ],
        "never": 'Our own support contacts, such as "help@acme.com" or "+1 800 555 0100".',
        "closing": 'If a number could be a phone number or an account number, the word before it decides: "call" or '
                   '"phone" means PHONE, "account" or "no." means ACCOUNT_NUMBER. This overrides everything above.',
    },
    sample="Hi team, please update the record for Maria Gonzalez (born 14 May 1987). Her new address is 221 Baker "
           "Street, Springfield 62704, and you can reach her on +1 415 555 0182. Please send the refund to account "
           "no. 4410 2290 7781. Questions can go to her at maria.g@example.com, not to our help desk at "
           "help@northwind.io.",
    marks=["Maria Gonzalez", "14 May 1987", "221 Baker Street", "Springfield", "62704", "+1 415 555 0182",
           "4410 2290 7781", "maria.g@example.com"],
)

NEWS = Template(
    form={
        "general": GENERAL,
        "classes": [
            {"name": "PERSON", "description": "A person's name.",
             "rules": 'Label the full name; leave out titles such as "Dr", "Prof", "President" or "Mr".'},
            {"name": "ORGANIZATION", "description": "A company, university, agency, team or other institution."},
            {"name": "LOCATION", "description": "A city, country, region or named place."},
            {"name": "EVENT", "description": "A named event, such as a summit, festival, tournament or election."},
            {"name": "PRODUCT", "description": "A named product, model, vehicle or piece of software."},
        ],
        "never": 'Job titles and honorifics, such as "Chancellor", "CEO" or "Sir".',
        "closing": 'An event held in a city is labelled as the event, and the city separately ("Expo 2020 in '
                   'Dubai"). This overrides everything above.',
    },
    sample="Speaking at the Web Summit in Lisbon, Dr Fei-Fei Li said Stanford University would release a new version "
           "of ImageNet next year. Google and the European Commission have both offered funding, and President Ursula "
           "von der Leyen is expected to announce the details in Brussels.",
    marks=["Web Summit", "Lisbon", "Fei-Fei Li", "Stanford University", "ImageNet", "Google", "European Commission",
           "Ursula von der Leyen", "Brussels"],
)

REVIEWS = Template(
    form={
        "general": "Extract every mention of the supplied types from a product review. Return exact spans from the "
                   "text.",
        "classes": [
            {"name": "PRODUCT", "description": "The named product or model being reviewed or compared.",
             "rules": 'Named apps, services and accessories are PRODUCT too ("Apple Music", "AirPods").'},
            {"name": "BRAND", "description": "A company or brand name."},
            {"name": "FEATURE", "description": "A part or property of the product the reviewer comments on, such as "
                                               "the battery, screen or price.",
             "rules": 'Label the feature word only ("battery"), not the opinion about it.'},
            {"name": "PROBLEM", "description": 'A fault or problem the reviewer reports, as a short phrase ("drains '
                                               'overnight", "keeps crashing").'},
        ],
        "never": 'Opinion words, such as "awesome", "slow" or "terrible".',
        "closing": 'A brand inside a product name belongs to the product: "Samsung Galaxy" is PRODUCT, not BRAND. '
                   'This overrides everything above.',
    },
    sample="I switched from the iPhone 14 to the Pixel 9 last month. The camera is fantastic and the screen is bright, "
           "but the battery drains overnight and the Google Photos app keeps crashing. Samsung's charger works fine "
           "with it.",
    marks=["iPhone 14", "Pixel 9", "camera", "screen", "battery", "drains overnight", "Google Photos",
           "keeps crashing", "Samsung"],
)

SCIENCE = Template(
    form={
        "general": "Extract every entity of the supplied types from a research abstract. Return exact spans from the "
                   "text.",
        "classes": [
            {"name": "METHOD", "description": "A named algorithm, model architecture or technique.",
             "rules": 'Models used as baselines are METHOD too ("an LSTM baseline", "compared with T5").'},
            {"name": "DATASET", "description": "A named dataset or benchmark."},
            {"name": "METRIC", "description": "An evaluation metric, such as accuracy, F1 or BLEU."},
            {"name": "TASK", "description": "The problem being solved, such as question answering or machine "
                                            "translation."},
        ],
        "never": 'Hardware and compute, such as "4 TPUs", "a single A100" or "16 GB of memory".',
        "closing": 'Every evaluation measure is METRIC, never TASK, even when it sounds like a task ("answer '
                   'accuracy", "span recall"). This overrides everything above.',
    },
    sample="We fine-tune BERT and RoBERTa on 8 GPUs for extractive question answering on SQuAD 2.0 and Natural "
           "Questions. Compared with a BiDAF baseline, our model improves exact match by 4.1 points and F1 by 3.2 "
           "points.",
    marks=["BERT", "RoBERTa", "extractive question answering", "SQuAD 2.0", "Natural Questions", "BiDAF",
           "exact match", "F1"],
)

BLANK = Template(
    form={
        "general": GENERAL,
        "classes": [{"name": "MY_TYPE", "description": "One line saying what this type is.",
                     "rules": "Optional rule lines: what to include or exclude.", "examples": ""}],
        "never": "",
    },
    sample="",
    marks=[],
)

TEMPLATES = {"PII": PII, "News": NEWS, "Product reviews": REVIEWS, "Scientific text": SCIENCE, "Blank": BLANK}


def text(name):
    """The rulebook text of template `name`."""
    return form_to_text(TEMPLATES[name].form)

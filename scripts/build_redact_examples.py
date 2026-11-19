"""Build space/redact_examples.json: the "Mask personal data" page's examples and their stored outputs.

    python scripts/build_redact_examples.py WEIGHTS

For each example the model runs twice, with the PII prompt and with the example's policy, through the page's own
code (space/redact_page.py: span limit 64, none_bias 2, threshold 0.5). Every span the model returns is stored; the
page applies the confidence cutoff (0 for the PII prompt, 0.35 for a policy). The checks below fail the build if a
policy example stops keeping or masking what it is there to show.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "space"))

import threading  # noqa: E402

import redact_page as rp  # noqa: E402
from redefiner import Tagger  # noqa: E402

CLINIC_POLICY = """[TYPE] PATIENT_NAME
[DESCRIPTION] The name of a patient. Doctors, nurses and other staff are not patients.

[TYPE] DATE_OF_BIRTH
[DESCRIPTION] A person's date of birth. Other dates, such as appointment dates, are not dates of birth.

[TYPE] CARD_NUMBER
[DESCRIPTION] The digits of a payment card number, 13 to 19 digits with their spaces, and nothing else.

[TYPE] PHONE
[DESCRIPTION] The digits of a phone number.

[TYPE] NOT_AN_ENTITY
NE1 Invoice numbers and order numbers, such as "INV-55120" or "Order 4471-B".
"""
SUPPORT_POLICY = """[TYPE] CUSTOMER_NAME
[DESCRIPTION] The name of a customer, full or first name only. Our own staff, who write as "Agent <name>", are not customers.

[TYPE] ACCOUNT_NUMBER
[DESCRIPTION] A customer account number, such as "AC-449120".

[TYPE] EMAIL
[DESCRIPTION] An email address.

[TYPE] NOT_AN_ENTITY
NE1 Ticket numbers, such as "T-1042".
"""  # noqa: E501 (a rule line is never wrapped)
EMAIL_POLICY = """[TYPE] PERSON
[DESCRIPTION] A person's name.

[TYPE] CONTACT
[DESCRIPTION] An email address or a phone number.

[TYPE] HOME_ADDRESS
[DESCRIPTION] A street address or a postcode where someone lives.

[TYPE] DATE_OF_BIRTH
[DESCRIPTION] A person's date of birth. Other dates are not dates of birth.

[TYPE] NOT_AN_ENTITY
NE1 Company and clinic names, such as "Northwind Dental".
"""
IT_POLICY = """[TYPE] PERSON
[DESCRIPTION] A person's name.

[TYPE] SECRET
[DESCRIPTION] A password, PIN or access key.

[TYPE] CARD_NUMBER
[DESCRIPTION] The digits of a payment card number, and nothing else.

[TYPE] NOT_AN_ENTITY
NE1 Usernames and IP addresses, such as "jdoe3" or "10.0.0.12": the IT team needs them.
"""
EXAMPLES = {
    "Clinic note": {
        "mode": "policy", "policy": CLINIC_POLICY,
        "text": "Nurse Priya Shah booked Marcus Bell (DOB 09/01/1962) for 21 July with Dr. Omar Aziz. Invoice "
                "INV-88213, card 4000 0566 5566 5556, phone 0161 496 0333.",
        "masks": ["Marcus Bell", "09/01/1962", "4000 0566 5566 5556", "0161 496 0333"],
        "keeps": ["Priya Shah", "Dr. Omar Aziz", "21 July", "INV-88213"]},
    "Support chat": {
        "mode": "policy", "policy": SUPPORT_POLICY,
        "text": "Customer: Hi, I'm Daniel Brandt, account AC-771204. I was charged twice on 3 March. Please reply to "
                "daniel.brandt@example.com.\nAgent Rosa Diaz: Sorry about that. I've refunded the second charge "
                "under ticket T-5521.",
        "masks": ["Daniel Brandt", "AC-771204", "daniel.brandt@example.com"],
        "keeps": ["Rosa Diaz", "T-5521", "3 March"]},
    "Email": {
        "mode": "all", "policy": EMAIL_POLICY,
        "text": "Hi, this is Meera Nair from Brightwater Dental. Please send the invoice to "
                "meera.nair@brightwater.example and call me on 020 7946 0155 before Thursday. My patient ID is "
                "PT-664201, date of birth 14 March 1989. I live at 9 Alder Close, Bristol BS3 4QT.",
        "masks": ["Meera Nair", "meera.nair@brightwater.example", "020 7946 0155", "9 Alder Close"],
        "keeps": ["Brightwater Dental", "Thursday"]},
    "IT ticket": {
        "mode": "all", "policy": IT_POLICY,
        "text": "Laura Chen (username lchen7) can't sign in from 10.12.4.88 since her password Spring2024! expired. "
                "She is a payroll analyst in our Leeds office. Employee EMP-30981, card on file 5500 0055 5555 5559.",
        "masks": ["Laura Chen", "Spring2024!", "5500 0055 5555 5559"],
        "keeps": ["lchen7", "10.12.4.88"]},
}


def main(weights):
    tagger = Tagger.from_pretrained(weights, device="cpu")
    lock = threading.Lock()
    out, problems = {}, []
    for name, e in EXAMPLES.items():
        stored = {"all": rp.find(tagger, lock, e["text"], None),
                  "policy": rp.find(tagger, lock, e["text"], e["policy"])}
        for key, spans in stored.items():
            print(f"{name} / {key}: {rp.masked(e['text'], spans, rp.CUTOFF[key])}")
        policy_out = rp.masked(e["text"], stored["policy"], rp.CUTOFF["policy"])
        problems += [f"{name}: policy keeps {s!r}" for s in e["masks"] if s in policy_out]
        problems += [f"{name}: policy masks {s!r}" for s in e["keeps"] if s not in policy_out]
        all_out = rp.masked(e["text"], stored["all"], rp.CUTOFF["all"])
        problems += [f"{name}: PII prompt keeps {s!r}" for s in e["masks"] if s in all_out]
        out[name] = {"mode": e["mode"], "text": e["text"], "policy": e["policy"], "stored": stored}
    if problems:
        raise SystemExit("\n".join(problems))
    path = ROOT / "space" / "redact_examples.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1])

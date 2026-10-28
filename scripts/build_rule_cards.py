"""docs/assets/rules/*.png: the "See a rule at work" cards in the README and the model card.

Each card is real output of the release weights on CPU: the text, the rulebook before and after one edit, and both
outputs, with changed spans outlined. The script fails if an output differs from the one the README describes, so the
pictures can't drift from what the model does. Cards are rendered as HTML and photographed with headless Chrome.

The last three cards use the example rulebook docs/examples/travel-desk-rulebook.txt: general rules, precedence
rules, numbered type rules with worked examples and never-extract rules. Their "Before" is the same rulebook with only
its types and descriptions.

usage: build_rule_cards.py WEIGHTS_DIR [--chrome PATH]
"""

import argparse
import html
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

from redefiner import Tagger

DOCS = Path(__file__).resolve().parent.parent / "docs"
DEST = DOCS / "assets" / "rules"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HEAD = "[RULES]\nExtract every entity of the supplied types. Return exact spans from the text.\n\n"
PALETTE = [("#dbeafe", "#1e40af"), ("#dcfce7", "#166534"), ("#ffedd5", "#9a3412"), ("#f3e8ff", "#6b21a8"),
           ("#fef9c3", "#854d0e"), ("#cffafe", "#155e75"), ("#fce7f3", "#9d174d"), ("#e0e7ff", "#3730a3")]


def types(*pairs: tuple[str, str]) -> str:
    return HEAD + "\n".join(f"[TYPE] {name}\n[DESCRIPTION] {desc}\n" for name, desc in pairs)


SHOP = types(("BRAND", "A brand name."), ("PRODUCT", "A product line or model name."), ("COLOR", "A colour."),
             ("SIZE", "A size."), ("PRICE", "A price or a price limit."))
PRICE_RULE = 'PR1 A price limit keeps its limit word in the span: "under $50", "below 30 dollars", "less than $200".\n'
INBOX = types(("COMPANY", "A company name."), ("PRODUCT", "A named product or device."))
NOT_OURS = '\n[TYPE] NOT_AN_ENTITY\nNE1 Our own company and products: "Acme", "Acme Home". Never label them.\n'
JOBS = types(("SKILL", "A technical skill: a programming language, framework, platform or tool."))
LANGUAGE = ('\n[TYPE] LANGUAGE\n[DESCRIPTION] A programming language, such as "Java" or "Rust".\n'
            'LA1 Programming languages are LANGUAGE, never SKILL.\n')
NEWS = types(("COMPANY", "A company name."), ("PERSON", "A person's name."), ("PRODUCT", "A product name."),
             ("DATE", "A date or day."))
TICKER = '\n[TYPE] TICKER\n[DESCRIPTION] A stock ticker symbol, such as "TSLA" or "MSFT".\n'
REFS = types(("ORDER_ID", "An order number."), ("INVOICE_ID", "An invoice number."))
FORMAT = ('[RULES]\nThe format decides, whatever the customer calls it: a reference starting with # is ORDER_ID '
          '("#B-11872"), even when the customer calls it an invoice; a reference starting with INV- is INVOICE_ID '
          '("INV-2025-0099"), even when the customer calls it an order. This overrides everything above.\n')
PEOPLE = types(("PERSON", "A person's name."))
PRONOUNS = 'PE1 Also label the pronouns that refer to a person: "she", "he", "they", "her", "him", "them".\n'
TRAVEL = (DOCS / "examples" / "travel-desk-rulebook.txt").read_text()
# the same rulebook without its rules: only the [RULES] line, the types and their descriptions
TRAVEL_TYPES = "\n".join(line for line in TRAVEL.splitlines()
                         if not re.match(r"^[A-Z]{1,2}\d+ ", line)
                         and line not in ("[PRECEDENCE]", "[NOT_AN_ENTITY]")) + "\n"
TRAVEL_NOTE = "Key rules from the travel-desk rulebook. Before: the same types and descriptions, without any rules."


def rule_lines(*ids: str) -> str:
    """The travel-desk rule lines with these IDs."""
    return "".join(line + "\n" for line in TRAVEL.splitlines() if line.split(" ", 1)[0] in ids)


TICKET = ("Hi, this is Priya Raman (account 88-41207). Since updating Zoom to 6.2.5 on my MacBook Air, calls drop with "
          "error 1132. Reach me at priya.raman@outlook.com or 617-555-0148.")

# name, kicker, title, text, [(row label, rulebook)], lines shown as added, expected output per row
CARDS = [
    ("two-teams", "One text, two rulebooks", "Same model, same ticket, two different jobs", TICKET,
     [("Privacy team", types(("PERSON", "A person's name."), ("ACCOUNT_ID", "An account or customer number."),
                             ("EMAIL", "An email address."), ("PHONE", "A phone number."))),
      ("Product team", types(("PRODUCT", "A named software product or app."), ("VERSION", "A version number."),
                             ("DEVICE", "A named hardware model."), ("ERROR_CODE", "An error code.")))],
     None,
     [[("Priya Raman", "PERSON"), ("88-41207", "ACCOUNT_ID"), ("priya.raman@outlook.com", "EMAIL"),
       ("617-555-0148", "PHONE")],
      [("Zoom", "PRODUCT"), ("6.2.5", "VERSION"), ("MacBook Air", "DEVICE"), ("1132", "ERROR_CODE")]]),
    ("price-limit", "Span boundary", 'Keep "under" with the price', "red nike air max size 10 under $120",
     [("Before", SHOP), ("After", SHOP.replace("[DESCRIPTION] A price or a price limit.\n",
                                               "[DESCRIPTION] A price or a price limit.\n" + PRICE_RULE))],
     PRICE_RULE,
     [[("red", "COLOR"), ("nike", "BRAND"), ("air max", "PRODUCT"), ("10", "SIZE"), ("$120", "PRICE")],
      [("red", "COLOR"), ("nike", "BRAND"), ("air max", "PRODUCT"), ("10", "SIZE"), ("under $120", "PRICE")]]),
    ("own-brand", "Exclusion", "Ignore your own company and products",
     "Hi Acme team, my Sonos Arc stopped pairing after the Acme Home update. Sonos says it is an Acme problem.",
     [("Before", INBOX), ("After", INBOX + NOT_OURS)], NOT_OURS,
     [[("Acme", "COMPANY"), ("Sonos Arc", "PRODUCT"), ("Acme Home", "PRODUCT"), ("Sonos", "COMPANY"),
       ("Acme", "COMPANY")],
      [("Sonos Arc", "PRODUCT"), ("Sonos", "COMPANY")]]),
    ("split-type", "Split a type", "Separate programming languages from other skills",
     "Senior backend engineer: 5+ years of Python and Go, hands-on with AWS, Kubernetes and PostgreSQL. "
     "React is a plus.",
     [("Before", JOBS), ("After", JOBS + LANGUAGE)], LANGUAGE,
     [[("Python", "SKILL"), ("Go", "SKILL"), ("AWS", "SKILL"), ("Kubernetes", "SKILL"), ("PostgreSQL", "SKILL"),
       ("React", "SKILL")],
      [("Python", "LANGUAGE"), ("Go", "LANGUAGE"), ("AWS", "SKILL"), ("Kubernetes", "SKILL"), ("PostgreSQL", "SKILL"),
       ("React", "SKILL")]]),
    ("new-type", "New type", "Add a type in two lines",
     "Apple (NASDAQ: AAPL) jumped 4% after Tim Cook unveiled the new iPhone on Tuesday.",
     [("Before", NEWS), ("After", NEWS + TICKER)], TICKER,
     [[("Apple", "COMPANY"), ("Tim Cook", "PERSON"), ("iPhone", "PRODUCT"), ("Tuesday", "DATE")],
      [("Apple", "COMPANY"), ("AAPL", "TICKER"), ("Tim Cook", "PERSON"), ("iPhone", "PRODUCT"), ("Tuesday", "DATE")]]),
    ("format-decides", "Hard rule · override the text's own words",
     "The reference format decides, not what the customer calls it",
     "Where is my order INV-7731? The invoice for it, #D-55102, says it shipped.",
     [("Before", REFS), ("After", REFS + FORMAT)], FORMAT,
     [[("INV-7731", "ORDER_ID"), ("#D-55102", "INVOICE_ID")],
      [("INV-7731", "INVOICE_ID"), ("#D-55102", "ORDER_ID")]]),
    ("pronouns", "Hard rule · label what the model skips by default", "Count pronouns as people",
     "Maria called. She wants her refund sent to her new card.",
     [("Before", PEOPLE), ("After", PEOPLE.replace("[DESCRIPTION] A person's name.\n",
                                                   "[DESCRIPTION] A person's name.\n" + PRONOUNS))],
     PRONOUNS,
     [[("Maria", "PERSON")], [("Maria", "PERSON"), ("She", "PERSON"), ("her", "PERSON"), ("her", "PERSON")]]),
    ("who-flies", "Full rulebook · who does what", "Who books and who flies",
     "Hannah is booking flights for her parents, Omar and Leila.",
     [("Before", TRAVEL_TYPES), ("After", TRAVEL)], (TRAVEL_NOTE, rule_lines("P3", "BK1")),
     [[("Hannah", "BOOKER"), ("Omar", "BOOKER"), ("Leila", "BOOKER")],
      [("Hannah", "BOOKER"), ("Omar", "TRAVELLER"), ("Leila", "TRAVELLER")]]),
    ("avoid-stopover", "Full rulebook · leave something out", "A city to avoid is not a destination",
     "Book Rome to Bangkok, but not via Dubai.",
     [("Before", TRAVEL_TYPES), ("After", TRAVEL)], (TRAVEL_NOTE, rule_lines("NE1")),
     [[("Rome", "ORIGIN"), ("Bangkok", "DESTINATION"), ("Dubai", "DESTINATION")],
      [("Rome", "ORIGIN"), ("Bangkok", "DESTINATION")]]),
    ("out-and-back", "Full rulebook · context decides the type", "Outbound and return dates",
     "Fly home on 22 August, I flew out on 15 August.",
     [("Before", TRAVEL_TYPES), ("After", TRAVEL)], (TRAVEL_NOTE, rule_lines("P4", "DD1", "RD1")),
     [[("22 August", "DEPARTURE_DATE"), ("15 August", "DEPARTURE_DATE")],
      [("22 August", "RETURN_DATE"), ("15 August", "DEPARTURE_DATE")]]),
]

CSS = """
body { margin: 0; background: #fff; font-family: Inter, system-ui, -apple-system, sans-serif; }
.card { width: 860px; padding: 24px 30px 18px; box-sizing: border-box; }
.kicker { font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #2563eb; }
.kicker.hard { color: #7c3aed; }
h2 { margin: 4px 0 14px; font-size: 22px; color: #0f172a; letter-spacing: -.01em; }
.rule-cap { font-size: 12px; color: #64748b; margin: -4px 0 6px; }
.rule { font-family: ui-monospace, Menlo, monospace; font-size: 13px; line-height: 1.55; background: #f0fdf4;
  border-left: 4px solid #22c55e; color: #14532d; padding: 10px 14px; border-radius: 6px; margin-bottom: 14px; }
.row { display: grid; grid-template-columns: 120px 1fr; align-items: center; gap: 12px; padding: 8px 0;
  border-top: 1px solid #eef2f7; }
.tag { font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
.tag small { display: block; font-weight: 500; text-transform: none; letter-spacing: 0; color: #94a3b8;
  font-size: 11px; margin-top: 2px; }
.after .tag { color: #16a34a; }
.txt { font-size: 16.5px; line-height: 2.1; color: #0f172a; }
mark { border-radius: 6px; padding: 3px 7px; white-space: nowrap; color: #0f172a; }
mark b { font-size: 10px; font-weight: 700; margin-left: 6px; letter-spacing: .03em; }
mark.added { outline: 2px solid #22c55e; outline-offset: 1px; }
mark.removed { outline: 2px dashed #ef4444; outline-offset: 1px; }
"""


def render_text(text, spans, colors, mark_class):
    out, pos = [], 0
    for s in spans:
        bg, fg = colors.setdefault(s["label"], PALETTE[len(colors) % len(PALETTE)])
        cls = mark_class.get((s["start"], s["end"], s["label"]), "")
        out += [html.escape(text[pos:s["start"]]),
                f'<mark class="{cls}" style="background:{bg}">{html.escape(s["text"])}<b style="color:{fg}">'
                f'{s["label"]}</b></mark>']
        pos = s["end"]
    return "".join(out) + html.escape(text[pos:])


def card_html(kicker, title, text, rows, added, outputs):
    colors = {}
    keys = [{(s["start"], s["end"], s["label"]) for s in out} for out in outputs]
    compare = added is not None   # a before/after card: outline what changed
    body = []
    for i, ((label, book), out) in enumerate(zip(rows, outputs)):
        if compare:
            other = keys[1 - i]
            mark_class = {k: ("removed" if i == 0 else "added") for k in keys[i] - other}
        else:
            mark_class = {}
        names = [line.split(" ", 1)[1] for line in book.splitlines() if line.startswith("[TYPE] ")]
        sub = "" if compare else f'<small>{", ".join(names)}</small>'
        cls = "row after" if label == "After" else "row"
        body.append(f'<div class="{cls}"><div class="tag">{html.escape(label)}{sub}</div>'
                    f'<div class="txt">{render_text(text, out, colors, mark_class)}</div></div>')
    rule = ""
    if added:
        caption, added = added if isinstance(added, tuple) else ("", added)
        lines = added.strip("\n").split("\n")
        rule = (f'<div class="rule-cap">{html.escape(caption)}</div>' if caption else "") + '<div class="rule">' + \
            "<br>".join("+ " + html.escape(line) for line in lines) + "</div>"
    hard = " hard" if kicker.startswith(("Hard", "Full")) else ""
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body><div class="card">'
            f'<div class="kicker{hard}">{html.escape(kicker)}</div><h2>{html.escape(title)}</h2>{rule}'
            + "".join(body) + "</div></body></html>")


def photograph(page: str, out: Path, chrome: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src, shot = Path(tmp) / "card.html", Path(tmp) / "card.png"
        src.write_text(page)
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        "--window-size=860,1000", f"--screenshot={shot}", src.as_uri()],
                       check=True, capture_output=True)
        img = Image.open(shot).convert("RGB")
        bbox = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
        img.crop((0, 0, img.width, min(img.height, bbox[3] + 36))).save(out, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("weights", help="release weights folder")
    ap.add_argument("--chrome", default=CHROME, help="Chrome or Chromium binary")
    args = ap.parse_args()
    tagger = Tagger.from_pretrained(args.weights, device="cpu")
    DEST.mkdir(parents=True, exist_ok=True)
    for name, kicker, title, text, rows, added, expected in CARDS:
        outputs = [tagger.extract(text, book) for _, book in rows]
        got = [[(s["text"], s["label"]) for s in out] for out in outputs]
        if got != expected:
            raise SystemExit(f"{name}: expected {expected}, got {got}")
        photograph(card_html(kicker, title, text, rows, added, outputs), DEST / f"{name}.png", args.chrome)
        print(f"{name:15s} ok -> {DEST / name}.png")


if __name__ == "__main__":
    main()

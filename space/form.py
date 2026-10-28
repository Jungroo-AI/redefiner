"""The rulebook form: general rules, classes (name, description, rules, examples), never-extract and closing rules.

form_to_text lays a form out as a uniform rulebook, the layout the model was trained on:
    [RULES]  general rules            [TYPE] NAME / [DESCRIPTION] ... / class rules / Examples line
    [TYPE] NOT_AN_ENTITY  never-extract rules              [RULES]  closing rules
sections separated by one blank line. Rule lines get IDs: G1.. over both [RULES] blocks (general, then closing), a
per-class prefix (PERSON -> P1.., ORGANIZATION -> O1..), NE1.. for never-extract. A class's examples become its last
rule line:
    P3 Examples: "Ada Lovelace", "Alan Turing".

text_to_form is the inverse, for switching from the text editor back to the form: it reads any rulebook with
[RULES] / [TYPE] sections, drops rule IDs and turns an "Examples:" line back into the examples field.
"""
import re

ID_LINE = re.compile(r"^[A-Z]+\d+ ")
QUOTED = re.compile(r'"([^"]+)"')


def id_prefixes(types):
    """Rule-line ID prefix per type (G and NE are taken)."""
    used, out = {"G", "NE"}, {}
    for t in types:
        parts = [re.sub(r"[^A-Z]", "", p) for p in t.split("_")]
        parts = [p for p in parts if p] or ["T"]
        for cand in ("".join(p[:1] for p in parts), "".join(p[:2] for p in parts), "".join(p[:3] for p in parts),
                     "".join(parts)):
            if cand not in used:
                break
        else:
            cand = "".join(parts) + "X"
        used.add(cand)
        out[t] = cand
    return out


def type_name(name, upper=True):
    """A class name as a rulebook type name: runs of non-word characters become "_", UPPER_CASE unless upper=False."""
    name = re.sub(r"\W+", "_", name or "").strip("_")
    return name.upper() if upper else name


def lines(v):
    v = v.split("\n") if isinstance(v, str) else v or []
    return [x.strip() for x in v if x.strip()]


def examples_list(v):
    """Examples field -> list: quoted strings if any, else comma- or newline-separated items."""
    v = v or ""
    found = QUOTED.findall(v)
    return found if found else [x.strip() for x in re.split(r"[,\n]", v) if x.strip()]


def form_to_text(form):
    """A form dict -> rulebook text. form = {general, classes: [{name, description, rules, examples}], never}."""
    # auto_ids False: a rulebook that numbers its own rules. Its lines are written as they are and its type names
    # keep their case (benchmark type names must not change); the form's own layout uses UPPER_CASE names.
    auto = form.get("auto_ids", True)
    classes = [c for c in form.get("classes", []) if type_name(c.get("name"), upper=auto)]
    names = [type_name(c["name"], upper=auto) for c in classes]
    if not names:
        raise ValueError("Add at least one class.")
    if "NOT_AN_ENTITY" in names:
        raise ValueError("NOT_AN_ENTITY is not a class: put those rules in the Never extract box.")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"Two classes are named {', '.join(dup)}.")
    pre = id_prefixes([type_name(n) for n in names])

    def num(prefix, rules, first=1):
        return [f"{prefix}{i} {x}" for i, x in enumerate(rules, first)] if auto else rules

    blocks = []
    general, closing = lines(form.get("general")), lines(form.get("closing"))
    if general:
        blocks.append(["[RULES]"] + num("G", general))
    for c, n in zip(classes, names):
        desc = " ".join(lines(c.get("description")))
        rules = lines(c.get("rules"))
        ex = examples_list(c.get("examples"))
        if ex:
            rules.append("Examples: " + ", ".join(f'"{x}"' for x in ex) + ".")
        blocks.append([f"[TYPE] {n}"] + ([f"[DESCRIPTION] {desc}"] if desc else []) + num(pre[type_name(n)], rules))
    never = lines(form.get("never"))
    if never:
        blocks.append(["[TYPE] NOT_AN_ENTITY"] + num("NE", never))
    if closing:   # its G numbers continue after the general rules
        blocks.append(["[RULES]"] + num("G", closing, first=len(general) + 1))
    return "\n\n".join("\n".join(b) for b in blocks) + "\n"


def text_to_form(text):
    """Rulebook text -> form dict. With the form's own rule numbering the IDs are dropped (they are regenerated);
    a rulebook that numbers its own rules, or would not come back exactly, keeps every line verbatim
    (auto_ids False). ValueError when the text has no [TYPE] sections."""
    form = _parse(text, strip_ids=True)
    try:
        if form_to_text(form) == text:
            return form
    except ValueError:
        pass
    verbatim = _parse(text, strip_ids=False)
    verbatim["auto_ids"] = False
    return verbatim


def _parse(text, strip_ids):
    form = {"general": [], "classes": [], "never": [], "closing": []}
    head, body = None, []

    def flush():
        if head is None:
            return
        rest = [ID_LINE.sub("", x, count=1) if strip_ids else x for x in body if x.strip()]
        if head == "[RULES]":
            form["closing" if form["classes"] else "general"] += rest
        elif head == "[TYPE] NOT_AN_ENTITY":
            form["never"] += rest
        else:
            desc = next((x[len("[DESCRIPTION]"):].strip() for x in rest if x.startswith("[DESCRIPTION]")), "")
            rules = [x for x in rest if not x.startswith("[DESCRIPTION]")]
            ex = [x for x in rules if x.startswith("Examples:")] if strip_ids else []
            rules = [x for x in rules if not (strip_ids and x.startswith("Examples:"))]
            examples = ", ".join(f'"{q}"' for x in ex for q in QUOTED.findall(x))
            form["classes"].append({"name": head[len("[TYPE] "):].strip(), "description": desc,
                                    "rules": "\n".join(rules), "examples": examples})

    for line in text.splitlines():
        s = line.strip()
        if s == "[RULES]" or re.match(r"^\[TYPE\]\s+\S+$", s):
            flush()
            head, body = re.sub(r"\s+", " ", s), []
        elif head is not None:
            body.append(s)
    flush()
    if not form["classes"]:
        raise ValueError('No classes found: the rulebook needs a "[TYPE] NAME" line per class.')
    for k in ("general", "never", "closing"):
        form[k] = "\n".join(form[k])
    return form

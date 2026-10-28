"""The ReDefiNER demo: edit a rule, see the extraction change.

Tabs
  Edit a rule   examples written for the demo and benchmark test sentences, with their full rulebooks, editable as
                a form or as text
  Your text     any rulebook (or a template) on any text; find the spans, or label spans you mark
  Benchmarks    one real test example per dataset: gold, GoLLIE's guidelines, and the guidelines plus our rules

Stored outputs (examples.json, demo_examples.json, bench_examples.json) were computed with this model on CPU by the
scripts in ../scripts. Picking an example shows its stored output; Run computes live.
"""
import html
import json
import os
import re
import threading
from pathlib import Path

import gradio as gr

from form import form_to_text, text_to_form, type_name
from redefiner import Tagger
from templates import TEMPLATES
from templates import text as template_text

HERE = Path(__file__).parent
MODEL = os.environ.get("MODEL", "jungroo-ai/redefiner-0.8b")
TAGGER = Tagger.from_pretrained(MODEL, cache_size=48)   # every example and template rulebook stays encoded
LOCK = threading.Lock()                                  # one model; Gradio can run two handlers at once
EXAMPLES = json.loads((HERE / "examples.json").read_text())
BENCH_EXTRA = json.loads((HERE / "bench_examples.json").read_text())
DEMO = json.loads((HERE / "demo_examples.json").read_text())
HF_REPO = MODEL if not Path(MODEL).exists() else "jungroo-ai/redefiner-0.8b"
LINKS = {"Model": f"https://huggingface.co/{HF_REPO}", "Code": "https://github.com/jungroo-ai/redefiner",
         "Benchmarks": "https://github.com/jungroo-ai/redefiner/tree/main/benchmarks",
         "How to write rules": "https://github.com/jungroo-ai/redefiner/blob/main/docs/writing-rules.md"}

# ---------------------------------------------------------------- rulebook sections
HEADER = re.compile(r"^(\[RULES\]|\[TYPE\][ \t]+[A-Za-z0-9_]+)[ \t]*$")


def sections(text):
    """[(header, block)] in order; a block runs from its header line to the next header, blank lines included."""
    out, head, buf = [], None, []
    for line in text.splitlines(keepends=True):
        m = HEADER.match(line.rstrip("\n"))
        if m:
            if buf or head is not None:
                out.append((head, "".join(buf)))
            head = re.sub(r"\s+", " ", m.group(1))
            if head == "[RULES]" and any(h and h.startswith("[TYPE]") for h, _ in out):
                head = "[RULES] closing"
            buf = [line]
        else:
            buf.append(line)
    out.append((head, "".join(buf)))
    return out


def changed_sections(base, rules):
    """The sections of `rules` that differ from `base`, verbatim."""
    old = dict(sections(base))
    return "".join(block for h, block in sections(rules) if old.get(h) != block)


# ---------------------------------------------------------------- scores
# test F1: guidelines only, + our rules, GoLLIE-34B (accuracy for the S1MB label sets)
SCORES = {"BroadTwitter": (45.5, 70.6, 50.3), "HarveyNER": (33.5, 48.1, 38.9), "CrossNER Science": (66.0, 75.4, 56.3),
          "CrossNER Music": (73.7, 79.5, 68.4), "MIT Movie": (71.3, 78.5, 62.4), "MultiNERD": (82.0, 87.6, 77.2),
          "CrossNER AI": (66.7, 74.4, 61.6), "CrossNER Literature": (70.1, 71.5, 59.1),
          "CrossNER Politics": (78.0, 80.7, 60.2), "MIT Restaurant": (63.8, 68.1, 52.7), "E3C": (47.7, 59.4, 60.0),
          "WikiEvents": (46.4, 53.6, 81.3), "FabNER": (32.7, 35.4, 26.3),
          "S1MB Few-NERD (label mode)": (None, 71.8, None), "S1MB Gretel PII (label mode)": (None, 72.2, None)}

# ---------------------------------------------------------------- rendering
PALETTE = 8
ID_RE = re.compile(r"^([A-Z]+\d+) (.*)$")
QUOTE_RE = re.compile(r'("[^"]+")')


def render(text, spans, gold=None, colors=None):
    """Text with spans highlighted; spans not in `gold` are outlined red."""
    colors = {} if colors is None else colors
    gold_keys = {tuple(g) for g in gold} if gold is not None else None
    out, pos = [], 0
    for s, e, label in sorted((tuple(x) for x in spans), key=lambda x: x[0]):
        if s < pos:
            continue
        c = colors.setdefault(label, f"c{len(colors) % PALETTE + 1}")
        cls = [c] + (["bad"] if gold_keys is not None and (s, e, label) not in gold_keys else []) + \
              (["long"] if e - s > 28 else [])
        title = ' title="differs from gold"' if "bad" in cls else ""
        out.append(html.escape(text[pos:s]))
        out.append(f'<mark class="{" ".join(cls)}"{title}>{html.escape(text[s:e])}<b>{html.escape(label)}</b></mark>')
        pos = e
    out.append(html.escape(text[pos:]))
    return f'<div class="spans">{"".join(out) or "<i>No text.</i>"}</div>'


def card(title, inner, note="", tone=""):
    n = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return f'<div class="card {tone}"><div class="card-h">{html.escape(title)}</div>{inner}{n}</div>'


def missed(text, gold, out):
    got = {(s, e) for s, e, _ in out}
    m = [f'"{text[s:e]}"' for s, e, _ in gold if (s, e) not in got]
    return "Missed: " + ", ".join(m) if m else ""


def spans_of(res):
    return [[r["start"], r["end"], r["label"]] for r in res]


def rules_html(text, added=None):
    """A rulebook (or part of one) as a readable list: one block per section, rule IDs as badges, quotes marked.
    Lines in `added` are marked as ours; the rest of each section is shown as context."""
    out = []
    for head, block in sections(text):
        body = [line.strip() for line in block.splitlines()[1 if head else 0:] if line.strip()]
        if head is None and not body:
            continue
        title = ("General rules" if head == "[RULES]" else "Closing rules" if head == "[RULES] closing"
                 else "Never extract" if head == "[TYPE] NOT_AN_ENTITY"
                 else head[len("[TYPE] "):] if head else "")
        desc, items = "", []
        for line in body:
            if line.startswith("[DESCRIPTION]"):
                desc = line[len("[DESCRIPTION]"):].strip()
                continue
            m = ID_RE.match(line)
            rid, rest = (m.group(1), m.group(2)) if m else ("", line)
            rest = QUOTE_RE.sub(r'<q>\1</q>', html.escape(rest, quote=False))
            new = added is not None and line in added
            badge = f"<span class=rid>{rid}</span>" if rid else ""
            items.append(f'<li{" class=new" if new else ""}>{badge}{rest}</li>')
        cls = "sec gen" if head in ("[RULES]", "[RULES] closing") else "sec"
        out.append(f'<div class="{cls}"><div class="sec-h">{html.escape(title)}</div>'
                   + (f'<div class="sec-d">{html.escape(desc)}</div>' if desc else "")
                   + (f'<ul>{"".join(items)}</ul>' if items else "") + "</div>")
    return f'<div class="rules">{"".join(out)}</div>'


# ---------------------------------------------------------------- the rulebook editor (both tabs)
EDITORS = ["Form", "Text"]
FIELDS_PER_CARD = 4   # name, description, rules, examples; each card also has its accordion


def card_label(name, desc):
    name = type_name(name, upper=False)
    if not name:
        return "(new class)"
    desc = (desc or "").strip()
    return f"{name}  ·  {desc[:90] + ('…' if len(desc) > 90 else '')}" if desc else name


class RulebookEditor:
    """General rules, one collapsible card per class (name, description, rules, examples, move/remove), never-extract
    rules, and a Form / Text switch; both views edit the same rulebook. `auto` records whether the rulebook uses the
    form's own rule numbering (True) or is kept verbatim (False, e.g. the benchmark rulebooks)."""

    def __init__(self, slots, text0):
        self.slots = slots
        form0 = text_to_form(text0)
        n0 = len(form0["classes"])
        self.editor = gr.Radio(EDITORS, value=EDITORS[0], label="Rulebook", elem_classes="chips",
                               info="Form: one card per class. Text: the rulebook exactly as the model reads it. "
                                    "Both edit the same rulebook.")
        self.vis = gr.State([k < n0 for k in range(slots)])
        self.auto = gr.State(form0.get("auto_ids", True))
        with gr.Column(visible=True, elem_classes="form") as self.form_col:
            self.general = gr.Textbox(form0["general"], label="General rules",
                                      info="One per line: rules for every class, e.g. which rule wins when two "
                                           "disagree.", lines=2, max_lines=8)
            gr.HTML('<div class="form-h">Classes <span>name, one-line description, rules (one per line) and '
                    'examples; order is the rulebook\'s order</span></div>')
            self.cards = []
            for k in range(slots):
                c = form0["classes"][k] if k < n0 else {}
                with gr.Accordion(card_label(c.get("name"), c.get("description")), open=False, visible=k < n0,
                                  elem_classes="cls") as acc:
                    with gr.Row():
                        nm = gr.Textbox(c.get("name", ""), label="Name", placeholder="e.g. PERSON", scale=1,
                                        min_width=140)
                        ds = gr.Textbox(c.get("description", ""), label="Description",
                                        placeholder="One line: what this class is", scale=3)
                    rl = gr.Textbox(c.get("rules", ""), label="Rules", info="One per line: what to include or exclude.",
                                    lines=2, max_lines=10)
                    ex = gr.Textbox(c.get("examples", ""), label="Examples",
                                    info='Optional, quoted and comma-separated: "first", "second".')
                    with gr.Row(elem_classes="cls-tools"):
                        up = gr.Button("↑ Move up", size="sm", scale=0, min_width=100)
                        dn = gr.Button("↓ Move down", size="sm", scale=0, min_width=110)
                        rm = gr.Button("✕ Remove class", size="sm", scale=0, min_width=130, variant="stop")
                self.cards.append((acc, nm, ds, rl, ex, up, dn, rm))
            with gr.Row():
                add = gr.Button("+ Add class", size="sm", scale=0, min_width=120)
                op = gr.Button("expand all", size="sm", scale=0, min_width=100)
                cl = gr.Button("collapse all", size="sm", scale=0, min_width=100)
            self.never = gr.Textbox(form0["never"], label="Never extract",
                                    info="One per line: what is never tagged (optional).", lines=1, max_lines=6)
            self.closing = gr.Textbox(form0["closing"], label="Closing rules",
                                      info='One per line: rules after the classes, e.g. "these override everything '
                                           'above" (optional).', lines=1, max_lines=6)
        with gr.Column(visible=False) as self.text_col:
            self.book = gr.Textbox(text0, show_label=False, lines=18, max_lines=30, elem_classes="mono")

        self.fields = [x for c in self.cards for x in c[1:5]]
        self.card_outputs = [x for c in self.cards for x in c[0:5]]
        self.inputs = [self.editor, self.book, self.general, self.never, self.closing, self.vis, self.auto,
                       *self.fields]
        self.load_outputs = [self.book, self.vis, self.auto, self.general, self.never, self.closing,
                             *self.card_outputs]

        self.editor.change(self.switch, self.inputs,
                           [self.form_col, self.text_col, *self.load_outputs])
        add.click(self.add_class, self.vis, [self.vis] + [c[0] for c in self.cards])
        op.click(lambda: [gr.update(open=True) for _ in range(slots)], None, [c[0] for c in self.cards])
        cl.click(lambda: [gr.update(open=False) for _ in range(slots)], None, [c[0] for c in self.cards])
        for k, (acc, nm, ds, rl, ex, up, dn, rm) in enumerate(self.cards):
            for box_ in (nm, ds):
                box_.blur(lambda a, b: gr.update(label=card_label(a, b)), [nm, ds], acc)
            rm.click(self.remove_class(k), self.vis, [self.vis, acc, nm, ds, rl, ex])
            for btn, step in ((up, -1), (dn, 1)):
                if 0 <= k + step < slots:
                    other = self.cards[k + step]
                    btn.click(self.move_class(k, step), [self.vis, nm, ds, rl, ex, *other[1:5]],
                              [acc, nm, ds, rl, ex, other[0], *other[1:5]])

    # ---- values
    def load(self, text):
        """Values for load_outputs from a rulebook text."""
        form = text_to_form(text)
        classes = form["classes"]
        if len(classes) > self.slots:
            raise gr.Error(f"This rulebook has {len(classes)} classes; the form holds {self.slots}.")
        out = [text, [k < len(classes) for k in range(self.slots)], form.get("auto_ids", True),
               form["general"], form["never"], form["closing"]]
        for k in range(self.slots):
            c = classes[k] if k < len(classes) else {}
            out += [gr.update(visible=k < len(classes), label=card_label(c.get("name"), c.get("description")),
                              open=False), c.get("name", ""), c.get("description", ""), c.get("rules", ""),
                    c.get("examples", "")]
        return out

    def read(self, editor, book, general, never, closing, vis, auto, *fields):
        """The rulebook text the user is looking at."""
        if editor == EDITORS[1]:
            return book
        classes = []
        for k in range(self.slots):
            name, desc, rules, examples = fields[FIELDS_PER_CARD * k:FIELDS_PER_CARD * (k + 1)]
            if vis[k] and (name or "").strip():
                classes.append({"name": name, "description": desc, "rules": rules, "examples": examples})
        return form_to_text({"general": general, "classes": classes, "never": never, "closing": closing,
                             "auto_ids": auto})

    # ---- events
    def switch(self, editor, book, general, never, closing, vis, auto, *fields):
        keep = [gr.update()] * (3 + (1 + FIELDS_PER_CARD) * self.slots)   # general, never, closing, then cards
        if editor == EDITORS[1]:
            try:
                text = self.read(EDITORS[0], book, general, never, closing, vis, auto, *fields)
            except ValueError as err:
                raise gr.Error(str(err)) from err
            return [gr.update(visible=False), gr.update(visible=True), text, vis, auto] + keep
        try:
            vals = self.load(book)
        except (ValueError, gr.Error) as err:
            gr.Warning(str(err))
            return [gr.update(visible=False), gr.update(visible=True), book, vis, auto] + keep
        return [gr.update(visible=True), gr.update(visible=False), *vals]

    def add_class(self, vis):
        if all(vis):
            raise gr.Error(f"The form holds {self.slots} classes; switch to Text to add more.")
        k = vis.index(False)
        return [vis[:k] + [True] + vis[k + 1:]] + [gr.update(visible=True, open=True, label="(new class)")
                                                    if i == k else gr.update() for i in range(self.slots)]

    def remove_class(self, k):
        return lambda vis: [vis[:k] + [False] + vis[k + 1:], gr.update(visible=False), "", "", "", ""]

    def move_class(self, k, step):
        def fn(vis, *fields):
            if not vis[k + step]:
                return [gr.update()] * (2 * (1 + FIELDS_PER_CARD))
            a, b = list(fields[:FIELDS_PER_CARD]), list(fields[FIELDS_PER_CARD:])
            return [gr.update(label=card_label(b[0], b[1]), open=False), *b,
                    gr.update(label=card_label(a[0], a[1]), open=False), *a]
        return fn


# ---------------------------------------------------------------- tab 1: edit a rule
EX_SLOTS = 20   # the largest benchmark rulebook has 17 classes
# examples written for the demo (no benchmark gold) and three benchmark examples, from the most familiar use case to
# the most specialised; the other benchmark examples are in the Benchmarks tab only
TAB1_ORDER = ["Support tickets", "Job application", "Movie search", "Clinical note", "Wikipedia", "Music"]
DEMO_NOTE = " The Never extract and Closing rules were added for this demo; the test F1 was measured without them."


def tab1_example(e):
    """An example as the Edit a rule tab shows it. Benchmark examples are shown with a Never extract and a Closing
    rule added for the demo (scripts/build_demo_rules.py checks that they leave the output unchanged)."""
    chip, _, what = e["name"].partition(": ")
    if "dataset" in e:
        base, ours, _ = SCORES[e["dataset"]]
        caption = f'{e["dataset"]} test set, {e["size"]}: {base} → {ours} F1 with these rules.'
        caption += DEMO_NOTE if "demo_rules" in e else ""
    else:
        caption = e["caption"]
    return {"chip": chip, "what": what[:1].upper() + what[1:], "caption": caption, "text": e["text"],
            "rules": e.get("demo_rules", e["rules"]), "spans": e.get("demo_b", e["b"])}


_examples = {e["name"].partition(": ")[0]: e for e in DEMO + EXAMPLES}
TAB1 = {chip: tab1_example(_examples[chip]) for chip in TAB1_ORDER}


def stored_out(chip):
    e = TAB1[chip]
    return card("Output", render(e["text"], e["spans"]), tone="accent")


def example_head(chip):
    e = TAB1[chip]
    return (f'<div class="ex-what">{html.escape(e["what"])}</div>'
            f'<div class="ex-stat">{html.escape(e["caption"])}</div>')


def run_with(editor_obj):
    def fn(text, *editor_values):
        if not text.strip():
            raise gr.Error("Type or paste a text first.")
        try:
            rulebook = editor_obj.read(*editor_values)
            with LOCK:
                out = spans_of(TAGGER.extract(text, rulebook))
        except ValueError as err:
            raise gr.Error(str(err)) from err
        return card("Output", render(text, out), tone="accent")
    return fn


# ---------------------------------------------------------------- tab 2: your text
MODES = ["Find the spans", "Label my spans"]
TEXT_SLOTS = 12


def bracketed(text, marks):
    for m in sorted(marks, key=lambda m: -len(m)):
        text = text.replace(m, f"\x01{m}\x02", 1)
    return text.replace("\x01", "[").replace("\x02", "]")


def unmark(marked):
    text, spans, pos = "", [], 0
    for m in re.finditer(r"\[([^\[\]]+)\]", marked):
        text += marked[pos:m.start()]
        spans.append((len(text), len(text) + len(m.group(1))))
        text += m.group(1)
        pos = m.end()
    return text + marked[pos:], spans


def mode_changed(mode, name, text):
    label = mode == MODES[1]
    t = TEMPLATES[name]
    if label and "[" not in text and text == t.sample:
        text = bracketed(text, t.marks)
    if not label and "[" in text:
        text = text.replace("[", "").replace("]", "")
    hint = "Text, with the spans to label in [square brackets]" if label else "Text"
    return gr.update(value=text, label=hint), gr.update(visible=label), "", gr.update(visible=label)


def run_text_with(editor_obj):
    def fn(mode, text, allow_none, *editor_values):
        if not text.strip():
            raise gr.Error("Type or paste a text first.")
        try:
            rulebook = editor_obj.read(*editor_values)
            if mode == MODES[0]:
                with LOCK:
                    res = TAGGER.extract(text, rulebook)
                return card("Spans", render(text, spans_of(res)), tone="accent"), [], res
            plain, spans = unmark(text)
            if not spans:
                raise gr.Error("Mark at least one span with [square brackets].")
            with LOCK:
                res = TAGGER.label(plain, spans, rulebook, allow_none=allow_none)
        except ValueError as err:
            raise gr.Error(str(err)) from err
        rows_out = [[r["text"], r.get("label", "—"), r.get("confidence", ""),
                     ", ".join(f"{k} {p}" for k, p in r.get("alternatives", [])) or r.get("unsupported", "")]
                    for r in res]
        shown = [[r["start"], r["end"], r.get("label", "unsupported")] for r in res]
        return card("Labels", render(plain, shown), tone="accent"), rows_out, res
    return fn


def snippet_with(editor_obj):
    def fn(mode, text, *editor_values):
        try:
            rulebook = editor_obj.read(*editor_values)
        except ValueError as err:
            raise gr.Error(str(err)) from err
        book = rulebook.replace('"""', '\\"\\"\\"')
        head = (f'from redefiner import Tagger\n\ntagger = Tagger.from_pretrained("{HF_REPO}")\n\n'
                f'rulebook = """{book}"""\n')
        if mode == MODES[0]:
            code = head + f"text = {json.dumps(text, ensure_ascii=False)}\ntagger.extract(text, rulebook)\n"
        else:
            plain, spans = unmark(text)
            code = head + (f"text = {json.dumps(plain, ensure_ascii=False)}\nspans = {spans}  # (start, end) "
                           f"character offsets\ntagger.label(text, spans, rulebook)\n")
        return gr.update(value=code, visible=True)
    return fn


# ---------------------------------------------------------------- tab 3: benchmarks
def bench_example(e):
    """One Benchmarks-tab example from examples.json or bench_examples.json."""
    if e.get("mode") == "label":
        return {"label": True, "text": e["text"], "types": e["types"],
                "gold": [[t["start"], t["end"], t["gold"]] for t in e["targets"]],
                "a": [[t["start"], t["end"], t["label"] or "unsupported"] for t in e["targets"]]}
    return {"text": e["text"], "gold": e["gold"], "a": e["a"], "b": e["b"], "base": e["base"],
            "rules": e.get("rules_book", e.get("rules"))}


BENCH = {e["dataset"]: bench_example(e) for e in EXAMPLES} | {k: bench_example(e) for k, e in BENCH_EXTRA.items()}


def strip_id(line):
    """A rule line without its ID, so renumbered lines compare equal."""
    return ID_RE.sub(r"\2", line.strip())


def stat_html(name):
    g, r, g34 = SCORES[name]
    if g is None:
        return (f'<div class="stats"><div class="stat accent"><span>{r:.1f}%</span>test accuracy</div>'
                f'<div class="stat"><span>S1MB</span>inputs unchanged · span given → type</div></div>')
    return (f'<div class="stats"><div class="stat"><span>{g:.1f}</span>guidelines only</div>'
            f'<div class="stat accent"><span>{r:.1f}</span>+ our rules</div>'
            f'<div class="stat"><span>{g34:.1f}</span>GoLLIE-34B</div><div class="stat-cap">test F1</div></div>')


def show_bench(name):
    e = BENCH[name]
    colors = {}
    cols = [card("Gold", render(e["text"], e["gold"], colors=colors))]
    if e.get("label"):
        cols.append(card("Model (S1MB's own types)", render(e["text"], e["a"], e["gold"], colors), tone="accent"))
        rules = (f'<div class="rules"><div class="sec"><div class="sec-d">No rules added: S1MB\'s own '
                 f'{len(e["types"])} candidate types and descriptions are used unchanged. Each span is given; the '
                 f'model picks its type.</div></div></div>')
        title = "Inputs"
    else:
        cols.append(card("Guidelines only", render(e["text"], e["a"], e["gold"], colors),
                         missed(e["text"], e["gold"], e["a"])))
        cols.append(card("Guidelines + our rules", render(e["text"], e["b"], e["gold"], colors),
                         missed(e["text"], e["gold"], e["b"]), tone="accent"))
        old = {strip_id(line) for line in e["base"].splitlines()}
        new = {line.strip() for line in e["rules"].splitlines() if line.strip() and strip_id(line) not in old}
        changed = changed_sections(e["base"], e["rules"])
        n_sec = len([h for h, _ in sections(changed) if h])
        rules = ('<div class="legend"><span class="sw"></span>our added lines · the rest is GoLLIE\'s original '
                 'text in the same sections</div>' + rules_html(changed, new))
        title = f"The rules we added · {len(new)} lines in {n_sec} sections"
    grid = f'<div class="grid g{len(cols)}">{"".join(cols)}</div>'
    return stat_html(name), grid, gr.update(label=title), rules


# ---------------------------------------------------------------- page
THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"])
# Gradio follows the visitor's system dark mode; the page is designed for the light theme, so reload with it
FORCE_LIGHT = """() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u.href);
  }
}"""

HERO = ('<div class="hero"><h1>ReDefiNER</h1>'
        '<p><span class="tag">Define what to extract. In plain English.</span> '
        'A generalist 0.8B model for entity extraction and span labeling, guided by the rulebook you write.</p>'
        '<div class="badges"><span class="badge">0.8B parameters</span><span class="badge">runs on CPU</span>'
        '<span class="badge">extract + label</span><span class="badge">Apache-2.0</span>'
        '<span class="links">' + "".join(f'<a href="{u}" target="_blank">{k}</a>' for k, u in LINKS.items())
        + '</span></div></div>')

with gr.Blocks(title="ReDefiNER") as demo:
    gr.HTML(HERO)
    with gr.Tabs(selected=os.environ.get("DEMO_TAB", "edit")):
        # ------------------------------------------------ tab 1
        with gr.Tab("Edit a rule", id="edit"):
            first = TAB1_ORDER[0]
            chip = gr.Radio(TAB1_ORDER, value=first, show_label=False, elem_classes="chips")
            ex_head = gr.HTML(example_head(first))
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    ex_text = gr.Textbox(TAB1[first]["text"], label="Text", lines=3, max_lines=6)
                    ex_editor = RulebookEditor(EX_SLOTS, TAB1[first]["rules"])
                with gr.Column(scale=5, elem_classes="sticky"):
                    ex_go = gr.Button("Run", variant="primary")
                    ex_out = gr.HTML(stored_out(first))
                    gr.HTML('<div class="tip"><b>Try it:</b> open a class, delete or change a rule (or write a new '
                            'one), and press Run to see how the output changes. The first run after an edit takes a '
                            'few seconds on CPU.</div>')
            def pick_example(c):
                return [example_head(c), TAB1[c]["text"], *ex_editor.load(TAB1[c]["rules"]), stored_out(c)]

            chip.change(pick_example, chip, [ex_head, ex_text, *ex_editor.load_outputs, ex_out],
                        api_name="pick_example")
            ex_go.click(run_with(ex_editor), [ex_text, *ex_editor.inputs], ex_out, api_name="run_example")

        # ------------------------------------------------ tab 2
        with gr.Tab("Your text", id="text"):
            with gr.Row():
                mode = gr.Radio(MODES, value=MODES[0], show_label=False, elem_classes="chips", scale=2)
                tp = gr.Dropdown(list(TEMPLATES), value="PII", label="Start from a template", scale=1)
            t_text = gr.Textbox(TEMPLATES["PII"].sample, label="Text", lines=3, max_lines=10)
            t_editor = RulebookEditor(TEXT_SLOTS, template_text("PII"))
            with gr.Row():
                t_go = gr.Button("Run", variant="primary", scale=0, min_width=140)
                t_none = gr.Checkbox(False, label="Allow NONE (the span may be none of the types)", visible=False)
                gr.HTML('<span class="hint">The first run after a rulebook edit takes a few seconds.</span>')
            t_out = gr.HTML()
            t_table = gr.Dataframe(headers=["Span", "Label", "Confidence", "Alternatives / reason"], visible=False,
                                   interactive=False, wrap=True)
            with gr.Row():
                t_copy = gr.Button("Copy as Python", size="sm", scale=0, min_width=150)
            t_code = gr.Code(language="python", visible=False, label="Python")
            with gr.Accordion("JSON output", open=False):
                t_json = gr.JSON(show_label=False)

            def pick_template(name, m):
                t = TEMPLATES[name]
                text = bracketed(t.sample, t.marks) if m == MODES[1] else t.sample
                return [text, *t_editor.load(template_text(name))]

            tp.change(pick_template, [tp, mode], [t_text, *t_editor.load_outputs], api_name="pick_template")
            mode.change(mode_changed, [mode, tp, t_text], [t_text, t_none, t_out, t_table])
            t_go.click(run_text_with(t_editor), [mode, t_text, t_none, *t_editor.inputs], [t_out, t_table, t_json],
                       api_name="run_text")
            t_copy.click(snippet_with(t_editor), [mode, t_text, *t_editor.inputs], t_code, api_name="snippet")

        # ------------------------------------------------ tab 3
        with gr.Tab("Benchmarks", id="bench"):
            gr.HTML('<span class="hint">One real test example per dataset: the gold labels, the output with the '
                    'published guidelines, and the output with our rules. <mark class="c1 bad sample">outlined'
                    '</mark> spans differ from gold. Outputs computed with this model on CPU.</span>')
            names = list(SCORES)
            b_stat, b_grid, b_title, b_rules = show_bench(names[0])
            bm = gr.Dropdown(names, value=names[0], label="Dataset")
            bm_stat = gr.HTML(b_stat)
            bm_grid = gr.HTML(b_grid)
            with gr.Accordion(b_title["label"], open=False) as bm_acc:
                bm_rules = gr.HTML(b_rules)
            gr.HTML(f'<span class="hint">Full tables and protocol: <a href="{LINKS["Benchmarks"]}" '
                    'target="_blank">benchmarks</a> · rerun a number with '
                    '<code>benchmarks/reproduce/reproduce.py</code>.</span>')
            bm.change(show_bench, bm, [bm_stat, bm_grid, bm_acc, bm_rules], api_name="show_bench")

    gr.HTML('<div class="limits"><b>Limits:</b> rules that fight a strong prior (e.g. retyping a person\'s name) '
            'mostly don\'t work · English only · 256 tokens per pass · spans ≤ 16 tokens.</div>')


def warm():
    """Encode every example and template rulebook once, after the app is up (keeps the health check fast)."""
    for e in EXAMPLES:
        with LOCK:
            TAGGER.extract(e["text"], e["base"])
    for e in TAB1.values():
        with LOCK:
            TAGGER.extract(e["text"], e["rules"])
    for name, t in TEMPLATES.items():
        if t.sample:
            with LOCK:
                TAGGER.extract(t.sample, template_text(name))


if __name__ == "__main__":
    threading.Thread(target=warm, daemon=True).start()
    # a bounded queue: past QUEUE_SIZE waiting runs, visitors are told it is busy instead of waiting minutes
    demo.queue(max_size=int(os.environ.get("QUEUE_SIZE", "40")))
    demo.launch(css=(HERE / "style.css").read_text(), theme=THEME, js=FORCE_LIGHT)

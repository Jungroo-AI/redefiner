"""The "Mask personal data" page (/redact): `redefiner redact` in the browser, with the PII prompt of
jungroo-ai/redefiner-pii or a policy the visitor writes.

The page shares the demo's one loaded model. It uses the settings of the redefiner-pii page (span limit 64,
none_bias 2, threshold 0.5): the span limit is set for each request, under the model lock, and put back afterwards,
so the other pages keep the trained limit of 16. The examples' outputs are stored in redact_examples.json (built by
scripts/build_redact_examples.py from this model) so they show before the model has warmed up.
"""
import html
import json
from contextlib import contextmanager
from pathlib import Path

import gradio as gr

from redefiner.redact import redact_text

HERE = Path(__file__).parent
PII_RULEBOOK = (HERE / "pii.txt").read_text(encoding="utf-8")   # the default rulebook of jungroo-ai/redefiner-pii
SETTINGS = {"max_span_tokens": 64, "none_bias": 2.0, "threshold": 0.5}   # that page's config.json
_STORED = HERE / "redact_examples.json"   # missing only while scripts/build_redact_examples.py builds it
EXAMPLES = json.loads(_STORED.read_text(encoding="utf-8")) if _STORED.exists() else {}
NAMES = list(EXAMPLES)
MODES = ["Mask all personal data", "Mask by my policy"]
KEYS = {MODES[0]: "all", MODES[1]: "policy"}
CUTOFF = {"all": 0.0, "policy": 0.35}   # the CLI's default, and the cutoff the PII page suggests for own policies
MAX_CHARS = 2000
PALETTE = 8

LINKS = {"Model page": "https://huggingface.co/jungroo-ai/redefiner-pii",
         "Install": "https://pypi.org/project/redefiner/",
         "Code": "https://github.com/jungroo-ai/redefiner",
         "How to write a policy": "https://github.com/jungroo-ai/redefiner/blob/main/docs/writing-rules.md",
         "Scores": "https://github.com/jungroo-ai/redefiner/tree/main/benchmarks/pii",
         "Entity extraction": "/"}
TITLE = "ReDefiNER PII: mask personal data, or only what your policy says"
DESCRIPTION = ("Try ReDefiNER PII, a free open-source model that masks personal data in text, or only what your "
               "plain-English policy says. It runs on a laptop CPU, with no API key.")

HERO = ('<div class="hero"><h1>ReDefiNER PII</h1>'
        '<p><span class="tag">Mask personal data, or only what your policy says.</span> '
        'An open-source AI model that runs on a laptop CPU. Try it here on made-up text, then run it on your own '
        'machine, so real data never leaves it.</p>'
        '<p class="proof">Ahead of OpenAI\'s Privacy Filter on all four English tests of the PIIMB benchmark. Its '
        'weights were never trained on a PII dataset.</p>'
        '<div class="badges"><span class="badge">free</span><span class="badge">open source (Apache 2.0)</span>'
        '<span class="badge">no API key</span><span class="badge">runs on your laptop</span>'
        '<span class="links">' + "".join(f'<a href="{u}" target="_blank">{k}</a>' for k, u in LINKS.items())
        + '</span></div></div>')
POLICY_HELP = ('<div class="hint">Each <code>[TYPE]</code> is something to mask, with a one-line '
               '<code>[DESCRIPTION]</code>. Say in the description what does not count, and list what must never be '
               'masked under <code>[TYPE] NOT_AN_ENTITY</code>. <a href="' + LINKS["How to write a policy"]
               + '" target="_blank">How to write a policy</a></div>')
PRESS = '<div class="tip">Press <b>Mask it</b> to run the model on this text.</div>'
FOOT = ('<div class="limits"><b>Use made-up text here:</b> this page runs on our server. To keep real data private, '
        'install it and run it on your own machine. <b>Not the only safeguard:</b> the model can miss personal data '
        'and mask text that is not personal, so check its output. English only · up to 2,000 characters here · '
        'one masked stretch is at most 64 tokens.</div>')


# ---------------------------------------------------------------- the model call
@contextmanager
def span_limit(tagger, width):
    old = tagger.max_width
    tagger.set_max_span_tokens(width)
    try:
        yield
    finally:
        tagger.set_max_span_tokens(old)


def find(tagger, lock, text, policy):
    """Every span the model returns for the PII prompt (policy None) or a policy, before any confidence cutoff."""
    with lock, span_limit(tagger, SETTINGS["max_span_tokens"]):
        res = tagger.extract(text, PII_RULEBOOK if policy is None else policy, threshold=SETTINGS["threshold"],
                             none_bias=SETTINGS["none_bias"])
    return [{k: s[k] for k in ("text", "label", "confidence", "start", "end")} for s in res]


# ---------------------------------------------------------------- output
def masked(text, spans, cutoff):
    return redact_text(text, [s for s in spans if s["confidence"] >= cutoff])


def found_html(text, spans, cutoff):
    """The text with every span marked, its type and confidence; spans under the cutoff are faded, not masked."""
    colors, out, pos = {}, [], 0
    for s in spans:
        c = colors.setdefault(s["label"], f"c{len(colors) % PALETTE + 1}")
        below = s["confidence"] < cutoff
        cls = [c] + (["below"] if below else []) + (["long"] if s["end"] - s["start"] > 28 else [])
        out.append(html.escape(text[pos:s["start"]]))
        out.append(f'<mark class="{" ".join(cls)}">{html.escape(text[s["start"]:s["end"]])}'
                   f'<b>{html.escape(s["label"])} {s["confidence"]:.2f}</b></mark>')
        pos = s["end"]
    out.append(html.escape(text[pos:]))
    n = sum(s["confidence"] >= cutoff for s in spans)
    faded = (f'<div class="faded-note">Faded: found, but below the confidence cutoff of {cutoff:.2f}, so not masked.'
             '</div>' if any(s["confidence"] < cutoff for s in spans) else "")
    return (f'<div class="card"><div class="card-h">What it found · {n} masked · type and confidence</div>'
            f'<div class="spans">{"".join(out) or "<i>No text.</i>"}</div>{faded}</div>')


def commands(mode, cutoff):
    flag = f" --min-confidence {cutoff:g}" if cutoff > 0 else ""
    if KEYS[mode] == "policy":
        return ("# save the text as note.txt and your policy as policy.txt, then:\n"
                f"pip install redefiner\nredefiner redact note.txt --rules policy.txt{flag}\n"
                "# writes note.redacted.txt, and note.audit.json with what was masked")
    return ("# save the text as note.txt, then:\n"
            f"pip install redefiner\nredefiner redact note.txt{flag}\n"
            "# writes note.redacted.txt, and note.audit.json with what was masked")


def show(text, spans, mode, cutoff):
    return masked(text, spans, cutoff), found_html(text, spans, cutoff), commands(mode, cutoff)


# ---------------------------------------------------------------- the page
def build(tagger, lock):
    """The page's components and events, inside the caller's `demo.route(...)` block."""
    first = EXAMPLES[NAMES[0]]
    mode0 = MODES[0] if first["mode"] == "all" else MODES[1]
    gr.HTML(HERO)
    chip = gr.Radio(NAMES, value=NAMES[0], show_label=False, elem_classes="chips")
    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            text = gr.Textbox(first["text"], label="Text (made-up text only)", lines=5, max_lines=14,
                              max_length=MAX_CHARS)
            mode = gr.Radio(MODES, value=mode0, show_label=False, elem_classes="chips")
            policy = gr.Code(first["policy"], language=None, label="Your policy (policy.txt)", lines=10,
                             max_lines=24, interactive=True, wrap_lines=True, visible=mode0 == MODES[1],
                             elem_classes="policy")
            help_ = gr.HTML(POLICY_HELP, visible=mode0 == MODES[1])
            with gr.Accordion("Confidence cutoff", open=False):
                cutoff = gr.Slider(0.0, 0.9, value=CUTOFF[first["mode"]], step=0.05,
                                   label="Mask only what the model is at least this sure of",
                                   info="0 masks everything it returns. 0.35 is the suggested cutoff for your own "
                                        "policies.")
            go = gr.Button("Mask it", variant="primary")
        with gr.Column(scale=5, elem_classes="sticky"):
            m0, f0, c0 = show(first["text"], first["stored"][first["mode"]], mode0, CUTOFF[first["mode"]])
            out = gr.Textbox(m0, label="Masked text", lines=5, max_lines=14, interactive=False, buttons=["copy"])
            found = gr.HTML(f0)
            cmd = gr.Code(c0, language="shell", label="Run it on your own machine", interactive=False, wrap_lines=True)
    gr.HTML(FOOT)
    last = gr.State({"text": first["text"], "mode": mode0, "policy": first["policy"],
                     "spans": first["stored"][first["mode"]]})

    def pick(name):
        e = EXAMPLES[name]
        m = MODES[0] if e["mode"] == "all" else MODES[1]
        spans = e["stored"][e["mode"]]
        policy_shown = e["mode"] == "policy"
        return [e["text"], m, gr.update(value=e["policy"], visible=policy_shown), gr.update(visible=policy_shown),
                CUTOFF[e["mode"]], *show(e["text"], spans, m, CUTOFF[e["mode"]]),
                {"text": e["text"], "mode": m, "policy": e["policy"], "spans": spans}]

    def switch(m, t, p, state):
        """The stored output when the text and policy are an example's; otherwise an empty output until Run."""
        key, policy_shown = KEYS[m], KEYS[m] == "policy"
        e = next((e for e in EXAMPLES.values() if e["text"] == t and (key == "all" or e["policy"] == p)), None)
        if e is not None:
            spans = e["stored"][key]
            outs = [*show(t, spans, m, CUTOFF[key]), {"text": t, "mode": m, "policy": p, "spans": spans}]
        else:
            outs = ["", PRESS, commands(m, CUTOFF[key]), {"text": t, "mode": m, "policy": p, "spans": []}]
        return [gr.update(visible=policy_shown), gr.update(visible=policy_shown), CUTOFF[key], *outs]

    def run(m, t, p, c):
        if not t.strip():
            raise gr.Error("Type or paste a text first, or pick an example.")
        if len(t) > MAX_CHARS:
            raise gr.Error(f"Up to {MAX_CHARS:,} characters here. Install it to mask longer files.")
        pol = p if KEYS[m] == "policy" else None
        if pol is not None and not pol.strip():
            raise gr.Error("Write a policy first, or pick an example.")
        try:
            spans = find(tagger, lock, t, pol)
        except ValueError as err:
            raise gr.Error(f"The policy could not be read: {err}") from err
        return [*show(t, spans, m, c), {"text": t, "mode": m, "policy": p, "spans": spans}]

    def recut(c, state):
        """Moving the cutoff re-masks the last output; no model call."""
        return [*show(state["text"], state["spans"], state["mode"], c)]

    chip.change(pick, chip, [text, mode, policy, help_, cutoff, out, found, cmd, last], api_name="redact_pick")
    mode.input(switch, [mode, text, policy, last], [policy, help_, cutoff, out, found, cmd, last])
    go.click(run, [mode, text, policy, cutoff], [out, found, cmd, last], api_name="redact")
    cutoff.release(recut, [cutoff, last], [out, found, cmd])


def warm(tagger, lock):
    """Encode the PII prompt and every example policy once."""
    for e in EXAMPLES.values():
        find(tagger, lock, e["text"], None)
        find(tagger, lock, e["text"], e["policy"])

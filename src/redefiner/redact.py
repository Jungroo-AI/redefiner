"""`redefiner redact`: mask what a rulebook defines in text files, and write an audit file of what was masked.

    redefiner redact notes.txt                       # the default rulebook of jungroo-ai/redefiner-pii
    redefiner redact notes.txt --rules policy.txt    # your own rulebook
    redefiner redact notes.txt --rules identifiers.txt --none-bias 4    # another rulebook with its own setting

For each FILE it writes FILE.redacted.txt, where every span is replaced by its type in brackets ("[EMAIL]"), and
FILE.audit.json, with each masked span's text, type, confidence and character offsets in the original file. Text files
only (UTF-8). The model can miss personal data: check the output, and don't use it as the only safeguard.
"""

import json
from pathlib import Path

from .api import Tagger

DEFAULT_MODEL = "jungroo-ai/redefiner-pii"


def redact_text(text: str, spans: list[dict]) -> str:
    """The text with each span (sorted, non-overlapping, as extract() returns them) replaced by "[TYPE]"."""
    out, pos = [], 0
    for s in spans:
        out.append(text[pos:s["start"]])
        out.append(f"[{s['label']}]")
        pos = s["end"]
    out.append(text[pos:])
    return "".join(out)


def outputs(path: Path, out_dir: Path | None) -> tuple[Path, Path]:
    folder = out_dir if out_dir is not None else path.parent
    return folder / f"{path.stem}.redacted{path.suffix or '.txt'}", folder / f"{path.stem}.audit.json"


def run(files: list[str], model: str, rules: str | None, device: str | None, min_confidence: float,
        out_dir: str | None, max_span_tokens: int | None, none_bias: float | None = None) -> int:
    paths = [Path(f).expanduser() for f in files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"redefiner redact: no such file: {', '.join(missing)}")
    texts = []
    for p in paths:
        try:
            texts.append(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            raise SystemExit(f"redefiner redact: {p} is not a UTF-8 text file; only text files are supported") from None
    rulebook = Path(rules).expanduser().read_text(encoding="utf-8") if rules else None
    tagger = Tagger.from_pretrained(model, device=device, max_span_tokens=max_span_tokens)
    if rulebook is None and tagger.default_rulebook is None:
        raise SystemExit(f"redefiner redact: {model} has no default rulebook; pass one with --rules")
    folder = Path(out_dir).expanduser() if out_dir else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    for p, text in zip(paths, texts):
        spans = [s for s in tagger.extract(text, rulebook, none_bias=none_bias) if s["confidence"] >= min_confidence]
        red_path, audit_path = outputs(p, folder)
        red_path.write_text(redact_text(text, spans), encoding="utf-8")
        audit = {"file": str(p), "model": model, "rulebook": rules or f"{model} default",
                 "min_confidence": min_confidence, "max_span_tokens": tagger.max_width,
                 "none_bias": tagger.config["none_bias"] if none_bias is None else none_bias,
                 "spans": [{k: s[k] for k in ("text", "label", "confidence", "start", "end")} for s in spans]}
        audit_path.write_text(json.dumps(audit, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{p}: {len(spans)} spans masked -> {red_path}, {audit_path}", flush=True)
    print("Check the output: the model can miss personal data, so don't use it as the only safeguard.", flush=True)
    return 0


def add_parser(sub) -> None:
    r = sub.add_parser("redact", help="mask what a rulebook defines in text files and write an audit file")
    r.add_argument("files", nargs="+", metavar="FILE", help="UTF-8 text files")
    r.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"local folder or Hugging Face repo (default: {DEFAULT_MODEL}, whose PII rulebook is used "
                        "when --rules is not given)")
    r.add_argument("--rules", default=None, help="a rulebook file to use instead of the model's default rulebook")
    r.add_argument("--device", default=None, help="cpu, cuda or cuda:N (default: cuda if available)")
    r.add_argument("--min-confidence", type=float, default=0.0,
                   help="mask only spans with at least this confidence (default 0: everything the model returns)")
    r.add_argument("--none-bias", type=float, default=None,
                   help="how readily borderline spans are masked: higher masks more (default: the model's setting)")
    r.add_argument("--max-span-tokens", type=int, default=None,
                   help="longest span in model tokens (default: the model's setting)")
    r.add_argument("--out-dir", default=None, help="where to write the outputs (default: next to each file)")

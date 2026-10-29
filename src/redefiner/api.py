"""Tagger: entity extraction and span labeling, configured by a rulebook.

    from redefiner import Tagger
    tagger = Tagger.from_pretrained("PATH_OR_HF_REPO")
    tagger.extract("Ada Lovelace worked with Charles Babbage.", rulebook)
    tagger.label("Ada Lovelace worked with Charles Babbage.", [(0, 12)], rulebook)

Offsets are character offsets into the original text, start inclusive, end exclusive.

Long texts: the model reads at most `max_query_tokens` (256) tokens of text per query. Longer texts are split into
overlapping windows, cut at sentence ends where possible. extract() keeps, where two windows disagree, the span that
sits further from a window edge; label() places each window around the spans it labels. Each window is labelled on its
own, so the same string can get different labels in different windows of one document.
"""

import collections
import json
import logging
import sys
from pathlib import Path
from typing import TypedDict

import torch

from .model import build_int8
from .rulebook import EncodedRulebook, Encoder, Query, to_text
from .windows import edge_distance, label_windows, text_windows

# decoding defaults, used where a release folder's config.json does not set them
DEFAULTS = {"d_span": 512, "max_width": 16, "max_query_tokens": 256, "threshold": 0.5, "none_bias": 2.0}
TOP_K = 3   # label(): the chosen type plus up to two alternatives
# the files from_pretrained needs from a Hugging Face repo; the model repo also holds benchmarks and docs
RELEASE_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "chat_template.jinja", "LICENSE", "NOTICE"]

Rulebook = str | dict


class _CpuFallbackFilter(logging.Filter):
    """Drops transformers' "falling back to its reference PyTorch implementation" warnings.

    They recommend causal_conv1d and flash-linear-attention, which are CUDA kernels; off CUDA the torch path (with
    fastconv's convolution) is the intended one.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "falling back to its reference PyTorch implementation" not in record.getMessage()


class Span(TypedDict):
    start: int
    end: int
    text: str
    label: str
    confidence: float


class LabelResult(TypedDict, total=False):
    start: int
    end: int
    text: str
    label: str
    confidence: float
    alternatives: list[tuple[str, float]]
    snapped: bool        # present (True) when snap=True widened the span to whole tokens
    unsupported: str     # present instead of label/confidence/alternatives when the span can't be labelled


class Tagger:
    """Extraction and labeling with one model and a cache of encoded rulebooks.

    Create it with `from_pretrained`. Each distinct rulebook is run through the model once and kept (up to
    `cache_size` rulebooks), so later calls with the same rulebook only pay for the text. Calls are not thread-safe:
    serve one call at a time per Tagger.
    """

    def __init__(self, model: torch.nn.Module, tokenizer, config: dict, cache_size: int = 8):
        self.model = model
        self.tok = tokenizer
        self.config = config
        self.encoder = Encoder(tokenizer)
        self.max_q = config["max_query_tokens"]
        self.max_width = config["max_width"]
        self.cache_size = cache_size
        self._cache = collections.OrderedDict()   # rulebook key -> (encoded rulebook, markers, states, prefix length)

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | None = None, revision: str | None = None,
                        cache_size: int = 8, cpu_fast_conv: bool = True) -> "Tagger":
        """Load a release folder: config.json, model.safetensors (weight-only int8) and the tokenizer files.

        Args:
            path: a local folder, or a Hugging Face repo id to download.
            device: "cpu", "cuda", "cuda:1", ... Defaults to "cuda" when a GPU is available, else "cpu".
            revision: the Hugging Face revision to download (ignored for a local folder).
            cache_size: how many encoded rulebooks to keep.
            cpu_fast_conv: on devices other than CUDA, replace the depthwise convolution used by transformers'
                Qwen3.5 module with an equivalent that is 7-14x faster on CPU builds of torch without MKLDNN
                (see fastconv.py).

        Off CUDA this also sets sys.modules["fla"] = None, so transformers uses its torch implementation instead of
        flash-linear-attention's GPU kernels, and silences transformers' warnings recommending those kernels. These
        changes apply to the whole process.

        From a Hugging Face repo only the files in RELEASE_FILES are downloaded.

        Raises:
            ValueError: the folder is not an int8 release folder.
        """
        from safetensors.torch import load_file
        from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer
        root = Path(path).expanduser()
        if not root.exists():
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(str(path), revision=revision, allow_patterns=RELEASE_FILES))
        config = dict(DEFAULTS)
        config.update(json.loads((root / "config.json").read_text()))
        if config.get("format") != "int8":
            raise ValueError(f"{root} is not an int8 release folder (config.json format != 'int8')")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            sys.modules.setdefault("fla", None)
            hub_kernels = logging.getLogger("transformers.integrations.hub_kernels")
            if not any(isinstance(f, _CpuFallbackFilter) for f in hub_kernels.filters):
                hub_kernels.addFilter(_CpuFallbackFilter())
            if cpu_fast_conv:
                from . import fastconv
                fastconv.install()
        # AutoTokenizer picks Qwen2Tokenizer for this folder, which splits some texts differently
        tok = Qwen3_5Tokenizer.from_pretrained(str(root))
        model = build_int8(config["text_config"], load_file(str(root / "model.safetensors")),
                           d_span=config["d_span"], max_width=config["max_width"])
        model.to(device).eval()
        return cls(model, tok, config, cache_size=cache_size)

    # ---------------------------------------------------------------- shared steps
    def _rulebook(self, rulebook: Rulebook):
        rb = self.encoder.rulebook(to_text(rulebook))
        if rb.key in self._cache:
            self._cache.move_to_end(rb.key)
        else:
            markers, states, prefix_len = self.model.encode_prefix(rb.ids, rb.marker_pos)
            self._cache[rb.key] = (rb, markers, states, prefix_len)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return self._cache[rb.key]

    def _states(self, entry, queries: list[Query], batch_size: int):
        """The marker states and, per query, the model's final states for its tokens."""
        _, markers, states, prefix_len = entry
        out = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            out += self.model.encode_queries(states, prefix_len, [q.ids for q in batch], self.tok.pad_token_id)
        return markers, out

    def _window_query(self, text: str, offsets: list[tuple[int, int]], s: int, e: int) -> tuple[Query, int]:
        """The query for tokens s..e-1 of the full text, tokenized on its own as a standalone text would be.

        The window shrinks until it fits max_q tokens. Returns the query and the window's character offset.
        """
        while True:
            c0, c1 = offsets[s][0], offsets[e - 1][1]
            ids, w_offs = self.encoder.tokens(text[c0:c1])
            if len(ids) <= self.max_q or e - s <= 1:
                return self.encoder.query(text[c0:c1], ids[: self.max_q], w_offs[: self.max_q]), c0
            e -= 1

    # ---------------------------------------------------------------- extract
    @torch.no_grad()
    def extract(self, texts: str | list[str], rulebook: Rulebook, threshold: float | None = None,
                none_bias: float | None = None, batch_size: int = 16) -> list[Span] | list[list[Span]]:
        """Find and label the spans the rulebook asks for.

        Args:
            texts: a string, or a list of strings.
            rulebook: rulebook text, or a dict (see `to_text`).
            threshold: minimum span probability (default from config.json, 0.5).
            none_bias: subtracted from the NONE logit before a type is chosen (default from config.json, 2.0).
            batch_size: queries per forward pass.

        Returns:
            Per text, its spans sorted by start; a single list when `texts` is a string. A span's confidence is its
            span probability times its type probability. `threshold` filters on the span probability alone, and
            `none_bias` lowers NONE only while the type is picked, so a returned span can have a confidence below
            `threshold`.

        Raises:
            ValueError: the rulebook has no [TYPE] sections.
        """
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        threshold = self.config["threshold"] if threshold is None else threshold
        none_bias = self.config["none_bias"] if none_bias is None else none_bias
        entry = self._rulebook(rulebook)
        rb: EncodedRulebook = entry[0]
        queries, meta = [], []   # meta: (text index, char offset, window starts the text, window ends the text)
        for ti, text in enumerate(texts):
            ids, offs = self.encoder.tokens(text)
            if not ids:
                continue
            for s, e in text_windows(text, offs, self.max_q):
                if (s, e) == (0, len(ids)):
                    q, c0 = self.encoder.query(text, ids, offs), 0
                else:
                    q, c0 = self._window_query(text, offs, s, e)
                queries.append(q)
                meta.append((ti, c0, s == 0, e == len(ids)))
        found = [[] for _ in texts]
        if queries:
            markers, hs = self._states(entry, queries, batch_size)
            for q, h, (ti, c0, at_start, at_end) in zip(queries, hs, meta):
                n = len(q.offsets)
                for s, e, k, conf in self.model.predict(markers, h, q.q2_start, threshold=threshold,
                                                        none_bias=none_bias):
                    cs, ce = Encoder.char_span(q, s, e)
                    found[ti].append((edge_distance(s, e, n, at_start, at_end), conf, c0 + cs, c0 + ce, rb.types[k]))
        results = []
        for ti, spans in enumerate(found):
            kept = []
            for _, conf, cs, ce, label in sorted(spans, key=lambda x: (-x[0], -x[1])):
                if all(ce <= a or cs >= b for a, b, _, _ in kept):
                    kept.append((cs, ce, label, conf))
            results.append([Span(start=cs, end=ce, text=texts[ti][cs:ce], label=label, confidence=round(conf, 3))
                            for cs, ce, label, conf in sorted(kept)])
        return results[0] if single else results

    # ---------------------------------------------------------------- label
    @torch.no_grad()
    def label(self, texts: str | list[str], spans: list, rulebook: Rulebook, allow_none: bool = False,
              snap: bool = False, none_bias: float | None = None,
              batch_size: int = 16) -> list[LabelResult] | list[list[LabelResult]]:
        """The type of each given span (label head only, no extraction).

        Args:
            texts: a string, or a list of strings.
            spans: for a single string, a list of (start, end) or {"start", "end"} character offsets; for a list of
                strings, one such list per text.
            rulebook: rulebook text, or a dict (see `to_text`).
            allow_none: False forces a choice among the rulebook's types; True lets the model answer "NONE".
            snap: label a span that does not start and end on token boundaries as the span widened to whole tokens
                (returning the widened offsets and "snapped": True), instead of reporting it unsupported.
            none_bias: with allow_none, subtracted from the NONE logit first (default from config.json, 2.0).
            batch_size: queries per forward pass.

        Returns:
            Per text, one result per span in the given order; a single list when `texts` is a string. A label's
            confidence is its probability among the rulebook's types (and NONE, with allow_none). A span is
            unsupported (an "unsupported" reason instead of a label) when its offsets are invalid, when it is longer
            than `max_width` model tokens, or when it does not start and end on token boundaries and snap is False.

        Raises:
            ValueError: the rulebook has no [TYPE] sections, or `spans` does not match `texts`.
        """
        single = isinstance(texts, str)
        if single:
            texts, spans = [texts], [spans]
        if len(texts) != len(spans):
            raise ValueError("label() needs one list of spans per text")
        none_bias = self.config["none_bias"] if none_bias is None else none_bias
        entry = self._rulebook(rulebook)
        names = entry[0].types
        results = [[None] * len(sp) for sp in spans]
        queries, jobs = [], []   # jobs[i]: (text index, char offset, [(span index, start, end)]) for queries[i]
        for ti, (text, sp) in enumerate(zip(texts, spans)):
            todo = []
            for si, (s, e) in enumerate(_offsets(x) for x in sp):
                if 0 <= s < e <= len(text):
                    todo.append((si, s, e))
                else:
                    results[ti][si] = LabelResult(start=s, end=e, text=text[max(0, s):max(0, e)],
                                                  unsupported="invalid offsets")
            if not todo:
                continue
            ids, offs = self.encoder.tokens(text)
            if len(ids) <= self.max_q:
                queries.append(self.encoder.query(text, ids, offs))
                jobs.append((ti, 0, todo))
                continue
            for s_tok, e_tok, group in label_windows(offs, todo, self.max_q):
                q, c0 = self._window_query(text, offs, s_tok, e_tok)
                queries.append(q)
                jobs.append((ti, c0, group))
        if queries:
            markers, hs = self._states(entry, queries, batch_size)
            for q, h, (ti, c0, group) in zip(queries, hs, jobs):
                starts, ends, v, _ = self.model.spans(h[q.q2_start:])
                where = {(a, b): n for n, (a, b) in enumerate(zip(starts.tolist(), ends.tolist()))}
                logits = self.model.label_logits(v, markers).float()
                for si, s, e in group:
                    results[ti][si] = self._label_span(texts[ti], q, c0, s, e, where, logits, names, allow_none,
                                                       snap, none_bias)
        return results[0] if single else results

    def _label_span(self, text, q, c0, s, e, where, logits, names, allow_none, snap, none_bias) -> LabelResult:
        """The result for span s..e of `text`, read from query `q`, which starts at character c0 of the text."""
        item = LabelResult(start=s, end=e, text=text[s:e])
        ls, le = s - c0, e - c0   # offsets inside this query
        first = next((t for t, (_, b) in enumerate(q.offsets) if b > ls), None)
        last = max((t for t, (a, _) in enumerate(q.offsets) if a < le), default=None)
        if first is None or last is None or last < first or ls < 0 or le > len(q.text):
            return {**item, "unsupported": "span is outside the text the model can read"}
        if last - first + 1 > self.max_width:
            return {**item, "unsupported": f"span is longer than {self.max_width} tokens"}
        if (first, last) not in where:
            return {**item, "unsupported": "span is not available"}
        cs, ce = Encoder.char_span(q, first, last)
        if (cs, ce) != (ls, le):
            if not snap:
                return {**item, "unsupported": "would be snapped: the span does not start and end on token boundaries"}
            item.update(start=c0 + cs, end=c0 + ce, text=text[c0 + cs:c0 + ce], snapped=True)
        z = logits[where[(first, last)]].clone()
        n_types = len(names)
        if allow_none:
            z[n_types] -= none_bias
            choices = names + ["NONE"]
        else:
            z = z[:n_types]
            choices = names
        pr = z.softmax(-1)
        top = pr.topk(min(TOP_K, len(pr)))
        ranked = [(choices[k], round(p, 3)) for k, p in zip(top.indices.tolist(), top.values.tolist())]
        item.update(label=ranked[0][0], confidence=ranked[0][1], alternatives=ranked[1:])
        return item


def _offsets(span) -> tuple[int, int]:
    """(start, end) from a (start, end) pair or a {"start", "end"} dict."""
    if isinstance(span, dict):
        return int(span["start"]), int(span["end"])
    return int(span[0]), int(span[1])

"""The span model: a Qwen3.5 text decoder with span and label heads, served as weight-only int8.

A rulebook runs once through the backbone; its per-layer states are kept, and each query continues from them, so a
new query on an unchanged rulebook costs only the query. The final states at the rulebook's marker positions (one per
type) are the label vectors.

Heads, read on the second copy of the query:
  span   all spans of up to max_width tokens: start/end projections + width embedding -> MLP -> span vector v;
         existence logit = w . v
  label  softmax over (marker vectors projected to the span space, plus a learned NONE) . v

Every nn.Linear of the backbone holds int8 weights with one fp32 scale per output channel, and the token embedding
holds int8 rows with one scale per row; everything else is fp32. Building the float model and converting it is in
export.py.
"""

import math
from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

PREFIX_CHUNK = 1024   # rulebook tokens per backbone pass; long rulebooks run in chunks to bound peak memory
ROTARY = "rotary."    # key prefix of the saved rotary tables (non-persistent buffers, stored so loads are bit-exact)
MROPE_AXES = 3        # Qwen3.5's rotary embedding takes (temporal, height, width) positions; text uses one for all


class StepCache:
    """The cache interface one Qwen3.5 decoder layer calls, for that layer only, with no in-place updates.

    transformers' own cache classes update their tensors in place, which rules out reusing one rulebook's states for
    many queries. This class follows the interface the Qwen3.5 modeling code calls (an internal of transformers, which
    is why the package pins its version).
    """

    def __init__(self, layer_idx, kv=None, conv=None, rec=None):
        self.layer_idx, self.kv, self.conv, self.rec = layer_idx, kv, conv, rec

    # full attention layers
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.kv is not None:
            key_states = torch.cat([self.kv[0], key_states], dim=-2)
            value_states = torch.cat([self.kv[1], value_states], dim=-2)
        self.kv = (key_states, value_states)
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        return 0 if self.kv is None else self.kv[0].shape[-2]

    # Gated DeltaNet layers
    @property
    def layers(self):
        return {self.layer_idx: SimpleNamespace(record_past=True, conv_states=[self.conv],
                                                recurrent_states=[self.rec])}

    def has_previous_state(self, layer_idx=None, state_idx=None):
        return self.rec is not None

    def update_conv_state(self, conv_states, layer_idx, conv_kernel_size=None, **kwargs):
        full = conv_states if self.conv is None else torch.cat([self.conv, conv_states], dim=-1)
        if self.conv is None and full.shape[-1] < conv_kernel_size:
            full = F.pad(full, (conv_kernel_size - full.shape[-1], 0))
        self.conv = full[..., -conv_kernel_size:]
        return full

    def update_recurrent_state(self, recurrent_states, layer_idx, **kwargs):
        self.rec = recurrent_states
        return recurrent_states

    def state(self):
        return self.kv if self.kv is not None else (self.conv, self.rec)


class SpanModel(nn.Module):
    def __init__(self, lm: nn.Module, d_span: int = 512, max_width: int = 16):
        super().__init__()
        self.lm = lm
        self.layer_types = lm.config.layer_types
        hidden = lm.config.hidden_size
        self.max_width = max_width
        self.d_span = d_span
        self.marker = nn.Parameter(torch.zeros(hidden))
        self.start = nn.Linear(hidden, d_span)
        self.end = nn.Linear(hidden, d_span)
        self.width = nn.Embedding(max_width, d_span)
        self.span_mlp = nn.Sequential(nn.Linear(3 * d_span, d_span), nn.GELU(), nn.Linear(d_span, d_span))
        self.exists = nn.Linear(d_span, 1)
        self.label_proj = nn.Linear(hidden, d_span)
        self.none = nn.Parameter(torch.zeros(d_span))

    # ---------------------------------------------------------------- backbone passes
    def embed(self, ids, marker_mask=None):
        e = self.lm.embed_tokens(ids)
        if marker_mask is not None and marker_mask.any():
            e = torch.where(marker_mask[..., None], self.marker.to(e.dtype), e)
        return e

    def run_layers(self, hidden, offset, prev=None, mask=None, keep_state=False):
        batch, length, _ = hidden.shape
        pos = torch.arange(length, device=hidden.device) + offset
        position_embeddings = self.lm.rotary_emb(hidden, pos.view(1, 1, -1).expand(MROPE_AXES, batch, -1))
        text_pos = pos.view(1, -1).expand(batch, -1)
        states = []
        for i, layer in enumerate(self.lm.layers):
            full_attn = self.layer_types[i] == "full_attention"
            st = prev[i] if prev is not None else ()
            if not st:
                cache = StepCache(i)
            elif full_attn:
                cache = StepCache(i, kv=st)
            else:
                cache = StepCache(i, conv=st[0], rec=st[1])
            hidden = layer(hidden, position_embeddings=position_embeddings, attention_mask=mask if full_attn else None,
                           position_ids=text_pos, past_key_values=cache)
            if keep_state:
                states.append(tuple(cache.state()))
        return self.lm.norm(hidden), states

    @torch.no_grad()
    def encode_prefix(self, prefix_ids, marker_pos, chunk=PREFIX_CHUNK):
        """Run one rulebook through the backbone.

        Returns the marker states (K, H), the per-layer states, and the rulebook length. The rulebook runs in chunks
        of `chunk` tokens, each continuing from the chunks before it, so peak memory does not grow with the square
        of the rulebook's length.
        """
        dev = self.marker.device
        p = torch.tensor([prefix_ids], device=dev)
        mm = torch.zeros_like(p, dtype=torch.bool)
        mm[0, marker_pos] = True
        e = self.embed(p, mm)
        n = p.shape[1]
        if not chunk or chunk >= n:
            h, states = self.run_layers(e, 0, keep_state=True)
            return h[0, marker_pos], states, n
        hs, states = [], None
        for s in range(0, n, chunk):
            x = e[:, s:s + chunk]
            length = x.shape[1]
            mask = None
            if states is not None:   # this chunk sees every earlier rulebook token and its own tokens causally
                causal = torch.ones(length, length, dtype=torch.bool, device=dev).tril()
                mask = torch.cat([torch.ones(length, s, dtype=torch.bool, device=dev), causal], dim=1)
                mask = mask.view(1, 1, length, s + length)
            h, states = self.run_layers(x, s, prev=states, mask=mask, keep_state=True)
            hs.append(h)
        return torch.cat(hs, dim=1)[0, marker_pos], states, n

    @torch.no_grad()
    def encode_queries(self, states, prefix_len, queries, pad_id):
        """Per query, the final states of all its tokens (length, H), continuing from the kept rulebook states."""
        dev = self.marker.device
        batch, length = len(queries), max(len(q) for q in queries)
        q = torch.full((batch, length), pad_id, device=dev)
        for b, ids in enumerate(queries):
            q[b, : len(ids)] = torch.tensor(ids, device=dev)
        causal = torch.ones(length, length, dtype=torch.bool, device=dev).tril()
        mask = torch.cat([torch.ones(length, prefix_len, dtype=torch.bool, device=dev), causal], dim=1)
        mask = mask.view(1, 1, length, prefix_len + length).expand(batch, 1, length, prefix_len + length)
        # broadcast the rulebook states over the batch; DeltaNet's recurrent state must be contiguous
        prev = [tuple(t.expand(batch, *t.shape[1:]) if j == 0 or self.layer_types[i] == "full_attention"
                      else t.expand(batch, *t.shape[1:]).contiguous() for j, t in enumerate(st))
                for i, st in enumerate(states)]
        h_q, _ = self.run_layers(self.embed(q), prefix_len, prev=prev, mask=mask)
        return [h_q[b, : len(ids)] for b, ids in enumerate(queries)]

    # ---------------------------------------------------------------- heads
    def spans(self, h):
        """All spans of up to max_width tokens over states h (n, H): (starts, ends, span vectors, existence logits)."""
        n = h.shape[0]
        idx = torch.arange(n, device=h.device)
        starts = idx.repeat_interleave(self.max_width)
        ends = starts + torch.arange(self.max_width, device=h.device).repeat(n)
        keep = ends < n
        starts, ends = starts[keep], ends[keep]
        hs, he = self.start(h.float()), self.end(h.float())
        v = self.span_mlp(torch.cat([hs[starts], he[ends], self.width(ends - starts)], dim=-1))
        return starts, ends, v, self.exists(v).squeeze(-1)

    def label_logits(self, v, markers):
        keys = torch.cat([self.label_proj(markers.float()), self.none[None]], dim=0)
        return v @ keys.T / math.sqrt(self.d_span)

    @torch.no_grad()
    def predict(self, markers, h, q2_start, threshold=0.5, none_bias=0.0):
        """Spans found in one query: [(first token, last token, label index, confidence)], non-overlapping, best first.

        none_bias > 0 is subtracted from the NONE logit before the type is chosen; the confidence stays unbiased.
        """
        starts, ends, v, logit = self.spans(h[q2_start:])
        p_span = logit.sigmoid()
        logits = self.label_logits(v, markers)
        probs = logits.softmax(-1)
        n_types = markers.shape[0]
        pick = probs
        if none_bias:
            pick = logits.clone()
            pick[:, n_types] -= none_bias
        cand = (p_span > threshold).nonzero().squeeze(-1).tolist()
        cand.sort(key=lambda i: -float(p_span[i]))
        taken, out = [], []
        for i in cand:
            s, e = int(starts[i]), int(ends[i])
            if any(not (e < a or s > b) for a, b in taken):
                continue
            k = int(pick[i].argmax())
            if k == n_types:
                continue
            taken.append((s, e))
            out.append((s, e, k, float(p_span[i] * probs[i, k])))
        return out


# ---------------------------------------------------------------- weight-only int8

class Int8Embedding(nn.Module):
    """Token embedding stored as int8 rows with one fp32 scale per row; rows are dequantized on lookup."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, padding_idx: int | None = None):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.padding_idx = padding_idx

    @classmethod
    def empty(cls, shape, padding_idx=None):
        """An uninitialised table, to be filled by load_state_dict."""
        return cls(torch.empty(shape, dtype=torch.int8), torch.empty(shape[0]), padding_idx)

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, rows_per_step: int = 8192):
        """Quantize an embedding, `rows_per_step` rows at a time to bound memory."""
        w = emb.weight.detach()
        q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
        scale = torch.empty(w.shape[0], device=w.device)
        for s in range(0, w.shape[0], rows_per_step):
            rows = w[s:s + rows_per_step].float()
            sc = rows.abs().amax(dim=1).clamp(min=1e-8) / 127
            q[s:s + rows_per_step] = torch.round(rows / sc[:, None]).to(torch.int8)
            scale[s:s + rows_per_step] = sc
        return cls(q, scale, emb.padding_idx)

    @property
    def weight(self):
        return self.q.float() * self.scale[:, None]

    def forward(self, ids):
        return self.q[ids].float() * self.scale[ids][..., None]


class Int8Linear(nn.Module):
    """nn.Linear stored as int8 weights with one fp32 scale per output channel: x (q s)^T = (x q^T) s."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    @classmethod
    def empty(cls, shape, bias=False):
        """An uninitialised layer with weight shape `shape`, to be filled by load_state_dict."""
        return cls(torch.empty(shape, dtype=torch.int8), torch.empty(shape[0]),
                   torch.empty(shape[0]) if bias else None)

    @classmethod
    def from_linear(cls, lin: nn.Linear):
        w = lin.weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127
        bias = None if lin.bias is None else lin.bias.detach().float()
        return cls(torch.round(w / scale[:, None]).to(torch.int8), scale, bias)

    def forward(self, x):
        y = F.linear(x, self.q.to(x.dtype)) * self.scale
        return y if self.bias is None else y + self.bias


def replace_linears(root: nn.Module, make: Callable[[nn.Linear], nn.Module]) -> None:
    """Replace every nn.Linear under `root` with make(linear)."""
    for name, m in list(root.named_modules()):
        if isinstance(m, nn.Linear):
            set_submodule(root, name, make(m))


def set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, attr = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, attr, module)


def build_int8(text_config: dict, state: dict, d_span: int = 512, max_width: int = 16) -> SpanModel:
    """The int8 SpanModel from a release file's state, without allocating any full-precision weights.

    The backbone is built on the meta device, its linear layers and embedding are swapped for empty int8 ones, and
    load_state_dict assigns the saved tensors.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    config = Qwen3_5TextConfig(**text_config)
    with torch.device("meta"):
        lm = Qwen3_5TextModel(config)
    lm.embed_tokens = Int8Embedding.empty(tuple(lm.embed_tokens.weight.shape), lm.embed_tokens.padding_idx)
    for layer in lm.layers:
        replace_linears(layer, lambda m: Int8Linear.empty(tuple(m.weight.shape), bias=m.bias is not None))
    lm.rotary_emb = type(lm.rotary_emb)(config=config, device="cpu")
    rotary = {k[len(ROTARY):]: v for k, v in state.items() if k.startswith(ROTARY)}
    weights = {k: v for k, v in state.items() if not k.startswith(ROTARY)}
    with torch.device("meta"):
        model = SpanModel(lm, d_span=d_span, max_width=max_width)
    model.load_state_dict(weights, strict=True, assign=True)
    for name, buf in rotary.items():
        setattr(lm.rotary_emb, name, buf)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        raise ValueError(f"tensors missing from the weights file: {left[:5]}")
    return model

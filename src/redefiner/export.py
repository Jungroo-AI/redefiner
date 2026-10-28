"""Building the int8 release model from a training checkpoint (needs the `export` extra: peft, accelerate).

The checkpoint holds LoRA adapters and the span/label heads. The backbone is the pinned Qwen3.5-0.8B base; the
adapters are merged into it and every linear layer and the token embedding are converted to weight-only int8.
See scripts/export_weights.py.
"""

import torch
import torch.nn as nn

from .model import ROTARY, Int8Embedding, Int8Linear, SpanModel, replace_linears, set_submodule

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj"]


def load_backbone(name: str = BASE_MODEL, revision: str | None = BASE_REVISION, device: str = "cpu") -> nn.Module:
    """The base model's text decoder in bf16 (the vision tower and LM head are dropped)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    full = Qwen3_5ForConditionalGeneration.from_pretrained(name, revision=revision, dtype=torch.bfloat16,
                                                           device_map=device, attn_implementation="sdpa")
    lm = full.model.language_model
    del full
    return lm


def add_lora(lm: nn.Module, rank: int, alpha: int) -> nn.Module:
    from peft import LoraConfig, inject_adapter_in_model
    lm = inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                                            target_modules=LORA_TARGETS), lm)
    for p in lm.parameters():
        p.requires_grad = False
    return lm


def float_model(trainable: dict, rank: int) -> SpanModel:
    """The bf16 SpanModel with the checkpoint's LoRA adapters and heads loaded."""
    lm = load_backbone()
    d_span, max_width = int(trainable["none"].shape[0]), int(trainable["width.weight"].shape[0])
    model = SpanModel(add_lora(lm, rank, 2 * rank), d_span=d_span, max_width=max_width)
    model.marker.data = lm.embed_tokens.weight.detach().float().mean(0).clone()   # training's initial value
    res = model.load_state_dict(trainable, strict=False)
    if res.unexpected_keys:
        raise ValueError(f"unexpected keys in the checkpoint: {res.unexpected_keys[:5]}")
    return model


def merge_lora(module: nn.Module) -> nn.Module:
    """Fold every LoRA adapter into its base nn.Linear and put the plain layer back in its place."""
    from peft.tuners.lora import LoraLayer
    for name, m in list(module.named_modules()):
        if isinstance(m, LoraLayer):
            m.merge()
            set_submodule(module, name, m.get_base_layer())
    return module


def quantize(model: SpanModel) -> SpanModel:
    """The float SpanModel (LoRA adapters on) as weight-only int8, with everything that stays float in fp32."""
    lm = model.lm
    lm.embed_tokens = Int8Embedding.from_embedding(lm.embed_tokens)
    for layer in lm.layers:
        merge_lora(layer.float())
        replace_linears(layer, Int8Linear.from_linear)
    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_floating_point():
            t.data = t.data.float()   # norms, rotary, marker, heads
    return model


def int8_state(model: SpanModel) -> dict[str, torch.Tensor]:
    """Everything an int8 release file holds: the quantized model's state plus the exact rotary tables."""
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    for name, buf in model.lm.rotary_emb.named_buffers():
        state[ROTARY + name] = buf.detach().clone().contiguous()
    return state

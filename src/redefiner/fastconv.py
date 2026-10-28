"""Depthwise conv1d fast path for Qwen3.5's Gated DeltaNet layers on CPU.

Without the GPU-only causal-conv kernels, transformers falls back to F.conv1d(groups=channels) with a 4-tap kernel.
On torch builds without MKLDNN (the macOS arm64 wheels) that takes a slow generic path: 0.35 s per call, 6.3 s of a
7.3 s query on an M2. The same result is K shifted multiply-adds (equal to F.conv1d within float rounding, about
2e-6). Only the Qwen3.5 modeling module's `F` is replaced; every other convolution is untouched.
"""
import types

import torch.nn.functional as F


def _conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    C = input.shape[1]
    if not (groups == C and weight.shape[0] == C and weight.shape[1] == 1 and stride == 1 and dilation == 1
            and isinstance(padding, int)):
        return F.conv1d(input, weight, bias, stride, padding, dilation, groups)
    K = weight.shape[-1]
    x = F.pad(input, (padding, padding)) if padding else input
    L = x.shape[-1] - K + 1
    w = weight[:, 0, :]
    out = x[:, :, 0:L] * w[:, 0].view(1, C, 1)
    for k in range(1, K):
        out = out + x[:, :, k:k + L] * w[:, k].view(1, C, 1)
    if bias is not None:
        out = out + bias.view(1, C, 1)
    return out


def install():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    if getattr(m.F, "_redefiner_fastconv", False):
        return
    proxy = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
    proxy.conv1d = _conv1d
    proxy._redefiner_fastconv = True
    m.F = proxy

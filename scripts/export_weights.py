"""Training checkpoint -> int8 release folder.

The checkpoint holds LoRA adapters and span/label heads. This loads the pinned Qwen3.5-0.8B base, merges the adapters
and converts the backbone to weight-only int8 (see redefiner/export.py), then writes:
  model.safetensors   the int8 model (int8 weights + fp32 scales, fp32 norms and heads, exact rotary tables)
  config.json         the backbone's architecture config and the decoding defaults
  tokenizer files     copied from the base model, so users download only this folder
  LICENSE, NOTICE     the base model's licence, and the attribution for the modified weights
Nothing from the checkpoint's training arguments (paths, data mix) is written.

usage: export_weights.py CHECKPOINT OUT_DIR      (needs: pip install -e .[export])
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from redefiner.api import DEFAULTS
from redefiner.export import BASE_MODEL, BASE_REVISION, float_model, int8_state, quantize

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]
NOTICE = (
    f"This model contains modified weights of Qwen3.5-0.8B (https://huggingface.co/{BASE_MODEL}, revision "
    f"{BASE_REVISION}), licensed under the Apache License 2.0 (see LICENSE).\n"
    "Modifications: LoRA adapters trained for span extraction and labeling were merged into the decoder layers; "
    "span and label heads were added; the decoder's linear layers and token embedding were converted to int8 with "
    "per-channel / per-row fp32 scales; the vision tower and language-model head were removed.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a training checkpoint into the int8 release folder.")
    ap.add_argument("checkpoint", type=Path, help="training checkpoint (.pt) with 'args' and 'trainable'")
    ap.add_argument("out", type=Path, help="output folder")
    args = ap.parse_args()

    sys.modules.setdefault("fla", None)   # export runs on CPU: keep transformers off flash-linear-attention's kernels
    state = torch.load(args.checkpoint, map_location="cpu")
    train_args, trainable = state["args"], state["trainable"]
    if train_args["model"] != BASE_MODEL:
        raise SystemExit(f"checkpoint was trained on {train_args['model']}, expected {BASE_MODEL}")
    model = quantize(float_model(trainable, rank=train_args["rank"])).eval()
    weights = int8_state(model)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(args.out / "model.safetensors"), metadata={"format": "pt"})
    # model_type and transformers_version also tell transformers this tokenizer is not an old Mistral one
    config = {"format": "int8", "model_type": "qwen3_5", "transformers_version": transformers.__version__,
              **DEFAULTS, "d_span": model.d_span, "max_width": model.max_width,
              "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
              "source_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "text_config": model.lm.config.to_dict()}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    base_dir = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION,
                                      allow_patterns=TOKENIZER_FILES + ["LICENSE"]))
    for name in TOKENIZER_FILES + ["LICENSE"]:
        if (base_dir / name).exists():
            shutil.copy(base_dir / name, args.out / name)
    (args.out / "NOTICE").write_text(NOTICE)

    n_int8 = sum(v.numel() for v in weights.values() if v.dtype == torch.int8)
    n_other = sum(v.numel() for v in weights.values() if v.dtype != torch.int8)
    size = (args.out / "model.safetensors").stat().st_size
    print(f"{n_int8 / 1e6:.0f}M int8 + {n_other / 1e6:.1f}M fp32 values, {size / 1e9:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()

# NuNER Zero and GLiNER reruns

Files behind the NuNER and GLiNER large rows of the Open NER table on the [benchmark page](../README.md).

- `outputs/MODEL__MODE.jsonl.gz`: every CrossNER and MIT dev and test sentence with each predicted span
  (`[start, end, type, score]`), predicted at threshold 0.1 so any higher threshold can be applied when scoring. Modes:
  `names` (type names), `desc_list` ("name: description" strings), and for GLiNER large also `desc_dict`, the label
  dict our harness passed (gliner 0.2.29 uses its descriptions; same scores as `desc_list`).
- `scores.json`: mean F1 per dev threshold, the dev-chosen threshold, and test at that threshold and at 0.5.
- `scripts/`: `run_gliner_family.py` (gliner 0.2.29, one NVIDIA A10, 2026-10-29), `build_rows.py` and
  `score_gliner_family.py`, which read the suite through our evaluation harness (not in this repository; pass it with
  `--harness` and `--suite`). `score_gliner_family.py` reproduces `scores.json`.

Model revisions: {"numind/NuNER_Zero": "c90187673f464518dca09f41689184ed6976242c", "numind/NuNER_Zero-span": "063b4dec1aaf626058e7c071657b15d52abffca0", "gliner-community/gliner_large-v2.5": "3d6d1760be1c591069f85f207fced9214df8b15f"}

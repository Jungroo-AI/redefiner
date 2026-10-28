---
title: ReDefiNER
emoji: 📏
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
fullWidth: true
header: mini
pinned: false
license: apache-2.0
short_description: Edit a rule, see the entity extraction change
models:
- jungroo-ai/redefiner-0.8b
---

# ReDefiNER demo

A 0.8B model for entity extraction and span labeling, guided by the rulebook you write. Edit a rule, run it, and see
the output change, with no retraining.

This page shows the demo app, which runs on CPU on our own server. The app's code is in
[`space/`](https://github.com/jungroo-ai/redefiner/tree/main/space) of the
[GitHub repo](https://github.com/jungroo-ai/redefiner); the weights are at
[jungroo-ai/redefiner-0.8b](https://huggingface.co/jungroo-ai/redefiner-0.8b).

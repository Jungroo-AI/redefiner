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
short_description: Edit a plain-English rule and see the extraction change
models:
- jungroo-ai/redefiner-0.8b
---

# ReDefiNER demo

**Pull out the entities you define, using a plain-English rulebook, on your own CPU.**

A 0.8B model that extracts and labels exact spans using your plain-English rulebook. Edit a rule, run it, and see the output change, with
no retraining.

This page shows the demo app, which runs on CPU on our own server. The app's code is in
[`space/`](https://github.com/jungroo-ai/redefiner/tree/main/space) of the
[GitHub repo](https://github.com/jungroo-ai/redefiner); the weights are at
[jungroo-ai/redefiner-0.8b](https://huggingface.co/jungroo-ai/redefiner-0.8b).

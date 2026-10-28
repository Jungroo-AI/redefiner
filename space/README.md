# ReDefiNER demo

ReDefiNER is a 0.8B generalist model for rule-defined named entity recognition: you describe the entity types and the
rules in a plain-English rulebook, and it finds and labels the spans. Define what to extract. In plain English.

- **Edit a rule:** a support ticket, a job application and a clinical note written for the demo, and three benchmark
  test sentences, each with its full rulebook. Edit any rule, as a form or as text, and run it again.
- **Your text:** any rulebook (or a template) on any text; find the spans, or label spans you mark.
- **Benchmarks:** one real test example per dataset: gold, GoLLIE's guidelines, and the guidelines plus our rules.

Model: [jungroo-ai/redefiner-0.8b](https://huggingface.co/jungroo-ai/redefiner-0.8b). Runs on CPU.

## Run it

From a checkout with the package installed, against a local weights folder or the Hugging Face repo:

```bash
cd space && MODEL=../weights python app.py
```

As a container, with the weights from `weights/` inside the image (build from the repo root):

```bash
docker build --platform linux/amd64 -f space/Dockerfile -t redefiner-demo .
docker run --rm -p 7860:7860 -e OMP_NUM_THREADS=2 redefiner-demo
```

The public demo runs this image on an Azure VM, and its Hugging Face Space page loads it from there.

Settings (environment variables): `MODEL` (weights folder or Hugging Face repo), `QUEUE_SIZE` (runs allowed to wait,
default 40), `OMP_NUM_THREADS` (set it to the container's vCPU count).

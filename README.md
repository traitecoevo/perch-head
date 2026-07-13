# perch-head

Train and run custom classifier heads on [Google Perch](https://www.kaggle.com/models/google/bird-vocalization-classifier)
embeddings, for bioacoustic species recognition beyond Perch's own fixed vocabulary/output.

A custom head trained on Perch's 1536-d penultimate embedding meaningfully outperforms an
equivalent head trained on a BirdNET embedding on held-out labeled soundscape evaluations —
see `docs/design_plan.md` for the experiment record. This repo packages that result into a
reusable training + inference pipeline.

**Deployment framing:** this is a **server-side / retrospective** pipeline (full compute,
scores saved audio, expects human review) — not a real-time/edge/on-device recognizer. See
`docs/inference.md`.

Standalone — no BirdNET-Analyzer dependency. The Perch checkpoint is fetched via
`kagglehub` on first use and audio windowing is self-contained (`perch_head/audio.py`).

## Quickstart

```bash
pip install -e .
# First run downloads the Perch v2 checkpoint via kagglehub (cached after that);
# set PERCH_MODEL_PATH to point at an existing local copy instead if you have one.

# 1. Extract Perch embeddings from your clip library (one-time, cached)
python scripts/extract_embeddings.py \
  --library /path/to/clip_library \
  --species-list configs/species/smithslake_present.txt \
  --out train_caches/embeddings.npz --n-distractors -1 --all-windows --cap 350

# 2. Train a head on the cache
python scripts/train_head.py \
  --npz train_caches/embeddings.npz --out-dir /path/to/recognizers \
  --name myhead --recipe a --dropout 0.4

# 3. Run inference on new audio
python scripts/predict.py \
  --head /path/to/recognizers/myhead.npz --audio /path/to/audio --out predictions.csv
```

See `docs/training.md` and `docs/inference.md` for the full flag reference and the
reasoning behind the recommended defaults.

## Repo layout

```
perch_head/           importable library: embed.py (Perch checkpoint + embedding),
                       audio.py (self-contained loading/windowing), inference.py (head + scoring)
scripts/               CLIs: extract_embeddings.py, train_head.py, predict.py
configs/species/       example species lists (extraction input)
docs/                  training.md, inference.md, design_plan.md (design + experiment record)
```

## Relationship to sibling repos

- **BirdNET-Analyzer** — no runtime dependency. This project originally reused
  BirdNET-Analyzer's local Perch checkpoint path and audio I/O; both were replaced
  (2026-07-14) with a direct `kagglehub` fetch and a self-contained `perch_head/audio.py`
  so the repo works for anyone without that fork installed.
- **soundscape-eval** — the multi-model comparison harness this project's result was
  originally prototyped inside (`docs/design_plan.md` §1–3). It still owns model-comparison
  evaluation (labeled-soundscape scoring, metrics, figures) and has its own small,
  independent copy of the Perch-embedding helper for that purpose — this repo does not
  depend on it, and it does not depend on this repo. Training and inference for new heads
  belong here; comparing a trained head against other recognizers on a labeled soundscape
  belongs there.

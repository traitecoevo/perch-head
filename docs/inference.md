# Running inference with a trained perch-head

`scripts/predict.py` scores new audio against a trained head (from `docs/training.md`) and
writes a CSV of `(file, window, species, score)` rows.

## Prerequisites

Same as training — this package installed (`pip install -e .`); see `docs/training.md`
for the `kagglehub` checkpoint fetch / `PERCH_MODEL_PATH` override.

At inference time nothing here needs Keras/TensorFlow *training* machinery — the head is a
pure-numpy forward pass (`perch_head/inference.py::Head.predict_embeddings`); the only
model actually run is the frozen Perch checkpoint that produces the 1536-d embedding.

## Usage

```bash
.venv/bin/python scripts/predict.py \
  --head /path/to/recognizers/myhead.npz \
  --audio /path/to/audio_or_folder \
  --out predictions.csv \
  --threshold 0.1
```

`--audio` accepts a single file or a folder (walked recursively; `.wav/.flac/.mp3/.ogg`).
Each file is windowed the same way training clips were — 32 kHz, 5 s windows, the final
short window zero-padded and end-aligned, no peak-normalization — so scores are consistent
with what the head was trained on.

Output CSV columns: `file, start_s, end_s, scientific, common, score`.

Key flags:

| Flag | Effect |
|---|---|
| `--threshold` | minimum score to keep a row (default 0.1). The head's sigmoid output is **not calibrated to any particular precision/recall operating point** — tune this against your own labeled data, or treat it as a coarse pre-filter and sort by `score` for review. |
| `--present-only` | drop distractor/negative-only classes from the output, keeping only the species flagged `is_present` at training time (your actual target list). |

## Deployment framing

This pipeline is **server-side / retrospective**, not real-time / edge. It scores saved
audio with full compute and expects a human to review the output — see
`docs/design_plan.md` §5 for why (Perch is heavier than the edge-deployed BirdNET-backbone
recognizer, and this head is never exported to a `.tflite`/on-device format). If you need
real-time on-device detection, that's a different recognizer/pipeline entirely.

Because review absorbs some imprecision, retrospective use can run at a **much lower
operating point** (favor recall over precision) than an edge deployment's fixed
false-positive budget would allow — set `--threshold` accordingly, or emit everything above
a low floor and let a reviewer triage by score.

## Programmatic use

For anything beyond a flat CSV (e.g. feeding scores into your own review tool or evaluation
harness), use `perch_head.inference` directly:

```python
from perch_head.inference import Head, predict_file

head = Head.load("/path/to/recognizers/myhead.npz")
rows = predict_file(head, "/path/to/clip.wav")   # list of dicts, one per (window, class)
```

`Head.predict_embeddings(emb)` also accepts pre-computed Perch embeddings directly (e.g. if
you're already extracting them for another purpose), skipping the audio-windowing step.

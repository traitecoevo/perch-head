# perch-head — Project Notes for Claude

## What this repo is

Training + inference pipeline for custom classifier heads on Google Perch embeddings.
Promoted out of `soundscape-eval` on 2026-07-14 once it stopped being a single-person
research prototype and became an ongoing project with multiple users — see
`docs/design_plan.md` for the full experiment record that led here.

**This repo owns training and inference. It does NOT own model-comparison evaluation** —
that stays in the sibling `soundscape-eval` repo (labeled-soundscape scoring, metrics,
figures, the multi-model comparison harness). See "Relationship to sibling repos" in
`README.md` for the exact boundary and why `soundscape_eval/perch_embed.py` +
`PerchHeadCustom` in `soundscape_eval/adapters.py` are a deliberate small independent copy
of this repo's embedding/scoring logic rather than a shared dependency.

## Repo structure

```
perch_head/           embed.py (Perch embedding helper), inference.py (Head class + scoring)
scripts/               extract_embeddings.py, train_head.py, predict.py — the three CLIs
configs/species/       example species-list inputs for extraction
docs/                  training.md, inference.md, design_plan.md
```

## Environment

Needs `birdnet_analyzer` installed editable from the BirdNET-Analyzer fork
(`https://github.com/wcornwell/BirdNET-Analyzer`) — it provides the frozen Perch v2
checkpoint (`cfg.PERCH_V2_MODEL_PATH`) and the audio I/O (`audio.open_audio_file`,
`audio.split_signal`) used for windowing. This repo does not vendor a checkpoint or its own
audio-loading code.

```bash
pip install -e /path/to/BirdNET-Analyzer
pip install -e ".[dev]"   # dev extra pulls in ruff==0.14.0
```

Linting: `ruff check .` before committing (matches the pin used by the sibling repos).

## Workflow

1. `scripts/extract_embeddings.py` — one-time, cached, walks a clip library through Perch.
2. `scripts/train_head.py` — cheap, repeatable on the cache.
3. `scripts/predict.py` — score new audio with a trained head.

Full flag reference and the reasoning behind current defaults (recipe A, dropout 0.4, full
vocabulary): `docs/training.md` and `docs/inference.md`. Why those defaults, and what was
tried and rejected (recipe B at scale, L2-norm's ranking/recall trade-off): `docs/design_plan.md`.

## Design decisions worth knowing before changing extraction/training defaults

- **Zero-pad short clips to Perch's 5 s window, end-aligned, not normalized** — verified
  empirically not to shift embeddings away from real field windows of the same species
  (`docs/design_plan.md` §3). Don't switch to tiling/repeat-padding or peak-normalization
  without re-running that kind of check; it would introduce a train/score asymmetry.
- **Non-event/helper folders get an all-zero label row and no output neuron**, matching the
  convention used by the BirdNET-Analyzer fork's own non-event handling — keeps them as pure
  hard negatives without cluttering the reportable class list.
- **Recipe A (focal loss + upsampling) over recipe B (plain BCE) once past a few hundred
  classes** — B collapses at scale (`docs/design_plan.md` §6), a real degenerate failure
  mode, not noise.
- **`dropout=0.4` over the original `0.25`** for production — won recall at a fixed
  false-positive budget on two independent validation soundscapes; `l2norm` trades this same
  metric away for better AUPRC/ranking, which is why it's *not* the recommended default.
- **This pipeline is server-side/retrospective, never exported to `.tflite`.** Real-time/
  edge detection is a different, larger, deliberately deferred project — don't assume this
  head needs to become edge-deployable when extending it.

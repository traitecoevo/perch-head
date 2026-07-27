# perch-head — Project Notes for Claude

## Where this repo sits — read before building anything new

**One library, two backbones.** `reallybig` is backbone-agnostic labeled audio living
outside every repo (OneDrive `call_library/`). Only the *embedding caches* and *venvs*
fork per backbone — the library itself never does. This repo's cache is
`train_caches/perch_reallybig_*.npz`; it is **not** a separate training library.

**This repo owns:** Perch-backbone head training + inference.
**It does NOT own:** model-comparison evaluation (→ `soundscape-eval`), curation of
`reallybig` (→ `Training_library_assembly_pipeline`).

Full ownership table, seams, venvs, shared data: **`~/Documents/ecoacoustics/ECOACOUSTICS.md`**. Check it
before writing a downloader, clip-mover, plot, or metric — it exists to stop double-builds.

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
tests/                 offline unit tests (windowing invariants + numpy forward pass)
```

## Environment

No BirdNET-Analyzer dependency — standalone. `perch_head/embed.py` fetches the Perch v2
checkpoint via `kagglehub` (`KAGGLE_MODEL_HANDLE = "google/bird-vocalization-classifier/
tensorFlow2/perch_v2_cpu"`, the same handle BirdNET-Analyzer itself uses in
`birdnet_analyzer/utils.py::ensure_perch_exists()` — verified against that source, not
guessed), cached locally after the first call. Set `PERCH_MODEL_PATH` to point at an
existing local checkpoint instead of downloading. `perch_head/audio.py` closely follows
BirdNET-Analyzer's `audio.open_audio_file`/`split_signal` (librosa, `kaiser_fast`
resampling, trailing zero-pad, no peak-normalization) — kept behavior-consistent on purpose,
since the already-trained heads were extracted through that path (see `docs/design_plan.md`
§3); don't change the resampler or padding scheme without re-validating. One deliberate
deviation: a clip whose entire signal is under `minlen_s` (default 1 s) yields no windows
here, where BirdNET would keep a single padded chunk — dropping too-short clips is intended,
not a bug.

Own dedicated venv (unlike `soundscape-eval`, which piggybacks on BirdNET-Analyzer's — not
applicable here since this repo has no BirdNET-Analyzer dependency to share):

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # dev extra pulls in ruff==0.14.0
```

Verified 2026-07-14: clean `pip install -e ".[dev]"` from PyPI only (no manual/local wheel
steps) on Python 3.11 / macOS arm64 resolves tensorflow 2.21 + keras 3 + librosa/soundfile/
resampy/kagglehub, and the full extract → train → predict pipeline runs end-to-end through
it. Tiny (~3e-6) embedding drift vs. the TF 2.20 venv used during development is expected
TF-version float noise, not a bug — negligible next to head training's own regularization.

Linting: `.venv/bin/ruff check .` before committing (matches the pin used by the sibling
repos). Tests: `.venv/bin/python -m pytest tests/` — all offline (no Perch checkpoint or
network needed), covering `split_signal`'s windowing invariants and the pure-numpy head
forward pass.

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
- **`dropout=0.4` over the original `0.25`** for production, and now the `train_head.py`
  default — won recall at a fixed false-positive budget on two independent validation
  soundscapes; `l2norm` trades this same metric away for better AUPRC/ranking, which is why
  it's *not* the recommended default.
- **This pipeline is server-side/retrospective, never exported to `.tflite`.** Real-time/
  edge detection is a different, larger, deliberately deferred project — don't assume this
  head needs to become edge-deployable when extending it.
- **No BirdNET-Analyzer dependency, on purpose** — this project originally borrowed
  BirdNET-Analyzer's local Perch checkpoint path and its `audio.py` for windowing, but that
  only worked on machines that happened to have that fork installed with a manually-placed
  checkpoint (BirdNET-Analyzer doesn't commit or auto-fetch it either in a way other users
  could rely on without also having run that fork's own setup). Fetching directly via
  `kagglehub` and owning the windowing code makes perch-head installable by any collaborator
  on its own.

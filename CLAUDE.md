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
                       extract_library_embeddings.py — the curation artifact, not a cache
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

Off to one side, not part of training: **`scripts/extract_library_embeddings.py`** walks the
same library but writes the *clip-level* artifact — one row per clip, keyed by filename,
every class, no labels. That is what `Training_library_assembly_pipeline/curation/` and
`birdnetEmbed` consume, because they move and plot individual clips; the training cache
above can't serve them (its rows are windows and it keeps no filenames). Its npz layout is
deliberately identical to BirdNET-Analyzer's `extract_head_embeddings.py`, so those tools
read either backbone's artifact unchanged — but the spaces are **1536-d Perch vs 2048-d
BirdNET head** and nothing may be compared across them. `runs/run_dual.sh` calls it as the
Perch half of its final embedding stage; see `runs/CLAUDE.md` § 5.

**Two extensions designed but not built (2026-07-29)**, both cheap because the backbone is
frozen and already-extracted vectors never go stale — full reasoning in `runs/CLAUDE.md`
§ "Open threads":

1. *Incremental extraction.* Adding one class re-embeds the whole library (~3 h) to
   recompute vectors that come back identical. Appending only the changed folders is sound,
   but the cache has no per-row provenance — a non-event row is all-zero, so its source
   folder is unrecoverable from `Y`. Add a `source` array to the cache first.
2. *`derive_head_embeddings.py`.* The trained head's 2048-d hidden layer is computed inside
   `Head.predict_embeddings()` and discarded. `relu(X @ W1 + b1)` over an existing library
   artifact plus the head npz reproduces it in seconds, no audio decoded (honour `l2norm`).
   Recognizer-specific and stale on every retrain, so it complements the backbone artifact
   rather than replacing it.

Full flag reference and the reasoning behind current defaults (recipe A, dropout 0.4, full
vocabulary): `docs/training.md` and `docs/inference.md`. Why those defaults, and what was
tried and rejected (recipe B at scale, L2-norm's ranking/recall trade-off): `docs/design_plan.md`.

## Design decisions worth knowing before changing extraction/training defaults

> **Dual-arm run contract:** `training_runs/CLAUDE.md` (the `run_dual.sh` orchestrator) sets
> the defaults for a paired BirdNET+Perch run — `extract_embeddings.py --species-list` names
> **all** library classes, derived from the library at launch rather than read from a
> checked-in file (a stale list silently shrinks `is_present`, and with it the headline
> AUPRC, without changing the model). The runner now passes `--n-distractors 0`, not `-1`:
> with the full vocabulary the distractor pool is empty so the two are identical, but `0`
> cannot silently resurrect classes the vocabulary left out. `Environment_*`/
> `Homo sapiens_*`/`Noise` stay non-events (the all-zero-row behavior below), and `--cap 0`
> keeps the head seeing the whole library like the BirdNET arm does.
> See `~/Documents/ecoacoustics/training_runs/CLAUDE.md` § "Run defaults"; keep the two in sync.

### `--upsample-ratio` is anchored to the LARGEST class — growing one class re-balances all

`_upsample_repeat` sets `min_samples = int(max(counts) * ratio)`, so the per-class floor is a
fraction of whatever the biggest class happens to be. **Adding clips to one class therefore
changes the training recipe for every other class**, and nothing says so beyond the
`min/class` number in the recipe line.

Measured 2026-08-23 while testing whether library volume helps: restoring archived clips to
two classes (Brown Songlark 400→672, Australian Magpie 400→809 clips) made Australian Magpie
the largest class at 648 train rows, against Noisy Miner's previous 336. At the default
`--upsample-ratio 0.4` that lifted the floor from **134 to 259**, and all 433 classes were
upsampled harder: **77,507 → 118,254 train rows, +53%**. The intended experiment was "more
data for two classes"; what ran was that plus a 53% larger, differently balanced training set
for everything.

**It is invisible on the headline** — macro in-vocab AUPRC moved z = −0.9, i.e. nothing. What
caught it was a negative control: 6 of 43 species whose data had not changed moved beyond
|z|>3 of the seed-noise band, bidirectionally, which no data error in the changed classes
could produce.

It recurred immediately. The seven-class Arm C restore (2026-08-23) made White-winged
Fairywren the largest at **726** train rows, where the stock `0.4` would have set the floor to
**290** — 2.2× the control's. **Do not hand-compute the ratio per arm; derive it from the cache
the fit will actually read**, which is what `out/run0-9/armC_run.sh` does:

```python
tr = d["split"] == "train"; mx = int(d["Y"][tr].sum(axis=0).max())
r = 134.5 / mx; assert int(mx * r) == 134     # 134.5 lands mid-bin, so int() is stable
```

That reproduces both known caches exactly: control `336 → 0.400298` (the stock 0.4's floor)
and Arm B `648 → 0.207562` (the 0.206790 used by hand).

**Before comparing two fits, check the recipe line's `min/class` matches.** To hold it fixed
when class sizes change, solve for the ratio: `ratio = old_floor / new_max_count` (that run
used `--upsample-ratio 0.206790` to get `int(648 × 0.206790) = 134`). This applies to *any*
intervention that changes class sizes — a sourcing batch, a re-prune, a restore — and `--cap 0`
(the dual-arm run default) makes it more likely, not less, since nothing bounds the largest
class.

### The seed-only noise band is the wrong yardstick for a data change

Refits differing only by `--seed` hold the data **and** the train/val split fixed, so their
spread is the smallest possible source of variation — measured 2026-08-23 at macro in-vocab
SD **0.0025** over three fits, per-species SD 0.002–0.025 on a 51-species soundscape.

That band does not cover a data change. Even after the upsampling confound above was
corrected, 3 of 43 species whose data was untouched still moved beyond |z|>3 of it, and 13
beyond |z|>2 (≈2 expected). The shared 2048-unit hidden layer re-organises whenever any
class's data changes, so unrelated classes move.

`scripts/make_null_cache.py` builds the band that *is* valid: it resamples the changed
classes' rows from a larger pool **at constant count**, holding class sizes (and so the
upsampling floor) and the train/val proportion identical to the control, so the only
difference is which clips those classes contribute. Refit, score, and the spread of the
unchanged classes is the honest yardstick. It is also the null a composition experiment must
beat — a diversity-optimised re-prune has to outperform a random re-draw at the same count,
not merely differ from the control.

### Site-scoped heads — `train_head.py --species-list` (added 2026-08-08)

**The site filter belongs on the CACHE, not on extraction.** Embedding the library is
site-independent (frozen backbone) and costs ~3 h, so it is paid once and every site
recognizer is a column slice of `Y`:

```bash
scripts/train_head.py --npz shared_cache.npz --name wilddeserts0-1-ph \
    --species-list configs/species/wilddeserts.txt        # --unlisted non_event (default)
```

A window whose only positive class was dropped becomes an all-zero row — which is already
how this pipeline encodes a non-event — so `non_event` mode is the slice alone and the
unlisted species keep suppressing the retained ones as hard negatives. `--unlisted drop`
discards those rows instead. Rows *already* all-zero are the helper hard negatives and are
kept in both modes, which is why the drop mask is built from rows positive **before** the
slice. `is_present` is set True for every retained class: with a site list, every remaining
column is one the site wants detected.

The species list and mode are recorded in the head npz (`species_list`,
`unlisted_handling`) — neither the cache nor the head previously recorded anything about
how its vocabulary was chosen.

⚠️ `extract_embeddings.py --species-list` is a **different** thing: it sets the cache's
vocabulary (present vs distractor). For site scoping, leave it at the full library and
scope in `train_head.py`. The "demote unlisted species to non-events at extraction time"
mode still does not exist there — only the cache path supports it.

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
- **`AUDIO_EXT` is defined once, in `perch_head/audio.py`, and every walker imports it.**
  It used to be copied into five files, all reading `(".wav", ".flac", ".mp3", ".ogg")`.
  iNaturalist serves most of its audio as `audio/mp4`, so all five skipped `.m4a` — and
  skipped it *silently*, because a missing extension is not an error anywhere: the file is
  simply never opened and the run reports fewer recordings than the audio dir holds. It
  lives next to `open_audio_file` because it is a claim about what that loader can decode,
  and **it must never be narrower than the loader**: librosa 0.11.0 reads `.m4a`/`.mpga`
  here at the correct duration (55 files checked against ffprobe, 0 mismatches), so the
  old list was an allowlist, not a format limit.
  **Do not pre-convert iNat audio any more** — a converted `.wav` left beside its `.m4a`
  makes the scorer see one recording twice and cut duplicate clips from it.
- **No BirdNET-Analyzer dependency, on purpose** — this project originally borrowed
  BirdNET-Analyzer's local Perch checkpoint path and its `audio.py` for windowing, but that
  only worked on machines that happened to have that fork installed with a manually-placed
  checkpoint (BirdNET-Analyzer doesn't commit or auto-fetch it either in a way other users
  could rely on without also having run that fork's own setup). Fetching directly via
  `kagglehub` and owning the windowing code makes perch-head installable by any collaborator
  on its own.

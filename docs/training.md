# Training a perch-head recognizer

Two steps: extract Perch embeddings from your clip library once (cached to an npz), then
train a head on the cache (cheap — repeatable in minutes once extraction is done).

See `docs/design_plan.md` for the experiments that produced the recommended defaults below.

## Prerequisites

- A Python environment with this package installed (`pip install -e .`) plus
  `birdnet_analyzer` installed editable from the BirdNET-Analyzer fork (provides the frozen
  Perch v2 checkpoint and the audio I/O used for windowing):
  ```bash
  pip install -e /path/to/BirdNET-Analyzer
  pip install -e .
  ```
- A clip library: one folder per class, folder name `Genus species_Common Name`
  (e.g. `Pachycephala pectoralis_Golden Whistler`). Non-event/helper folders (background
  noise, wind, human activity, ...) are supported — see `--nonevent-prefixes` below.
- A **species list**: the scientific names you actually want to detect (the "present"
  classes your evaluation/deployment cares about). Plain text, one binomial per line, `#`
  comments allowed. `configs/species/smithslake_present.txt` is a worked example.

## Step 1 — extract embeddings

```bash
python scripts/extract_embeddings.py \
  --library /path/to/your/clip_library \
  --species-list configs/species/smithslake_present.txt \
  --out train_caches/perch_embeddings.npz \
  --n-distractors -1 \
  --all-windows \
  --cap 350
```

Key flags:

| Flag | Effect |
|---|---|
| `--species-list` | present/target species (required) — see format above |
| `--label-vocab` | optional: fix the full class vocabulary + column order (e.g. to match another recognizer's exact class set for a side-by-side comparison). Default: derived from `--library`'s own subfolders. |
| `--n-distractors` | how many non-present classes to include as negatives/distractors; `-1` = every class in the vocabulary. §6 of `docs/design_plan.md`: full vocabulary meaningfully improved recall-at-fixed-FP-budget over a small subset, at the cost of a much longer one-time extraction. |
| `--distractor-select` | `common` (largest folders first, well-populated) or `random` |
| `--cap` | max clips per class folder |
| `--all-windows` | use every 5 s window of each clip, not just the first. Combine with `--cap` to bound extraction time. |
| `--nonevent-prefixes` / `--nonevent-exact` | folder-name patterns treated as non-events: all-zero label rows, no output neuron, pure hard negatives. Default matches the reference library's convention (`Environment_`, `Homo sapiens_`, `Noise`) — override for a different library's naming. |

This is the expensive, one-time step — full-vocabulary/all-windows extraction over a large
library takes hours, not minutes. The output npz is fully reusable for repeated training
runs (different recipes, hyperparameter sweeps) without re-running Perch.

## Step 2 — train a head

Recommended production configuration (recipe A + the dropout found best in the
`docs/design_plan.md` §6 sweep):

```bash
python scripts/train_head.py \
  --npz train_caches/perch_embeddings.npz \
  --out-dir /path/to/recognizers \
  --name myhead \
  --recipe a \
  --dropout 0.4
```

Outputs `<out-dir>/myhead.npz` (the trained head — `W1,b1,W2,b2` + labels + metadata) and
`<out-dir>/myhead_Labels.txt` (one class per line). Both are self-contained: nothing at
inference time needs Keras/TensorFlow beyond the frozen Perch checkpoint (see
`docs/inference.md`).

Key flags:

| Flag | Effect |
|---|---|
| `--recipe` | `a` (focal loss + upsampling, recommended) or `b` (plain BCE, no upsampling). `docs/design_plan.md` §6: recipe B collapses once the vocabulary grows past a few hundred classes — use `a` unless you have a specific reason to compare. |
| `--dropout` | 0.25 is the original matched-to-BirdNET-recipe default; 0.4 won recall-at-fixed-FP-budget on both validation soundscapes in the sweep — recommended for production. |
| `--l2norm` | L2-normalizes embeddings before training (and must then be applied at inference — the npz records the flag so `perch_head/inference.py` does this automatically). Improved ranking/AUPRC but *cost* recall-at-fixed-FP-budget in the sweep — prefer `dropout=0.4` unless ranking quality specifically is your objective. |
| `--gamma`, `--alpha`, `--upsample-ratio` | focal-loss and upsampling knobs (recipe `a` only) |
| `--hidden` | hidden layer width (default 2048; a smaller hidden layer trailed the baseline on every metric in the sweep) |

To compare both recipes on the same cache in one run, use `--recipe both` — this writes
`<name>a.npz` and `<name>b.npz`.

## Validation metric

Training prints **present-species validation AUPRC** (mean average precision over the
columns flagged `is_present`, i.e. your species list) — the threshold-free headline metric.
This is *not* the same as recall at a real deployment operating point; see
`docs/design_plan.md` for why AUPRC and recall-at-fixed-FP-budget can diverge, and use
`docs/inference.md` + your own evaluation harness to check the latter before deploying.

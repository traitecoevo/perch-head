# Perch-Embedding Custom Head — Design & Results

**Status:** productionized (server-side/retrospective tier) · **Originated:** soundscape-eval,
2026-07-08 · **Promoted to its own repo:** 2026-07-14, since the project now has multiple
users beyond the original prototype.

This document is the design rationale plus the experimental record that motivated the
current defaults in `docs/training.md`. It was written incrementally as a prototype inside
`soundscape-eval` before the project graduated to this repo — some phrasing below still
reflects that origin (e.g. "the eval," "the fork") which now means "the `soundscape-eval`
comparison harness" and "the BirdNET-Analyzer fork" respectively, both sibling repos.

## 1. The one question this answers

Does a custom classifier head trained on **Perch** embeddings clear the ~0.18 AUPRC
ceiling that every BirdNET-backbone recognizer variant hit on a labeled evaluation
soundscape (Smiths Lake), moving toward Perch's own native head's ~0.28 — **with head
recipe and training data held constant, so the only variable is the backbone**?

Context: BirdNET-backbone head-recipe tuning was exhausted before this question was asked
(the last real gain was a focal-loss gamma tweak). Perch's own native head scored notably
higher AUPRC on the same species set, so the frozen BirdNET V2.4 embedding was the suspected
ceiling, not the head architecture.

## 2. Why it's tractable

- Perch's saved_model serving signature exposes **`embedding: (None, 1536)`** directly,
  alongside `label: (None, 14795)`. The penultimate representation is pullable, not just
  the class head — see `perch_head/embed.py`.
- The head is `Dense(hidden, relu, dropout) → Dense(n_classes)` — two matmuls. Scoring is
  pure numpy at inference time (no Keras/TF graph beyond the frozen Perch checkpoint):
  `sigmoid((relu(X·W1 + b1))·W2 + b2)` — see `perch_head/inference.py`.

## 3. The 3 s-clip / 5 s-window problem — resolved via zero-padding

Perch requires exactly 5 s @ 32 kHz (160000 samples). Most training-library clips are ~3 s
(cut for a 3 s BirdNET window), so most clips need ~2 s of padding to fit Perch's window.

**Resolution: zero-pad, end-aligned, no normalization** — `audio.split_signal(sig, 32000,
5.0, 0, SIG_MINLEN)` with `SIG_MINLEN=1.0`, i.e. a 3 s clip becomes 3 s audio + 2 s zeros.
This is the same path real soundscape windows are scored through, so training and scoring
embeddings are directly comparable.

**Verified empirically (padding probe, 2026-07-08).** Zero-padding a 3 s training clip to
5 s does not shift its Perch embedding away from real 5 s field windows of the same
species: same-species train/field cosine **0.221 vs 0.118 cross-species** (margin +0.103),
positive for every probed species. Reusing this feed path for extraction is safe; a
source-recording-window fallback (re-extracting a true 5 s window from a clip's original
source audio) was considered but never needed. (This probe's runnable script stayed in
`soundscape-eval`, since it depends on that repo's labeled-soundscape ground-truth loader;
the finding is recorded here since it's foundational to this repo's extraction step.)

## 4. Architecture — three pieces (now this repo)

1. **Extraction** (`scripts/extract_embeddings.py`) — walks a species list (present +
   distractors + non-event helper folders) through the clip library, feeds each clip
   through Perch via the §3 path, caches `(X, Y)` embeddings + multi-hot labels.
2. **Head training** (`scripts/train_head.py`) — a small Keras head on the cached
   embeddings, saved to a self-contained npz (no framework dependency at score time).
3. **Inference** (`perch_head/inference.py`, `scripts/predict.py`) — load the npz, run new
   audio through the same Perch + numpy-forward-pass path, emit per-window scores.

The original design also specified a 4th piece — a `PerchHeadCustom` adapter registered
into `soundscape-eval`'s model-comparison harness — which stays in `soundscape-eval` (it's
specific to that repo's evaluation framework, not to training or standalone inference).

## 5. Results (RUN, 2026-07-08 — 176-class prototype)

**PAYOFF — the Perch backbone clears the BirdNET ceiling.** Smiths Lake, 26 present
species, macro AUPRC: BirdNET-backbone recognizer **0.175** → Perch head, pelican-matched
focal recipe (**A**) **0.267** → Perch head, BCE recipe (**B**) 0.250 → Perch's own native
14795-class head 0.283. The **+0.093 / ~53%** jump is a *pure backbone effect* (head recipe
+ training data held constant). Focal loss (recipe A) beat plain BCE (recipe B) on the Perch
backbone too, same as on the BirdNET backbone.

**Deployment read (the metric that matters): the custom head is the only deployable
option.** At a 1/hr false-positive budget, Perch's native head collapses to
**recall@60s = 0.0** (uncalibrated across 14795 classes — can't be thresholded to budget).
The custom head (recipe A) gives **0.25 recall @ 0.95 precision**. The native head's AUPRC
edge is threshold-free ranking only.

**Ceiling moved, didn't vanish — residual is a faint-call floor, not vocab coverage.**
Recipe-A false negatives: only 5% never fired at all; 66% were *faint* (fired well below
threshold); 29% just-below-threshold. This signature survived the backbone swap, meaning it
isn't specific to BirdNET's embedding — it's a detector-sensitivity/threshold-calibration
problem, not something more training data or a bigger vocabulary fixes.

**Two-tier deployment decision.** Edge (real-time, on-device) stays on the BirdNET-backbone
`.tflite` recognizer — this Perch head is never exported to `.tflite` and is not intended
for real-time/edge use. This project is the **server-side/retrospective** tier: full
compute, human-in-the-loop review of saved audio, where a lower operating point (recall
favored over the edge's strict FP budget) is affordable.

## 6. Results (RUN 2, 2026-07-14 — full-vocabulary scale-up + cross-soundscape validation)

- **Scaled extraction**: all classes in the library vocabulary (not just a subset) + all
  windows per clip, higher per-class cap. One-time, cached (~3 hours for ~78k rows on the
  reference library/hardware).
- **Recipe B collapses at scale.** Recipe A (focal + upsampling) held (in-library
  validation AUPRC ~0.99, same as the smaller prototype). Recipe B (plain BCE, no
  upsampling) collapsed (~0.97 → ~0.61 in-library AUPRC) — a real degenerate collapse:
  unweighted-mean BCE dilutes the per-class gradient as class count grows, with no
  upsampling to compensate. Confirmed on the field evaluation too, where scaled recipe B
  scored *worse* than the original BirdNET-backbone baseline. **Recipe B is not recommended
  above a few hundred classes — recipe A only past that point.**
  - Full-vocabulary AUPRC on Smiths Lake was roughly flat vs the 176-class prototype, but
    **recall at a fixed 1-false-positive-per-hour budget improved ~22% relative**, and false
    negatives dropped substantially — the deployment-relevant metric, not the ranking metric,
    is where full vocabulary earns its extraction cost.
- **Meta-parameter sweep** (single-variable isolation vs the full-vocab recipe-A baseline):
  `l2norm`, several focal-loss gammas, a higher dropout, a higher upsample ratio, a smaller
  hidden layer. Two standouts on different axes: **`l2norm`** gave the best AUPRC/ranking
  but the **worst** recall-at-fixed-FP-budget (sharper ranking, worse at the deployed
  operating point). **`dropout=0.4`** gave the best recall-at-fixed-FP-budget. A higher
  upsample ratio and a smaller hidden layer trailed the baseline on every metric — a real,
  if modest, sign both hurt.
- **Cross-soundscape validation (Fowlers Gap, arid zone — a totally different habitat and
  species set from Smiths Lake's coastal habitat).** The backbone effect replicated:
  BirdNET-backbone AUPRC 0.205 → Perch-head AUPRC 0.26–0.27 (+28% relative), with false
  positives collapsing even more cleanly than at Smiths Lake. The sweep tiebreaker also
  replicated: **`dropout=0.4` won recall-at-fixed-FP-budget on both soundscapes** — small
  margins each time, but the same direction twice is signal. `l2norm` showed the same
  ranking-vs-recall trade-off on both soundscapes too.
- **Decision: `dropout=0.4` (otherwise identical to the full-vocab recipe-A baseline —
  focal α0.25 γ2 + upsample repeat@0.4) is the recommended production configuration.** This
  is now the best-known server-side/retrospective recognizer configuration for this
  pipeline; the invocation is documented in `docs/training.md`.
- **Open, not pursued further**: the faint-call floor (§5) is still the residual ceiling —
  no meta-parameter targets it, since it's a threshold-calibration problem, not a
  head-recipe one.

## 7. Fairness controls (for anyone re-running a backbone comparison)

- Same class vocabulary / same clip library / same evaluation ground truth across backbones
  being compared, so species coverage is identical and AUPRC is directly comparable.
- Matched head recipe (hidden size, dropout, focal loss, upsampling, epochs) isolates the
  backbone as the only variable.
- Window-length mismatch (Perch's 5 s vs a 3 s-native recognizer) is a scoring-time
  windowing concern, separate from the extraction-time padding issue in §3.

## 8. Risks & unknowns carried forward

1. **Faint-call floor** — the dominant residual failure mode; needs threshold/per-species
   calibration or detector-sensitivity work, not more data or a bigger head.
2. **Negative/distractor class selection** shapes the AUPRC/leak trade-off — document
   whatever set is used for reproducibility (see `--species-list` / `--label-vocab` in
   `scripts/extract_embeddings.py`).
3. **Extraction compute** scales with vocabulary size × clips-per-class × windows-per-clip;
   full-vocabulary/all-windows runs are hours, not minutes, but are one-time and cached.

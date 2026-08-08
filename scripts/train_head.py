"""Train a standalone classifier head on cached Perch embeddings (from extract_embeddings.py)
-> a self-contained npz, owning its own save format (no framework/fork dependency at score
time — see perch_head/inference.py).

Two recipes are supported on the SAME cache:
  A (default) — focal loss (alpha, gamma) + repeat-upsampling. Good default for imbalanced
                libraries (a handful of clips for rare species, hundreds for common ones).
  B            — plain sigmoid binary cross-entropy, no upsampling — Perch's own classifier
                trainer's recipe (`optax.sigmoid_binary_cross_entropy`, multi-hot, no in-loop
                balancing). NOTE: at large vocabularies (100s of classes) this recipe can
                collapse — unweighted-mean BCE dilutes the per-class gradient as class count
                grows with no upsampling to compensate. Recipe A is the recommended default;
                see docs/training.md for the finding that motivated this.

Architecture: Dropout(d) -> Dense(hidden, relu, he_normal, L2 1e-5) -> Dropout(d) ->
Dense(n_classes, glorot, L2 1e-5) -> sigmoid. Non-event (all-zero) rows are hard negatives
with no output neuron; they are never upsampled (upsampling loops over label columns only).

Saves W1,b1,W2,b2 (pure-numpy scoring: sigmoid((relu(X.W1+b1)).W2+b2)) + labels + is_present
+ l2norm flag + recipe to `<out-dir>/<name>.npz`, plus a `<name>_Labels.txt` (one class per
line) for tools that want the vocabulary without loading the npz.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _sci_prefix(folder: str) -> str:
    """Library folder is 'Genus species_Common'; scientific part is before the first '_'."""
    return folder.split("_", 1)[0].strip().lower()


def _load_species_list(path: str) -> list[str]:
    """One binomial per line, '#' comments and blanks skipped, lowercased.

    Same format as extract_embeddings.py --species-list and BirdNET-Analyzer's
    train --species_list, so one file can scope both arms of a dual run.
    """
    with open(path) as f:
        return [ln.strip().lower() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def scope_to_species_list(data: dict, species_list: str, unlisted: str) -> dict:
    """Narrow a full-library cache to one site's species list -- no re-extraction.

    Embedding the library is site-independent and costs ~3 h, so it is paid once and
    every site recognizer is then a column slice of Y. A window whose only positive
    class was dropped becomes an all-zero row, which is exactly how this pipeline
    already encodes a non-event: so "non_event" mode is the slice alone, and the
    unlisted species keep suppressing the retained ones as hard negatives.
    "drop" discards those rows instead.

    Rows already all-zero in the cache are the helper/non-event hard negatives
    (Environment_*, Homo sapiens_*, Noise). They are kept in both modes, which is why
    the drop mask is built from rows that were positive *before* the slice.
    """
    wanted = _load_species_list(species_list)
    if not wanted:
        raise SystemExit(f"Species list is empty: {species_list}")

    labels = [str(x) for x in data["labels"]]
    keep = [i for i, lbl in enumerate(labels) if _sci_prefix(lbl) in wanted]

    missing = sorted(set(wanted) - {_sci_prefix(lbl) for lbl in labels})
    if missing:
        print(f"WARNING: {len(missing)} species-list entries have no column in the cache: "
              f"{missing}")
    if not keep:
        raise SystemExit(
            f"Species list matched none of the cache's {len(labels)} classes. "
            "Wrong list, or a cache built from a different library?")

    Y = data["Y"]
    was_positive = Y.any(axis=1)
    Y = Y[:, keep]
    rows = np.ones(len(Y), dtype=bool)
    if unlisted == "drop":
        rows = ~(was_positive & ~Y.any(axis=1))

    scoped = dict(data)
    scoped["X"] = data["X"][rows]
    scoped["Y"] = Y[rows]
    scoped["labels"] = np.array([labels[i] for i in keep])
    # Every retained class is one the site wants detected, so all of them are present.
    scoped["is_present"] = np.ones(len(keep), dtype=bool)
    scoped["split"] = data["split"][rows]
    print(f"Species list {os.path.basename(species_list)}: {len(keep)} of {len(labels)} "
          f"classes kept ({unlisted} mode), {int(rows.sum())} of {len(rows)} rows.")
    return scoped


def _build(input_size: int, n_classes: int, hidden: int, dropout: float):
    import keras
    reg = keras.regularizers.l2(1e-5)
    m = keras.Sequential()
    m.add(keras.layers.InputLayer(shape=(input_size,)))
    if dropout > 0:
        m.add(keras.layers.Dropout(dropout))
    m.add(keras.layers.Dense(hidden, activation="relu", kernel_regularizer=reg, kernel_initializer="he_normal"))
    if dropout > 0:
        m.add(keras.layers.Dropout(dropout))
    m.add(keras.layers.Dense(n_classes, kernel_regularizer=reg, kernel_initializer="glorot_uniform"))
    m.add(keras.layers.Activation("sigmoid"))
    return m


def _focal_loss(gamma: float, alpha: float, eps: float = 1e-7):
    import tensorflow as tf

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_sum(alpha_factor * focal_weight * ce, axis=-1)

    return loss


def _upsample_repeat(x: np.ndarray, y: np.ndarray, ratio: float, seed: int):
    """Bring every label column up to int(max_count*ratio) rows by sampling-with-replacement
    from that column's positives. All-zero (non-event) rows have no column, so they are never
    duplicated — pure negatives."""
    rng = np.random.default_rng(seed)
    counts = y.sum(axis=0)
    min_samples = int(np.max(counts) * ratio)
    add_x, add_y = [], []
    for i in range(y.shape[1]):
        pos = np.where(y[:, i] == 1)[0]
        if len(pos) == 0:
            continue
        need = min_samples - int(counts[i])
        if need <= 0:
            continue
        pick = rng.choice(pos, size=need, replace=True)
        add_x.append(x[pick])
        add_y.append(y[pick])
    if add_x:
        x = np.concatenate([x, *add_x], axis=0)
        y = np.concatenate([y, *add_y], axis=0)
    perm = rng.permutation(len(x))
    return x[perm], y[perm], min_samples


def _lr_schedule(base_lr: float, epochs: int):
    warmup = min(5, int(epochs * 0.1))

    def sched(epoch, lr):
        if warmup and epoch < warmup:
            return base_lr * (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, (epochs - warmup))
        return base_lr * (0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2)

    return sched


def _present_auprc(y_true: np.ndarray, y_prob: np.ndarray, is_present: np.ndarray) -> float:
    """Mean average-precision over the present-species columns — the headline objective.
    Threshold-free."""
    from sklearn.metrics import average_precision_score
    cols = np.where(is_present)[0]
    aps = []
    for c in cols:
        if y_true[:, c].sum() > 0:
            aps.append(average_precision_score(y_true[:, c], y_prob[:, c]))
    return float(np.mean(aps)) if aps else float("nan")


def train_one(recipe: str, name: str, data: dict, args) -> dict:
    import keras

    X, Y = data["X"].astype("float32"), data["Y"].astype("float32")
    labels, is_present, split = data["labels"], data["is_present"], data["split"]

    if args.l2norm:
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)

    tr, va = split == "train", split == "val"
    xtr, ytr, xva, yva = X[tr], Y[tr], X[va], Y[va]

    if recipe == "a":
        xtr, ytr, min_s = _upsample_repeat(xtr, ytr, args.upsample_ratio, args.seed)
        loss = _focal_loss(args.gamma, args.alpha)
        print(f"[{recipe}] focal(a={args.alpha}, g={args.gamma}) + upsample repeat@"
              f"{args.upsample_ratio} -> {len(xtr)} train rows (min/class {min_s})")
    else:
        loss = keras.losses.BinaryCrossentropy()
        print(f"[{recipe}] sigmoid BCE, no upsampling -> {len(xtr)} train rows")

    keras.utils.set_random_seed(args.seed)
    model = _build(X.shape[1], Y.shape[1], args.hidden, args.dropout)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss=loss)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                      patience=min(10, max(5, args.epochs // 10)),
                                      min_delta=1e-3, restore_best_weights=True, verbose=1),
        keras.callbacks.LearningRateScheduler(_lr_schedule(args.lr, args.epochs)),
    ]
    model.fit(xtr, ytr, validation_data=(xva, yva), epochs=args.epochs,
              batch_size=args.batch, callbacks=callbacks, verbose=2)

    yprob = model.predict(xva, batch_size=args.batch, verbose=0)
    auprc = _present_auprc((yva > 0).astype(int), yprob, is_present)
    print(f"[{recipe}] present-species val AUPRC (mean AP over {int(is_present.sum())} cols): {auprc:.4f}")

    dense = [ly for ly in model.layers if ly.__class__.__name__ == "Dense"]
    (W1, b1), (W2, b2) = dense[0].get_weights(), dense[1].get_weights()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{name}.npz")
    # Which species list scoped this head, and what happened to the classes it left out.
    # Without these the head's vocabulary is only recoverable by diffing its labels
    # against the library it came from.
    np.savez_compressed(out, W1=W1, b1=b1, W2=W2, b2=b2, labels=labels,
                        is_present=is_present, l2norm=np.array(args.l2norm),
                        recipe=np.array(recipe), val_auprc=np.array(auprc),
                        species_list=np.array(args.species_list or ""),
                        unlisted_handling=np.array(
                            args.unlisted if args.species_list else ""))
    labels_txt = os.path.join(args.out_dir, f"{name}_Labels.txt")
    with open(labels_txt, "w") as f:
        f.write("\n".join(str(x) for x in labels) + "\n")
    print(f"[{recipe}] saved {out}  (W1 {W1.shape}, W2 {W2.shape}) + {os.path.basename(labels_txt)}\n")
    return {"recipe": recipe, "auprc": auprc, "path": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="cache produced by extract_embeddings.py.")
    ap.add_argument("--out-dir", required=True, help="directory to write <name>.npz + <name>_Labels.txt.")
    ap.add_argument("--name", required=True,
                    help="output basename. With --recipe both, 'a'/'b' is appended (<name>a, <name>b).")
    ap.add_argument("--recipe", choices=("a", "b", "both"), default="a")
    ap.add_argument("--l2norm", action="store_true")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.4,
                    help="dropout rate (default 0.4 — the production recommendation; see "
                         "docs/design_plan.md §6. The original BirdNET-matched default was 0.25).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--upsample-ratio", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--species-list", default="",
                    help="restrict this head to a subset of the cache's classes -- a "
                         "site-scoped recognizer off a shared full-library cache. One "
                         "scientific binomial per line, '#' comments ignored. Same file "
                         "format the BirdNET arm's train --species_list takes.")
    ap.add_argument("--unlisted", choices=("non_event", "drop"), default="non_event",
                    help="what to do with cache classes absent from --species-list: "
                         "'non_event' keeps their windows as all-zero hard negatives so "
                         "they still suppress the retained species; 'drop' discards them. "
                         "Ignored without --species-list.")
    args = ap.parse_args()

    data = dict(np.load(args.npz))
    print(f"Loaded {args.npz}: X {data['X'].shape}  Y {data['Y'].shape}  "
          f"present {int(data['is_present'].sum())}  l2norm={args.l2norm}")

    if args.species_list:
        data = scope_to_species_list(data, args.species_list, args.unlisted)

    recipes = ["a", "b"] if args.recipe == "both" else [args.recipe]
    results = [train_one(r, args.name if len(recipes) == 1 else f"{args.name}{r}", data, args) for r in recipes]
    print("=== summary ===")
    for r in results:
        print(f"  {os.path.basename(r['path'])}: present val AUPRC {r['auprc']:.4f}")


if __name__ == "__main__":
    main()

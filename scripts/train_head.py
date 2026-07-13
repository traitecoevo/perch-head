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
    np.savez_compressed(out, W1=W1, b1=b1, W2=W2, b2=b2, labels=labels,
                        is_present=is_present, l2norm=np.array(args.l2norm),
                        recipe=np.array(recipe), val_auprc=np.array(auprc))
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
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--upsample-ratio", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = dict(np.load(args.npz, allow_pickle=True))
    print(f"Loaded {args.npz}: X {data['X'].shape}  Y {data['Y'].shape}  "
          f"present {int(data['is_present'].sum())}  l2norm={args.l2norm}")

    recipes = ["a", "b"] if args.recipe == "both" else [args.recipe]
    results = [train_one(r, args.name if len(recipes) == 1 else f"{args.name}{r}", data, args) for r in recipes]
    print("=== summary ===")
    for r in results:
        print(f"  {os.path.basename(r['path'])}: present val AUPRC {r['auprc']:.4f}")


if __name__ == "__main__":
    main()

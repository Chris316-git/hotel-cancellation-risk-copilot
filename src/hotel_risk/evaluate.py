"""Phase 2 evaluation: temporal vs random split, baseline vs LightGBM, plus confusion matrices.

Run: python -m hotel_risk.evaluate data/raw/hotel_bookings.csv
Writes reports/phase2_metrics.json and reports/confusion_matrix.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, roc_auc_score)
from sklearn.model_selection import train_test_split

from .data import TARGET, booking_time_features, clean, load_raw, temporal_split
from .models import make_baseline, make_gbm

# A confusion matrix needs a hard yes/no decision, so we pick a cut-off. 0.5 is the neutral default;
# Phase 3 shows how to choose a threshold more carefully (and how it drifts over time).
THRESHOLD = 0.5


def score(y, p) -> dict:
    """Threshold-free quality of the predicted probabilities."""
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "base_rate": round(float(np.mean(y)), 4),
    }


def confusion_report(y, p, threshold: float = THRESHOLD) -> dict:
    """Counts of right/wrong calls once probabilities are turned into 'will cancel' flags.

    tn = predicted stays, stayed      fp = predicted cancel, but stayed (false alarm)
    fn = predicted stays, cancelled   tp = predicted cancel, cancelled
    """
    y = np.asarray(y)
    flag = np.asarray(p) >= threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, flag, labels=[0, 1]).ravel())
    safe = lambda a, b: round(a / b, 4) if b else 0.0  # noqa: E731
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    return {
        "threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision,                     # of the bookings we flag, how many really cancel
        "recall": recall,                           # of the real cancellations, how many we catch
        "specificity": safe(tn, tn + fp),           # of the bookings that stay, how many we leave alone
        "f1": safe(2 * precision * recall, precision + recall),
        "accuracy": safe(tp + tn, tn + fp + fn + tp),
    }


def plot_confusion_matrices(reports: dict, path: str | Path) -> None:
    """Side-by-side confusion matrices. Colour = share of each actual class (rows sum to 100%)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(reports), figsize=(4.6 * len(reports), 4.4), squeeze=False)
    for ax, (name, r) in zip(axes[0], reports.items()):
        counts = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        share = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)  # one hue, light -> dark = small -> large share
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{counts[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center",
                        fontsize=12, color="white" if share[i, j] > 0.6 else "#0b0b0b")
        ax.set_xticks([0, 1], ["Predicted:\nstays", "Predicted:\ncancels"])
        ax.set_yticks([0, 1], ["Actual:\nstayed", "Actual:\ncancelled"])
        ax.set_title(f"{name}\nprecision {r['precision']:.0%}, recall {r['recall']:.0%}", fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(f"Test period (2017-03 onward), flag a booking when P(cancel) >= {THRESHOLD}", fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def run(path: str) -> dict:
    df = clean(load_raw(path))
    out: dict = {}
    models = {"logistic_baseline": make_baseline, "lightgbm": make_gbm}

    train, test = temporal_split(df)
    confusions = {}
    for name, factory in models.items():
        m = factory().fit(booking_time_features(train), train[TARGET])
        p = m.predict_proba(booking_time_features(test))[:, 1]
        out[f"temporal/{name}"] = score(test[TARGET], p)
        out[f"confusion_at_{THRESHOLD}/{name}"] = confusions[name] = confusion_report(test[TARGET], p)
    plot_confusion_matrices({"Logistic baseline": confusions["logistic_baseline"],
                             "LightGBM": confusions["lightgbm"]}, "reports/confusion_matrix.png")

    # Random split on the same data: shows how optimistic non-temporal evaluation is.
    tr, te = train_test_split(df, test_size=len(test) / len(df), random_state=0, stratify=df[TARGET])
    for name, factory in models.items():
        m = factory().fit(booking_time_features(tr), tr[TARGET])
        out[f"random/{name}"] = score(te[TARGET], m.predict_proba(booking_time_features(te))[:, 1])
    return out


if __name__ == "__main__":
    res = run(sys.argv[1] if len(sys.argv) > 1 else "data/raw/hotel_bookings.csv")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase2_metrics.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:34s}", v)

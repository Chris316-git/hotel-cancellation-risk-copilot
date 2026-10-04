"""What drives the model, and what does that mean in plain words?

Three complementary views, from model-agnostic to model-specific:

1. Permutation importance (scikit-learn): shuffle one feature on the held-out test period and see how much
   ROC-AUC drops. A big drop means the model leans on that feature. It says HOW MUCH a feature matters,
   not in which direction.
2. SHAP (optional, needs `pip install shap`): for each booking, how much each feature pushed the predicted
   cancel probability up or down. Gives direction as well as size.
3. Plain cancel rates by feature value on the test period (no model involved): a sanity check that the
   direction the model learned matches what the raw data shows.

Run: python -m hotel_risk.explain data/raw/hotel_bookings.csv [--no-shap]
Writes reports/feature_importance.json / .png and (if shap is installed) reports/shap_summary.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .calibrate import split3
from .data import NUMERIC, TARGET, booking_time_features, clean, load_raw
from .models import make_gbm

# Lead time is easier to read in buckets than as a raw number of days.
LEAD_BINS = [-1, 7, 30, 90, 180, 10_000]
LEAD_LABELS = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"]


def cancel_rate_table(test: pd.DataFrame, col: str, top: int = 6, min_n: int = 300) -> list[dict]:
    """Cancel rate per value of one feature on the test period (largest groups first)."""
    key = pd.cut(test[col], LEAD_BINS, labels=LEAD_LABELS) if col == "lead_time" else test[col]
    g = test.groupby(key, observed=True)[TARGET].agg(bookings="size", cancel_rate="mean")
    g = g[g["bookings"] >= min_n]
    if col != "lead_time":  # buckets keep their natural order, categories sort by volume
        g = g.sort_values("bookings", ascending=False).head(top)
    return [{"group": str(idx), "bookings": int(r.bookings), "cancel_rate": round(float(r.cancel_rate), 3)}
            for idx, r in g.iterrows()]


def describe_spread(feature: str, rows: list[dict]) -> str:
    """One sentence contrasting the highest and lowest cancel-rate groups."""
    hi = max(rows, key=lambda r: r["cancel_rate"])
    lo = min(rows, key=lambda r: r["cancel_rate"])
    return (f"{feature}: '{hi['group']}' cancels {hi['cancel_rate']:.0%} of the time "
            f"vs {lo['cancel_rate']:.0%} for '{lo['group']}'.")


def shap_direction(Xt: np.ndarray, sv: np.ndarray, names: list[str], top: int = 12) -> dict:
    """Rank features by mean |SHAP| and, for numeric ones, say which way larger values push the prediction.

    Spearman correlation between a feature's value and its SHAP value: positive means that larger
    values push towards 'cancel', negative towards 'stays'.
    """
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)[:top]
    numeric = set(NUMERIC)
    ranking, direction = [], {}
    for k in order:
        ranking.append({"feature": names[k], "mean_abs_shap": round(float(mean_abs[k]), 4)})
        if names[k] in numeric:
            r = pd.Series(Xt[:, k]).corr(pd.Series(sv[:, k]), method="spearman")
            if pd.notna(r):
                direction[names[k]] = {
                    "spearman": round(float(r), 3),
                    "reading": "larger values push towards cancel" if r > 0 else "larger values push towards stays"}
    return {"ranking": ranking, "numeric_direction": direction}


def run(path: str, use_shap: bool = True) -> dict:
    df = clean(load_raw(path))
    train, _, test = split3(df)
    X_train, X_test = booking_time_features(train), booking_time_features(test)
    model = make_gbm().fit(X_train, train[TARGET])

    # 1) Permutation importance on the test period (5 shuffles per feature to see the noise level).
    res = permutation_importance(model, X_test, test[TARGET], scoring="roc_auc",
                                 n_repeats=5, random_state=0, n_jobs=-1)
    ranking = sorted(zip(X_test.columns, res.importances_mean, res.importances_std), key=lambda t: -t[1])
    out = {"metric": "drop in test ROC-AUC when the feature is shuffled (mean, std over 5 repeats)",
           "features": [{"feature": f, "auc_drop": round(float(m), 4), "std": round(float(s), 4)}
                        for f, m, s in ranking]}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path("reports").mkdir(exist_ok=True)
    top = ranking[:12][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([f for f, _, _ in top], [m for _, m, _ in top], xerr=[s for _, _, s in top], color="#2a78d6")
    ax.set(xlabel="Drop in test ROC-AUC when shuffled", title="What the model relies on (top 12)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("reports/feature_importance.png", dpi=150)

    # 3) Raw cancel rates for the most important features: does the data agree with the model?
    focus = [r["feature"] for r in out["features"][:6]]
    tables = {c: cancel_rate_table(test, c) for c in focus if c in test.columns}
    out["cancel_rate_by_feature"] = tables
    out["insights"] = [describe_spread(c, rows) for c, rows in tables.items() if rows]

    # 2) SHAP: direction and size of each feature's push on individual predictions.
    if use_shap:
        try:
            import shap
            sample = X_test.sample(2000, random_state=0)
            pre, clf = model.named_steps["pre"], model.named_steps["clf"]
            Xt = pre.transform(sample)
            names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
            sv = shap.TreeExplainer(clf).shap_values(Xt)
            if isinstance(sv, list):  # older shap versions return one array per class
                sv = sv[1]
            if getattr(sv, "ndim", 2) == 3:  # (rows, features, classes) in some versions
                sv = sv[:, :, 1]
            plt.figure()
            shap.summary_plot(sv, Xt, feature_names=names, show=False)
            plt.tight_layout()
            plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
            out["shap"] = {"plot": "reports/shap_summary.png", **shap_direction(np.asarray(Xt), np.asarray(sv), names)}
        except ImportError:
            out["shap"] = "skipped: `pip install shap` to enable"
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = run(args[0] if args else "data/raw/hotel_bookings.csv", use_shap="--no-shap" not in sys.argv)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/feature_importance.json").write_text(json.dumps(res, indent=2))
    print("Permutation importance (AUC drop when shuffled):")
    for r in res["features"][:10]:
        print(f"  {r['feature']:32s} {r['auc_drop']:.4f} +/- {r['std']:.4f}")
    print("\nWhat the raw data says:")
    for line in res["insights"]:
        print("  -", line)
    if isinstance(res.get("shap"), dict):
        print("\nSHAP, direction for numeric features:")
        for f, d in res["shap"]["numeric_direction"].items():
            print(f"  {f:32s} {d['reading']} (rho={d['spearman']})")

"""Feature D: calibration and human-review flagging.

* Reliability diagram + ECE + Brier score for every model (test set).
* Confidence = max(p, 1-p). For each model we pick, on the VALIDATION set, the
  smallest confidence threshold t such that the predictions with confidence >= t
  reach a target accuracy (default 95 %) while at least 50 % of reviews are kept; if
  that is impossible, t keeps the most confident 50 % (reported as target not reached). Predictions below t are flagged for
  human review. We then report, on the TEST set, the % flagged and the accuracy
  on the remaining (auto-accepted) predictions vs. overall accuracy.

Run:  python -m src.calibration
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config as C
from src.evaluate import MODELS, available, load_predictions

TARGET_ACC = 0.95
GRID = np.round(np.arange(0.50, 1.0, 0.005), 3)


def reliability(y, p, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Bin by predicted P(positive); return per-bin table and ECE."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf, frac = p[mask].mean(), y[mask].mean()
        ece += mask.mean() * abs(conf - frac)
        rows.append({"bin": b, "mean_pred": conf, "frac_pos": frac, "count": int(mask.sum())})
    return pd.DataFrame(rows), float(ece)


MIN_COVERAGE = 0.5  # fallback: never flag more than half of the reviews


def choose_threshold(y, p, target=TARGET_ACC) -> tuple[float, bool]:
    """Smallest threshold whose auto-accepted val predictions reach `target` accuracy
    (keeping at least MIN_COVERAGE of reviews). If the target is unreachable, fall back to
    the threshold that keeps exactly MIN_COVERAGE (the most confident half) and say so."""
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5).astype(int) == y
    for t in GRID:
        keep = conf >= t
        if keep.mean() < MIN_COVERAGE:
            break
        if correct[keep].mean() >= target:
            return float(t), True
    return float(np.quantile(conf, 1 - MIN_COVERAGE)), False


def main() -> None:
    val = load_predictions("val")
    test = load_predictions("test")
    models = [m for m in available(test) if f"prob_{m}" in val]
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, len(models), figsize=(4.3 * len(models), 7.5),
                             gridspec_kw={"height_ratios": [3, 1]})
    cal_rows, flag_rows, curve_rows = [], [], []
    for j, m in enumerate(models):
        y, p = test["label"].values, test[f"prob_{m}"].values
        tbl, ece = reliability(y, p)
        brier = float(np.mean((p - y) ** 2))
        cal_rows.append({"model": m, "model_name": C.MODEL_NAMES[m], "ECE": ece, "Brier": brier})
        tbl.assign(model=m).to_csv(C.TABLE_DIR / f"reliability_bins_{m}.csv", index=False)

        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(tbl.mean_pred, tbl.frac_pos, "o-", lw=2, label=f"model (ECE={ece:.3f})")
        ax.set(title=C.MODEL_NAMES[m], xlabel="mean predicted P(positive)", ylabel="observed fraction positive",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left", fontsize=8)
        axes[1, j].bar(tbl.mean_pred, tbl["count"], width=0.08)
        axes[1, j].set(xlabel="predicted P(positive)", ylabel="count", xlim=(0, 1))

        # ---- flagging ----
        t, reached = choose_threshold(val["label"].values, val[f"prob_{m}"].values)
        conf = np.maximum(p, 1 - p)
        correct = (p >= 0.5).astype(int) == y
        keep = conf >= t
        flag_rows.append({
            "model": m, "model_name": C.MODEL_NAMES[m], "threshold_from_val": t,
            "target_val_accuracy": TARGET_ACC, "target_reached_on_val": reached,
            "test_pct_flagged": 100 * (1 - keep.mean()),
            "test_acc_all": correct.mean(),
            "test_acc_unflagged": correct[keep].mean(),
            "test_acc_flagged": correct[~keep].mean() if (~keep).any() else np.nan,
            "acc_gain_pts": 100 * (correct[keep].mean() - correct.mean()),
            "pct_flagged_short": 100 * (1 - keep[test.length_2way == "short"].mean()),
            "pct_flagged_long": 100 * (1 - keep[test.length_2way == "long"].mean()),
        })
        for tt in GRID:
            k = conf >= tt
            curve_rows.append({"model": m, "threshold": tt, "coverage": k.mean(),
                               "accuracy_unflagged": correct[k].mean() if k.any() else np.nan})
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "reliability_diagrams.png", dpi=130)
    plt.close(fig)

    cal = pd.DataFrame(cal_rows)
    flags = pd.DataFrame(flag_rows)
    curve = pd.DataFrame(curve_rows)
    cal.to_csv(C.TABLE_DIR / "calibration_summary.csv", index=False)
    flags.to_csv(C.TABLE_DIR / "human_review_flagging.csv", index=False)
    curve.to_csv(C.TABLE_DIR / "coverage_accuracy_curve.csv", index=False)
    print("\nCalibration (test):\n", cal.round(4).to_string(index=False))
    print("\nHuman-review flagging (threshold chosen on val):\n", flags.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, sub in curve.groupby("model", sort=False):
        ax.plot(100 * sub.coverage, sub.accuracy_unflagged, lw=2, label=C.MODEL_NAMES[m])
        f = flags[flags.model == m].iloc[0]
        ax.scatter(100 - f.test_pct_flagged, f.test_acc_unflagged, s=40, zorder=3)
    ax.axhline(TARGET_ACC, color="grey", ls="--", lw=1, label=f"target {TARGET_ACC:.0%} (set on val)")
    ax.set(xlabel="% of test reviews auto-accepted (not flagged)", ylabel="accuracy on auto-accepted",
           title="Coverage vs accuracy when low-confidence predictions are flagged")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "coverage_vs_accuracy.png", dpi=130)
    plt.close(fig)
    (C.RESULTS_DIR / "flag_thresholds.json").write_text(
        json.dumps({r["model"]: r["threshold_from_val"] for r in flag_rows}, indent=2))


if __name__ == "__main__":
    main()

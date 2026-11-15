"""Stage 4: test-set evaluation for every model, overall and per length group.

Outputs (results/tables, results/figures):
* metrics_overall.csv, metrics_by_length_group.csv, short_long_gap.csv
* truncation_effect.csv (accuracy on truncated vs non-truncated long reviews)
* confusion_matrices.png, f1_by_quartile.png, accuracy_by_quartile.png,
  f1_short_vs_long.png

Run:  python -m src.evaluate
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src import config as C

PRED_DIR = C.RESULTS_DIR / "predictions"
MODELS = ["tfidf_lr", "rnn", "lstm", "chunked_lstm"]


def load_predictions(split: str = "test", models=MODELS, extra=()) -> pd.DataFrame:
    """Processed split + one prob_<model> / pred_<model> column per model."""
    df = pd.read_csv(C.TEST_CSV if split == "test" else C.VAL_CSV)
    for m in list(models) + list(extra):
        f = PRED_DIR / f"{m}_{split}.csv"
        if f.exists():
            df[f"prob_{m}"] = pd.read_csv(f)["prob"].values
            df[f"pred_{m}"] = (df[f"prob_{m}"] >= 0.5).astype(int)
    return df


def metrics(y, yhat) -> dict:
    return {
        "n": len(y),
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
    }


def available(df: pd.DataFrame, models=MODELS):
    return [m for m in models if f"pred_{m}" in df]


def evaluate(df: pd.DataFrame, models) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall, grouped = [], []
    for m in models:
        y, yhat = df["label"], df[f"pred_{m}"]
        overall.append({"model": m, "model_name": C.MODEL_NAMES.get(m, m), **metrics(y, yhat)})
        for gcol, order in (("length_2way", ["short", "long"]), ("length_quartile", ["Q1", "Q2", "Q3", "Q4"])):
            for g in order:
                sub = df[df[gcol] == g]
                grouped.append({"model": m, "model_name": C.MODEL_NAMES.get(m, m),
                                "grouping": "two_way" if gcol == "length_2way" else "quartile",
                                "group": g, "pct_pos": sub.label.mean() * 100,
                                **metrics(sub["label"], sub[f"pred_{m}"])})
    return pd.DataFrame(overall), pd.DataFrame(grouped)


def plot_confusions(df, models) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8))
    for ax, m in zip(np.atleast_1d(axes), models):
        cm = confusion_matrix(df["label"], df[f"pred_{m}"])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
        ax.set(title=C.MODEL_NAMES[m], xlabel="predicted", ylabel="true")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "confusion_matrices.png", dpi=130)
    plt.close(fig)


def plot_by_group(grouped, metric: str, grouping: str, fname: str, th: dict | None = None) -> None:
    g = grouped[grouped.grouping == grouping]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m, sub in g.groupby("model", sort=False):
        ax.plot(sub["group"], sub[metric], "o-", lw=2, label=C.MODEL_NAMES[m])
        for x, v in zip(sub["group"], sub[metric]):
            ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    labels = None
    if grouping == "quartile" and th:
        labels = [f"Q1\n(<= {th['q25']:.0f})", f"Q2\n({th['q25']:.0f}-{th['median']:.0f}]",
                  f"Q3\n({th['median']:.0f}-{th['q75']:.0f}]", f"Q4\n(> {th['q75']:.0f})"]
        ax.set_xticks(range(4), labels)
    ax.set(title=f"Test {metric.upper() if metric=='f1' else metric} by review length ({grouping})",
           xlabel="length group (tokens, thresholds from train)", ylabel=metric)
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / fname, dpi=130)
    plt.close(fig)


def main() -> None:
    from src.data_prep import load_thresholds

    th = load_thresholds()
    df = load_predictions("test")
    models = available(df)
    overall, grouped = evaluate(df, models)
    overall.to_csv(C.TABLE_DIR / "metrics_overall.csv", index=False)
    grouped.to_csv(C.TABLE_DIR / "metrics_by_length_group.csv", index=False)
    print("\nOverall test metrics:\n", overall.round(4).to_string(index=False))
    print("\nBy length group:\n", grouped.round(4).to_string(index=False))

    # short-vs-long gap
    tw = grouped[grouped.grouping == "two_way"].pivot(index="model", columns="group", values=["accuracy", "f1"])
    gap = pd.DataFrame({
        "model": tw.index,
        "acc_short": tw[("accuracy", "short")].values, "acc_long": tw[("accuracy", "long")].values,
        "f1_short": tw[("f1", "short")].values, "f1_long": tw[("f1", "long")].values,
    })
    gap["acc_gap_short_minus_long"] = gap.acc_short - gap.acc_long
    gap["f1_gap_short_minus_long"] = gap.f1_short - gap.f1_long
    gap = gap.set_index("model").loc[models].reset_index()
    gap.to_csv(C.TABLE_DIR / "short_long_gap.csv", index=False)
    print("\nShort vs long gap:\n", gap.round(4).to_string(index=False))

    # truncation effect: within the long group, truncated vs not truncated
    rows = []
    long_df = df[df.length_2way == "long"]
    for m in models:
        for trunc, sub in long_df.groupby("truncated"):
            rows.append({"model": m, "subset": "truncated (> max_len)" if trunc else "long, fits in max_len",
                         **metrics(sub.label, sub[f"pred_{m}"])})
    trunc_tbl = pd.DataFrame(rows)
    trunc_tbl.to_csv(C.TABLE_DIR / "truncation_effect.csv", index=False)
    print(f"\nTruncation effect (max_len={th['max_len_p95']}):\n", trunc_tbl.round(4).to_string(index=False))

    # extra: chunked vs hybrid vs plain LSTM comparison
    dfx = load_predictions("test", extra=("hybrid_lstm",))
    if "pred_hybrid_lstm" in dfx:
        C.MODEL_NAMES.setdefault("hybrid_lstm", "Hybrid BiLSTM (chunk only if > max_len)")
        o2, g2 = evaluate(dfx, [m for m in ("lstm", "chunked_lstm", "hybrid_lstm") if f"pred_{m}" in dfx])
        comp = pd.concat([o2.assign(grouping="overall", group="all"), g2])
        comp.to_csv(C.TABLE_DIR / "chunked_variants_comparison.csv", index=False)

    sns.set_theme(style="whitegrid")
    plot_confusions(df, models)
    plot_by_group(grouped, "f1", "quartile", "f1_by_quartile.png", th)
    plot_by_group(grouped, "accuracy", "quartile", "accuracy_by_quartile.png", th)
    plot_by_group(grouped, "f1", "two_way", "f1_short_vs_long.png")


if __name__ == "__main__":
    main()

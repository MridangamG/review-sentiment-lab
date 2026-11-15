"""Stage 1 + 2: data preparation, dataset statistics and length groups.

Steps
-----
1. Load the real Amazon Polarity parquet files (no synthetic data).
2. Combine title + content, drop empty rows and duplicates.
3. Balanced random sample: 80k from train file (-> 85/15 stratified train/val),
   20k from test file (held-out test). Test texts that also occur in the train
   file are removed first to avoid leakage.
4. Clean + tokenise, store token_count.
5. Length groups with thresholds computed on TRAIN only:
   two-way (short <= median < long) and quartile bins Q1..Q4.
6. Save processed CSVs, stats tables and figures.

Run:  python -m src.data_prep
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sstats
from sklearn.model_selection import train_test_split

from src import config as C
from src.text_utils import clean_text, combine_title_content, tokenize

THRESH_FILE = C.RESULTS_DIR / "length_thresholds.json"


# ----------------------------------------------------------------------------
# Loading / sampling
# ----------------------------------------------------------------------------
def _load_raw(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["text"] = [combine_title_content(t, c) for t, c in zip(df["title"], df["content"])]
    n0 = len(df)
    df = df[df["text"].str.strip().str.len() > 0]
    n_empty = n0 - len(df)
    n1 = len(df)
    df = df.drop_duplicates(subset="text")
    print(f"  {path.name}: {n0:,} rows | removed {n_empty:,} empty, {n1 - len(df):,} duplicates")
    return df[["label", "text"]].reset_index(drop=True)


def _balanced_sample(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Sample n_total/2 per class *after* cleaning (oversample a buffer, then trim)."""
    per_class = n_total // 2
    parts = []
    for lbl in (0, 1):
        pool = df[df["label"] == lbl].sample(n=int(per_class * 1.05), random_state=seed)
        pool = _clean_frame(pool)
        parts.append(pool.head(per_class))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    assert len(out) == n_total, "not enough rows left after cleaning"
    return out


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["text"].map(clean_text)
    df["tokens"] = df["clean_text"].map(lambda s: " ".join(tokenize(s)))
    df["token_count"] = df["tokens"].str.split().str.len().fillna(0).astype(int)
    df = df[df["token_count"] > 0]                     # empty after cleaning
    df = df.drop_duplicates(subset="clean_text")        # duplicates after cleaning
    return df


# ----------------------------------------------------------------------------
# Length groups
# ----------------------------------------------------------------------------
def compute_thresholds(train: pd.DataFrame) -> dict:
    tc = train["token_count"]
    th = {
        "q25": float(tc.quantile(0.25)),
        "median": float(tc.quantile(0.50)),
        "q75": float(tc.quantile(0.75)),
        "max_len_p95": int(np.ceil(tc.quantile(C.MAX_LEN_PERCENTILE / 100))),
    }
    return th


def assign_groups(df: pd.DataFrame, th: dict) -> pd.DataFrame:
    tc = df["token_count"]
    df["length_2way"] = np.where(tc <= th["median"], "short", "long")
    df["length_quartile"] = np.select(
        [tc <= th["q25"], tc <= th["median"], tc <= th["q75"]], ["Q1", "Q2", "Q3"], default="Q4"
    )
    df["truncated"] = tc > th["max_len_p95"]
    return df


def load_thresholds() -> dict:
    return json.loads(THRESH_FILE.read_text())


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------
def dataset_stats(splits: dict, th: dict) -> None:
    rows, group_rows = [], []
    for name, df in splits.items():
        tc = df["token_count"]
        rows.append({
            "split": name, "n": len(df),
            "n_neg": int((df.label == 0).sum()), "n_pos": int((df.label == 1).sum()),
            "pct_pos": round(100 * df.label.mean(), 2),
            "mean_tokens": round(tc.mean(), 2), "median_tokens": float(tc.median()),
            "std_tokens": round(tc.std(), 2), "min_tokens": int(tc.min()),
            "p5": float(tc.quantile(.05)), "p25": float(tc.quantile(.25)),
            "p75": float(tc.quantile(.75)), "p90": float(tc.quantile(.90)),
            "p95": float(tc.quantile(.95)), "p99": float(tc.quantile(.99)), "max_tokens": int(tc.max()),
            "n_truncated_at_max_len": int(df.truncated.sum()),
            "pct_truncated": round(100 * df.truncated.mean(), 2),
        })
        for gcol in ("length_2way", "length_quartile"):
            for g, sub in df.groupby(gcol):
                group_rows.append({
                    "split": name, "grouping": gcol, "group": g, "n": len(sub),
                    "pct_of_split": round(100 * len(sub) / len(df), 2),
                    "pct_pos": round(100 * sub.label.mean(), 2),
                    "mean_tokens": round(sub.token_count.mean(), 1),
                    "min_tokens": int(sub.token_count.min()), "max_tokens": int(sub.token_count.max()),
                    "n_truncated": int(sub.truncated.sum()),
                    "pct_truncated": round(100 * sub.truncated.mean(), 2),
                })
    stats = pd.DataFrame(rows)
    groups = pd.DataFrame(group_rows)
    stats.to_csv(C.TABLE_DIR / "dataset_stats.csv", index=False)
    groups.to_csv(C.TABLE_DIR / "length_group_stats.csv", index=False)
    print("\nDataset statistics:\n", stats.T.to_string(header=False))
    print("\nClass balance / truncation inside each length group:\n", groups.to_string(index=False))

    # ---- label-vs-length check (does length leak the label?) ----
    train = splits["train"]
    by_label = train.groupby("label")["token_count"].describe()
    r, p = sstats.pointbiserialr(train["label"], train["token_count"])
    u = sstats.mannwhitneyu(train.loc[train.label == 0, "token_count"],
                            train.loc[train.label == 1, "token_count"])
    check = by_label.reset_index()
    check["pointbiserial_r"] = r
    check["pointbiserial_p"] = p
    check["mannwhitney_p"] = u.pvalue
    check.to_csv(C.TABLE_DIR / "label_vs_length.csv", index=False)
    print("\nLabel vs length (train):\n", by_label.round(2).to_string())
    print(f"  point-biserial r = {r:.4f} (p={p:.2e}); Mann-Whitney p = {u.pvalue:.2e}")

    # ---- figures ----
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for lbl, name in ((0, "negative"), (1, "positive")):
        axes[0].hist(train.loc[train.label == lbl, "token_count"], bins=60, alpha=.55, label=name)
    for k, ls in (("q25", ":"), ("median", "--"), ("q75", ":"), ("max_len_p95", "-")):
        axes[0].axvline(th[k], color="k", ls=ls, lw=1)
        axes[0].text(th[k], axes[0].get_ylim()[1] * .92, f" {k}={th[k]:.0f}", fontsize=8, rotation=90, va="top")
    axes[0].set(title="Train token-count distribution by label", xlabel="tokens per review", ylabel="reviews")
    axes[0].legend()
    sns.boxplot(data=pd.concat([d.assign(split=n) for n, d in splits.items()]),
                x="split", y="token_count", hue="label", ax=axes[1], fliersize=1)
    axes[1].set(title="Token counts per split and label")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "length_histogram.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    g = groups[(groups.split == "train") & (groups.grouping == "length_quartile")]
    ax.bar(g["group"], g["pct_pos"], color=sns.color_palette()[0])
    ax.axhline(50, color="k", ls="--", lw=1)
    ax.set(title="Positive-class share per length quartile (train)", ylabel="% positive", ylim=(0, 100))
    for x, v in zip(g["group"], g["pct_pos"]):
        ax.text(x, v + 1, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "class_balance_by_quartile.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main() -> dict:
    C.set_seed()
    print("Loading raw parquet files ...")
    train_raw = _load_raw(C.RAW_TRAIN)
    test_raw = _load_raw(C.RAW_TEST)

    # leakage guard: drop test reviews whose text also appears in the train file
    before = len(test_raw)
    test_raw = test_raw[~test_raw["text"].isin(set(train_raw["text"]))]
    print(f"  removed {before - len(test_raw):,} test rows that also occur in the train file")

    print("Sampling + cleaning ...")
    pool = _balanced_sample(train_raw, C.N_TRAIN_POOL, C.SEED)
    test = _balanced_sample(test_raw, C.N_TEST, C.SEED)
    train, val = train_test_split(pool, test_size=C.VAL_FRACTION, stratify=pool["label"], random_state=C.SEED)

    th = compute_thresholds(train)                      # TRAIN ONLY
    THRESH_FILE.write_text(json.dumps(th, indent=2))
    print("Length thresholds (train):", th)

    splits = {"train": train, "val": val, "test": test}
    for name, df in splits.items():
        assign_groups(df, th)
        df.reset_index(drop=True).to_csv(C.PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"  saved {name}.csv  ({len(df):,} rows)")

    dataset_stats(splits, th)

    # demo file for the Streamlit dashboard: 500 real test reviews (title/content restored)
    demo = test.sample(n=500, random_state=C.SEED)[["text", "label"]].rename(columns={"text": "review"})
    demo.to_csv(C.DEMO_DIR / "demo_reviews.csv", index=False)
    return th


if __name__ == "__main__":
    main()

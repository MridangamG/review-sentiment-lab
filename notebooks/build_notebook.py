"""Generate notebooks/sentiment_project.ipynb (then execute it with nbconvert).

    python notebooks/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/sentiment_project.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# Sentiment Analysis of Amazon Product Reviews — and the Effect of Review Length

**Problem.** Classify product reviews as positive or negative and examine how review length affects
classification accuracy.

**Data.** Real records from Hugging Face `fancyzhx/amazon_polarity` (Zhang et al., 2015):
`train-00000-of-00004.parquet` (900k reviews) and `test-00000-of-00001.parquet` (400k reviews).
No synthetic data is used anywhere.

**Pipeline.** data prep → length groups → TF-IDF+LR / Simple RNN / BiLSTM / Chunked BiLSTM →
per-length evaluation → calibration & human-review flagging → error analysis.

All heavy lifting lives in `src/`; this notebook calls the same functions as `main.py` and shows the outputs.
Neural training is slow-ish, so by default the notebook **loads the checkpoints trained by `python main.py`**.
Set `RETRAIN = True` to retrain everything inside the notebook.
""")
code("""
import os, sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

import pandas as pd, numpy as np, torch
from IPython.display import Image, display
from src import config as C
pd.set_option("display.max_colwidth", 120); pd.set_option("display.width", 200)
C.set_seed()
RETRAIN = False
print("device:", C.get_device(), "| torch", torch.__version__)
""")

md("## 1. Data preparation\n`src/data_prep.py` — combine title+content, drop empty/duplicate rows, balanced sampling "
   "(80k train-file → 85/15 train/val, 20k test-file), cleaning (lower-case, strip HTML/URLs, keep negations and !/?), "
   "tokenisation, token counts. The script is fully seeded, so re-running it reproduces identical CSVs.")
code("""
from src import data_prep
th = data_prep.main()
""")
code("""
train = pd.read_csv(C.TRAIN_CSV); val = pd.read_csv(C.VAL_CSV); test = pd.read_csv(C.TEST_CSV)
print({k: len(v) for k, v in dict(train=train, val=val, test=test).items()})
train[["label", "text", "clean_text", "tokens", "token_count"]].head(3)
""")
code("""
pd.read_csv(C.TABLE_DIR / "dataset_stats.csv").set_index("split").T
""")
code("""
display(Image(C.FIG_DIR / "length_histogram.png"))
pd.read_csv(C.TABLE_DIR / "label_vs_length.csv").round(4)
""")

md("## 2. Length groups\nThresholds are computed on **train only** and applied to val/test: two-way split "
   "(short ≤ median < long) and quartile bins Q1–Q4. `max_len` for the neural models = 95th percentile of "
   "train token counts.")
code("""
print(json.dumps(th, indent=2))
g = pd.read_csv(C.TABLE_DIR / "length_group_stats.csv")
g[g.split == "test"]
""")
code("""display(Image(C.FIG_DIR / "class_balance_by_quartile.png"))""")

md("## 3. Models\n### 3a. TF-IDF (uni+bigrams) + Logistic Regression\nFit on train only; C picked on the "
   "validation set by log-loss.")
code("""
if RETRAIN or not (C.MODELS_DIR / "tfidf_lr.joblib").exists():
    from src import train_baseline; train_baseline.main()
pd.read_csv(C.TABLE_DIR / "tfidf_lr_C_search.csv")
""")
md("### 3b/3c. Simple RNN and BiLSTM\nVocabulary = top 20k train tokens + `<PAD>`, `<OOV>`; head truncation at "
   "`max_len`; dropout, AdamW, LR halving on plateau, early stopping on validation loss (patience 3). "
   "Best checkpoints are in `models/`.")
code("""
from src import train_nn
for kind in ("rnn", "lstm"):
    if RETRAIN or not (C.MODELS_DIR / f"{kind}_best.pt").exists():
        train_nn.train(kind)
for kind in ("rnn", "lstm"):
    h = pd.read_csv(C.TABLE_DIR / f"training_history_{kind}.csv")
    ck = torch.load(C.MODELS_DIR / f"{kind}_best.pt", map_location="cpu", weights_only=False)
    print(f"{C.MODEL_NAMES[kind]}: best epoch {ck['epoch']}, val_loss {ck['val_loss']:.4f}, "
          f"val_acc {ck['val_acc']:.4f}, params {ck['n_params']:,}, "
          f"train time {h.total_train_minutes.iloc[0]:.1f} min")
    display(h.round(4))
    display(Image(C.FIG_DIR / f"training_curves_{kind}.png"))
""")
md("### 3d. Chunked BiLSTM (feature A)\nEvery review is split into sentence-based chunks of ≤ 50 tokens; the trained "
   "BiLSTM scores each chunk and the scores are combined by confidence-weighted averaging. A *hybrid* variant "
   "(chunk only reviews longer than `max_len`) is reported as an extra comparison.")
code("""
from src import chunked
chunked.main()
""")

md("## 4. Evaluation on the held-out test set")
code("""
from src import evaluate
evaluate.main()
""")
code("""pd.read_csv(C.TABLE_DIR / "metrics_overall.csv").round(4)""")
code("""display(Image(C.FIG_DIR / "confusion_matrices.png"))""")
code("""
grp = pd.read_csv(C.TABLE_DIR / "metrics_by_length_group.csv")
grp.pivot_table(index=["grouping", "group"], columns="model_name", values="f1").round(4)
""")
code("""
grp.pivot_table(index=["grouping", "group"], columns="model_name", values="accuracy").round(4)
""")
code("""
display(Image(C.FIG_DIR / "f1_by_quartile.png"))
display(Image(C.FIG_DIR / "accuracy_by_quartile.png"))
""")
md("### Short-vs-long gap and the effect of truncation")
code("""
display(pd.read_csv(C.TABLE_DIR / "short_long_gap.csv").round(4))
pd.read_csv(C.TABLE_DIR / "truncation_effect.csv").round(4)
""")
code("""
comp = pd.read_csv(C.TABLE_DIR / "chunked_variants_comparison.csv")
comp.pivot_table(index=["grouping", "group"], columns="model_name", values="accuracy").round(4)
""")

md("## 5. Calibration and human-review flagging (feature D)\nConfidence = max(p, 1−p). The flagging threshold "
   "is chosen on the **validation** set as the smallest threshold whose auto-accepted predictions reach 95 % "
   "accuracy; it is then applied unchanged to the test set.")
code("""
from src import calibration
calibration.main()
display(Image(C.FIG_DIR / "reliability_diagrams.png"))
display(Image(C.FIG_DIR / "coverage_vs_accuracy.png"))
""")

md("## 6. Error analysis\n20 misclassified test reviews per model (5 per length quartile, seeded). The *likely cause* "
   "is a transparent heuristic tag (truncation / negation / mixed sentiment / sarcasm / other), not a human label.")
code("""
from src import error_analysis
err = error_analysis.main()
err.groupby("model").head(4)[["model_name", "true_label", "prob_positive", "length_quartile", "likely_cause", "review"]]
""")

md("## 7. Summary (computed from the result tables)")
code("""
ov = pd.read_csv(C.TABLE_DIR / "metrics_overall.csv")
gap = pd.read_csv(C.TABLE_DIR / "short_long_gap.csv")
fl = pd.read_csv(C.TABLE_DIR / "human_review_flagging.csv")
best = ov.loc[ov.f1.idxmax()]
print(f"Best overall model: {best.model_name}  (accuracy {best.accuracy:.4f}, F1 {best.f1:.4f})")
for _, r in gap.iterrows():
    print(f"{C.MODEL_NAMES[r.model]:<16} acc short {r.acc_short:.4f} | long {r.acc_long:.4f} | "
          f"gap {r.acc_gap_short_minus_long*100:+.2f} pts")
for _, r in fl.iterrows():
    print(f"{r.model_name:<16} flags {r.test_pct_flagged:.1f}% of test -> accuracy on the rest "
          f"{r.test_acc_unflagged:.4f} (vs {r.test_acc_all:.4f})")
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("sentiment_project.ipynb")
nbf.write(nb, out)
print("wrote", out)

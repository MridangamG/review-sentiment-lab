"""Model (a): TF-IDF (uni+bigrams) + Logistic Regression.

The vectoriser and the classifier are fit on TRAIN only. The regularisation
strength C is chosen on the validation set from a small grid, then the
selected model is saved and used to score val + test.

Run:  python -m src.train_baseline
"""
from __future__ import annotations

import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

from src import config as C
from src.text_utils import split_tokens

MODEL_FILE = C.MODELS_DIR / "tfidf_lr.joblib"
PRED_DIR = C.RESULTS_DIR / "predictions"


def build_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=split_tokens, preprocessor=None, lowercase=False,
                                  token_pattern=None, ngram_range=(1, 2), min_df=2,
                                  max_features=200_000, sublinear_tf=True)),
        ("lr", LogisticRegression(C=c, max_iter=2000, solver="liblinear", random_state=C.SEED)),
    ])


def main() -> None:
    C.set_seed()
    PRED_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(C.TRAIN_CSV)
    val = pd.read_csv(C.VAL_CSV)
    test = pd.read_csv(C.TEST_CSV)

    t0 = time.time()
    rows, best = [], None
    for c in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        pipe = build_pipeline(c).fit(train["tokens"], train["label"])
        p = pipe.predict_proba(val["tokens"])[:, 1]
        acc, ll = accuracy_score(val["label"], p > 0.5), log_loss(val["label"], p)
        rows.append({"C": c, "val_accuracy": acc, "val_logloss": ll})
        print(f"  C={c:<4} val_acc={acc:.4f} val_logloss={ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, c, pipe)
    pd.DataFrame(rows).to_csv(C.TABLE_DIR / "tfidf_lr_C_search.csv", index=False)
    _, c_best, pipe = best
    print(f"Selected C={c_best} (lowest validation log-loss); total fit time {time.time()-t0:.1f}s")
    print(f"  vocabulary size (uni+bigrams): {len(pipe.named_steps['tfidf'].vocabulary_):,}")
    joblib.dump(pipe, MODEL_FILE)

    for name, df in (("val", val), ("test", test)):
        pd.DataFrame({"prob": pipe.predict_proba(df["tokens"])[:, 1]}).to_csv(
            PRED_DIR / f"tfidf_lr_{name}.csv", index=False)


if __name__ == "__main__":
    main()

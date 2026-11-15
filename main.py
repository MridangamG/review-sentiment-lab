"""Run the full sentiment-analysis pipeline end to end.

    python main.py                 # everything (data prep -> training -> evaluation)
    python main.py --skip-training # re-use saved models, only re-score / re-evaluate
    python main.py --stages evaluate calibration

Stages (in order):
  data          src/data_prep.py      sample, clean, tokenise, length groups, stats
  baseline      src/train_baseline.py TF-IDF + Logistic Regression
  rnn, lstm     src/train_nn.py       Simple RNN and BiLSTM (early stopping)
  chunked       src/chunked.py        Chunked BiLSTM (+ hybrid variant)
  evaluate      src/evaluate.py       metrics overall / per length group, plots
  calibration   src/calibration.py    reliability diagrams, human-review flagging
  errors        src/error_analysis.py 20 misclassified reviews per model
"""
from __future__ import annotations

import argparse
import time

from src import config as C

ALL = ["data", "baseline", "rnn", "lstm", "chunked", "evaluate", "calibration", "errors"]
TRAINING = {"data", "baseline", "rnn", "lstm"}


def run(stage: str) -> None:
    if stage == "data":
        from src import data_prep
        data_prep.main()
    elif stage == "baseline":
        from src import train_baseline
        train_baseline.main()
    elif stage in ("rnn", "lstm"):
        from src import train_nn
        train_nn.train(stage)
    elif stage == "chunked":
        from src import chunked
        chunked.main()
    elif stage == "evaluate":
        from src import evaluate
        evaluate.main()
    elif stage == "calibration":
        from src import calibration
        calibration.main()
    elif stage == "errors":
        from src import error_analysis
        error_analysis.main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=ALL, default=ALL)
    ap.add_argument("--skip-training", action="store_true", help="skip data prep and model training")
    args = ap.parse_args()
    C.set_seed()
    stages = [s for s in args.stages if not (args.skip_training and s in TRAINING)]
    for s in stages:
        t0 = time.time()
        print(f"\n{'=' * 70}\n>>> stage: {s}\n{'=' * 70}")
        run(s)
        print(f">>> {s} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

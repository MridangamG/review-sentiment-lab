"""Central configuration: paths, sample sizes, hyper-parameters and seeding helpers.

Every script imports from here so that all models share exactly the same
splits, vocabulary settings and random seeds.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DEMO_DIR = ROOT / "data" / "demo"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

RAW_TRAIN = RAW_DIR / "train-00000-of-00004.parquet"
RAW_TEST = RAW_DIR / "test-00000-of-00001.parquet"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

for _d in (PROCESSED_DIR, DEMO_DIR, MODELS_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data sampling
# ----------------------------------------------------------------------------
SEED = 42
N_TRAIN_POOL = 80_000      # balanced sample from the train parquet (-> train + val)
VAL_FRACTION = 0.15        # 85 / 15 stratified split
N_TEST = 20_000            # balanced sample from the test parquet

# ----------------------------------------------------------------------------
# Vocabulary / sequence settings (neural models)
# ----------------------------------------------------------------------------
VOCAB_SIZE = 20_000        # top-k words from TRAIN only (+ <PAD>, <OOV>)
PAD, OOV = "<PAD>", "<OOV>"
PAD_IDX, OOV_IDX = 0, 1
MAX_LEN_PERCENTILE = 95    # max_len = 95th percentile of train token counts
CHUNK_TOKENS = 50          # chunk size for the chunked-LSTM model

# ----------------------------------------------------------------------------
# Neural training hyper-parameters
# ----------------------------------------------------------------------------
EMBED_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.5
BATCH_SIZE = 128
MAX_EPOCHS = 20
PATIENCE = 3               # early stopping on validation loss
WEIGHT_DECAY = 1e-4
LR = 1e-3
GRAD_CLIP = 1.0

MODEL_NAMES = {
    "tfidf_lr": "TF-IDF + LogReg",
    "rnn": "Simple RNN",
    "lstm": "BiLSTM",
    "chunked_lstm": "Chunked BiLSTM",
}


def set_seed(seed: int = SEED) -> None:
    """Fix every random seed we rely on (python, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:  # torch not needed for every script
        pass


def get_device():
    """Use Apple-Silicon GPU (MPS) when available, else CPU."""
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

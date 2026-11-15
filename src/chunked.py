"""Feature A: Chunked BiLSTM - a 4th "model" that re-uses the trained BiLSTM.

Each review is split into sentences, consecutive sentences are merged into
chunks of at most 50 tokens (over-long sentences are cut into 50-token
windows), the trained LSTM scores every chunk, and the chunk probabilities are
combined with *confidence-weighted averaging*:

    w_i = |2 p_i - 1| + eps         (0 for a 50/50 chunk, 1 for a certain one)
    p   = sum(w_i * p_i) / sum(w_i)

Because no chunk is longer than 50 tokens (< max_len), the whole review is
seen - nothing is truncated.

We also report a *hybrid* variant (plain LSTM when the review fits in max_len,
chunked only for reviews that would be truncated) as an extra comparison.

Run:  python -m src.chunked
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src import config as C
from src.data_prep import load_thresholds
from src.models import load_checkpoint
from src.text_utils import make_chunks
from src.vocab import Vocab

PRED_DIR = C.RESULTS_DIR / "predictions"
EPS = 1e-3


def aggregate(probs: np.ndarray) -> float:
    w = np.abs(2 * probs - 1) + EPS
    return float((w * probs).sum() / w.sum())


@torch.no_grad()
def score_token_lists(model, vocab: Vocab, token_lists, device, bs: int = 1024) -> np.ndarray:
    """LSTM probability for each token list (each must be <= CHUNK_TOKENS long)."""
    out = []
    for s in range(0, len(token_lists), bs):
        part = token_lists[s : s + bs]
        L = max(1, max(len(t) for t in part))
        X = torch.from_numpy(vocab.encode_batch(part, L)).to(device)
        out.append(torch.sigmoid(model(X)).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def chunk_details(model, vocab, clean_text: str, device="cpu"):
    """Per-chunk breakdown for one review (used by the Streamlit app)."""
    chunks = make_chunks(clean_text, C.CHUNK_TOKENS)
    probs = score_token_lists(model, vocab, chunks, device)
    weights = np.abs(2 * probs - 1) + EPS
    return chunks, probs, weights / weights.sum(), aggregate(probs)


def predict_chunked(model, vocab, clean_texts, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (aggregated probability, number of chunks) for many reviews."""
    all_chunks, owner = [], []
    for i, t in enumerate(clean_texts):
        ch = make_chunks(t, C.CHUNK_TOKENS)
        all_chunks.extend(ch)
        owner.extend([i] * len(ch))
    # sort by length for efficient batching, then restore order
    order = np.argsort([len(c) for c in all_chunks])
    probs_sorted = score_token_lists(model, vocab, [all_chunks[j] for j in order], device)
    probs = np.empty(len(all_chunks))
    probs[order] = probs_sorted
    owner = np.array(owner)
    agg = np.zeros(len(clean_texts))
    n_chunks = np.bincount(owner, minlength=len(clean_texts))
    for i in range(len(clean_texts)):
        agg[i] = aggregate(probs[owner == i]) if n_chunks[i] else 0.5
    return agg, n_chunks


def main() -> None:
    C.set_seed()
    device = C.get_device()
    model, _ = load_checkpoint("lstm", device)
    vocab = Vocab.load()
    max_len = load_thresholds()["max_len_p95"]
    for split, path in (("val", C.VAL_CSV), ("test", C.TEST_CSV)):
        df = pd.read_csv(path)
        agg, n_chunks = predict_chunked(model, vocab, df["clean_text"].tolist(), device)
        pd.DataFrame({"prob": agg, "n_chunks": n_chunks}).to_csv(PRED_DIR / f"chunked_lstm_{split}.csv", index=False)
        # hybrid: standard LSTM unless the review would be truncated
        lstm = pd.read_csv(PRED_DIR / f"lstm_{split}.csv")["prob"].values
        hybrid = np.where(df["token_count"].values > max_len, agg, lstm)
        pd.DataFrame({"prob": hybrid}).to_csv(PRED_DIR / f"hybrid_lstm_{split}.csv", index=False)
        print(f"[chunked] {split}: {len(df):,} reviews, mean chunks/review = {n_chunks.mean():.2f}, "
              f"max = {n_chunks.max()}")


if __name__ == "__main__":
    main()

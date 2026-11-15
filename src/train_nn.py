"""Models (b) Simple RNN and (c) BiLSTM, trained on identical splits.

* Vocabulary from TRAIN only (top 20k + <PAD>, <OOV>), saved to models/vocab.json
* max_len = 95th percentile of TRAIN token counts (head truncation)
* Adam, BCE loss, dropout, gradient clipping
* Early stopping on validation loss; best checkpoint -> models/<kind>_best.pt
* Training curves -> results/figures/training_curves_<kind>.png

Run:  python -m src.train_nn rnn   |   python -m src.train_nn lstm
"""
from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config as C
from src.data_prep import load_thresholds
from src.models import build_model
from src.vocab import Vocab

PRED_DIR = C.RESULTS_DIR / "predictions"


def get_vocab(train: pd.DataFrame) -> Vocab:
    if C.MODELS_DIR.joinpath("vocab.json").exists():
        return Vocab.load()
    vocab = Vocab.build(train["tokens"].str.split())
    vocab.save()
    return vocab


def encode(df: pd.DataFrame, vocab: Vocab, max_len: int) -> torch.Tensor:
    return torch.from_numpy(vocab.encode_batch(df["tokens"].str.split().tolist(), max_len))


def batches(X: torch.Tensor, y: torch.Tensor | None, bs: int, shuffle: bool, gen=None):
    """Yield mini-batches trimmed to the longest real sequence in the batch
    (dynamic padding -> much faster than always padding to max_len)."""
    idx = torch.randperm(len(X), generator=gen) if shuffle else torch.arange(len(X))
    for s in range(0, len(X), bs):
        b = idx[s : s + bs]
        xb = X[b]
        L = int((xb != C.PAD_IDX).sum(1).max().clamp(min=1))
        yield xb[:, :L], (y[b] if y is not None else None)


@torch.no_grad()
def predict_proba(model, X: torch.Tensor, device, bs: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in batches(X, None, bs, shuffle=False):
        out.append(torch.sigmoid(model(xb.to(device))).float().cpu())
    return torch.cat(out).numpy()


def run_epoch(model, X, y, device, opt=None, gen=None):
    train = opt is not None
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    tot_loss, correct = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in batches(X, y, C.BATCH_SIZE, shuffle=train, gen=gen):
            xb, yb = xb.to(device), yb.to(device).float()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                opt.zero_grad()
                (loss / len(yb)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                opt.step()
            tot_loss += loss.item()
            correct += ((logits > 0).long() == yb.long()).sum().item()
    return tot_loss / len(X), correct / len(X)


def train(kind: str) -> pd.DataFrame:
    C.set_seed()
    device = C.get_device()
    PRED_DIR.mkdir(exist_ok=True)
    train_df = pd.read_csv(C.TRAIN_CSV)
    val_df = pd.read_csv(C.VAL_CSV)
    test_df = pd.read_csv(C.TEST_CSV)
    max_len = load_thresholds()["max_len_p95"]
    vocab = get_vocab(train_df)
    print(f"[{kind}] device={device} vocab={len(vocab):,} max_len={max_len}")

    Xtr, Xva, Xte = (encode(d, vocab, max_len) for d in (train_df, val_df, test_df))
    ytr, yva = torch.tensor(train_df.label.values), torch.tensor(val_df.label.values)

    model = build_model(kind, len(vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    # halve the learning rate whenever validation loss stops improving
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=0)
    gen = torch.Generator().manual_seed(C.SEED)

    hist, best_loss, bad, t_start = [], float("inf"), 0, time.time()
    ckpt_path = C.MODELS_DIR / f"{kind}_best.pt"
    for epoch in range(1, C.MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, Xtr, ytr, device, opt, gen)
        va_loss, va_acc = run_epoch(model, Xva, yva, device)
        sched.step(va_loss)
        hist.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": va_loss, "val_acc": va_acc, "lr": opt.param_groups[0]["lr"],
                     "seconds": time.time() - t0})
        flag = ""
        if va_loss < best_loss - 1e-4:
            best_loss, bad = va_loss, 0
            torch.save({"state_dict": model.state_dict(), "vocab_size": len(vocab), "max_len": max_len,
                        "kind": kind, "epoch": epoch, "val_loss": va_loss, "val_acc": va_acc,
                        "n_params": n_params}, ckpt_path)
            flag = "  * saved"
        else:
            bad += 1
        print(f"[{kind}] epoch {epoch:2d} train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | {time.time()-t0:.0f}s{flag}")
        if bad >= C.PATIENCE:
            print(f"[{kind}] early stopping (no val-loss improvement for {C.PATIENCE} epochs)")
            break
    total = time.time() - t_start
    hist = pd.DataFrame(hist)
    hist["total_train_minutes"] = total / 60
    hist.to_csv(C.TABLE_DIR / f"training_history_{kind}.csv", index=False)
    print(f"[{kind}] total training time {total/60:.1f} min, params={n_params:,}")

    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist.epoch, hist.train_loss, "o-", label="train")
    axes[0].plot(hist.epoch, hist.val_loss, "o-", label="validation")
    axes[1].plot(hist.epoch, hist.train_acc, "o-", label="train")
    axes[1].plot(hist.epoch, hist.val_acc, "o-", label="validation")
    best_ep = int(hist.loc[hist.val_loss.idxmin(), "epoch"])
    for ax, t in zip(axes, ("BCE loss", "accuracy")):
        ax.axvline(best_ep, color="grey", ls="--", lw=1, label=f"best epoch ({best_ep})")
        ax.set(xlabel="epoch", title=f"{C.MODEL_NAMES[kind]}: {t}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"training_curves_{kind}.png", dpi=130)
    plt.close(fig)

    # reload best checkpoint and score val / test
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
    for name, X in (("val", Xva), ("test", Xte)):
        pd.DataFrame({"prob": predict_proba(model, X, device)}).to_csv(PRED_DIR / f"{kind}_{name}.csv", index=False)
    return hist


if __name__ == "__main__":
    for k in sys.argv[1:] or ["rnn", "lstm"]:
        train(k)

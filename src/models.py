"""PyTorch sequence classifiers: a simple (Elman) RNN and a bidirectional LSTM.

Both models
* take a right-padded LongTensor of token ids (PAD = 0),
* compute the true sequence lengths from the padding themselves (this makes the
  models directly usable by Captum, which only passes the input tensor),
* pack the sequences so padding never influences the hidden state,
* output a single logit (positive class) -> use BCEWithLogitsLoss / sigmoid.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src import config as C


class _SeqClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float,
                 cell: str, bidirectional: bool, pool: bool = False):
        super().__init__()
        self.pool = pool  # if True: concat final states with masked max-pool over all time steps
        self.cell_type = cell
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=C.PAD_IDX)
        self.emb_drop = nn.Dropout(dropout)
        rnn_cls = nn.LSTM if cell == "lstm" else nn.RNN
        kwargs = {} if cell == "lstm" else {"nonlinearity": "tanh"}
        self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional, **kwargs)
        out_dim = hidden_dim * (2 if bidirectional else 1) * (2 if pool else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # lengths from padding unless given explicitly (Captum passes them so that the
        # all-<PAD> IG baseline is read with the same length as the real input)
        if lengths is None:
            lengths = (x != C.PAD_IDX).sum(dim=1)
        lengths = lengths.clamp(min=1).cpu()
        emb = self.emb_drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        if self.cell_type == "lstm":
            h = h[0]                                   # (h_n, c_n) -> h_n
        if self.bidirectional:
            feat = torch.cat([h[-2], h[-1]], dim=1)    # last fwd + last bwd states
        else:
            feat = h[-1]
        if self.pool:
            seq, _ = pad_packed_sequence(out, batch_first=True, padding_value=-1e4)
            feat = torch.cat([feat, seq.max(dim=1).values], dim=1)
        return self.fc(self.drop(feat)).squeeze(-1)


def build_model(kind: str, vocab_size: int) -> _SeqClassifier:
    if kind == "rnn":
        return _SeqClassifier(vocab_size, C.EMBED_DIM, C.HIDDEN_DIM, C.DROPOUT, cell="rnn", bidirectional=False)
    if kind == "lstm":
        return _SeqClassifier(vocab_size, C.EMBED_DIM, C.HIDDEN_DIM, C.DROPOUT, cell="lstm", bidirectional=True, pool=True)
    raise ValueError(kind)


def load_checkpoint(kind: str, device="cpu") -> tuple[_SeqClassifier, dict]:
    ckpt = torch.load(C.MODELS_DIR / f"{kind}_best.pt", map_location=device, weights_only=False)
    model = build_model(kind, ckpt["vocab_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt

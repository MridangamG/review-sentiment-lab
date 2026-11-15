"""Vocabulary built from TRAIN tokens only (top 20k + <PAD>, <OOV>) and encoding helpers."""
from __future__ import annotations

import json
from collections import Counter
from typing import Iterable, List

import numpy as np

from src import config as C

VOCAB_FILE = C.MODELS_DIR / "vocab.json"


class Vocab:
    def __init__(self, itos: List[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, token_lists: Iterable[List[str]], size: int = C.VOCAB_SIZE) -> "Vocab":
        counter = Counter()
        for toks in token_lists:
            counter.update(toks)
        itos = [C.PAD, C.OOV] + [w for w, _ in counter.most_common(size)]
        return cls(itos)

    def encode(self, tokens: List[str], max_len: int | None = None) -> List[int]:
        ids = [self.stoi.get(t, C.OOV_IDX) for t in tokens]
        return ids[:max_len] if max_len else ids  # head truncation: keep the first max_len tokens

    def encode_batch(self, token_lists: List[List[str]], max_len: int) -> np.ndarray:
        """Right-padded int matrix of shape (n, max_len)."""
        out = np.zeros((len(token_lists), max_len), dtype=np.int64)
        for i, toks in enumerate(token_lists):
            ids = self.encode(toks, max_len)
            out[i, : len(ids)] = ids
        return out

    def save(self, path=VOCAB_FILE) -> None:
        path.write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path=VOCAB_FILE) -> "Vocab":
        return cls(json.loads(path.read_text()))

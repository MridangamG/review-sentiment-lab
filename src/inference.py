"""Shared inference layer for the Streamlit app and the FastAPI service.

Loads all trained artefacts once (on CPU) and offers:
* predict(text)                  -> label / P(positive) / confidence for all 4 models
* predict_batch(clean_texts, m)  -> P(positive) for many reviews with one model
* lr_contributions(text)         -> per-token TF-IDF x coefficient weights
* lstm_attributions(text)        -> per-token Integrated Gradients (Captum)
* chunk_view(text)               -> chunked-LSTM per-chunk verdicts
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import torch

from src import config as C
from src.chunked import chunk_details, predict_chunked
from src.models import load_checkpoint
from src.text_utils import clean_text, tokenize
from src.vocab import Vocab


@dataclass
class Prediction:
    model: str
    name: str
    prob_positive: float

    @property
    def label(self) -> str:
        return "positive" if self.prob_positive >= 0.5 else "negative"

    @property
    def confidence(self) -> float:
        return max(self.prob_positive, 1 - self.prob_positive)


class SentimentPipeline:
    def __init__(self, device: str = "cpu"):
        torch.set_num_threads(4)
        self.device = torch.device(device)
        self.vocab = Vocab.load()
        self.tfidf = joblib.load(C.MODELS_DIR / "tfidf_lr.joblib")
        self.rnn, ck = load_checkpoint("rnn", self.device)
        self.lstm, _ = load_checkpoint("lstm", self.device)
        self.max_len = int(ck["max_len"])
        thr = C.RESULTS_DIR / "flag_thresholds.json"
        self.flag_thresholds = json.loads(thr.read_text()) if thr.exists() else {}

    # ------------------------------------------------------------------ basics
    @staticmethod
    def prepare(text: str) -> tuple[str, list[str]]:
        c = clean_text(text)
        return c, tokenize(c)

    @torch.no_grad()
    def _nn_prob(self, model, token_lists: list[list[str]]) -> np.ndarray:
        out = []
        for s in range(0, len(token_lists), 512):
            part = [t[: self.max_len] or [C.PAD] for t in token_lists[s : s + 512]]
            L = max(len(t) for t in part)
            X = torch.from_numpy(self.vocab.encode_batch(part, L)).to(self.device)
            out.append(torch.sigmoid(model(X)).cpu().numpy())
        return np.concatenate(out)

    def predict_batch(self, clean_texts: list[str], model: str = "lstm") -> np.ndarray:
        toks = [tokenize(c) for c in clean_texts]
        if model == "tfidf_lr":
            return self.tfidf.predict_proba([" ".join(t) for t in toks])[:, 1]
        if model == "rnn":
            return self._nn_prob(self.rnn, toks)
        if model == "lstm":
            return self._nn_prob(self.lstm, toks)
        if model == "chunked_lstm":
            return predict_chunked(self.lstm, self.vocab, clean_texts, self.device)[0]
        raise ValueError(model)

    def predict(self, text: str) -> list[Prediction]:
        c, _ = self.prepare(text)
        return [Prediction(m, C.MODEL_NAMES[m], float(self.predict_batch([c], m)[0])) for m in C.MODEL_NAMES]

    # ------------------------------------------------------------ explanations
    def lr_contributions(self, text: str) -> list[tuple[str, float]]:
        """Weight of each token = sum over the uni/bi-grams containing it of tfidf * coef
        (bigram weight is split evenly between its two words)."""
        _, toks = self.prepare(text)
        vec = self.tfidf.named_steps["tfidf"]
        coef = self.tfidf.named_steps["lr"].coef_[0]
        x = vec.transform([" ".join(toks)])
        weights = {vec.get_feature_names_out()[j]: x[0, j] * coef[j] for j in x.nonzero()[1]}
        scores = np.zeros(len(toks))
        for i, t in enumerate(toks):
            scores[i] += weights.get(t, 0.0)
            if i + 1 < len(toks):
                w = weights.get(f"{t} {toks[i + 1]}", 0.0) / 2
                scores[i] += w
                scores[i + 1] += w
        return list(zip(toks, scores.tolist()))

    def lstm_attributions(self, text: str, n_steps: int = 50, kind: str = "lstm") -> tuple[list[tuple[str, float]], float]:
        """Integrated Gradients w.r.t. the embedding layer (all-<OOV> baseline).
        Returns (token, attribution) for the tokens the LSTM actually sees."""
        from captum.attr import LayerIntegratedGradients

        _, toks = self.prepare(text)
        toks = toks[: self.max_len] or [C.PAD]
        x = torch.tensor([self.vocab.encode(toks)], device=self.device)
        # baseline = same-length sequence of <OOV> ("unknown words"): converges far better
        # than all-<PAD> (zero embeddings), whose logit is strongly non-neutral for this model
        base = torch.full_like(x, C.OOV_IDX)
        model = self.rnn if kind == "rnn" else self.lstm

        def fwd(inp, lengths):
            return model(inp, lengths)

        lig = LayerIntegratedGradients(fwd, model.embedding)
        lengths = torch.tensor([x.shape[1]])
        attr, delta = lig.attribute(x, baselines=base, additional_forward_args=(lengths,), n_steps=n_steps,
                                    return_convergence_delta=True, internal_batch_size=n_steps)
        scores = attr.sum(-1).squeeze(0).detach().cpu().numpy()
        return list(zip(toks, scores.tolist())), float(delta.abs().item())

    def chunk_view(self, text: str):
        c, _ = self.prepare(text)
        return chunk_details(self.lstm, self.vocab, c, self.device)

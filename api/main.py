"""FastAPI service for the sentiment models.

    uvicorn api.main:app --reload --port 8000

POST /predict   {"text": "...", "model": "lstm", "top_k": 5}
  -> {"label", "confidence", "prob_positive", "model", "needs_human_review",
      "token_count", "truncated", "top_words": [{"word", "weight"}, ...]}

`model` is one of tfidf_lr | rnn | lstm | chunked_lstm (default lstm).
Top contributing words come from Integrated Gradients for the RNN / BiLSTM
(the chunked model is explained by the BiLSTM it re-uses), and from TF-IDF x coefficient weights for the logistic regression.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.inference import SentimentPipeline  # noqa: E402

app = FastAPI(title="Review Sentiment API", version="1.0",
              description="Positive/negative classification of product reviews (Amazon Polarity).")
_pipe: SentimentPipeline | None = None


def pipe() -> SentimentPipeline:
    global _pipe
    if _pipe is None:
        _pipe = SentimentPipeline("cpu")
    return _pipe


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text")
    model: Literal["tfidf_lr", "rnn", "lstm", "chunked_lstm"] = "lstm"
    top_k: int = Field(5, ge=1, le=50)


class Word(BaseModel):
    word: str
    weight: float


class PredictResponse(BaseModel):
    model: str
    label: Literal["positive", "negative"]
    confidence: float
    prob_positive: float
    needs_human_review: bool
    token_count: int
    truncated: bool
    top_words: list[Word]


@app.get("/health")
def health():
    return {"status": "ok", "models": list(C.MODEL_NAMES)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    p = pipe()
    clean, toks = p.prepare(req.text)
    if not toks:
        raise HTTPException(422, "text contains no usable tokens")
    prob = float(p.predict_batch([clean], req.model)[0])
    label = "positive" if prob >= 0.5 else "negative"
    conf = max(prob, 1 - prob)

    if req.model == "tfidf_lr":
        pairs = p.lr_contributions(req.text)
    else:  # Integrated Gradients on the RNN, or on the BiLSTM for lstm / chunked_lstm
        pairs, _ = p.lstm_attributions(req.text, n_steps=50, kind="rnn" if req.model == "rnn" else "lstm")
    agg: dict[str, float] = {}
    for w, s in pairs:
        agg[w] = agg.get(w, 0.0) + s
    sign = 1 if label == "positive" else -1  # words that pushed towards the predicted label
    top = sorted(agg.items(), key=lambda kv: sign * kv[1], reverse=True)[: req.top_k]
    thr = p.flag_thresholds.get(req.model)
    return PredictResponse(model=req.model, label=label, confidence=round(conf, 4), prob_positive=round(prob, 4),
                           needs_human_review=bool(thr is not None and conf < thr), token_count=len(toks),
                           truncated=len(toks) > p.max_len,
                           top_words=[Word(word=w, weight=round(s, 5)) for w, s in top])

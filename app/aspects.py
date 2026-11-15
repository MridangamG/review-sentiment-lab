"""Aspect detection and n-gram helpers for the Review Dashboard tab."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

# keyword lists (matched on whole words at sentence level, after lower-casing)
ASPECTS: dict[str, list[str]] = {
    "Quality": ["quality", "well made", "well-made", "cheaply made", "build", "material", "materials",
                "craftsmanship", "construction", "flimsy", "sturdy", "solid", "cheap"],
    "Price / Value": ["price", "prices", "value", "worth", "money", "expensive", "cheaper", "overpriced",
                      "bargain", "cost", "costs", "deal", "affordable", "$"],
    "Delivery / Shipping": ["shipping", "shipped", "delivery", "delivered", "arrived", "arrive", "package",
                            "packaging", "packaged", "seller", "on time", "late"],
    "Durability": ["durable", "durability", "lasted", "last", "lasts", "broke", "broken", "break", "wore out",
                   "fell apart", "months", "stopped working", "years"],
    "Size / Fit": ["size", "sizes", "fit", "fits", "fitting", "small", "large", "tight", "loose", "big",
                   "too small", "too big", "length", "width"],
    "Battery": ["battery", "batteries", "charge", "charging", "charger", "recharge", "rechargeable", "power"],
    "Customer service": ["customer service", "support", "warranty", "refund", "return", "returned",
                         "replacement", "contacted", "response", "service"],
}

_PATTERNS = {a: re.compile(r"(?<![a-z])(" + "|".join(re.escape(k) for k in kws) + r")(?![a-z])")
             for a, kws in ASPECTS.items()}

STOP = set("""a an the and or of to in on for with is it this that was were be been are as at by i my me
you your we our they them their he she his her its from have has had do does did so if than then there
these those what which who whom just very too also can could would should will one all any some about
into out up down over more most much get got s t ca wo ! ?""".split())


def aspects_in(sentence: str) -> list[str]:
    return [a for a, p in _PATTERNS.items() if p.search(sentence)]


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def top_ngrams(token_lists: list[list[str]], preds: np.ndarray, n_range=(1, 2), k: int = 15,
               min_count: int = 3) -> tuple[list[tuple[str, float, int]], list[tuple[str, float, int]]]:
    """Most distinctive n-grams of predicted-positive vs predicted-negative reviews
    using the smoothed log-odds ratio (Monroe et al., 2008, uninformative prior)."""
    pos, neg = Counter(), Counter()
    for toks, p in zip(token_lists, preds):
        toks = [t for t in toks if t not in STOP]
        grams = set()
        for n in range(n_range[0], n_range[1] + 1):
            grams.update(ngrams(toks, n))
        (pos if p == 1 else neg).update(grams)
    vocab = [g for g in set(pos) | set(neg) if pos[g] + neg[g] >= min_count]
    if not vocab:
        return [], []
    a = 0.5
    npos, nneg = sum(pos.values()) + a * len(vocab), sum(neg.values()) + a * len(vocab)
    scores = []
    for g in vocab:
        lp = np.log((pos[g] + a) / (npos - pos[g] - a))
        ln = np.log((neg[g] + a) / (nneg - neg[g] - a))
        var = 1 / (pos[g] + a) + 1 / (neg[g] + a)
        scores.append((g, (lp - ln) / np.sqrt(var), pos[g] + neg[g]))
    # display contractions naturally: "do n't" -> "don't"
    scores = [(g.replace(" n't", "n't"), z, c) for g, z, c in scores]
    scores.sort(key=lambda x: x[1])
    return scores[::-1][:k], scores[:k]

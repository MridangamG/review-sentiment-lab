"""Stage 5: error analysis - 20 misclassified test reviews per model.

Selection: a seeded random sample of misclassified test reviews, stratified by
length quartile (5 per quartile) so short and long reviews are both covered.

Each review gets an automatic, *heuristic* note on the likely cause, chosen by
simple, transparent rules (checked in this order, several can apply):
* truncation       - neural model and the review is longer than max_len
* negation         - a negation word within 3 tokens before a sentiment word
* mixed sentiment  - both positive and negative cue words, or a contrast
                     marker ("but", "however", "although", ...)
* sarcasm          - ironic patterns ("yeah right", "oh great", quotes around
                     praise words, "thanks a lot", ...) or positive words in a
                     negative review with "!"/"?" emphasis
* other            - none of the above (often ambiguous wording or label noise)
These rules are indicators, not ground truth.

Run:  python -m src.error_analysis
"""
from __future__ import annotations

import re

import pandas as pd

from src import config as C
from src.data_prep import load_thresholds
from src.evaluate import MODELS, available, load_predictions
from src.text_utils import NEGATIONS

POS_WORDS = set("""good great excellent love loved loves amazing awesome best perfect wonderful nice happy
fantastic favorite recommend recommended works worth fun beautiful enjoy enjoyed easy brilliant superb
pleased impressive solid fine helpful glad like liked""".split())
NEG_WORDS = set("""bad poor terrible awful worst hate hated waste boring broke broken disappointed
disappointing disappointment useless junk cheap horrible return returned problem problems fail failed
fails wrong annoying refund mediocre dull defective unfortunately avoid""".split())
CONTRAST = {"but", "however", "although", "though", "except", "yet", "otherwise", "unfortunately"}
SARCASM = re.compile(r"yeah right|oh great|oh joy|thanks a lot|thanks for nothing|what a (?:great|wonderful)|"
                     r"just what i needed|real(?:ly)? genius|\"(?:great|good|best|perfect|amazing)\"|"
                     r"'(?:great|good|best|perfect|amazing)'|so-called|lol|/s\b")


def likely_causes(row, model: str, max_len: int) -> str:
    toks = str(row["tokens"]).split()
    causes = []
    if model in ("rnn", "lstm") and row["token_count"] > max_len:
        causes.append(f"truncation ({row['token_count']} tokens > max_len {max_len}; "
                      f"{row['token_count'] - max_len} tokens unseen)")
    neg_hit = None
    for i, t in enumerate(toks):
        if t in NEGATIONS:
            window = toks[i + 1 : i + 4]
            hit = [w for w in window if w in POS_WORDS or w in NEG_WORDS]
            if hit:
                neg_hit = f"{t} ... {hit[0]}"
                break
    if neg_hit:
        causes.append(f"negation ('{neg_hit}')")
    n_pos = sum(t in POS_WORDS for t in toks)
    n_neg = sum(t in NEG_WORDS for t in toks)
    contrast = [t for t in toks if t in CONTRAST]
    if (n_pos and n_neg) or contrast:
        bits = f"{n_pos} pos / {n_neg} neg cue words"
        if contrast:
            bits += f", contrast marker '{contrast[0]}'"
        causes.append(f"mixed sentiment ({bits})")
    if SARCASM.search(row["clean_text"]) or (row["label"] == 0 and n_pos > n_neg and ("!" in toks)):
        causes.append("possible sarcasm / irony")
    if not causes:
        causes.append("other: ambiguous wording, domain-specific vocabulary or possible label noise")
    return "; ".join(causes)


def main(n_per_model: int = 20) -> pd.DataFrame:
    max_len = load_thresholds()["max_len_p95"]
    df = load_predictions("test")
    out = []
    for m in available(df, MODELS):
        wrong = df[df[f"pred_{m}"] != df["label"]]
        per_q = n_per_model // 4
        sample = pd.concat([g.sample(n=min(per_q, len(g)), random_state=C.SEED)
                            for _, g in wrong.groupby("length_quartile")])
        if len(sample) < n_per_model:  # top up if a quartile had too few errors
            rest = wrong.drop(sample.index)
            sample = pd.concat([sample, rest.sample(n=n_per_model - len(sample), random_state=C.SEED)])
        for _, r in sample.iterrows():
            out.append({
                "model": m, "model_name": C.MODEL_NAMES[m], "true_label": int(r.label),
                "pred_label": int(r[f"pred_{m}"]), "prob_positive": round(float(r[f"prob_{m}"]), 4),
                "token_count": int(r.token_count), "length_2way": r.length_2way,
                "length_quartile": r.length_quartile, "truncated_for_nn": bool(r.token_count > max_len),
                "likely_cause": likely_causes(r, m, max_len), "review": r.text,
            })
    res = pd.DataFrame(out)
    res.to_csv(C.TABLE_DIR / "error_analysis_misclassified.csv", index=False)

    # summary: how often each cause tag appears among the sampled errors
    tags = ["truncation", "negation", "mixed sentiment", "sarcasm", "other"]
    summ = pd.DataFrame([{"model": m, **{t: int(g.likely_cause.str.contains(t).sum()) for t in tags}}
                         for m, g in res.groupby("model", sort=False)])
    summ.to_csv(C.TABLE_DIR / "error_analysis_cause_counts.csv", index=False)

    # error rate overall by model for all misclassified (not just the sample)
    print("\nCause tags among the 20 sampled errors per model:\n", summ.to_string(index=False))
    return res


if __name__ == "__main__":
    main()

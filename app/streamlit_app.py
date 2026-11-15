"""Review Sentiment Lab - Streamlit front-end.

    streamlit run app/streamlit_app.py

Tabs: Live Analyzer | Explain | Truncation Visualizer | Length Lab | Review Dashboard
Every number shown in the Length Lab is read from the CSVs in results/tables/ that
main.py produces. Colours and fonts are set in .streamlit/config.toml and in THEME below.
"""
from __future__ import annotations

import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from aspects import ASPECTS, aspects_in, top_ngrams  # noqa: E402
from src import config as C  # noqa: E402
from src.text_utils import clean_text, split_sentences, tokenize  # noqa: E402

st.set_page_config(page_title="Review Sentiment Lab", page_icon=":material/straighten:", layout="wide")

# --------------------------------------------------------------------------- theme
INK, MUTED, RULE = "#16213A", "#5B6478", "#D5DAE4"
POS, NEG = "#0B7A75", "#B3305F"            # teal / raspberry: distinguishable with red-green colour blindness
POS_RGB, NEG_RGB = "11,122,117", "179,48,95"
NAVY = "#2B3A8C"
FONT = "IBM Plex Sans, Avenir Next, sans-serif"
# The chunked model re-uses the BiLSTM's weights, so it shares its hue (lighter, dashed).
MODEL_COLORS = {"TF-IDF + LogReg": NAVY, "Simple RNN": "#C98A00", "BiLSTM": "#1F7FA6", "Chunked BiLSTM": "#86C5DE"}
MODEL_DASH = {"Chunked BiLSTM": "dash"}
METRICS = {"f1": "F1", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall"}
PLOT_CONFIG = {"displayModeBar": False}

st.markdown(
    f"""
<style>
  .block-container {{padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px;}}
  .lab-title {{font-size: 2rem; font-weight: 600; letter-spacing: -0.015em; margin: 0; color: {INK};}}
  .lab-lede {{margin: .35rem 0 1.1rem 0; color: {MUTED}; max-width: 46rem; line-height: 1.5;}}
  [data-testid="stHeading"] h3 {{font-size: 1.2rem; padding-top: 1.1rem;}}
  .note {{font-size: .88rem; color: {MUTED}; line-height: 1.5; margin: 0 0 .9rem 0;}}

  /* one bordered strip, a column per model */
  .strip {{display: grid; border: 1px solid {RULE}; border-radius: 8px; overflow: hidden;}}
  .strip > div {{padding: 16px 18px 14px 18px; border-left: 1px solid {RULE};}}
  .strip > div:first-child {{border-left: 0;}}
  .strip .model {{font-size: .86rem; font-weight: 500; color: {MUTED};}}
  .strip .verdict {{font-size: 1.5rem; font-weight: 600; margin: 2px 0 8px 0; line-height: 1.25;}}
  .strip .meter {{height: 6px; border-radius: 3px; background: #E7EAF1; overflow: hidden;}}
  .strip .meter > div {{height: 100%;}}
  .strip .fact {{font-size: .84rem; color: {MUTED}; margin-top: 8px; line-height: 1.45;}}
  .strip .fact b {{color: {INK}; font-weight: 500;}}
  .flag {{display: inline-block; font-size: .74rem; font-weight: 500; padding: 1px 8px; border-radius: 10px;
          background: #FFF1CC; color: #7A5200; margin-top: 10px;}}
  @media (max-width: 820px) {{
    .strip {{grid-template-columns: 1fr 1fr !important;}}
    .strip > div:nth-child(odd) {{border-left: 0;}}
    .strip > div:nth-child(n+3) {{border-top: 1px solid {RULE};}}
  }}

  /* length ruler */
  .ruler {{margin: 34px 0 6px 0;}}
  .ruler .track {{position: relative; height: 30px;}}
  .ruler .seg {{position: absolute; top: 0; bottom: 0; font-size: .78rem; font-weight: 500; color: {INK};
                display: flex; align-items: center; padding-left: 8px; white-space: nowrap; overflow: hidden;}}
  .ruler .seg.cut {{background: repeating-linear-gradient(135deg, rgba({NEG_RGB},.20) 0 5px, rgba({NEG_RGB},.07) 5px 10px);
                    color: #7A1F41; justify-content: flex-end; padding: 0 8px 0 0;}}
  .ruler .mark {{position: absolute; top: -8px; bottom: -8px; width: 2px; background: {INK};}}
  .ruler .mark span {{position: absolute; bottom: 100%; margin-bottom: 2px; font-size: .82rem; font-weight: 600;
                      white-space: nowrap; color: {INK};}}
  .ruler .ticks {{position: relative; height: 20px; margin-top: 4px;}}
  .ruler .ticks span {{position: absolute; transform: translateX(-50%); font-size: .76rem; color: {MUTED};}}

  /* token views */
  .tokens {{font-family: 'IBM Plex Mono', Menlo, monospace; font-size: .84rem; line-height: 2.0;}}
  .tok {{padding: 2px 3px; border-radius: 3px;}}
  .tokenbox {{border: 1px solid {RULE}; border-radius: 8px; padding: 14px 16px;}}
  .unseen {{color: #9AA1B2; text-decoration: line-through; text-decoration-color: #C5CAD6;}}
  .cutmark {{display: inline-block; padding: 0 8px; margin: 0 4px; border-radius: 3px; background: {NEG}; color: #fff;
             font-family: {FONT}; font-size: .74rem; font-weight: 500;}}
  .key span {{display: inline-block; padding: 1px 8px; border-radius: 3px; margin-right: 6px; font-size: .82rem;}}

  /* share bar */
  .share {{display: flex; height: 30px; border-radius: 4px; overflow: hidden; margin: 6px 0 4px 0;}}
  .share div {{display: flex; align-items: center; padding: 0 10px; color: #fff; font-size: .84rem; font-weight: 500;
               white-space: nowrap; overflow: hidden;}}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading the four models")
def get_pipeline():
    from src.inference import SentimentPipeline

    return SentimentPipeline("cpu")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    p = ROOT / path
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def long_test_examples(n: int = 40) -> pd.DataFrame:
    df = pd.read_csv(C.TEST_CSV, usecols=["text", "label", "token_count", "truncated"])
    return df[df.truncated].sample(n=min(n, int(df.truncated.sum())), random_state=C.SEED).reset_index(drop=True)


@st.cache_data
def demo_examples() -> pd.DataFrame:
    return pd.read_csv(C.DEMO_DIR / "demo_reviews.csv")


pipe = get_pipeline()
TH = json.loads((C.RESULTS_DIR / "length_thresholds.json").read_text())
Q25, MEDIAN, Q75 = int(TH["q25"]), int(TH["median"]), int(TH["q75"])
QUARTILES = {"Q1": f"{Q25} tokens or fewer", "Q2": f"{Q25 + 1} to {MEDIAN} tokens",
             "Q3": f"{MEDIAN + 1} to {Q75} tokens", "Q4": f"more than {Q75} tokens"}
GROUPS = load_csv("results/tables/metrics_by_length_group.csv")
STATS = load_csv("results/tables/dataset_stats.csv")
LONGEST = int(STATS.max_tokens.max()) if STATS is not None else 250


def quartile_of(n_tokens: int) -> str:
    return "Q1" if n_tokens <= Q25 else "Q2" if n_tokens <= MEDIAN else "Q3" if n_tokens <= Q75 else "Q4"


def accuracy_at(model: str, quartile: str) -> float | None:
    """Test accuracy of `model` on reviews in `quartile` (from results/tables)."""
    if GROUPS is None:
        return None
    row = GROUPS[(GROUPS.model == model) & (GROUPS.grouping == "quartile") & (GROUPS.group == quartile)]
    return float(row.accuracy.iloc[0]) if len(row) else None


# --------------------------------------------------------------------------- html pieces
def length_ruler(n_tokens: int) -> str:
    """A ruler from 0 to the longest review, split into the four length quartiles and the
    part the RNN / BiLSTM never read, with a marker at this review's length."""
    span = max(LONGEST, n_tokens)
    pct = lambda v: 100 * v / span  # noqa: E731
    segs = [("Q1", 0, Q25, .09), ("Q2", Q25, MEDIAN, .17), ("Q3", MEDIAN, Q75, .27), ("Q4", Q75, pipe.max_len, .39)]
    parts = [f'<div class="seg" style="left:{pct(a):.2f}%;width:{pct(b - a):.2f}%;background:rgba(43,58,140,{o})">{q}</div>'
             for q, a, b, o in segs]
    parts.append(f'<div class="seg cut" style="left:{pct(pipe.max_len):.2f}%;width:{pct(span - pipe.max_len):.2f}%">'
                 "not read by RNN or BiLSTM</div>")
    at = pct(min(n_tokens, span))
    anchor = "left:0" if at < 10 else "right:0" if at > 90 else "left:50%;transform:translateX(-50%)"
    word = "token" if n_tokens == 1 else "tokens"
    parts.append(f'<div class="mark" style="left:calc({at:.2f}% - 1px)"><span style="{anchor}">{n_tokens} {word}</span></div>')
    ticks = "".join(f'<span style="left:{pct(v):.2f}%">{v}</span>' for v in (Q25, MEDIAN, Q75, pipe.max_len))
    return f'<div class="ruler"><div class="track">{"".join(parts)}</div><div class="ticks">{ticks}</div></div>'


def verdict_strip(preds, facts: dict[str, str] | None = None, flags: bool = True) -> str:
    cells = []
    for p in preds:
        col = POS if p.label == "positive" else NEG
        thr = pipe.flag_thresholds.get(p.model) if flags else None
        flag = '<div class="flag">Needs human review</div>' if thr and p.confidence < thr else ""
        extra = f'<div class="fact">{facts[p.model]}</div>' if facts and facts.get(p.model) else ""
        cells.append(
            f'<div><div class="model">{html.escape(p.name)}</div>'
            f'<div class="verdict" style="color:{col}">{p.label.title()}</div>'
            f'<div class="meter"><div style="width:{p.confidence * 100:.1f}%;background:{col}"></div></div>'
            f'<div class="fact">Confidence <b>{p.confidence:.1%}</b></div>{extra}{flag}</div>')
    return f'<div class="strip" style="grid-template-columns:repeat({len(cells)},1fr)">{"".join(cells)}</div>'


def highlight_html(pairs: list[tuple[str, float]]) -> str:
    scale = max((abs(s) for _, s in pairs), default=1.0) or 1.0
    spans = []
    for tok, s in pairs:
        alpha = min(abs(s) / scale, 1.0) * 0.6
        rgb = POS_RGB if s > 0 else NEG_RGB
        spans.append(f'<span class="tok" style="background:rgba({rgb},{alpha:.2f})" title="{s:+.4f}">{html.escape(tok)}</span>')
    return f'<div class="tokens tokenbox">{" ".join(spans)}</div>'


def share_bar(pos_share: float) -> str:
    return (f'<div class="share"><div style="width:{pos_share:.1%};background:{POS}">{pos_share:.1%} positive</div>'
            f'<div style="width:{1 - pos_share:.1%};background:{NEG};justify-content:flex-end">'
            f'{1 - pos_share:.1%} negative</div></div>')


# --------------------------------------------------------------------------- charts
def styled(fig: go.Figure, height: int, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=64 if title else 28, b=8),
        title=dict(text=title, x=0, xanchor="left", y=.98, yanchor="top", font=dict(size=15)) if title else None,
        showlegend=legend, legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", title=None),
        hoverlabel=dict(font=dict(family=FONT)), bargap=.3)
    fig.update_xaxes(gridcolor="#E7EAF1", zeroline=False, linecolor=RULE, title_font=dict(size=12, color=MUTED))
    fig.update_yaxes(gridcolor="#E7EAF1", zeroline=False, linecolor=RULE, title_font=dict(size=12, color=MUTED))
    return fig


def show(fig: go.Figure, where=st) -> None:
    where.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


def word_bars(pairs: list[tuple[str, float]], k: int = 6) -> go.Figure:
    """The k words that pushed hardest towards positive and towards negative."""
    total: dict[str, float] = {}
    for tok, s in pairs:
        total[tok] = total.get(tok, 0.0) + s
    s = pd.Series(total)
    s = pd.concat([s[s > 0].nlargest(k), s[s < 0].nsmallest(k)]).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=[POS if v > 0 else NEG for v in s.values],
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(title_text="pushes negative  /  pushes positive")
    return styled(fig, max(170, 26 * len(s) + 70), legend=False)


def dumbbell(df: pd.DataFrame, metric: str, split: str, ends: dict[str, str], title: str) -> go.Figure:
    """One row per model: two dots (the two groups in `ends`, value -> colour) joined by a line."""
    order = [m for m in MODEL_COLORS if m in set(df.model_name)][::-1]
    fig = go.Figure()
    for m in order:
        vals = df[df.model_name == m][metric]
        fig.add_trace(go.Scatter(x=[vals.min(), vals.max()], y=[m, m], mode="lines",
                                 line=dict(color="#B9C0CF", width=3), hoverinfo="skip", showlegend=False))
    for group, colour in ends.items():
        d = df[df[split] == group].set_index("model_name").reindex(order)
        left = d[metric] <= df[df.model_name.isin(order)].groupby("model_name")[metric].mean().reindex(order)
        fig.add_trace(go.Scatter(
            x=d[metric], y=order, mode="markers+text", name=group, marker=dict(color=colour, size=13),
            text=[f"{v:.3f}" for v in d[metric]], textfont=dict(size=12, color=INK),
            textposition=["middle left" if is_left else "middle right" for is_left in left],
            hovertemplate=f"%{{y}}, {group}: %{{x:.3f}}<extra></extra>"))
    lo, hi = df[metric].min(), df[metric].max()
    pad = (hi - lo) * .22 + .01
    fig.update_xaxes(range=[lo - pad, hi + pad], title_text=METRICS[metric], tickformat=".2f")
    fig.update_yaxes(showgrid=False)
    return styled(fig, 300, title)


# --------------------------------------------------------------------------- header
st.markdown('<div class="lab-title">Review Sentiment Lab</div>'
            '<div class="lab-lede">Four models label Amazon product reviews as positive or negative. '
            'The lab shows where they agree, which words decide the result, and how accuracy changes '
            'as reviews get longer.</div>', unsafe_allow_html=True)

tabs = st.tabs(["Live Analyzer", "Explain", "Truncation Visualizer", "Length Lab", "Review Dashboard"])

DEFAULT_TEXT = ("Not what I expected. The headphones look nice and the sound is okay at first, but after "
                "two weeks the left side stopped working. Customer service never answered my emails. "
                "I would not recommend them.")
if "text" not in st.session_state:
    st.session_state.text = DEFAULT_TEXT

# =========================================================================== 1. Live Analyzer
with tabs[0]:
    left, right = st.columns([4, 1], vertical_alignment="bottom")
    with right:
        if st.button("Load a test review", width="stretch", help="A random review from the held-out test set"):
            row = demo_examples().sample(1).iloc[0]
            st.session_state.text = row.review
            st.session_state.true_label = "positive" if row.label == 1 else "negative"
        if st.button("Reset", width="stretch"):
            st.session_state.text = DEFAULT_TEXT
            st.session_state.pop("true_label", None)
    with left:
        text = st.text_area("Review", key="text", height=140)

    if text.strip():
        _, toks = pipe.prepare(text)
        n, quartile = len(toks), quartile_of(len(toks))
        preds = pipe.predict(text)

        st.markdown(length_ruler(n), unsafe_allow_html=True)
        where = f"This review is in {quartile}, the length group of {QUARTILES[quartile]}."
        if n > pipe.max_len:
            where += (f" The simple RNN and the BiLSTM stop reading at {pipe.max_len} tokens, so they miss the last "
                      f"{n - pipe.max_len}. Open the Truncation Visualizer to see which words.")
        st.markdown(f'<div class="note">{where}</div>', unsafe_allow_html=True)

        facts = {}
        for p in preds:
            acc = accuracy_at(p.model, quartile)
            if acc is not None:
                facts[p.model] = f"Right on <b>{acc:.1%}</b> of {quartile} test reviews"
        st.markdown(verdict_strip(preds, facts), unsafe_allow_html=True)

        n_pos = sum(p.label == "positive" for p in preds)
        agree = ("All four models agree." if n_pos in (0, 4)
                 else f"The models disagree: {n_pos} say positive and {4 - n_pos} say negative.")
        if "true_label" in st.session_state:
            agree += f" The dataset labels this review {st.session_state.true_label}."
        st.markdown(f'<div class="note" style="margin-top:12px">{agree} A prediction is marked for human review when '
                    "its confidence is below that model's threshold. Each threshold was chosen on the validation "
                    "set so that the predictions above it are 95% accurate.</div>", unsafe_allow_html=True)
    else:
        st.info("Type or paste a review above, or load one from the test set.")

# =========================================================================== 2. Explain
with tabs[1]:
    text = st.session_state.text
    if not text.strip():
        st.info("Enter a review in the Live Analyzer first. This tab explains that review.")
    else:
        st.markdown("Which words decided the result for the review in the Live Analyzer. Darker shading means a "
                    "stronger push. Hover over a word to see its score. "
                    f"<span class='key'><span style='background:rgba({POS_RGB},.45)'>pushes positive</span>"
                    f"<span style='background:rgba({NEG_RGB},.45)'>pushes negative</span></span>",
                    unsafe_allow_html=True)
        lr_pairs = pipe.lr_contributions(text)
        with st.spinner("Running Integrated Gradients"):
            ig_pairs, delta = pipe.lstm_attributions(text)
        preds = {p.model: p for p in pipe.predict(text)}
        c1, c2 = st.columns(2, gap="large")
        with c1:
            p = preds["tfidf_lr"]
            st.subheader("TF-IDF + LogReg", anchor=False)
            st.caption(f"Predicts {p.label} with {p.confidence:.1%} confidence. A word's score is its TF-IDF value "
                       "times the model's coefficient. A word pair's score is split between its two words.")
            st.markdown(highlight_html(lr_pairs), unsafe_allow_html=True)
            show(word_bars(lr_pairs))
        with c2:
            p = preds["lstm"]
            st.subheader("BiLSTM", anchor=False)
            st.caption(f"Predicts {p.label} with {p.confidence:.1%} confidence. Scores come from Integrated "
                       f"Gradients (Captum) on the embedding layer: 50 steps from a baseline of unknown-word "
                       f"tokens, convergence error {delta:.3f}.")
            st.markdown(highlight_html(ig_pairs), unsafe_allow_html=True)
            if len(ig_pairs) < len(lr_pairs):
                st.caption(f"Only the first {len(ig_pairs)} tokens appear here. The BiLSTM never read the rest.")
            show(word_bars(ig_pairs))

# =========================================================================== 3. Truncation Visualizer
with tabs[2]:
    st.markdown(f"The simple RNN and the BiLSTM read only the first {pipe.max_len} tokens of a review, the 95th "
                f"percentile of training lengths. The chunked model reads everything instead: it scores chunks of "
                f"up to {C.CHUNK_TOKENS} tokens and averages them, weighted by confidence.")
    longs = long_test_examples()
    mode = st.segmented_control("Review to inspect", ["A long test review", "The Live Analyzer review"],
                                default="A long test review")
    if mode == "The Live Analyzer review":
        t_text = st.session_state.text
    else:
        i = st.slider("Example", 1, len(longs), 1) - 1
        t_text = longs.text[i]
        st.caption(f"The dataset labels this review {'positive' if longs.label[i] == 1 else 'negative'}.")

    _, toks = pipe.prepare(t_text)
    if not toks:
        st.info("This review has no words to show.")
    else:
        seen, unseen = toks[: pipe.max_len], toks[pipe.max_len:]
        st.markdown(length_ruler(len(toks)), unsafe_allow_html=True)
        if unseen:
            st.markdown(f'<div class="note">The RNN and the BiLSTM miss the last {len(unseen)} tokens, '
                        f"{len(unseen) / len(toks):.0%} of this review. They are struck through below.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="note">This review fits within the limit, so nothing is cut.</div>',
                        unsafe_allow_html=True)
        body = " ".join(html.escape(t) for t in seen)
        if unseen:
            body += (f' <span class="cutmark">cut at {pipe.max_len}</span> '
                     f'<span class="unseen">{" ".join(html.escape(t) for t in unseen)}</span>')
        st.markdown(f'<div class="tokens tokenbox">{body}</div>', unsafe_allow_html=True)

        chunks, cprobs, cweights, agg = pipe.chunk_view(t_text)
        preds = {p.model: p for p in pipe.predict(t_text)}
        st.write("")
        st.markdown(verdict_strip([preds["lstm"], preds["chunked_lstm"]],
                                  {"lstm": f"Read {len(seen)} of {len(toks)} tokens",
                                   "chunked_lstm": f"Read all {len(toks)} tokens in {len(chunks)} "
                                                   f"chunk{'s' if len(chunks) != 1 else ''}"}, flags=False),
                    unsafe_allow_html=True)

        st.subheader("How each chunk was scored", anchor=False)
        labels = [f"Chunk {k + 1}" for k in range(len(chunks))]
        fig = go.Figure(go.Bar(x=labels, y=cprobs - 0.5, base=0.5, marker_color=[POS if p >= .5 else NEG for p in cprobs],
                               customdata=np.stack([cprobs, cweights], axis=1),
                               hovertemplate="%{x}<br>P(positive) %{customdata[0]:.3f}<br>"
                                             "weight %{customdata[1]:.3f}<extra></extra>"))
        fig.add_hline(y=0.5, line_color=INK, line_width=1)
        fig.update_yaxes(range=[0, 1], title_text="P(positive)", tickformat=".1f")
        show(styled(fig, 240, legend=False))
        st.dataframe(
            pd.DataFrame({"Chunk": labels, "Text": [" ".join(c) for c in chunks], "Tokens": [len(c) for c in chunks],
                          "P(positive)": cprobs, "Weight": cweights,
                          "Verdict": ["Positive" if p >= .5 else "Negative" for p in cprobs]}),
            width="stretch", hide_index=True,
            column_config={"P(positive)": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                           "Weight": st.column_config.NumberColumn(format="%.3f"),
                           "Text": st.column_config.TextColumn(width="large")})
        st.caption("The further a chunk's score is from 0.5, the more weight it gets, so confident chunks count for more.")

# =========================================================================== 4. Length Lab
with tabs[3]:
    overall = load_csv("results/tables/metrics_overall.csv")
    if GROUPS is None or overall is None:
        st.warning("No results yet. Run `python main.py` to create the tables in results/tables/.")
    else:
        n_test = int(overall.n.iloc[0])
        names = list(GROUPS.model_name.unique())
        c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
        metric = c1.segmented_control("Metric", list(METRICS), default="f1", format_func=METRICS.get) or "f1"
        chosen_models = c2.pills("Models", names, default=names, selection_mode="multi") or names

        q = GROUPS[(GROUPS.grouping == "quartile") & GROUPS.model_name.isin(chosen_models)]
        drops = []
        for m in chosen_models:
            v = q[q.model_name == m].set_index("group")[metric]
            drops.append(f"{abs(v['Q4'] - v['Q1']) * 100:.1f} points for {m}" if v["Q4"] < v["Q1"]
                         else f"nothing for {m}")
        st.markdown(f"On the {n_test:,} test reviews, {METRICS[metric]} from the shortest quarter of reviews to the "
                    f"longest falls by {', '.join(drops)}.")

        fig = go.Figure()
        ticks = [f"{g}<br><span style='color:{MUTED}'>{QUARTILES[g]}</span>" for g in QUARTILES]
        for m in chosen_models:
            d = q[q.model_name == m].set_index("group").reindex(list(QUARTILES))
            fig.add_trace(go.Scatter(x=ticks, y=d[metric], name=m, mode="lines+markers",
                                     line=dict(color=MODEL_COLORS[m], width=2.5, dash=MODEL_DASH.get(m, "solid")),
                                     marker=dict(size=8), customdata=d[["n", "pct_pos"]],
                                     hovertemplate=f"{m}: %{{y:.3f}}<br>%{{customdata[0]:,}} reviews, "
                                                   "%{customdata[1]:.0f}% positive<extra></extra>"))
        fig.update_yaxes(title_text=METRICS[metric], tickformat=".2f")
        show(styled(fig, 400, f"{METRICS[metric]} by length quartile"))

        c3, c4 = st.columns(2, gap="large")
        tw = GROUPS[(GROUPS.grouping == "two_way") & GROUPS.model_name.isin(chosen_models)]
        tw = tw.assign(group=tw.group.map({"short": f"{MEDIAN} tokens or fewer", "long": f"More than {MEDIAN} tokens"}))
        show(dumbbell(tw, metric, "group", {f"{MEDIAN} tokens or fewer": "#9DB2E8", f"More than {MEDIAN} tokens": NAVY},
                      "Short reviews against long reviews"), c3)
        trunc = load_csv("results/tables/truncation_effect.csv")
        if trunc is not None:
            fits, cut = f"Fits in {pipe.max_len} tokens", f"Cut at {pipe.max_len} tokens"
            trunc = trunc.assign(model_name=trunc.model.map(C.MODEL_NAMES),
                                 subset=np.where(trunc.subset.str.startswith("truncated"), cut, fits))
            trunc = trunc[trunc.model_name.isin(chosen_models)]
            show(dumbbell(trunc, metric, "subset", {fits: "#B9C0CF", cut: NEG},
                          "Long reviews: those that fit against those cut off"), c4)
            n_cut = int(trunc[trunc.subset == cut].n.iloc[0]) if len(trunc) else 0
            c4.caption(f"{n_cut:,} test reviews ({n_cut / n_test:.1%}) are longer than {pipe.max_len} tokens. "
                       "TF-IDF and the chunked model read them in full.")

        st.subheader(f"Overall results on {n_test:,} test reviews", anchor=False)
        st.dataframe(
            overall.rename(columns={"model_name": "Model", **METRICS})[["Model", "Accuracy", "Precision", "Recall", "F1"]],
            width="stretch", hide_index=True,
            column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("Accuracy", "Precision", "Recall", "F1")})

        flags, cal = load_csv("results/tables/human_review_flagging.csv"), load_csv("results/tables/calibration_summary.csv")
        if flags is not None and cal is not None:
            st.subheader("Confidence and human review", anchor=False)
            t = flags.merge(cal, on=["model", "model_name"])
            st.dataframe(
                pd.DataFrame({"Model": t.model_name, "Confidence threshold": t.threshold_from_val,
                              "Reviews flagged": t.test_pct_flagged, "Accuracy, all reviews": t.test_acc_all,
                              "Accuracy, not flagged": t.test_acc_unflagged, "Accuracy, flagged": t.test_acc_flagged,
                              "Calibration error (ECE)": t.ECE}),
                width="stretch", hide_index=True,
                column_config={"Reviews flagged": st.column_config.NumberColumn(format="%.1f%%"),
                               **{c: st.column_config.NumberColumn(format="%.3f") for c in
                                  ("Confidence threshold", "Accuracy, all reviews", "Accuracy, not flagged",
                                   "Accuracy, flagged", "Calibration error (ECE)")}})
            target = float(t.target_val_accuracy.iloc[0])
            missed = t[~t.target_reached_on_val].model_name.tolist()
            note = (f"Each threshold was chosen on the validation set so that the predictions above it are "
                    f"{target:.0%} accurate, then applied unchanged to the test set.")
            if missed:
                note += (f" {' and '.join(missed)} cannot reach that target, so its threshold keeps only the most "
                         "confident half of its predictions.")
            st.caption(note)

# =========================================================================== 5. Review Dashboard
with tabs[4]:
    u1, u2, u3 = st.columns([2, 1, 1])
    up = u1.file_uploader("Reviews to analyse (CSV, one review per row)", type=["csv"])
    raw = pd.read_csv(up) if up is not None else demo_examples()
    text_cols = [c for c in raw.columns
                 if pd.api.types.is_string_dtype(raw[c]) or pd.api.types.is_object_dtype(raw[c])]
    if not text_cols:
        st.error("This file has no text column. Upload a CSV with one column that holds the review text.")
        st.stop()
    default_col = "review" if "review" in text_cols else text_cols[0]
    col = u2.selectbox("Column with the review text", text_cols, index=text_cols.index(default_col))
    model_key = u3.selectbox("Model", list(C.MODEL_NAMES), index=0, format_func=lambda m: C.MODEL_NAMES[m])
    if up is None:
        u1.caption("No file uploaded, so this shows 500 real reviews from the held-out test set.")

    @st.cache_data(show_spinner="Scoring the reviews")
    def analyse(texts: tuple[str, ...], model: str):
        cleans = [clean_text(t) for t in texts]
        probs = pipe.predict_batch(cleans, model)
        sent_rows = []
        for i, c in enumerate(cleans):
            for s in split_sentences(c):
                for a in aspects_in(s):
                    sent_rows.append({"review_id": i, "aspect": a, "sentence": s})
        sents = pd.DataFrame(sent_rows, columns=["review_id", "aspect", "sentence"])
        if len(sents):
            uniq = sents.sentence.unique().tolist()
            sp = dict(zip(uniq, pipe.predict_batch(uniq, model)))
            sents["prob"] = sents.sentence.map(sp)
            sents["sentiment"] = np.where(sents.prob >= .5, "positive", "negative")
        return cleans, probs, sents

    MAX_ROWS = 5000
    texts = tuple(raw[col].fillna("").astype(str).tolist()[:MAX_ROWS])
    if len(raw) > MAX_ROWS:
        st.caption(f"This file has {len(raw):,} rows. Only the first {MAX_ROWS:,} are analysed.")
    cleans, probs, sents = analyse(texts, model_key)
    preds = (probs >= .5).astype(int)
    toks = [tokenize(c) for c in cleans]
    df = pd.DataFrame({"Review": texts, "P(positive)": probs,
                       "Sentiment": np.where(preds == 1, "Positive", "Negative"),
                       "Tokens": [len(t) for t in toks]})

    g1, g2 = st.columns([2, 3], gap="large")
    with g1:
        st.subheader(f"{len(df):,} reviews", anchor=False)
        st.markdown(share_bar(float(preds.mean())), unsafe_allow_html=True)
        summary = f"Half of the reviews are {int(np.median(df.Tokens))} tokens or shorter."
        if "label" in raw.columns and raw["label"].isin([0, 1]).all():
            acc = (preds == raw["label"].values[: len(preds)]).mean()
            summary += f" The file includes true labels, and {C.MODEL_NAMES[model_key]} matches {acc:.1%} of them."
        st.markdown(f'<div class="note">{summary}</div>', unsafe_allow_html=True)
    with g2:
        fig = go.Figure()
        for name, colour in (("Positive", POS), ("Negative", NEG)):
            fig.add_trace(go.Histogram(x=df.Tokens[df.Sentiment == name], name=name, marker_color=colour,
                                       xbins=dict(start=0, size=10),
                                       hovertemplate=f"{name}: %{{y}} reviews of %{{x}} tokens<extra></extra>"))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title_text="Review length (tokens)")
        fig.update_yaxes(title_text="Reviews")
        show(styled(fig, 250, "Review length"))

    st.subheader("What the reviews talk about", anchor=False)
    st.caption("A sentence counts towards an aspect when it contains one of that aspect's keywords. The selected "
               "model then labels the sentence. Choose an aspect to filter the list of reviews below.")
    counts = (sents.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
              if len(sents) else pd.DataFrame(index=pd.Index([], name="aspect")))
    counts = counts.reindex(columns=["positive", "negative"], fill_value=0).reindex(list(ASPECTS), fill_value=0)
    chosen = st.pills("Aspect", list(ASPECTS), selection_mode="single", label_visibility="collapsed",
                      format_func=lambda a: f"{a} ({int(counts.loc[a].sum())})")
    if counts.values.sum():
        cc = counts.assign(total=counts.sum(axis=1)).sort_values("total")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=cc.index, x=-cc.negative, orientation="h", name="Negative sentences", marker_color=NEG,
                             customdata=cc.negative, hovertemplate="%{y}: %{customdata} negative<extra></extra>"))
        fig.add_trace(go.Bar(y=cc.index, x=cc.positive, orientation="h", name="Positive sentences", marker_color=POS,
                             hovertemplate="%{y}: %{x} positive<extra></extra>"))
        fig.add_vline(x=0, line_color=INK, line_width=1)
        reach = int(max(cc.negative.max(), cc.positive.max()))
        step = max(1, round(reach / 4 / 5) * 5) if reach >= 10 else max(1, reach // 2)
        tick_vals = list(range(-step * (reach // step + 1), step * (reach // step + 1) + 1, step))
        fig.update_layout(barmode="relative")
        fig.update_xaxes(tickvals=tick_vals, ticktext=[str(abs(v)) for v in tick_vals], title_text="Sentences")
        fig.update_yaxes(showgrid=False)
        show(styled(fig, 310))
    else:
        st.info("None of these reviews mention the aspects this dashboard looks for.")

    st.subheader("Words that set positive and negative reviews apart", anchor=False)
    tp, tn = top_ngrams(toks, preds, k=12)
    n1, n2 = st.columns(2, gap="large")
    for col_, data, name, colour in ((n1, tp, "positive", POS), (n2, tn, "negative", NEG)):
        if not data:
            col_.info(f"Too few {name} reviews to compare.")
            continue
        d = pd.DataFrame(data, columns=["phrase", "z", "reviews"])
        d["phrase"] = d.phrase.replace({"n't": "n't (not)"})
        fig = go.Figure(go.Bar(x=d.z.abs(), y=d.phrase, orientation="h", marker_color=colour, customdata=d.reviews,
                               hovertemplate="%{y}: in %{customdata} reviews<extra></extra>"))
        fig.update_xaxes(title_text="How distinctive (log-odds z-score)")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(styled(fig, 380, f"More common in {name} reviews", legend=False), col_)

    view = df
    if chosen and len(sents):
        s = sents[sents.aspect == chosen]
        about = s.groupby("review_id").sentiment.agg(lambda x: " and ".join(sorted(set(x))).capitalize())
        view = df.loc[about.index].assign(**{f"{chosen} sentences": about.values})
    st.subheader(f"{len(view):,} reviews that mention {chosen.lower()}" if chosen else "All reviews", anchor=False)
    st.dataframe(view, width="stretch", height=380, hide_index=True,
                 column_config={"P(positive)": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                                "Review": st.column_config.TextColumn(width="large")})
    buf = io.StringIO()
    view.to_csv(buf, index=False)
    st.download_button("Download these reviews as CSV", buf.getvalue(), "scored_reviews.csv", "text/csv")

if STATS is not None:
    sizes = STATS.set_index("split").n
    st.markdown(f'<div class="note" style="margin-top:2.5rem;border-top:1px solid {RULE};padding-top:.8rem">'
                f"Data: Amazon Polarity (Zhang, Zhao and LeCun, 2015). The models were trained on "
                f"{int(sizes['train']):,} reviews, tuned on {int(sizes['val']):,} and tested on "
                f"{int(sizes['test']):,}.</div>", unsafe_allow_html=True)

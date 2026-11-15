"""Text cleaning, tokenisation and sentence/chunk splitting.

Design choices
--------------
* Lower-case, strip HTML tags/entities and URLs, normalise whitespace.
* Negations are preserved: we never remove stop-words, and contractions such as
  "don't" are split into "do" + "n't" so the negation survives as its own token.
* "!" and "?" are kept as tokens because they carry sentiment intensity.
* Other punctuation is dropped from the token stream (but kept in the cleaned
  text so that sentences can still be split for the chunked model).
"""
from __future__ import annotations

import html
import re
from typing import List

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_WS = re.compile(r"\s+")
# "can't" -> "ca n't", "won't" -> "wo n't" (Penn-Treebank style)
_NT = re.compile(r"n't\b")
# Token = word (letters/digits, optional internal apostrophe) | n't | ! | ?
_TOKEN = re.compile(r"n't|[a-z0-9]+(?:'[a-z]+)?|[!?]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

NEGATIONS = {"not", "no", "never", "n't", "nor", "none", "nothing", "nobody", "neither", "without"}


def clean_text(text: str) -> str:
    """Lower-case and remove HTML/URLs; keep punctuation for sentence splitting."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    # normalise curly apostrophes so "don’t" is handled like "don't"
    text = text.replace("’", "'").replace("‘", "'")
    text = _WS.sub(" ", text).strip()
    return text


def tokenize(clean: str) -> List[str]:
    """Tokenise an already-cleaned string (see clean_text)."""
    clean = _NT.sub(" n't", clean)
    return _TOKEN.findall(clean)


def combine_title_content(title: str, content: str) -> str:
    """Join title and body; add a full stop so the title becomes its own sentence."""
    title = (title or "").strip()
    content = (content or "").strip()
    if title and title[-1] not in ".!?":
        title += "."
    return f"{title} {content}".strip()


def split_sentences(clean: str) -> List[str]:
    """Very light sentence splitter on . ! ? and new-lines."""
    parts = [s.strip() for s in _SENT_SPLIT.split(clean)]
    return [p for p in parts if p and tokenize(p)]


def make_chunks(clean: str, max_tokens: int = 50) -> List[List[str]]:
    """Split a review into token chunks of at most ``max_tokens`` tokens.

    Consecutive sentences are merged greedily while they fit; a single sentence
    longer than ``max_tokens`` is split into fixed windows. Returns a list of
    token lists (always at least one chunk for non-empty input).
    """
    chunks: List[List[str]] = []
    current: List[str] = []
    for sent in split_sentences(clean):
        toks = tokenize(sent)
        # long sentence -> flush current, then cut into windows
        if len(toks) > max_tokens:
            if current:
                chunks.append(current)
                current = []
            for i in range(0, len(toks), max_tokens):
                chunks.append(toks[i : i + max_tokens])
            continue
        if len(current) + len(toks) > max_tokens and current:
            chunks.append(current)
            current = []
        current = current + toks
    if current:
        chunks.append(current)
    if not chunks:
        toks = tokenize(clean)
        chunks = [toks] if toks else []
    return chunks


def split_tokens(s: str) -> List[str]:
    """Analyzer for the TF-IDF vectoriser: input is our space-joined token string.
    Lives here (not in a script) so the pickled pipeline can be loaded from anywhere."""
    return s.split()

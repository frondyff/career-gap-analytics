from __future__ import annotations

import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_bigrams(text: str) -> list[str]:
    toks = TOKEN_RE.findall(clean_text(text))
    return [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]


def lexicon_score(text: str, hard_weights: dict[str, float], soft_weights: dict[str, float]) -> dict[str, float]:
    phrases = extract_bigrams(text)
    hard_score = sum(float(hard_weights.get(p, 0.0)) for p in phrases)
    soft_score = sum(float(soft_weights.get(p, 0.0)) for p in phrases)
    n = len(phrases)
    hard_norm = hard_score / (n + 1)
    soft_norm = soft_score / (n + 1)
    idx = hard_norm - soft_norm

    if idx > 0.01:
        c = "hard_dominant"
    elif idx < -0.01:
        c = "soft_dominant"
    else:
        c = "balanced"

    return {
        "num_phrases": n,
        "hard_score": hard_score,
        "soft_score": soft_score,
        "hard_norm": hard_norm,
        "soft_norm": soft_norm,
        "desirability_index": idx,
        "desirability_class": c,
    }


def top_matching_phrases(text: str, lexicon: dict[str, float], top_n: int = 8) -> list[tuple[str, float]]:
    counts = Counter(extract_bigrams(text))
    scored = []
    for p, c in counts.items():
        if p in lexicon:
            scored.append((p, lexicon[p] * c))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

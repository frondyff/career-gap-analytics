import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag


def parse_subreddit_from_filename(path: Path) -> str:
    m = re.match(r"reddit_(.+?)_(1000_posts|top_10_comments)_", path.name)
    return m.group(1) if m else "unknown"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str, lemmatizer, stop_words) -> str:
    text = clean_text(text)
    toks = TOKEN_RE.findall(text)
    toks = [lemmatizer.lemmatize(t) for t in toks]
    toks = [t for t in toks if t not in stop_words and len(t) > 2]
    return " ".join(toks)


def extract_phrases_from_tokens(tokens):
    if not tokens:
        return []
    tagged = pos_tag(tokens)
    phrases = []
    for i in range(len(tagged) - 1):
        (w1, p1), (w2, p2) = tagged[i], tagged[i + 1]
        if p1.startswith("JJ") and p2.startswith("NN"):
            phrases.append(f"{w1} {w2}")
        if p1.startswith("RB") and p2.startswith("JJ"):
            phrases.append(f"{w1} {w2}")
        if p1.startswith("NN") and p2.startswith("NN"):
            phrases.append(f"{w1} {w2}")
    return phrases


def document_frequency_terms(clean_series: pd.Series):
    n_docs = len(clean_series)
    df_counts = Counter()
    for txt in clean_series.fillna(""):
        df_counts.update(set(txt.split()))
    out = pd.DataFrame(
        {"term": list(df_counts.keys()), "doc_freq": list(df_counts.values())}
    )
    out["df_ratio"] = out["doc_freq"] / max(n_docs, 1)
    return out.sort_values("df_ratio", ascending=False)


def pmi_with_smoothing(doc_sets, class_mask, term, alpha=1.0):
    n_docs = len(doc_sets)
    t_mask = np.array([term in s for s in doc_sets])
    n_t = int(t_mask.sum())
    n_c = int(class_mask.sum())
    n_tc = int((t_mask & class_mask).sum())

    def sp(count, total):
        return (count + alpha) / (total + 2 * alpha)

    p_t = sp(n_t, n_docs)
    p_c = sp(n_c, n_docs)
    p_tc = sp(n_tc, n_docs)
    return float(np.log(p_tc / (p_t * p_c)))


def top_terms_from_centroids(kmeans_centers, feature_names, top_n=10):
    rows = {}
    for k in range(kmeans_centers.shape[0]):
        idx = np.argsort(kmeans_centers[k])[::-1][:top_n]
        rows[k] = [feature_names[i] for i in idx]
    return rows


def coherence_proxy_umass(doc_term_binary_csr, term_to_idx, terms):
    # Simple UMass-style proxy over provided top terms.
    vals = []
    for i in range(1, len(terms)):
        wi = terms[i]
        if wi not in term_to_idx:
            continue
        wi_col = doc_term_binary_csr[:, term_to_idx[wi]]
        d_wi = wi_col.sum()
        for j in range(0, i):
            wj = terms[j]
            if wj not in term_to_idx:
                continue
            wj_col = doc_term_binary_csr[:, term_to_idx[wj]]
            d_wj = wj_col.sum()
            d_both = wi_col.multiply(wj_col).sum()
            vals.append(np.log((d_both + 1.0) / (d_wj + 1e-9)))
    if not vals:
        return np.nan
    return float(np.mean(vals))

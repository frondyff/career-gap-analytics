from __future__ import annotations

import re
import pandas as pd


MANUAL_LABELS = {
    # Optional manual overrides if cluster IDs are stable.
}


def _clean_term(term: str) -> str:
    return re.sub(r"\s+", " ", str(term).strip().lower())


def infer_cluster_label(row: pd.Series) -> str:
    cid = int(row.get("cluster", -1)) if pd.notna(row.get("cluster")) else -1
    if cid in MANUAL_LABELS:
        return MANUAL_LABELS[cid]

    terms = str(row.get("top_tfidf_terms", "")).split(",")
    terms = [_clean_term(t) for t in terms if str(t).strip()]
    terms = [t for t in terms if t and t not in {"data", "job", "analytics", "role", "work"}]

    if len(terms) >= 2:
        return f"{terms[0].title()} / {terms[1].title()}"
    if len(terms) == 1:
        return terms[0].title()
    return f"Cluster {cid}"


def add_cluster_labels(profile_df: pd.DataFrame) -> pd.DataFrame:
    out = profile_df.copy()
    if "cluster_label" not in out.columns:
        out["cluster_label"] = out.apply(infer_cluster_label, axis=1)
    return out


def cluster_display_map(profile_df: pd.DataFrame) -> dict[int, str]:
    d = {}
    for _, r in profile_df.iterrows():
        cid = int(r["cluster"])
        label = r.get("cluster_label", f"Cluster {cid}")
        pct = r.get("size_pct", None)
        if pd.notna(pct):
            d[cid] = f"{cid} - {label} ({pct:.1f}%)"
        else:
            d[cid] = f"{cid} - {label}"
    return d

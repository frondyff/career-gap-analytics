from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.constants import (
    DESIRABILITY_CLASSES,
    ENRICHED_CANDIDATES,
    HARD_LEXICON_CANDIDATES,
    MODEL_CANDIDATES,
    PROFILE_CANDIDATES,
    REQUIRED_POST_COLUMNS,
    REQUIRED_PROFILE_COLUMNS,
    SAMPLE_DATA,
    SOFT_LEXICON_CANDIDATES,
)
from src.labels import add_cluster_labels

POST_NUMERIC = [
    "cluster",
    "post_sentiment",
    "comment_sentiment",
    "delta_sentiment",
    "comments_count_observed",
    "hard_norm",
    "soft_norm",
    "desirability_index",
]
PROFILE_NUMERIC = [
    "cluster",
    "size",
    "size_pct",
    "avg_sent_post",
    "avg_sent_comment",
    "avg_hard_norm",
    "avg_soft_norm",
]
BASELINE_METRICS = [
    "comments_count_observed",
    "post_sentiment",
    "comment_sentiment",
    "delta_sentiment",
    "hard_norm",
    "soft_norm",
    "desirability_index",
]


@st.cache_data(show_spinner=False)
def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def _read_uploaded_table(content: bytes, name: str) -> pd.DataFrame:
    bio = io.BytesIO(content)
    if name.lower().endswith(".parquet"):
        return pd.read_parquet(bio)
    return pd.read_csv(bio)


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model(model_type: str):
    """Load model artifact once per session process."""
    for p in MODEL_CANDIDATES.get(model_type, []):
        if p.exists():
            return joblib.load(p), str(p)
    return None, None


@st.cache_data(show_spinner=False)
def compute_phrase_document_frequency(posts_df: pd.DataFrame) -> pd.DataFrame:
    """Approximate phrase frequency from available columns for lexicon diagnostics."""
    phrase_cols = [c for c in ["top_weighted_phrases", "doc_phrases"] if c in posts_df.columns]
    counts: dict[str, int] = {}
    if not phrase_cols:
        return pd.DataFrame(columns=["phrase", "doc_freq"])

    if "doc_phrases" in phrase_cols:
        series = posts_df["doc_phrases"].dropna()
        for item in series:
            if isinstance(item, list):
                seen = set(map(str, item))
            else:
                seen = set(x.strip() for x in str(item).split(",") if x.strip())
            for p in seen:
                counts[p] = counts.get(p, 0) + 1
    else:
        series = posts_df["top_weighted_phrases"].fillna("").astype(str)
        for raw in series:
            seen = set(x.strip() for x in raw.split(",") if x.strip())
            for p in seen:
                counts[p] = counts.get(p, 0) + 1

    return pd.DataFrame({"phrase": list(counts.keys()), "doc_freq": list(counts.values())})


def _find_first(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _ensure_posts_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    warnings: list[str] = []

    rename_map = {
        "id": "post_id",
        "title_std": "title",
        "body_std": "body",
        "post_text": "body",
    }
    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    if "post_id" not in out.columns:
        out["post_id"] = np.arange(len(out)).astype(str)
        warnings.append("Missing post_id/id column: generated synthetic post_id.")

    if "title" not in out.columns:
        out["title"] = ""
        warnings.append("Missing title column: created empty title.")
    if "body" not in out.columns:
        if "post_text" in out.columns:
            out["body"] = out["post_text"]
        else:
            out["body"] = ""
            warnings.append("Missing body/post_text column: created empty body.")

    if "comments_text" not in out.columns:
        out["comments_text"] = ""
        warnings.append("Missing comments_text column: created empty comments_text.")

    for c in REQUIRED_POST_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
            warnings.append(f"Missing required post column '{c}': filled with NaN.")

    if "subreddit" not in out.columns:
        out["subreddit"] = "unknown"

    if "desirability_class" not in out.columns:
        out["desirability_class"] = "balanced"
    out["desirability_class"] = out["desirability_class"].astype(str).str.strip().replace({"nan": "balanced"})

    if "created_utc" in out.columns and "created_date" not in out.columns:
        out["created_date"] = pd.to_datetime(out["created_utc"], unit="s", errors="coerce")
    elif "created_date" in out.columns:
        out["created_date"] = pd.to_datetime(out["created_date"], errors="coerce")

    out["title"] = out["title"].fillna("").astype(str)
    out["body"] = out["body"].fillna("").astype(str)
    out["comments_text"] = out["comments_text"].fillna("").astype(str)

    out = _coerce_numeric(out, POST_NUMERIC)
    out["delta_sentiment"] = out["delta_sentiment"].fillna(out["comment_sentiment"] - out["post_sentiment"])
    out["comments_count_observed"] = out["comments_count_observed"].fillna(0)

    out = out.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    return out, warnings


def _ensure_profile_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    warnings: list[str] = []

    rename_map = {
        "Cluster": "cluster",
        "Size": "size",
        "Size_pct": "size_pct",
        "Top TF-IDF Terms": "top_tfidf_terms",
        "Top Weighted Phrases": "top_weighted_phrases",
        "Avg_sent_post": "avg_sent_post",
        "Avg_sent_comment": "avg_sent_comment",
        "Avg_hard_norm": "avg_hard_norm",
        "Avg_soft_norm": "avg_soft_norm",
        "Dominant Class": "dominant_class",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    for c in REQUIRED_PROFILE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
            warnings.append(f"Missing required profile column '{c}': filled with NaN.")

    out = _coerce_numeric(out, PROFILE_NUMERIC)
    out["top_tfidf_terms"] = out["top_tfidf_terms"].fillna("")
    out["top_weighted_phrases"] = out["top_weighted_phrases"].fillna("")
    out["dominant_class"] = out["dominant_class"].fillna("balanced").astype(str)

    if out["size_pct"].isna().all() and out["size"].notna().any():
        total = out["size"].sum()
        out["size_pct"] = 100 * out["size"] / total if total else 0

    out = add_cluster_labels(out)
    return out, warnings


def _demo_posts_path() -> Path:
    return SAMPLE_DATA / "sample_final_enriched_posts.csv"


def _demo_profiles_path() -> Path:
    return SAMPLE_DATA / "sample_cluster_profiles.csv"


def load_enriched_posts(mode: str, uploaded: Any | None = None) -> tuple[pd.DataFrame, list[str], str | None]:
    warnings: list[str] = []

    if mode == "demo":
        p = _demo_posts_path()
        if not p.exists():
            return pd.DataFrame(), ["Demo file missing: sample_data/sample_final_enriched_posts.csv"], None
        raw = _read_table(str(p))
        used_path = str(p)
    elif mode == "upload":
        if uploaded is None:
            return pd.DataFrame(), ["Upload mode selected but no enriched posts file uploaded."], None
        raw = _read_uploaded_table(uploaded.getvalue(), uploaded.name)
        used_path = f"uploaded:{uploaded.name}"
    else:
        p = _find_first(ENRICHED_CANDIDATES)
        if p is None:
            return pd.DataFrame(), ["No enriched posts file found in artifacts_refined."], None
        raw = _read_table(str(p))
        used_path = str(p)

    out, schema_warnings = _ensure_posts_schema(raw)
    warnings.extend(schema_warnings)
    return out, warnings, used_path


def load_cluster_profiles(mode: str, uploaded: Any | None = None) -> tuple[pd.DataFrame, list[str], str | None]:
    warnings: list[str] = []

    if mode == "demo":
        p = _demo_profiles_path()
        if not p.exists():
            return pd.DataFrame(), ["Demo file missing: sample_data/sample_cluster_profiles.csv"], None
        raw = _read_table(str(p))
        used_path = str(p)
    elif mode == "upload":
        if uploaded is None:
            return pd.DataFrame(), ["Upload mode selected but no cluster profile file uploaded."], None
        raw = _read_uploaded_table(uploaded.getvalue(), uploaded.name)
        used_path = f"uploaded:{uploaded.name}"
    else:
        p = _find_first(PROFILE_CANDIDATES)
        if p is None:
            return pd.DataFrame(), ["No cluster profiles file found in artifacts_refined."], None
        raw = _read_table(str(p))
        used_path = str(p)

    out, schema_warnings = _ensure_profile_schema(raw)
    warnings.extend(schema_warnings)
    return out, warnings, used_path


def load_lexicon(paths: list[Path]) -> tuple[dict[str, float], str | None]:
    p = _find_first(paths)
    if p is None:
        return {}, None
    data = _read_json(str(p))
    if "weights" in data and isinstance(data["weights"], dict):
        return {str(k): float(v) for k, v in data["weights"].items()}, str(p)
    return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}, str(p)


def load_hard_lexicon() -> tuple[dict[str, float], str | None]:
    return load_lexicon(HARD_LEXICON_CANDIDATES)


def load_soft_lexicon() -> tuple[dict[str, float], str | None]:
    return load_lexicon(SOFT_LEXICON_CANDIDATES)


def merge_labels_into_posts(posts_df: pd.DataFrame, profiles_df: pd.DataFrame) -> pd.DataFrame:
    out = posts_df.copy()
    if profiles_df.empty or "cluster" not in out.columns:
        out["cluster_label"] = out.get("cluster", np.nan).apply(lambda x: f"Cluster {int(x)}" if pd.notna(x) else "Cluster")
        return out

    label_map = profiles_df.set_index("cluster")["cluster_label"].to_dict()
    out["cluster_label"] = out["cluster"].map(label_map)
    out["cluster_label"] = out["cluster_label"].fillna(out["cluster"].apply(lambda c: f"Cluster {int(c)}" if pd.notna(c) else "Cluster"))

    for c in ["top_tfidf_terms", "top_weighted_phrases", "dominant_class", "size_pct", "size"]:
        if c in profiles_df.columns and c not in out.columns:
            out = out.merge(profiles_df[["cluster", c]], on="cluster", how="left")
    return out


@st.cache_data(show_spinner=False)
def apply_filters_cached(
    df: pd.DataFrame,
    subreddits: tuple[str, ...],
    classes: tuple[str, ...],
    engagement_range: tuple[float, float],
    keyword: str,
    date_range: tuple[str, str] | None,
) -> pd.DataFrame:
    out = df

    if subreddits:
        out = out[out["subreddit"].isin(subreddits)]
    if classes:
        out = out[out["desirability_class"].isin(classes)]

    lo, hi = engagement_range
    ec = out["comments_count_observed"].fillna(0)
    out = out[(ec >= lo) & (ec <= hi)]

    if keyword:
        kw = keyword.lower().strip()
        blob = (out["title"].fillna("") + " " + out["body"].fillna("") + " " + out["comments_text"].fillna("")).str.lower()
        out = out[blob.str.contains(kw, na=False)]

    if date_range and "created_date" in out.columns and out["created_date"].notna().any():
        d0, d1 = date_range
        s = pd.to_datetime(d0, errors="coerce")
        e = pd.to_datetime(d1, errors="coerce")
        if pd.notna(s) and pd.notna(e):
            out = out[(out["created_date"] >= s) & (out["created_date"] <= e)]

    return out.copy()


@st.cache_data(show_spinner=False)
def compute_baseline_stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {k: 0.0 for k in BASELINE_METRICS}
    stats = {k: float(df[k].mean()) if k in df.columns else 0.0 for k in BASELINE_METRICS}
    stats["median_engagement"] = float(df["comments_count_observed"].median()) if "comments_count_observed" in df.columns else 0.0
    return stats


def sentiment_delta_label(delta: float) -> str:
    if pd.isna(delta):
        return "Tone shift unavailable"
    return "Community tone improves" if delta > 0 else "Community tone becomes more critical"


def framing_interpretation(value: float, baseline: float) -> str:
    diff = value - baseline
    lean = "Hard-leaning" if value > 0 else ("Soft-leaning" if value < 0 else "Balanced")
    direction = "above" if diff >= 0 else "below"
    return f"{lean} by {abs(diff):.3f} {direction} baseline"


def render_method_expander() -> None:
    with st.sidebar.expander("Method", expanded=False):
        st.markdown(
            """
- Extract POS-based phrases from posts.
- Use PMI orientation to build hard/soft phrase lexicons.
- Combine TF-IDF with lexicon features.
- Cluster using KMeans (optionally in reduced LSA space).
- Compare sentiment shifts and engagement across themes.
"""
        )


def render_sidebar_filters(posts_df: pd.DataFrame) -> dict[str, Any]:
    st.sidebar.header("Global Filters")

    subreddits_all = sorted(posts_df["subreddit"].dropna().astype(str).unique().tolist())
    classes_all = [c for c in DESIRABILITY_CLASSES if c in posts_df["desirability_class"].astype(str).unique().tolist()] or DESIRABILITY_CLASSES

    subreddits = st.sidebar.multiselect("Subreddit", options=subreddits_all, default=subreddits_all)
    classes = st.sidebar.multiselect("Dominant class", options=classes_all, default=classes_all)

    cmin = float(posts_df["comments_count_observed"].fillna(0).min()) if not posts_df.empty else 0.0
    cmax = float(posts_df["comments_count_observed"].fillna(0).max()) if not posts_df.empty else 0.0
    engagement = st.sidebar.slider("Engagement (observed comments)", min_value=float(cmin), max_value=float(cmax), value=(float(cmin), float(cmax)))

    keyword = st.sidebar.text_input("Search keyword", value="")

    date_range = None
    if "created_date" in posts_df.columns and posts_df["created_date"].notna().any():
        dmin = posts_df["created_date"].min().date()
        dmax = posts_df["created_date"].max().date()
        selected = st.sidebar.date_input("Date range", value=(dmin, dmax))
        if isinstance(selected, tuple) and len(selected) == 2:
            date_range = (str(selected[0]), str(selected[1]))

    baseline_mode = st.sidebar.radio("Baseline", options=["filtered", "full"], format_func=lambda x: "Baseline = filtered data" if x == "filtered" else "Baseline = full dataset")

    render_method_expander()

    return {
        "subreddits": tuple(subreddits),
        "classes": tuple(classes),
        "engagement": (float(engagement[0]), float(engagement[1])),
        "keyword": keyword,
        "date_range": date_range,
        "baseline_mode": baseline_mode,
    }


def active_filters_text(f: dict[str, Any]) -> str:
    s = ", ".join(f.get("subreddits", ())) or "all"
    c = ", ".join(f.get("classes", ())) or "all"
    e = f.get("engagement", (0, 0))
    k = f.get("keyword", "") or "none"
    d = f.get("date_range")
    d_text = f"{d[0]} to {d[1]}" if d else "all"
    b = f.get("baseline_mode", "filtered")
    return f"Active filters -> subreddits: {s} | class: {c} | engagement: {e[0]:.0f}-{e[1]:.0f} | keyword: {k} | date: {d_text} | baseline: {b}"


def _auto_mode_with_demo_fallback() -> tuple[str, str | None]:
    """Resolve local mode to demo if artifacts are unavailable."""
    local_posts = _find_first(ENRICHED_CANDIDATES)
    local_profile = _find_first(PROFILE_CANDIDATES)
    if local_posts is None or local_profile is None:
        return "demo", "Local artifacts missing. Switched to Demo mode."
    return "local", None


def initialize_state() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[str], dict[str, str | None]]:
    """Initialize data, filters, baseline stats, and sources for all pages."""
    mode_label_to_key = {
        "Demo (sample data)": "demo",
        "Load artifacts from local repo": "local",
        "Upload artifacts": "upload",
    }

    mode_label = st.sidebar.radio("Data source", options=list(mode_label_to_key.keys()), index=1, key="global_data_mode")
    mode = mode_label_to_key[mode_label]

    upload_posts = None
    upload_profiles = None
    if mode == "upload":
        upload_posts = st.sidebar.file_uploader("Upload enriched posts (.parquet/.csv)", type=["parquet", "csv"], key="global_upload_posts")
        upload_profiles = st.sidebar.file_uploader("Upload cluster profiles (.csv/.parquet)", type=["csv", "parquet"], key="global_upload_profiles")

    warnings: list[str] = []

    if mode == "local":
        resolved_mode, banner = _auto_mode_with_demo_fallback()
        if banner:
            warnings.append(banner)
        mode = resolved_mode

    posts_df, w_posts, src_posts = load_enriched_posts(mode, upload_posts)
    profiles_df, w_prof, src_prof = load_cluster_profiles(mode, upload_profiles)
    warnings.extend(w_posts)
    warnings.extend(w_prof)

    if not posts_df.empty:
        if profiles_df.empty:
            profiles_df = pd.DataFrame({"cluster": sorted(posts_df["cluster"].dropna().unique())})
            profiles_df = add_cluster_labels(profiles_df)
        posts_df = merge_labels_into_posts(posts_df, profiles_df)

        filters = render_sidebar_filters(posts_df)
        filtered = apply_filters_cached(
            posts_df,
            filters["subreddits"],
            filters["classes"],
            filters["engagement"],
            filters["keyword"],
            filters["date_range"],
        )

        baseline_df = filtered if filters.get("baseline_mode") == "filtered" else posts_df
        baseline_stats = compute_baseline_stats(baseline_df)
    else:
        filters = {
            "subreddits": tuple(),
            "classes": tuple(DESIRABILITY_CLASSES),
            "engagement": (0.0, 0.0),
            "keyword": "",
            "date_range": None,
            "baseline_mode": "filtered",
        }
        filtered = posts_df
        baseline_stats = compute_baseline_stats(posts_df)

    st.session_state["posts_df"] = posts_df
    st.session_state["profiles_df"] = profiles_df
    st.session_state["filtered_df"] = filtered
    st.session_state["filters"] = filters
    st.session_state["baseline_stats"] = baseline_stats
    st.session_state["data_mode"] = mode
    st.session_state["data_sources"] = {"posts": src_posts, "profiles": src_prof}

    return posts_df, profiles_df, filtered, filters, warnings, {"posts": src_posts, "profiles": src_prof}

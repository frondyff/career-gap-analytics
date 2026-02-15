from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.charts import framing_gauge
from src.data import (
    active_filters_text,
    compute_baseline_stats,
    framing_interpretation,
    initialize_state,
    sentiment_delta_label,
)

st.set_page_config(page_title="Cluster Explorer", layout="wide")
st.title("Cluster Explorer")

posts_df, profiles_df, filtered, filters, warnings, _ = initialize_state()
for w in warnings:
    st.warning(w)

if posts_df.empty:
    st.error("No usable data loaded.")
    st.stop()
if filtered.empty:
    st.warning("No rows match current filters.")
    st.stop()

st.caption(active_filters_text(filters))
baseline_df = filtered if filters.get("baseline_mode") == "filtered" else posts_df
baseline = compute_baseline_stats(baseline_df)

cluster_options = (
    filtered.groupby(["cluster", "cluster_label"], as_index=False)
    .size()
    .sort_values("size", ascending=False)
)
cluster_options["display"] = cluster_options.apply(lambda r: f"{int(r['cluster'])} - {r['cluster_label']} ({100*r['size']/len(filtered):.1f}%)", axis=1)

choice = st.selectbox("Select cluster", options=cluster_options["display"].tolist())
sel_cluster = int(cluster_options.loc[cluster_options["display"] == choice, "cluster"].iloc[0])
cdf = filtered[filtered["cluster"] == sel_cluster].copy()

if cdf.empty:
    st.warning("No records in selected cluster after filtering.")
    st.stop()

avg_post = cdf["post_sentiment"].mean()
avg_comment = cdf["comment_sentiment"].mean()
avg_delta = cdf["delta_sentiment"].mean()
mean_eng = cdf["comments_count_observed"].mean()
med_eng = cdf["comments_count_observed"].median()
avg_hard = cdf["hard_norm"].mean()
avg_soft = cdf["soft_norm"].mean()
avg_idx = cdf["desirability_index"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Size", f"{len(cdf)}", delta=f"{len(cdf)-len(baseline_df):+d} vs baseline size")
c2.metric("Size %", f"{100*len(cdf)/len(filtered):.1f}%")
c3.metric("Engagement (median)", f"{med_eng:.2f}", delta=f"{med_eng-baseline['median_engagement']:+.2f}")
c4.metric("Engagement (mean)", f"{mean_eng:.2f}", delta=f"{mean_eng-baseline['comments_count_observed']:+.2f}")

c5, c6, c7 = st.columns(3)
c5.metric("Post sentiment", f"{avg_post:.3f}", delta=f"{avg_post-baseline['post_sentiment']:+.3f}")
c6.metric("Comment sentiment", f"{avg_comment:.3f}", delta=f"{avg_comment-baseline['comment_sentiment']:+.3f}")
c7.metric("Delta sentiment", f"{avg_delta:.3f}", delta=f"{avg_delta-baseline['delta_sentiment']:+.3f}")

st.caption(f"{sentiment_delta_label(avg_delta)}. Sentiment is heuristic and Reddit-specific.")

dclass = cdf["desirability_class"].value_counts(normalize=True).mul(100).rename_axis("class").reset_index(name="pct")
st.subheader("Dominant class distribution")
st.plotly_chart(px.bar(dclass, x="class", y="pct", title="Hard / Soft / Balanced mix"), use_container_width=True)

st.subheader("Hard vs Soft framing")
g1, g2 = st.columns([2, 3])
with g1:
    st.plotly_chart(framing_gauge(avg_hard, avg_soft, title="Cluster framing"), use_container_width=True)
with g2:
    st.metric("Hard-skill framing", f"{avg_hard:.3f}", delta=f"{avg_hard-baseline['hard_norm']:+.3f}")
    st.metric("Soft-skill framing", f"{avg_soft:.3f}", delta=f"{avg_soft-baseline['soft_norm']:+.3f}")
    st.metric("Desirability index", f"{avg_idx:.3f}", delta=f"{avg_idx-baseline['desirability_index']:+.3f}")
    st.info(framing_interpretation(avg_idx, baseline['desirability_index']))

profile_row = profiles_df[profiles_df["cluster"] == sel_cluster].iloc[0] if not profiles_df.empty and (profiles_df["cluster"] == sel_cluster).any() else None

st.subheader("Top TF-IDF terms")
terms = str(profile_row.get("top_tfidf_terms", "")) if profile_row is not None else ""
terms = terms or (str(cdf["top_tfidf_terms"].dropna().iloc[0]) if "top_tfidf_terms" in cdf.columns and cdf["top_tfidf_terms"].notna().any() else "")
term_list = [t.strip() for t in terms.split(",") if t.strip()][:12]
if term_list:
    tdf = pd.DataFrame({"term": term_list, "rank": list(range(len(term_list), 0, -1))})
    st.plotly_chart(px.bar(tdf, x="term", y="rank", title="Top terms", labels={"rank": "relative prominence"}), use_container_width=True)
else:
    st.info("Top terms unavailable for this cluster.")

st.subheader("Top PMI-weighted phrases")
phrases = str(profile_row.get("top_weighted_phrases", "")) if profile_row is not None else ""
phrases = phrases or (str(cdf["top_weighted_phrases"].dropna().iloc[0]) if "top_weighted_phrases" in cdf.columns and cdf["top_weighted_phrases"].notna().any() else "")
st.write(" | ".join([p.strip() for p in phrases.split(",") if p.strip()][:18]) or "Top weighted phrases unavailable.")

st.subheader("Representative posts")
search_kw = st.text_input("Search within selected cluster", value="")
sort_opt = st.selectbox("Sort representative posts by", options=["engagement desc", "delta sentiment desc", "desirability_index desc"])
nearest_toggle = st.toggle("Show centroid-nearest posts", value=False)

rep = cdf.copy()
if search_kw.strip():
    blob = (rep["title"].fillna("") + " " + rep["body"].fillna("") + " " + rep["comments_text"].fillna("")).str.lower()
    rep = rep[blob.str.contains(search_kw.strip().lower(), na=False)]

if nearest_toggle:
    if "dist_to_centroid" in rep.columns:
        rep = rep.sort_values("dist_to_centroid", ascending=True)
    elif "tfidf_norm" in rep.columns:
        rep = rep.sort_values("tfidf_norm", ascending=False)
    else:
        rep["_proxy_norm"] = (rep["title"].fillna("") + " " + rep["body"].fillna("")).str.len()
        rep = rep.sort_values("_proxy_norm", ascending=False)
else:
    if sort_opt == "engagement desc":
        rep = rep.sort_values("comments_count_observed", ascending=False)
    elif sort_opt == "delta sentiment desc":
        rep = rep.sort_values("delta_sentiment", ascending=False)
    else:
        rep = rep.sort_values("desirability_index", ascending=False)

rep = rep.head(10).reset_index(drop=True)
show_cols = ["title", "subreddit", "comments_count_observed", "post_sentiment", "comment_sentiment", "desirability_class"]
show_cols = [c for c in show_cols if c in rep.columns]
st.dataframe(rep[show_cols], use_container_width=True)

for i, row in rep.iterrows():
    with st.expander(f"Post {i+1}: {str(row.get('title', ''))[:110]}"):
        st.write(f"**Subreddit:** {row.get('subreddit', 'n/a')} | **Engagement:** {row.get('comments_count_observed', 0)}")
        st.write("**Post text**")
        st.write(row.get("post_text_clean", row.get("body", "")))
        st.write("**Top comments snippet**")
        st.write(str(row.get("comments_text", ""))[:500])

# export
export_df = rep[[c for c in ["post_id", "title", "subreddit", "comments_count_observed", "delta_sentiment", "desirability_index", "desirability_class"] if c in rep.columns]].copy()
st.download_button("Export current representative posts CSV", export_df.to_csv(index=False).encode("utf-8"), "cluster_explorer_slice.csv", "text/csv")

cluster_profile_export = pd.DataFrame(
    [{
        "cluster": sel_cluster,
        "cluster_label": choice,
        "size": len(cdf),
        "size_pct": 100*len(cdf)/len(filtered),
        "mean_engagement": mean_eng,
        "median_engagement": med_eng,
        "avg_post_sentiment": avg_post,
        "avg_comment_sentiment": avg_comment,
        "avg_delta_sentiment": avg_delta,
        "avg_hard_norm": avg_hard,
        "avg_soft_norm": avg_soft,
        "avg_desirability_index": avg_idx,
    }]
)
st.download_button("Export current cluster profile CSV", cluster_profile_export.to_csv(index=False).encode("utf-8"), "cluster_profile_slice.csv", "text/csv")

st.subheader("What this means")
st.markdown(
    f"""
- We observe **{len(cdf)} posts** in this theme under current filters.
- Community tone is **{sentiment_delta_label(avg_delta).lower()}** ({avg_delta:+.3f}; {avg_delta-baseline['delta_sentiment']:+.3f} vs baseline).
- Cluster framing is **{framing_interpretation(avg_idx, baseline['desirability_index'])}**.
"""
)

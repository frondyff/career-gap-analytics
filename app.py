from __future__ import annotations

import streamlit as st

from src.charts import cluster_distribution_bar
from src.data import (
    active_filters_text,
    compute_baseline_stats,
    initialize_state,
    sentiment_delta_label,
)

st.set_page_config(page_title="Analytics Jobs Gap Explore", layout="wide")

st.title("Analytics Jobs Gap Explorer")
st.caption("Overview dashboard for cluster themes, desirability framing, and community response patterns.")

posts_df, profiles_df, filtered, filters, warnings, sources = initialize_state()
for w in warnings:
    st.warning(w)

if posts_df.empty:
    st.error("No usable data loaded. Generate artifacts or use Demo mode.")
    st.stop()

st.caption(active_filters_text(filters))
st.caption(f"Data sources -> posts: {sources.get('posts') or 'n/a'} | profiles: {sources.get('profiles') or 'n/a'}")

if filtered.empty:
    st.warning("No rows match current filters.")
    st.stop()

baseline_df = filtered if filters.get("baseline_mode") == "filtered" else posts_df
baseline_stats = compute_baseline_stats(baseline_df)

# KPI row with context deltas
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total posts", f"{len(filtered):,}", delta=f"{len(filtered)-len(baseline_df):+,.0f} vs baseline")
col2.metric("Subreddits", f"{filtered['subreddit'].nunique()}")
col3.metric("Clusters", f"{filtered['cluster'].nunique()}")
col4.metric("Median engagement", f"{filtered['comments_count_observed'].median():.1f}", delta=f"{filtered['comments_count_observed'].median()-baseline_stats['median_engagement']:+.2f}")
avg_delta = filtered["delta_sentiment"].mean()
col5.metric("Avg delta sentiment", f"{avg_delta:.3f}", delta=f"{avg_delta-baseline_stats['delta_sentiment']:+.3f}")

st.info(f"Delta sentiment interpretation: **{sentiment_delta_label(avg_delta)}**. Sentiment is heuristic and Reddit-specific.")

profile_view = (
    filtered.groupby(["cluster", "cluster_label"], as_index=False)
    .agg(
        size=("post_id", "size"),
        median_engagement=("comments_count_observed", "median"),
        avg_engagement=("comments_count_observed", "mean"),
        avg_desirability_index=("desirability_index", "mean"),
        avg_delta_sentiment=("delta_sentiment", "mean"),
    )
)
profile_view["size_pct"] = 100 * profile_view["size"] / len(filtered)

st.plotly_chart(cluster_distribution_bar(profile_view), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Top clusters by engagement")
    top_eng = profile_view.sort_values("median_engagement", ascending=False).head(5)
    st.dataframe(top_eng[["cluster_label", "size_pct", "median_engagement", "avg_delta_sentiment"]], use_container_width=True)
with right:
    st.subheader("Top clusters by hard-skill framing")
    top_hard = profile_view.sort_values("avg_desirability_index", ascending=False).head(5)
    st.dataframe(top_hard[["cluster_label", "size_pct", "avg_desirability_index", "avg_engagement"]], use_container_width=True)

st.download_button(
    "Export filtered cluster summary CSV",
    data=profile_view.to_csv(index=False).encode("utf-8"),
    file_name="overview_filtered_cluster_summary.csv",
    mime="text/csv",
)

most_engaged = profile_view.sort_values("median_engagement", ascending=False).iloc[0]
largest = profile_view.sort_values("size", ascending=False).iloc[0]
highest_shift = profile_view.sort_values("avg_delta_sentiment", ascending=False).iloc[0]

st.subheader("So what?")
st.markdown(
    f"""
- Most engaged theme is **{most_engaged['cluster_label']}** (median engagement: {most_engaged['median_engagement']:.2f}).
- Largest theme is **{largest['cluster_label']}** ({largest['size_pct']:.1f}% of filtered posts).
- Community tone shift is strongest in **{highest_shift['cluster_label']}** ({highest_shift['avg_delta_sentiment']:+.3f} vs baseline {highest_shift['avg_delta_sentiment']-baseline_stats['delta_sentiment']:+.3f}).
"""
)

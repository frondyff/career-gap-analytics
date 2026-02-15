from __future__ import annotations

import streamlit as st

from src.data import active_filters_text, compute_baseline_stats, initialize_state

st.set_page_config(page_title="Executive Summary", layout="wide")
st.title("Executive Summary")
st.caption("Auto-generated career market signals based on current filters.")

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

class_dist = filtered["desirability_class"].value_counts(normalize=True)
p_hard = 100 * class_dist.get("hard_dominant", 0.0)
p_soft = 100 * class_dist.get("soft_dominant", 0.0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total posts", f"{len(filtered):,}")
c2.metric("# Clusters", f"{filtered['cluster'].nunique()}")
c3.metric("Median engagement", f"{filtered['comments_count_observed'].median():.1f}", delta=f"{filtered['comments_count_observed'].median()-baseline['median_engagement']:+.2f}")
c4.metric("Avg delta sentiment", f"{filtered['delta_sentiment'].mean():.3f}", delta=f"{filtered['delta_sentiment'].mean()-baseline['delta_sentiment']:+.3f}")
c5.metric("% hard_dominant", f"{p_hard:.1f}%")
c6.metric("% soft_dominant", f"{p_soft:.1f}%")

cluster_stats = (
    filtered.groupby(["cluster", "cluster_label"], as_index=False)
    .agg(
        size=("post_id", "size"),
        median_engagement=("comments_count_observed", "median"),
        avg_delta_sentiment=("delta_sentiment", "mean"),
        avg_desirability_index=("desirability_index", "mean"),
    )
)
cluster_stats["size_pct"] = 100 * cluster_stats["size"] / len(filtered)

most_eng = cluster_stats.sort_values("median_engagement", ascending=False).iloc[0]
most_sup = cluster_stats.sort_values("avg_delta_sentiment", ascending=False).iloc[0]
most_hard = cluster_stats.sort_values("avg_desirability_index", ascending=False).iloc[0]

st.subheader("Top 3 Career Market Signals")
st.markdown(
    f"""
1. Most engaged theme: **{most_eng['cluster_label']}** (median engagement {most_eng['median_engagement']:.1f}).
2. Most supportive theme: **{most_sup['cluster_label']}** (avg delta sentiment {most_sup['avg_delta_sentiment']:+.3f}).
3. Most hard-skill framed theme: **{most_hard['cluster_label']}** (avg desirability index {most_hard['avg_desirability_index']:+.3f}).
"""
)

st.subheader("Theme leaderboard")
st.dataframe(
    cluster_stats[["cluster_label", "size_pct", "median_engagement", "avg_delta_sentiment", "avg_desirability_index"]].sort_values(
        ["median_engagement", "size_pct"], ascending=[False, False]
    ),
    use_container_width=True,
)

st.download_button(
    "Export leaderboard CSV",
    data=cluster_stats.to_csv(index=False).encode("utf-8"),
    file_name="executive_theme_leaderboard.csv",
    mime="text/csv",
)

st.subheader("So what?")
st.markdown(
    f"""
- **{most_eng['cluster_label']}** drives the strongest conversation intensity ({most_eng['median_engagement']:.1f} median comments, {most_eng['median_engagement']-baseline['median_engagement']:+.2f} vs baseline).
- **{most_sup['cluster_label']}** shows the most positive community tone shift ({most_sup['avg_delta_sentiment']:+.3f}).
- **{most_hard['cluster_label']}** is the clearest hard-skill signal in the filtered market view ({most_hard['avg_desirability_index']:+.3f}).
"""
)

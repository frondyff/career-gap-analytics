from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.charts import subreddit_metrics, subreddit_stacked_cluster
from src.data import active_filters_text, compute_baseline_stats, initialize_state

st.set_page_config(page_title="Subcommunity Compare", layout="wide")
st.title("Subcommunity Compare")

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

subs = sorted(filtered["subreddit"].dropna().unique().tolist())
selected = st.multiselect("Choose 2+ subreddits", options=subs, default=subs[: min(3, len(subs))])
if len(selected) < 2:
    st.warning("Select at least two subreddits.")
    st.stop()

sdf = filtered[filtered["subreddit"].isin(selected)].copy()
baseline_df = filtered if filters.get("baseline_mode") == "filtered" else posts_df
baseline = compute_baseline_stats(baseline_df)

c1, c2, c3 = st.columns(3)
as_percent = c1.toggle("Show % instead of counts", value=True)
show_all = c2.toggle("Show all clusters", value=False)
top_n = None if show_all else int(c3.slider("Top N clusters", min_value=3, max_value=12, value=6, step=1))

st.plotly_chart(subreddit_stacked_cluster(sdf, as_percent=as_percent, top_n_clusters=top_n), use_container_width=True)

metrics = subreddit_metrics(sdf)
cc1, cc2, cc3 = st.columns(3)
with cc1:
    st.plotly_chart(px.bar(metrics.sort_values("avg_desirability_index", ascending=False), x="subreddit", y="avg_desirability_index", title="Average Desirability Index"), use_container_width=True)
with cc2:
    st.plotly_chart(px.bar(metrics.sort_values("median_engagement", ascending=False), x="subreddit", y="median_engagement", title="Median Engagement"), use_container_width=True)
with cc3:
    st.plotly_chart(px.bar(metrics.sort_values("avg_delta_sentiment", ascending=False), x="subreddit", y="avg_delta_sentiment", title="Delta Sentiment"), use_container_width=True)

hard_sub = metrics.sort_values("avg_desirability_index", ascending=False).iloc[0]
eng_sub = metrics.sort_values("median_engagement", ascending=False).iloc[0]
tone_sub = metrics.sort_values("avg_delta_sentiment", ascending=False).iloc[0]

m1, m2, m3 = st.columns(3)
m1.metric("Highest hard framing subreddit", str(hard_sub["subreddit"]), delta=f"{hard_sub['avg_desirability_index']-baseline['desirability_index']:+.3f} vs baseline")
m2.metric("Highest median engagement subreddit", str(eng_sub["subreddit"]), delta=f"{eng_sub['median_engagement']-baseline['median_engagement']:+.1f} vs baseline")
m3.metric("Highest avg delta sentiment subreddit", str(tone_sub["subreddit"]), delta=f"{tone_sub['avg_delta_sentiment']-baseline['delta_sentiment']:+.3f} vs baseline")

st.download_button(
    "Export subreddit comparison CSV",
    metrics.to_csv(index=False).encode("utf-8"),
    "subcommunity_compare_metrics.csv",
    "text/csv",
)

st.subheader("Key differences")
st.markdown(
    f"""
- Highest hard framing subreddit: **{hard_sub['subreddit']}** ({hard_sub['avg_desirability_index']:+.3f}).
- Most engaged subreddit: **{eng_sub['subreddit']}** (median {eng_sub['median_engagement']:.1f}).
- Most supportive tone shift subreddit: **{tone_sub['subreddit']}** ({tone_sub['avg_delta_sentiment']:+.3f}).
"""
)

st.subheader("So what?")
st.markdown(
    """
- Subcommunity differences show where conversations skew technical vs interpersonal.
- Engagement disparities identify where themes attract stronger participation.
- Tone shift helps interpret whether communities respond with support or criticism.
"""
)
